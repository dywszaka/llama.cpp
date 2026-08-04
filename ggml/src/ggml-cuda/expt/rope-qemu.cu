#include "rope-qemu.cuh"
#include "rope-qemu-cuda.cuh"
#include "rope-qemu-protocol.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>

#if defined(GGML_CUDA_ROPE_QEMU)
#include <zmq.h>
#endif

static const char * ROPE_MODE_ENV = "GGML_CUDA_ROPE_QEMU_MODE";
static const char * ROPE_ENDPOINT_ENV = "GGML_CUDA_ROPE_QEMU_ENDPOINT";
static const char * ROPE_TIMEOUT_ENV = "GGML_CUDA_ROPE_QEMU_TIMEOUT_MS";
static const char * ROPE_TABLE_ENV = "GGML_CUDA_ROPE_QEMU_TABLE";
static const char * ROPE_ARTIFACT_ENV = "GGML_CUDA_ROPE_QEMU_ARTIFACT";
static const char * ROPE_MISMATCH_ENV = "GGML_CUDA_ROPE_QEMU_MISMATCH_LOG";
static const char * ROPE_TIMING_ENV = "GGML_CUDA_ROPE_QEMU_TIMING";

static const char * rope_mode_name(ggml_cuda_rope_qemu_mode mode) {
    switch (mode) {
        case GGML_CUDA_ROPE_QEMU_MODE_CUDA:      return "cuda";
        case GGML_CUDA_ROPE_QEMU_MODE_QEMU:      return "qemu";
        case GGML_CUDA_ROPE_QEMU_MODE_QEMU_CUDA: return "qemu_cuda";
        case GGML_CUDA_ROPE_QEMU_MODE_COMPARE:   return "compare";
    }
    return "cuda";
}

static ggml_cuda_rope_qemu_mode parse_mode(const char * value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "cuda") == 0) {
        return GGML_CUDA_ROPE_QEMU_MODE_CUDA;
    }
    if (std::strcmp(value, "qemu") == 0) {
        return GGML_CUDA_ROPE_QEMU_MODE_QEMU;
    }
    if (std::strcmp(value, "qemu_cuda") == 0) {
        return GGML_CUDA_ROPE_QEMU_MODE_QEMU_CUDA;
    }
    if (std::strcmp(value, "compare") == 0 ||
            std::strcmp(value, "compare_cuda") == 0 ||
            std::strcmp(value, "compare_qemu") == 0) {
        return GGML_CUDA_ROPE_QEMU_MODE_COMPARE;
    }
    GGML_LOG_WARN("%s: unknown %s=%s; using cuda\n", __func__, ROPE_MODE_ENV, value);
    return GGML_CUDA_ROPE_QEMU_MODE_CUDA;
}

ggml_cuda_rope_qemu_mode ggml_cuda_rope_qemu_get_mode() {
    static const ggml_cuda_rope_qemu_mode mode =
            parse_mode(std::getenv(ROPE_MODE_ENV));
    return mode;
}

bool ggml_cuda_rope_qemu_enabled() {
    return ggml_cuda_rope_qemu_get_mode() != GGML_CUDA_ROPE_QEMU_MODE_CUDA;
}

static bool float_bits_equal(float left, float right) {
    uint32_t left_bits = 0;
    uint32_t right_bits = 0;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

bool ggml_cuda_rope_qemu_supported(const ggml_cuda_rope_qemu_params & params) {
    if (!params.forward || params.freq_factors != nullptr || params.positions == nullptr ||
            (params.src0_type != GGML_TYPE_F32 && params.src0_type != GGML_TYPE_F16) ||
            params.dst_type != params.src0_type || params.ne[0] < 128 ||
            (params.ne[0] & 1) != 0 || params.ne[1] <= 0 ||
            params.ne[2] <= 0 || params.ne[3] <= 0 || params.n_dims != 128 ||
            params.mode != 2 || params.n_ctx_orig != 40960) {
        return false;
    }
    if (params.s0[0] != 1 || params.sd[0] != 1 ||
            params.sd[1] != params.ne[0] ||
            params.sd[2] != params.ne[0] * params.ne[1] ||
            params.sd[3] != params.ne[0] * params.ne[1] * params.ne[2] ||
            params.s0[3] != params.s0[2] * params.ne[2]) {
        return false;
    }
    if (!float_bits_equal(params.freq_base, 1000000.0f) ||
            !float_bits_equal(params.freq_scale, 1.0f) ||
            !float_bits_equal(params.ext_factor, 0.0f) ||
            !float_bits_equal(params.attn_factor, 1.0f) ||
            !float_bits_equal(params.beta_fast, 32.0f) ||
            !float_bits_equal(params.beta_slow, 1.0f)) {
        return false;
    }
    return params.sections[0] == 0 && params.sections[1] == 0 &&
            params.sections[2] == 0 && params.sections[3] == 0;
}

static size_t rope_elements(const ggml_cuda_rope_qemu_params & params) {
    return (size_t) params.ne[0] * (size_t) params.ne[1] *
            (size_t) params.ne[2] * (size_t) params.ne[3];
}

static size_t rope_positions(const ggml_cuda_rope_qemu_params & params) {
    return (size_t) params.ne[2] * (size_t) params.ne[3];
}

static std::string env_or_default(const char * name, const char * fallback) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : fallback;
}

static int rpc_timeout_ms() {
    const char * value = std::getenv(ROPE_TIMEOUT_ENV);
    if (value == nullptr || value[0] == '\0') {
        return 300000;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0 && parsed <= INT_MAX ?
            (int) parsed : 300000;
}

static bool timing_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv(ROPE_TIMING_ENV);
        return value != nullptr && value[0] != '\0' &&
                std::strcmp(value, "0") != 0 &&
                std::strcmp(value, "false") != 0 &&
                std::strcmp(value, "off") != 0;
    }();
    return enabled;
}

static void log_mode_once(ggml_cuda_rope_qemu_mode mode) {
    static std::atomic<bool> logged(false);
    if (mode == GGML_CUDA_ROPE_QEMU_MODE_CUDA || logged.exchange(true)) {
        return;
    }
    const std::string table = env_or_default(ROPE_TABLE_ENV,
            "/home/lerong.chen/0729-rope-node4/rope-cos-sin-f32.bin");
    if (mode == GGML_CUDA_ROPE_QEMU_MODE_QEMU_CUDA) {
        GGML_LOG_INFO("%s: %s=qemu_cuda enabled; static_table=%s, "
                "table is loaded to each CUDA device once, per-call path is device-only, "
                "canonical_input=BF16_RZ timing=%s\n", __func__, ROPE_MODE_ENV,
                table.c_str(), timing_enabled() ? "on" : "off");
    } else {
        GGML_LOG_INFO("%s: %s=%s enabled; endpoint=%s static_table=%s "
                "canonical_input=BF16_RZ downstream=%s timing=%s\n", __func__,
                ROPE_MODE_ENV, rope_mode_name(mode),
                env_or_default(ROPE_ENDPOINT_ENV, "tcp://127.0.0.1:15587").c_str(),
                table.c_str(), mode == GGML_CUDA_ROPE_QEMU_MODE_COMPARE ?
                        "llama_cuda" : "qemu",
                timing_enabled() ? "on" : "off");
    }
}

static uint64_t elapsed_ns(
        std::chrono::steady_clock::time_point begin,
        std::chrono::steady_clock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            end - begin).count();
}

static float event_ms(cudaEvent_t begin, cudaEvent_t end) {
    float value = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&value, begin, end));
    return value;
}

static float bf16_to_float(uint16_t value) {
    const uint32_t bits = (uint32_t) value << 16;
    float output = 0.0f;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

struct rope_qemu_result {
    std::vector<uint16_t> input;
    std::vector<int32_t> positions;
    std::vector<uint16_t> output;
    uint64_t request_id = 0;
    uint64_t daemon_ns = 0;
    uint64_t d2h_ns = 0;
    uint64_t rpc_ns = 0;
};

struct rope_metrics {
    double mse = 0.0;
    double rmse = 0.0;
    double max_abs = 0.0;
    size_t bit_mismatches = 0;
    size_t first_mismatch = 0;
};

static float native_value(
        const std::vector<unsigned char> & data,
        ggml_type type,
        size_t index) {
    if (type == GGML_TYPE_F32) {
        float value = 0.0f;
        std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
        return value;
    }
    ggml_fp16_t value = 0;
    std::memcpy(&value, data.data() + index * sizeof(value), sizeof(value));
    return ggml_fp16_to_fp32(value);
}

static rope_metrics compare_outputs(
        const std::vector<unsigned char> & native,
        ggml_type native_type,
        const std::vector<uint16_t> & qemu,
        const std::vector<uint16_t> & qemu_cuda) {
    GGML_ASSERT(qemu.size() == qemu_cuda.size());
    rope_metrics metrics;
    double sum_square = 0.0;
    for (size_t index = 0; index < qemu.size(); ++index) {
        if (qemu[index] != qemu_cuda[index]) {
            if (metrics.bit_mismatches == 0) {
                metrics.first_mismatch = index;
            }
            ++metrics.bit_mismatches;
        }
        const double difference = (double) native_value(native, native_type, index) -
                (double) bf16_to_float(qemu[index]);
        sum_square += difference * difference;
        metrics.max_abs = std::max(metrics.max_abs, std::abs(difference));
    }
    metrics.mse = qemu.empty() ? 0.0 : sum_square / (double) qemu.size();
    metrics.rmse = std::sqrt(metrics.mse);
    return metrics;
}

static bool ensure_parent(const std::filesystem::path & path) {
    const auto parent = path.parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code error;
    std::filesystem::create_directories(parent, error);
    if (error) {
        GGML_LOG_WARN("%s: failed to create %s: %s\n", __func__,
                parent.string().c_str(), error.message().c_str());
        return false;
    }
    return true;
}

static void write_bits(std::ofstream & stream, const std::vector<uint16_t> & values) {
    stream << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << "\"0x" << std::hex << std::setw(4) << std::setfill('0')
               << (unsigned int) values[index] << std::dec << '"';
    }
    stream << ']';
}

static void write_compare_artifact(
        const ggml_tensor * dst,
        const ggml_cuda_rope_qemu_params & params,
        const rope_qemu_result & qemu,
        const rope_metrics & metrics,
        float llama_ms,
        float qemu_cuda_ms) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const std::filesystem::path path(env_or_default(
            ROPE_ARTIFACT_ENV, "experiments/rope-qemu-compare.jsonl"));
    if (!ensure_parent(path)) {
        return;
    }
    std::ofstream output(path, std::ios::app);
    if (!output) {
        return;
    }
    output << "{\"op\":\"ROPE\",\"dst\":\"" << dst->name
           << "\",\"request\":" << qemu.request_id
           << ",\"shape\":[" << params.ne[0] << ',' << params.ne[1] << ','
           << params.ne[2] << ',' << params.ne[3] << ']'
           << ",\"position\":" << (qemu.positions.empty() ? -1 : qemu.positions[0])
           << ",\"llama_qemu_mse\":" << metrics.mse
           << ",\"llama_qemu_rmse\":" << metrics.rmse
           << ",\"llama_qemu_max_abs\":" << metrics.max_abs
           << ",\"qemu_qemu_cuda_bit_mismatches\":" << metrics.bit_mismatches
           << ",\"llama_cuda_ms\":" << llama_ms
           << ",\"qemu_cuda_ms\":" << qemu_cuda_ms
           << ",\"qemu_daemon_ms\":" << (double) qemu.daemon_ns / 1.0e6
           << "}\n";
}

static void write_mismatch(
        const ggml_tensor * dst,
        const rope_qemu_result & qemu,
        const std::vector<uint16_t> & qemu_cuda,
        const rope_metrics & metrics) {
    if (metrics.bit_mismatches == 0) {
        return;
    }
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const std::filesystem::path path(env_or_default(
            ROPE_MISMATCH_ENV, "experiments/rope-qemu-cuda-mismatch.jsonl"));
    if (!ensure_parent(path)) {
        return;
    }
    std::ofstream output(path, std::ios::app);
    if (!output) {
        return;
    }
    output << "{\"op\":\"ROPE\",\"dst\":\"" << dst->name
           << "\",\"request\":" << qemu.request_id
           << ",\"mismatches\":" << metrics.bit_mismatches
           << ",\"first_mismatch\":" << metrics.first_mismatch
           << ",\"positions\":[";
    for (size_t index = 0; index < qemu.positions.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << qemu.positions[index];
    }
    output << "],\"input_bf16\":";
    write_bits(output, qemu.input);
    output << ",\"qemu_output_bf16\":";
    write_bits(output, qemu.output);
    output << ",\"qemu_cuda_output_bf16\":";
    write_bits(output, qemu_cuda);
    output << "}\n";
    GGML_LOG_ERROR("ROPE QEMU/qemu_cuda mismatch request=%llu count=%zu first=%zu log=%s\n",
            (unsigned long long) qemu.request_id, metrics.bit_mismatches,
            metrics.first_mismatch, path.string().c_str());
}

#if defined(GGML_CUDA_ROPE_QEMU)

static void send_frame(void * socket, const void * data, size_t bytes, bool more) {
    if (zmq_send(socket, data, bytes, more ? ZMQ_SNDMORE : 0) < 0) {
        GGML_ABORT("%s: ZMQ send failed: %s\n", __func__, zmq_strerror(zmq_errno()));
    }
}

static std::vector<unsigned char> receive_frame(void * socket, bool * more) {
    zmq_msg_t message;
    zmq_msg_init(&message);
    if (zmq_msg_recv(&message, socket, 0) < 0) {
        const std::string error = zmq_strerror(zmq_errno());
        zmq_msg_close(&message);
        GGML_ABORT("%s: ZMQ receive failed: %s\n", __func__, error.c_str());
    }
    const auto * begin = (const unsigned char *) zmq_msg_data(&message);
    std::vector<unsigned char> output(begin, begin + zmq_msg_size(&message));
    int has_more = 0;
    size_t option_bytes = sizeof(has_more);
    zmq_getsockopt(socket, ZMQ_RCVMORE, &has_more, &option_bytes);
    *more = has_more != 0;
    zmq_msg_close(&message);
    return output;
}

static void * rope_rpc_socket() {
    static void * socket = [] {
        void * context = zmq_ctx_new();
        if (context == nullptr) {
            GGML_ABORT("%s: zmq_ctx_new failed\n", __func__);
        }
        void * created = zmq_socket(context, ZMQ_REQ);
        if (created == nullptr) {
            GGML_ABORT("%s: zmq_socket failed: %s\n", __func__, zmq_strerror(zmq_errno()));
        }
        const int timeout = rpc_timeout_ms();
        zmq_setsockopt(created, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
        zmq_setsockopt(created, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
        const std::string endpoint = env_or_default(
                ROPE_ENDPOINT_ENV, "tcp://127.0.0.1:15587");
        if (zmq_connect(created, endpoint.c_str()) != 0) {
            GGML_ABORT("%s: failed to connect %s: %s\n", __func__,
                    endpoint.c_str(), zmq_strerror(zmq_errno()));
        }
        return created;
    }();
    return socket;
}

static rope_qemu_result call_qemu_rpc(
        const ggml_cuda_rope_qemu_params & params,
        std::vector<uint16_t> input,
        std::vector<int32_t> positions) {
    static std::mutex mutex;
    static std::atomic<uint64_t> next_request(1);
    std::lock_guard<std::mutex> lock(mutex);

    rope_fp32_rpc_request_v1 request{};
    request.magic = ROPE_FP32_RPC_MAGIC;
    request.version = ROPE_FP32_RPC_VERSION;
    request.header_bytes = sizeof(request);
    request.request_id = next_request.fetch_add(1);
    request.flags = ROPE_FP32_RPC_CANONICAL_DENSE |
            ROPE_FP32_RPC_STATIC_F32_TABLE;
    request.src0_type = ROPE_FP32_RPC_DTYPE_BF16;
    request.pos_type = ROPE_FP32_RPC_DTYPE_I32;
    request.dst_type = ROPE_FP32_RPC_DTYPE_BF16;
    request.ne0 = params.ne[0];
    request.ne1 = params.ne[1];
    request.ne2 = params.ne[2];
    request.ne3 = params.ne[3];
    request.n_dims = params.n_dims;
    request.mode = params.mode;
    request.n_ctx_orig = params.n_ctx_orig;
    request.position_count = (uint32_t) positions.size();
    request.table_positions = ROPE_FP32_TABLE_POSITIONS;
    request.table_channels = ROPE_FP32_TABLE_CHANNELS;
    request.freq_base = params.freq_base;
    request.freq_scale = params.freq_scale;
    request.ext_factor = params.ext_factor;
    request.attn_factor = params.attn_factor;
    request.beta_fast = params.beta_fast;
    request.beta_slow = params.beta_slow;
    std::memcpy(request.sections, params.sections, sizeof(request.sections));
    request.src0_bytes = input.size() * sizeof(uint16_t);
    request.position_bytes = positions.size() * sizeof(int32_t);
    request.dst_bytes = request.src0_bytes;

    const auto begin = std::chrono::steady_clock::now();
    void * socket = rope_rpc_socket();
    send_frame(socket, &request, sizeof(request), true);
    send_frame(socket, input.data(), request.src0_bytes, true);
    send_frame(socket, positions.data(), request.position_bytes, false);
    bool more = false;
    const auto header_frame = receive_frame(socket, &more);
    if (!more || header_frame.size() != sizeof(rope_fp32_rpc_response_v1)) {
        GGML_ABORT("%s: invalid ROPE RPC response header\n", __func__);
    }
    rope_fp32_rpc_response_v1 response{};
    std::memcpy(&response, header_frame.data(), sizeof(response));
    const auto output_frame = receive_frame(socket, &more);
    const uint64_t rpc_ns = elapsed_ns(begin, std::chrono::steady_clock::now());
    if (more || response.magic != ROPE_FP32_RPC_MAGIC ||
            response.version != ROPE_FP32_RPC_VERSION ||
            response.header_bytes != sizeof(response) ||
            response.request_id != request.request_id ||
            response.status != ROPE_FP32_RPC_STATUS_OK ||
            response.output_bytes != request.dst_bytes ||
            output_frame.size() != request.dst_bytes) {
        GGML_ABORT("%s: ROPE RPC failed status=%u error=%u output=%llu expected=%llu\n",
                __func__, response.status, response.error_code,
                (unsigned long long) response.output_bytes,
                (unsigned long long) request.dst_bytes);
    }

    rope_qemu_result result;
    result.input = std::move(input);
    result.positions = std::move(positions);
    result.output.resize(output_frame.size() / sizeof(uint16_t));
    std::memcpy(result.output.data(), output_frame.data(), output_frame.size());
    result.request_id = response.request_id;
    result.daemon_ns = response.elapsed_ns;
    result.rpc_ns = rpc_ns;
    return result;
}

#endif

static rope_qemu_result run_rpc_from_device(
        const ggml_cuda_rope_qemu_params & params,
        const uint16_t * input_bf16,
        cudaStream_t stream) {
#if !defined(GGML_CUDA_ROPE_QEMU)
    GGML_UNUSED(params);
    GGML_UNUSED(input_bf16);
    GGML_UNUSED(stream);
    GGML_ABORT("%s: llama.cpp was built without GGML_CUDA_ROPE_QEMU=ON\n", __func__);
#else
    std::vector<uint16_t> input(rope_elements(params));
    std::vector<int32_t> positions(rope_positions(params));
    const auto begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(input.data(), input_bf16,
            input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(positions.data(), params.positions,
            positions.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const uint64_t d2h_ns = elapsed_ns(begin, std::chrono::steady_clock::now());
    rope_qemu_result result = call_qemu_rpc(
            params, std::move(input), std::move(positions));
    result.d2h_ns = d2h_ns;
    return result;
#endif
}

static void log_rpc_timing(
        const ggml_tensor * dst,
        const rope_qemu_result & result,
        uint64_t return_ns,
        ggml_cuda_rope_qemu_mode mode) {
    if (!timing_enabled()) {
        return;
    }
    GGML_LOG_INFO("RVV_ROPE_TIMING request=%llu mode=%s dst=%s d2h_ms=%.3f "
            "rpc_ms=%.3f daemon_ms=%.3f return_ms=%.3f\n",
            (unsigned long long) result.request_id, rope_mode_name(mode), dst->name,
            (double) result.d2h_ns / 1.0e6, (double) result.rpc_ns / 1.0e6,
            (double) result.daemon_ns / 1.0e6, (double) return_ns / 1.0e6);
}

static void run_qemu_only(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        const ggml_cuda_rope_qemu_params & params) {
    const size_t elements = rope_elements(params);
    ggml_cuda_pool_alloc<uint16_t> input(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> output(ctx.pool(), elements);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_rope_qemu_cuda_preprocess(params, input.get(), stream);
    rope_qemu_result result = run_rpc_from_device(params, input.get(), stream);
    const auto begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(output.get(), result.output.data(),
            elements * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
    ggml_cuda_rope_qemu_cuda_output(params, output.get(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const uint64_t return_ns = elapsed_ns(begin, std::chrono::steady_clock::now());
    log_rpc_timing(dst, result, return_ns, GGML_CUDA_ROPE_QEMU_MODE_QEMU);
}

static void run_qemu_cuda_only(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        const ggml_cuda_rope_qemu_params & params) {
    const size_t elements = rope_elements(params);
    ggml_cuda_pool_alloc<uint16_t> input(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> output(ctx.pool(), elements);
    cudaStream_t stream = ctx.stream();
    cudaEvent_t begin = nullptr;
    cudaEvent_t preprocessed = nullptr;
    cudaEvent_t operated = nullptr;
    cudaEvent_t finished = nullptr;
    if (timing_enabled()) {
        CUDA_CHECK(cudaEventCreate(&begin));
        CUDA_CHECK(cudaEventCreate(&preprocessed));
        CUDA_CHECK(cudaEventCreate(&operated));
        CUDA_CHECK(cudaEventCreate(&finished));
        CUDA_CHECK(cudaEventRecord(begin, stream));
    }
    ggml_cuda_rope_qemu_cuda_preprocess(params, input.get(), stream);
    if (timing_enabled()) CUDA_CHECK(cudaEventRecord(preprocessed, stream));
    ggml_cuda_rope_qemu_cuda_run_bf16(params, input.get(), output.get(), stream);
    if (timing_enabled()) CUDA_CHECK(cudaEventRecord(operated, stream));
    ggml_cuda_rope_qemu_cuda_output(params, output.get(), stream);
    if (timing_enabled()) {
        CUDA_CHECK(cudaEventRecord(finished, stream));
        CUDA_CHECK(cudaEventSynchronize(finished));
        GGML_LOG_INFO("QEMU_CUDA_ROPE_TIMING dst=%s preprocess_ms=%.3f "
                "operator_ms=%.3f output_ms=%.3f total_ms=%.3f\n", dst->name,
                event_ms(begin, preprocessed), event_ms(preprocessed, operated),
                event_ms(operated, finished), event_ms(begin, finished));
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(preprocessed));
        CUDA_CHECK(cudaEventDestroy(operated));
        CUDA_CHECK(cudaEventDestroy(finished));
    }
}

static void run_compare(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        const ggml_cuda_rope_qemu_params & params,
        ggml_cuda_rope_launch_fn cuda_launch) {
    const size_t elements = rope_elements(params);
    const size_t native_bytes = elements * ggml_type_size(params.dst_type);
    ggml_cuda_pool_alloc<unsigned char> native_output(ctx.pool(), native_bytes);
    ggml_cuda_pool_alloc<unsigned char> qemu_cuda_native(ctx.pool(), native_bytes);
    ggml_cuda_pool_alloc<uint16_t> input(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> qemu_cuda_output(ctx.pool(), elements);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_rope_qemu_params native_params = params;
    native_params.dst = native_output.get();
    ggml_cuda_rope_qemu_params qemu_cuda_params = params;
    qemu_cuda_params.dst = qemu_cuda_native.get();

    cudaEvent_t cuda_begin = nullptr;
    cudaEvent_t cuda_end = nullptr;
    cudaEvent_t model_begin = nullptr;
    cudaEvent_t model_end = nullptr;
    CUDA_CHECK(cudaEventCreate(&cuda_begin));
    CUDA_CHECK(cudaEventCreate(&cuda_end));
    CUDA_CHECK(cudaEventCreate(&model_begin));
    CUDA_CHECK(cudaEventCreate(&model_end));

    CUDA_CHECK(cudaEventRecord(cuda_begin, stream));
    cuda_launch(native_params, stream);
    CUDA_CHECK(cudaEventRecord(cuda_end, stream));
    CUDA_CHECK(cudaEventRecord(model_begin, stream));
    ggml_cuda_rope_qemu_cuda_preprocess(params, input.get(), stream);
    ggml_cuda_rope_qemu_cuda_run_bf16(
            params, input.get(), qemu_cuda_output.get(), stream);
    ggml_cuda_rope_qemu_cuda_output(
            qemu_cuda_params, qemu_cuda_output.get(), stream);
    CUDA_CHECK(cudaEventRecord(model_end, stream));

    std::vector<unsigned char> native_host(native_bytes);
    std::vector<uint16_t> input_host(elements);
    std::vector<uint16_t> qemu_cuda_host(elements);
    std::vector<int32_t> positions_host(rope_positions(params));
    CUDA_CHECK(cudaMemcpyAsync(native_host.data(), native_output.get(), native_bytes,
            cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(input_host.data(), input.get(), elements * sizeof(uint16_t),
            cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(qemu_cuda_host.data(), qemu_cuda_output.get(),
            elements * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(positions_host.data(), params.positions,
            positions_host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

#if defined(GGML_CUDA_ROPE_QEMU)
    rope_qemu_result qemu = call_qemu_rpc(
            params, std::move(input_host), std::move(positions_host));
#else
    rope_qemu_result qemu;
    GGML_ABORT("%s: compare requires GGML_CUDA_ROPE_QEMU=ON\n", __func__);
#endif
    const rope_metrics metrics = compare_outputs(
            native_host, params.dst_type, qemu.output, qemu_cuda_host);
    write_compare_artifact(dst, params, qemu, metrics,
            event_ms(cuda_begin, cuda_end), event_ms(model_begin, model_end));
    write_mismatch(dst, qemu, qemu_cuda_host, metrics);

    CUDA_CHECK(cudaMemcpyAsync(params.dst, native_output.get(), native_bytes,
            cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    log_rpc_timing(dst, qemu, 0, GGML_CUDA_ROPE_QEMU_MODE_COMPARE);
    if (timing_enabled()) {
        GGML_LOG_INFO("LLAMA_CUDA_ROPE_TIMING dst=%s total_ms=%.3f\n",
                dst->name, event_ms(cuda_begin, cuda_end));
    }
    CUDA_CHECK(cudaEventDestroy(cuda_begin));
    CUDA_CHECK(cudaEventDestroy(cuda_end));
    CUDA_CHECK(cudaEventDestroy(model_begin));
    CUDA_CHECK(cudaEventDestroy(model_end));
}

void ggml_cuda_rope_qemu_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rope_qemu_params & params,
        ggml_cuda_rope_launch_fn cuda_launch) {
    const ggml_cuda_rope_qemu_mode mode = ggml_cuda_rope_qemu_get_mode();
    log_mode_once(mode);
    switch (mode) {
        case GGML_CUDA_ROPE_QEMU_MODE_QEMU:
            run_qemu_only(ctx, dst_tensor, params);
            return;
        case GGML_CUDA_ROPE_QEMU_MODE_QEMU_CUDA:
            run_qemu_cuda_only(ctx, dst_tensor, params);
            return;
        case GGML_CUDA_ROPE_QEMU_MODE_COMPARE:
            run_compare(ctx, dst_tensor, params, cuda_launch);
            return;
        case GGML_CUDA_ROPE_QEMU_MODE_CUDA:
            GGML_ABORT("%s: CUDA mode must use original ROPE dispatch\n", __func__);
    }
}
