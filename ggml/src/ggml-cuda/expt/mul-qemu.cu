#include "mul-qemu.cuh"
#include "mul-qemu-cuda.cuh"
#include "mul-qemu-protocol.h"

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
#include <thread>
#include <vector>

#if defined(GGML_CUDA_MUL_QEMU)
#include <zmq.h>
#endif

static const char * GGML_CUDA_MUL_QEMU_MODE_ENV = "GGML_CUDA_MUL_QEMU_MODE";
static const char * GGML_CUDA_MUL_QEMU_ENDPOINT_ENV = "GGML_CUDA_MUL_QEMU_ENDPOINT";
static const char * GGML_CUDA_MUL_QEMU_TIMEOUT_ENV = "GGML_CUDA_MUL_QEMU_TIMEOUT_MS";
static const char * GGML_CUDA_MUL_QEMU_ARTIFACT_ENV = "GGML_CUDA_MUL_QEMU_ARTIFACT";
static const char * GGML_CUDA_MUL_QEMU_MISMATCH_LOG_ENV = "GGML_CUDA_MUL_QEMU_MISMATCH_LOG";
static const char * GGML_CUDA_MUL_QEMU_TIMING_ENV = "GGML_CUDA_MUL_QEMU_TIMING";
static const char * GGML_CUDA_MUL_QEMU_DEFAULT_ENDPOINT = "tcp://127.0.0.1:15582";
static const char * GGML_CUDA_MUL_QEMU_DEFAULT_ARTIFACT =
        "experiments/mul-qemu-compare.jsonl";
static const char * GGML_CUDA_MUL_QEMU_DEFAULT_MISMATCH_LOG =
        "experiments/mul-qemu-cuda-mismatch.jsonl";

static const char * ggml_cuda_mul_qemu_mode_name(ggml_cuda_mul_qemu_mode mode) {
    switch (mode) {
        case GGML_CUDA_MUL_QEMU_MODE_CUDA:      return "cuda";
        case GGML_CUDA_MUL_QEMU_MODE_QEMU:      return "qemu";
        case GGML_CUDA_MUL_QEMU_MODE_QEMU_CUDA: return "qemu_cuda";
        case GGML_CUDA_MUL_QEMU_MODE_COMPARE:   return "compare";
    }
    return "cuda";
}

static ggml_cuda_mul_qemu_mode parse_mode(const char * value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "cuda") == 0) {
        return GGML_CUDA_MUL_QEMU_MODE_CUDA;
    }
    if (std::strcmp(value, "qemu") == 0) {
        return GGML_CUDA_MUL_QEMU_MODE_QEMU;
    }
    if (std::strcmp(value, "qemu_cuda") == 0) {
        return GGML_CUDA_MUL_QEMU_MODE_QEMU_CUDA;
    }
    if (std::strcmp(value, "compare") == 0 ||
            std::strcmp(value, "compare_cuda") == 0 ||
            std::strcmp(value, "compare_qemu") == 0) {
        return GGML_CUDA_MUL_QEMU_MODE_COMPARE;
    }
    GGML_LOG_WARN("%s: unknown %s=%s; using cuda\n",
            __func__, GGML_CUDA_MUL_QEMU_MODE_ENV, value);
    return GGML_CUDA_MUL_QEMU_MODE_CUDA;
}

ggml_cuda_mul_qemu_mode ggml_cuda_mul_qemu_get_mode() {
    static const ggml_cuda_mul_qemu_mode mode =
            parse_mode(std::getenv(GGML_CUDA_MUL_QEMU_MODE_ENV));
    return mode;
}

bool ggml_cuda_mul_qemu_enabled() {
    return ggml_cuda_mul_qemu_get_mode() != GGML_CUDA_MUL_QEMU_MODE_CUDA;
}

static std::string rpc_endpoint() {
    const char * value = std::getenv(GGML_CUDA_MUL_QEMU_ENDPOINT_ENV);
    return value != nullptr && value[0] != '\0' ? value :
            GGML_CUDA_MUL_QEMU_DEFAULT_ENDPOINT;
}

static int rpc_timeout_ms() {
    const char * value = std::getenv(GGML_CUDA_MUL_QEMU_TIMEOUT_ENV);
    if (value == nullptr || value[0] == '\0') {
        return 300000;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0 && parsed <= INT_MAX ?
            (int) parsed : 300000;
}

static std::string artifact_path() {
    const char * value = std::getenv(GGML_CUDA_MUL_QEMU_ARTIFACT_ENV);
    return value != nullptr && value[0] != '\0' ? value :
            GGML_CUDA_MUL_QEMU_DEFAULT_ARTIFACT;
}

static std::string mismatch_log_path() {
    const char * value = std::getenv(GGML_CUDA_MUL_QEMU_MISMATCH_LOG_ENV);
    return value != nullptr && value[0] != '\0' ? value :
            GGML_CUDA_MUL_QEMU_DEFAULT_MISMATCH_LOG;
}

static bool timing_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv(GGML_CUDA_MUL_QEMU_TIMING_ENV);
        return value != nullptr && value[0] != '\0' &&
                std::strcmp(value, "0") != 0 &&
                std::strcmp(value, "false") != 0 &&
                std::strcmp(value, "off") != 0;
    }();
    return enabled;
}

static void log_mode_once(ggml_cuda_mul_qemu_mode mode) {
    static std::atomic<bool> logged(false);
    if (mode == GGML_CUDA_MUL_QEMU_MODE_CUDA || logged.exchange(true)) {
        return;
    }
    if (mode == GGML_CUDA_MUL_QEMU_MODE_QEMU_CUDA) {
        GGML_LOG_INFO(
                "%s: %s=qemu_cuda enabled; BF16 MUL stays on the CUDA device, "
                "canonical_input=RZ, ZMQ/D2H/H2D are not used, timing=%s\n",
                __func__, GGML_CUDA_MUL_QEMU_MODE_ENV,
                timing_enabled() ? "on" : "off");
        return;
    }
    const std::string endpoint = rpc_endpoint();
    if (mode == GGML_CUDA_MUL_QEMU_MODE_COMPARE) {
        GGML_LOG_INFO(
                "%s: %s=compare enabled; downstream=llama CUDA, endpoint=%s, "
                "canonical_input=RZ, timing=%s, artifact=%s, mismatch_log=%s\n",
                __func__, GGML_CUDA_MUL_QEMU_MODE_ENV, endpoint.c_str(),
                timing_enabled() ? "on" : "off", artifact_path().c_str(),
                mismatch_log_path().c_str());
        return;
    }
    GGML_LOG_INFO(
            "%s: %s=qemu enabled; BF16 RVV MUL endpoint=%s, "
            "canonical_input=RZ, timing=%s\n",
            __func__, GGML_CUDA_MUL_QEMU_MODE_ENV, endpoint.c_str(),
            timing_enabled() ? "on" : "off");
}

static size_t mul_elements(const ggml_cuda_mul_qemu_params & params) {
    return (size_t) params.ne[0] * (size_t) params.ne[1] *
            (size_t) params.ne[2] * (size_t) params.ne[3];
}

static uint64_t elapsed_ns(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point finish) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            finish - start).count();
}

static float event_elapsed_ms(cudaEvent_t start, cudaEvent_t finish) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, finish));
    return milliseconds;
}

static float bf16_to_float(uint16_t value) {
    const uint32_t bits = (uint32_t) value << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static std::string json_escape(const char * value) {
    std::string escaped;
    for (const char * current = value; *current != '\0'; ++current) {
        switch (*current) {
            case '\\': escaped += "\\\\"; break;
            case '"':  escaped += "\\\""; break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:   escaped += *current; break;
        }
    }
    return escaped;
}

static bool create_parent_directories(const std::filesystem::path & path) {
    const std::filesystem::path parent = path.parent_path();
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

static void write_bf16_array(
        std::ofstream & output,
        const std::vector<uint16_t> & values) {
    output << '[';
    for (size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << "\"0x" << std::hex << std::setw(4) << std::setfill('0')
               << (unsigned int) values[index] << std::dec << '"';
    }
    output << ']';
}

struct mul_compare_metrics {
    double mse = 0.0;
    double rmse = 0.0;
    double max_abs = 0.0;
    size_t bit_mismatches = 0;
    size_t first_mismatch = 0;
};

struct mul_qemu_result {
    std::vector<uint16_t> src0_values;
    std::vector<uint16_t> src1_values;
    std::vector<uint16_t> output_values;
    uint64_t request_id = 0;
    uint64_t daemon_elapsed_ns = 0;
    uint64_t d2h_ns = 0;
    uint64_t rpc_roundtrip_ns = 0;
    uint64_t external_total_ns = 0;
};

struct mul_qemu_cuda_timing {
    float preprocess_ms = 0.0f;
    float operator_ms = 0.0f;
    float output_ms = 0.0f;
    float total_ms = 0.0f;
};

static float native_output_value(
        const std::vector<unsigned char> & values,
        ggml_type type,
        size_t index) {
    if (type == GGML_TYPE_F32) {
        float value = 0.0f;
        std::memcpy(&value, values.data() + index * sizeof(value), sizeof(value));
        return value;
    }
    ggml_fp16_t value = 0;
    std::memcpy(&value, values.data() + index * sizeof(value), sizeof(value));
    return ggml_fp16_to_fp32(value);
}

static mul_compare_metrics compare_outputs(
        const std::vector<unsigned char> & native,
        ggml_type native_type,
        const std::vector<uint16_t> & qemu,
        const std::vector<uint16_t> & qemu_cuda) {
    GGML_ASSERT(qemu.size() == qemu_cuda.size());
    mul_compare_metrics metrics;
    double sum_squared = 0.0;
    for (size_t index = 0; index < qemu.size(); ++index) {
        if (qemu[index] != qemu_cuda[index]) {
            if (metrics.bit_mismatches == 0) {
                metrics.first_mismatch = index;
            }
            ++metrics.bit_mismatches;
        }
        const double difference = (double) native_output_value(native, native_type, index) -
                (double) bf16_to_float(qemu[index]);
        sum_squared += difference * difference;
        metrics.max_abs = std::max(metrics.max_abs, std::abs(difference));
    }
    metrics.mse = qemu.empty() ? 0.0 : sum_squared / (double) qemu.size();
    metrics.rmse = std::sqrt(metrics.mse);
    return metrics;
}

static void write_comparison_artifact(
        const ggml_tensor * dst_tensor,
        const ggml_cuda_mul_qemu_params & params,
        const mul_qemu_result & qemu,
        const mul_compare_metrics & metrics,
        float llama_cuda_ms,
        const mul_qemu_cuda_timing & qemu_cuda_timing) {
    static std::mutex artifact_mutex;
    std::lock_guard<std::mutex> lock(artifact_mutex);
    const std::filesystem::path path(artifact_path());
    if (!create_parent_directories(path)) {
        return;
    }
    std::ofstream output(path, std::ios::app);
    if (!output) {
        GGML_LOG_WARN("%s: failed to open %s\n", __func__, path.string().c_str());
        return;
    }
    const char * tensor_name = dst_tensor->name[0] != '\0' ?
            dst_tensor->name : "(unnamed)";
    output << "{\"op\":\"" << json_escape(ggml_op_desc(dst_tensor))
           << "\",\"dst\":\"" << json_escape(tensor_name)
           << "\",\"request\":" << qemu.request_id
           << ",\"shape\":[" << params.ne[0] << ',' << params.ne[1] << ','
           << params.ne[2] << ',' << params.ne[3] << ']'
           << ",\"src0_type\":\"" << ggml_type_name(params.src0_type) << '"'
           << ",\"src1_type\":\"" << ggml_type_name(params.src1_type) << '"'
           << ",\"dst_type\":\"" << ggml_type_name(params.dst_type) << '"'
           << ",\"llama_qemu_mse\":" << metrics.mse
           << ",\"llama_qemu_rmse\":" << metrics.rmse
           << ",\"llama_qemu_max_abs\":" << metrics.max_abs
           << ",\"qemu_qemu_cuda_bit_mismatches\":" << metrics.bit_mismatches
           << ",\"qemu_elapsed_ms\":" << (double) qemu.daemon_elapsed_ns / 1.0e6
           << ",\"llama_cuda_ms\":" << llama_cuda_ms
           << ",\"qemu_cuda_ms\":" << qemu_cuda_timing.total_ms
           << "}\n";
}

static void write_mismatch_log(
        const ggml_tensor * dst_tensor,
        const ggml_cuda_mul_qemu_params & params,
        const mul_qemu_result & qemu,
        const std::vector<uint16_t> & qemu_cuda,
        const mul_compare_metrics & metrics) {
    if (metrics.bit_mismatches == 0) {
        return;
    }
    static std::mutex mismatch_mutex;
    std::lock_guard<std::mutex> lock(mismatch_mutex);
    const std::filesystem::path path(mismatch_log_path());
    if (!create_parent_directories(path)) {
        return;
    }
    std::ofstream output(path, std::ios::app);
    if (!output) {
        GGML_LOG_WARN("%s: failed to open %s\n", __func__, path.string().c_str());
        return;
    }
    const char * tensor_name = dst_tensor->name[0] != '\0' ?
            dst_tensor->name : "(unnamed)";
    output << "{\"op\":\"" << json_escape(ggml_op_desc(dst_tensor))
           << "\",\"dst\":\"" << json_escape(tensor_name)
           << "\",\"request\":" << qemu.request_id
           << ",\"shape\":[" << params.ne[0] << ',' << params.ne[1] << ','
           << params.ne[2] << ',' << params.ne[3] << ']'
           << ",\"mismatches\":" << metrics.bit_mismatches
           << ",\"first_mismatch\":" << metrics.first_mismatch
           << ",\"src0_bf16\":";
    write_bf16_array(output, qemu.src0_values);
    output << ",\"src1_bf16\":";
    write_bf16_array(output, qemu.src1_values);
    output << ",\"qemu_output_bf16\":";
    write_bf16_array(output, qemu.output_values);
    output << ",\"qemu_cuda_output_bf16\":";
    write_bf16_array(output, qemu_cuda);
    output << "}\n";
    GGML_LOG_ERROR(
            "QEMU_CUDA_MUL_BIT_MISMATCH request=%llu dst=%s mismatches=%zu "
            "first=%zu log=%s\n",
            (unsigned long long) qemu.request_id, tensor_name,
            metrics.bit_mismatches, metrics.first_mismatch,
            path.string().c_str());
}

static void log_qemu_timing(
        const ggml_tensor * dst_tensor,
        const mul_qemu_result & result,
        size_t elements,
        uint64_t return_copy_ns,
        ggml_cuda_mul_qemu_mode mode) {
    if (!timing_enabled()) {
        return;
    }
    const char * tensor_name = dst_tensor->name[0] != '\0' ?
            dst_tensor->name : "(unnamed)";
    GGML_LOG_INFO(
            "RVV_MUL_TIMING request=%llu mode=%s dst=%s elements=%zu "
            "d2h_ms=%.3f rpc_roundtrip_ms=%.3f daemon_request_ms=%.3f "
            "return_copy_ms=%.3f total_ms=%.3f\n",
            (unsigned long long) result.request_id,
            ggml_cuda_mul_qemu_mode_name(mode), tensor_name, elements,
            (double) result.d2h_ns / 1.0e6,
            (double) result.rpc_roundtrip_ns / 1.0e6,
            (double) result.daemon_elapsed_ns / 1.0e6,
            (double) return_copy_ns / 1.0e6,
            (double) (result.external_total_ns + return_copy_ns) / 1.0e6);
}

static void log_qemu_cuda_timing(
        const ggml_tensor * dst_tensor,
        const ggml_cuda_mul_qemu_params & params,
        const mul_qemu_cuda_timing & timing,
        ggml_cuda_mul_qemu_mode mode) {
    if (!timing_enabled()) {
        return;
    }
    static std::mutex stats_mutex;
    static uint64_t calls = 0;
    static double cumulative_ms = 0.0;
    uint64_t current_calls = 0;
    double average_ms = 0.0;
    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        ++calls;
        cumulative_ms += timing.total_ms;
        current_calls = calls;
        average_ms = cumulative_ms / (double) calls;
    }
    const char * tensor_name = dst_tensor->name[0] != '\0' ?
            dst_tensor->name : "(unnamed)";
    GGML_LOG_INFO(
            "QEMU_CUDA_MUL_TIMING mode=%s dst=%s shape=%lldx%lldx%lldx%lld "
            "preprocess_ms=%.3f operator_ms=%.3f output_ms=%.3f total_ms=%.3f "
            "calls=%llu average_total_ms=%.3f\n",
            ggml_cuda_mul_qemu_mode_name(mode), tensor_name,
            (long long) params.ne[0], (long long) params.ne[1],
            (long long) params.ne[2], (long long) params.ne[3],
            timing.preprocess_ms, timing.operator_ms, timing.output_ms,
            timing.total_ms, (unsigned long long) current_calls, average_ms);
}

#if defined(GGML_CUDA_MUL_QEMU)

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
    std::vector<unsigned char> result(begin, begin + zmq_msg_size(&message));
    int has_more = 0;
    size_t option_bytes = sizeof(has_more);
    zmq_getsockopt(socket, ZMQ_RCVMORE, &has_more, &option_bytes);
    *more = has_more != 0;
    zmq_msg_close(&message);
    return result;
}

static void * mul_rpc_socket() {
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
        const std::string endpoint = rpc_endpoint();
        if (zmq_connect(created, endpoint.c_str()) != 0) {
            GGML_ABORT("%s: failed to connect %s: %s\n", __func__,
                    endpoint.c_str(), zmq_strerror(zmq_errno()));
        }
        return created;
    }();
    return socket;
}

static mul_qemu_result call_qemu_rpc(
        const ggml_cuda_mul_qemu_params & params,
        std::vector<uint16_t> src0,
        std::vector<uint16_t> src1) {
    static std::mutex rpc_mutex;
    static std::atomic<uint64_t> next_request_id(1);
    std::lock_guard<std::mutex> lock(rpc_mutex);

    mul_rpc_request_v1 request = {};
    request.magic = MUL_RPC_MAGIC;
    request.version = MUL_RPC_VERSION;
    request.header_bytes = sizeof(request);
    request.request_id = next_request_id.fetch_add(1);
    request.flags = MUL_RPC_REQUEST_CANONICAL_DENSE;
    request.src0_type = MUL_RPC_DTYPE_BF16;
    request.src1_type = MUL_RPC_DTYPE_BF16;
    request.dst_type = MUL_RPC_DTYPE_BF16;
    request.ne0 = params.ne[0];
    request.ne1 = params.ne[1];
    request.ne2 = params.ne[2];
    request.ne3 = params.ne[3];
    request.src0_bytes = src0.size() * sizeof(uint16_t);
    request.src1_bytes = src1.size() * sizeof(uint16_t);
    request.dst_bytes = request.src0_bytes;

    const auto rpc_start = std::chrono::steady_clock::now();
    void * socket = mul_rpc_socket();
    send_frame(socket, &request, sizeof(request), true);
    send_frame(socket, src0.data(), request.src0_bytes, true);
    send_frame(socket, src1.data(), request.src1_bytes, false);

    bool more = false;
    const std::vector<unsigned char> response_frame = receive_frame(socket, &more);
    if (!more || response_frame.size() != sizeof(mul_rpc_response_v1)) {
        GGML_ABORT("%s: invalid QEMU RPC response header\n", __func__);
    }
    mul_rpc_response_v1 response = {};
    std::memcpy(&response, response_frame.data(), sizeof(response));
    const std::vector<unsigned char> output_frame = receive_frame(socket, &more);
    const uint64_t rpc_ns = elapsed_ns(rpc_start, std::chrono::steady_clock::now());
    if (more || response.magic != MUL_RPC_MAGIC ||
            response.version != MUL_RPC_VERSION ||
            response.header_bytes != sizeof(response) ||
            response.request_id != request.request_id ||
            response.status != MUL_RPC_STATUS_OK ||
            response.output_bytes != request.dst_bytes ||
            output_frame.size() != request.dst_bytes) {
        GGML_ABORT(
                "%s: MUL RPC failed status=%u error=%u output=%llu expected=%llu\n",
                __func__, response.status, response.error_code,
                (unsigned long long) response.output_bytes,
                (unsigned long long) request.dst_bytes);
    }

    mul_qemu_result result;
    result.src0_values = std::move(src0);
    result.src1_values = std::move(src1);
    result.output_values.resize(output_frame.size() / sizeof(uint16_t));
    std::memcpy(result.output_values.data(), output_frame.data(), output_frame.size());
    result.request_id = response.request_id;
    result.daemon_elapsed_ns = response.elapsed_ns;
    result.rpc_roundtrip_ns = rpc_ns;
    return result;
}

#endif

static mul_qemu_result run_qemu_rpc(
        const ggml_cuda_mul_qemu_params & params,
        const uint16_t * src0_bf16,
        const uint16_t * src1_bf16,
        size_t elements,
        cudaStream_t stream) {
#if !defined(GGML_CUDA_MUL_QEMU)
    GGML_UNUSED(params);
    GGML_UNUSED(src0_bf16);
    GGML_UNUSED(src1_bf16);
    GGML_UNUSED(elements);
    GGML_UNUSED(stream);
    GGML_ABORT("%s: llama.cpp was built without GGML_CUDA_MUL_QEMU=ON\n", __func__);
#else
    const auto total_start = std::chrono::steady_clock::now();
    std::vector<uint16_t> src0_host(elements);
    std::vector<uint16_t> src1_host(elements);
    const auto d2h_start = std::chrono::steady_clock::now();
    if (elements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(src0_host.data(), src0_bf16,
                elements * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(src1_host.data(), src1_bf16,
                elements * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const uint64_t d2h_ns = elapsed_ns(d2h_start, std::chrono::steady_clock::now());
    mul_qemu_result result = call_qemu_rpc(
            params, std::move(src0_host), std::move(src1_host));
    result.d2h_ns = d2h_ns;
    result.external_total_ns = elapsed_ns(total_start, std::chrono::steady_clock::now());
    return result;
#endif
}

static void run_qemu_cuda_only(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_mul_qemu_params & params,
        size_t elements) {
    ggml_cuda_pool_alloc<uint16_t> src0_bf16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> src1_bf16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> output_bf16(ctx.pool(), elements);
    cudaStream_t stream = ctx.stream();

    cudaEvent_t start_event = nullptr;
    cudaEvent_t preprocess_event = nullptr;
    cudaEvent_t operator_event = nullptr;
    cudaEvent_t finish_event = nullptr;
    if (timing_enabled()) {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&preprocess_event));
        CUDA_CHECK(cudaEventCreate(&operator_event));
        CUDA_CHECK(cudaEventCreate(&finish_event));
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    ggml_cuda_mul_qemu_cuda_preprocess(
            params, src0_bf16.get(), src1_bf16.get(), stream);
    if (timing_enabled()) {
        CUDA_CHECK(cudaEventRecord(preprocess_event, stream));
    }
    ggml_cuda_mul_qemu_cuda_run_bf16(
            params, src0_bf16.get(), src1_bf16.get(), output_bf16.get(), stream);
    if (timing_enabled()) {
        CUDA_CHECK(cudaEventRecord(operator_event, stream));
    }
    ggml_cuda_mul_qemu_cuda_output(params, output_bf16.get(), stream);

    if (timing_enabled()) {
        CUDA_CHECK(cudaEventRecord(finish_event, stream));
        CUDA_CHECK(cudaEventSynchronize(finish_event));
        const mul_qemu_cuda_timing timing = {
            event_elapsed_ms(start_event, preprocess_event),
            event_elapsed_ms(preprocess_event, operator_event),
            event_elapsed_ms(operator_event, finish_event),
            event_elapsed_ms(start_event, finish_event),
        };
        log_qemu_cuda_timing(
                dst_tensor, params, timing, GGML_CUDA_MUL_QEMU_MODE_QEMU_CUDA);
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(preprocess_event));
        CUDA_CHECK(cudaEventDestroy(operator_event));
        CUDA_CHECK(cudaEventDestroy(finish_event));
    }
}

static void run_qemu_only(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_mul_qemu_params & params,
        size_t elements) {
    ggml_cuda_pool_alloc<uint16_t> src0_bf16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> src1_bf16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> output_bf16(ctx.pool(), elements);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_mul_qemu_cuda_preprocess(
            params, src0_bf16.get(), src1_bf16.get(), stream);
    mul_qemu_result qemu = run_qemu_rpc(
            params, src0_bf16.get(), src1_bf16.get(), elements, stream);
    const auto copy_start = std::chrono::steady_clock::now();
    if (elements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(output_bf16.get(), qemu.output_values.data(),
                elements * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        ggml_cuda_mul_qemu_cuda_output(params, output_bf16.get(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const uint64_t return_copy_ns = elapsed_ns(
            copy_start, std::chrono::steady_clock::now());
    log_qemu_timing(
            dst_tensor, qemu, elements, return_copy_ns, GGML_CUDA_MUL_QEMU_MODE_QEMU);
}

static void run_compare(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_mul_qemu_params & params,
        ggml_cuda_mul_launch_fn cuda_launch,
        size_t elements) {
    const size_t native_bytes = elements * ggml_type_size(params.dst_type);
    const size_t bf16_bytes = elements * sizeof(uint16_t);
    ggml_cuda_pool_alloc<unsigned char> cuda_dst(ctx.pool(), native_bytes);
    ggml_cuda_pool_alloc<unsigned char> qemu_cuda_dst(ctx.pool(), native_bytes);
    ggml_cuda_pool_alloc<uint16_t> src0_bf16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> src1_bf16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> qemu_cuda_output_bf16(ctx.pool(), elements);

    ggml_cuda_mul_qemu_params cuda_params = params;
    cuda_params.dst = cuda_dst.get();
    ggml_cuda_mul_qemu_params qemu_cuda_params = params;
    qemu_cuda_params.dst = qemu_cuda_dst.get();

    cudaStream_t main_stream = ctx.stream();
    cudaStream_t cuda_stream = ctx.stream(ctx.device, 1);
    cudaStream_t qemu_cuda_stream = ctx.stream(ctx.device, 2);
    cudaStream_t qemu_stream = ctx.stream(ctx.device, 3);

    cudaEvent_t barrier_event = nullptr;
    cudaEvent_t preprocessed_event = nullptr;
    cudaEvent_t cuda_start_event = nullptr;
    cudaEvent_t cuda_finish_event = nullptr;
    cudaEvent_t qemu_cuda_start_event = nullptr;
    cudaEvent_t qemu_cuda_preprocess_event = nullptr;
    cudaEvent_t qemu_cuda_operator_event = nullptr;
    cudaEvent_t qemu_cuda_finish_event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&barrier_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&preprocessed_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&cuda_start_event));
    CUDA_CHECK(cudaEventCreate(&cuda_finish_event));
    CUDA_CHECK(cudaEventCreate(&qemu_cuda_start_event));
    CUDA_CHECK(cudaEventCreate(&qemu_cuda_preprocess_event));
    CUDA_CHECK(cudaEventCreate(&qemu_cuda_operator_event));
    CUDA_CHECK(cudaEventCreate(&qemu_cuda_finish_event));

    CUDA_CHECK(cudaEventRecord(barrier_event, main_stream));
    CUDA_CHECK(cudaStreamWaitEvent(cuda_stream, barrier_event, 0));
    CUDA_CHECK(cudaStreamWaitEvent(qemu_cuda_stream, barrier_event, 0));

    CUDA_CHECK(cudaEventRecord(cuda_start_event, cuda_stream));
    cuda_launch(cuda_params, cuda_stream);
    CUDA_CHECK(cudaEventRecord(cuda_finish_event, cuda_stream));
    std::vector<unsigned char> cuda_host(native_bytes);
    if (native_bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(cuda_host.data(), cuda_dst.get(), native_bytes,
                cudaMemcpyDeviceToHost, cuda_stream));
    }

    CUDA_CHECK(cudaEventRecord(qemu_cuda_start_event, qemu_cuda_stream));
    ggml_cuda_mul_qemu_cuda_preprocess(
            params, src0_bf16.get(), src1_bf16.get(), qemu_cuda_stream);
    CUDA_CHECK(cudaEventRecord(qemu_cuda_preprocess_event, qemu_cuda_stream));
    CUDA_CHECK(cudaEventRecord(preprocessed_event, qemu_cuda_stream));
    CUDA_CHECK(cudaStreamWaitEvent(qemu_stream, preprocessed_event, 0));
    ggml_cuda_mul_qemu_cuda_run_bf16(
            params, src0_bf16.get(), src1_bf16.get(),
            qemu_cuda_output_bf16.get(), qemu_cuda_stream);
    CUDA_CHECK(cudaEventRecord(qemu_cuda_operator_event, qemu_cuda_stream));
    ggml_cuda_mul_qemu_cuda_output(
            qemu_cuda_params, qemu_cuda_output_bf16.get(), qemu_cuda_stream);
    CUDA_CHECK(cudaEventRecord(qemu_cuda_finish_event, qemu_cuda_stream));
    std::vector<uint16_t> qemu_cuda_host(elements);
    if (bf16_bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(qemu_cuda_host.data(), qemu_cuda_output_bf16.get(),
                bf16_bytes, cudaMemcpyDeviceToHost, qemu_cuda_stream));
    }

    mul_qemu_result qemu_result;
    std::thread qemu_thread([&params, src0 = src0_bf16.get(), src1 = src1_bf16.get(),
            elements, qemu_stream, &qemu_result, device = ctx.device] {
        ggml_cuda_set_device(device);
        qemu_result = run_qemu_rpc(params, src0, src1, elements, qemu_stream);
    });

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(qemu_cuda_stream));
    qemu_thread.join();

    const float llama_cuda_ms = event_elapsed_ms(cuda_start_event, cuda_finish_event);
    const mul_qemu_cuda_timing qemu_cuda_timing = {
        event_elapsed_ms(qemu_cuda_start_event, qemu_cuda_preprocess_event),
        event_elapsed_ms(qemu_cuda_preprocess_event, qemu_cuda_operator_event),
        event_elapsed_ms(qemu_cuda_operator_event, qemu_cuda_finish_event),
        event_elapsed_ms(qemu_cuda_start_event, qemu_cuda_finish_event),
    };
    const mul_compare_metrics metrics = compare_outputs(
            cuda_host, params.dst_type, qemu_result.output_values, qemu_cuda_host);
    write_comparison_artifact(
            dst_tensor, params, qemu_result, metrics, llama_cuda_ms, qemu_cuda_timing);
    write_mismatch_log(
            dst_tensor, params, qemu_result, qemu_cuda_host, metrics);

    if (native_bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(params.dst, cuda_dst.get(), native_bytes,
                cudaMemcpyDeviceToDevice, main_stream));
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
    }

    if (timing_enabled()) {
        const char * tensor_name = dst_tensor->name[0] != '\0' ?
                dst_tensor->name : "(unnamed)";
        GGML_LOG_INFO(
                "LLAMA_CUDA_MUL_TIMING mode=compare dst=%s elements=%zu total_ms=%.3f\n",
                tensor_name, elements, llama_cuda_ms);
    }
    log_qemu_timing(
            dst_tensor, qemu_result, elements, 0, GGML_CUDA_MUL_QEMU_MODE_COMPARE);
    log_qemu_cuda_timing(
            dst_tensor, params, qemu_cuda_timing, GGML_CUDA_MUL_QEMU_MODE_COMPARE);

    CUDA_CHECK(cudaEventDestroy(barrier_event));
    CUDA_CHECK(cudaEventDestroy(preprocessed_event));
    CUDA_CHECK(cudaEventDestroy(cuda_start_event));
    CUDA_CHECK(cudaEventDestroy(cuda_finish_event));
    CUDA_CHECK(cudaEventDestroy(qemu_cuda_start_event));
    CUDA_CHECK(cudaEventDestroy(qemu_cuda_preprocess_event));
    CUDA_CHECK(cudaEventDestroy(qemu_cuda_operator_event));
    CUDA_CHECK(cudaEventDestroy(qemu_cuda_finish_event));
}

void ggml_cuda_mul_qemu_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_mul_qemu_params & params,
        ggml_cuda_mul_launch_fn cuda_launch) {
    const ggml_cuda_mul_qemu_mode mode = ggml_cuda_mul_qemu_get_mode();
    log_mode_once(mode);
    const size_t elements = mul_elements(params);
    GGML_ASSERT(elements == (size_t) ggml_nelements(dst_tensor));

    switch (mode) {
        case GGML_CUDA_MUL_QEMU_MODE_QEMU:
            run_qemu_only(ctx, dst_tensor, params, elements);
            return;
        case GGML_CUDA_MUL_QEMU_MODE_QEMU_CUDA:
            run_qemu_cuda_only(ctx, dst_tensor, params, elements);
            return;
        case GGML_CUDA_MUL_QEMU_MODE_COMPARE:
            run_compare(ctx, dst_tensor, params, cuda_launch, elements);
            return;
        case GGML_CUDA_MUL_QEMU_MODE_CUDA:
            GGML_ABORT("%s: CUDA-only mode must use the original MUL dispatch\n", __func__);
    }
}
