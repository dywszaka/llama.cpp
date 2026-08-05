#include "rms-norm-qemu.cuh"
#include "rms-norm-qemu-cuda.cuh"
#include "rms-norm-qemu-protocol.h"

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

#if defined(GGML_CUDA_RMS_NORM_QEMU)
#include <zmq.h>
#endif

static const char * GGML_CUDA_RMS_NORM_QEMU_MODE_ENV = "GGML_CUDA_RMS_NORM_QEMU_MODE";
static const char * GGML_CUDA_RMS_NORM_QEMU_ENDPOINT_ENV = "GGML_CUDA_RMS_NORM_QEMU_ENDPOINT";
static const char * GGML_CUDA_RMS_NORM_QEMU_TIMEOUT_ENV = "GGML_CUDA_RMS_NORM_QEMU_TIMEOUT_MS";
static const char * GGML_CUDA_RMS_NORM_QEMU_ARTIFACT_ENV = "GGML_CUDA_RMS_NORM_QEMU_ARTIFACT";
static const char * GGML_CUDA_RMS_NORM_QEMU_MISMATCH_LOG_ENV = "GGML_CUDA_RMS_NORM_QEMU_MISMATCH_LOG";
static const char * GGML_CUDA_RMS_NORM_QEMU_TIMING_ENV = "GGML_CUDA_RMS_NORM_QEMU_TIMING";
static const char * GGML_CUDA_RMS_NORM_QEMU_DEFAULT_ENDPOINT = "tcp://127.0.0.1:15583";
static const char * GGML_CUDA_RMS_NORM_QEMU_DEFAULT_ARTIFACT =
        "experiments/rms-norm-qemu-compare.jsonl";
static const char * GGML_CUDA_RMS_NORM_QEMU_DEFAULT_MISMATCH_LOG =
        "experiments/rms-norm-qemu-cuda-mismatch.jsonl";

static const char * ggml_cuda_rms_norm_qemu_mode_name(ggml_cuda_rms_norm_qemu_mode mode) {
    switch (mode) {
        case GGML_CUDA_RMS_NORM_QEMU_MODE_CUDA:      return "cuda";
        case GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU:      return "qemu";
        case GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU_CUDA: return "qemu_cuda";
        case GGML_CUDA_RMS_NORM_QEMU_MODE_COMPARE:   return "compare";
    }
    return "cuda";
}

static ggml_cuda_rms_norm_qemu_mode parse_mode(const char * value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "cuda") == 0) {
        return GGML_CUDA_RMS_NORM_QEMU_MODE_CUDA;
    }
    if (std::strcmp(value, "qemu") == 0) {
        return GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU;
    }
    if (std::strcmp(value, "qemu_cuda") == 0) {
        return GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU_CUDA;
    }
    if (std::strcmp(value, "compare") == 0 ||
            std::strcmp(value, "compare_cuda") == 0 ||
            std::strcmp(value, "compare_qemu") == 0) {
        return GGML_CUDA_RMS_NORM_QEMU_MODE_COMPARE;
    }
    GGML_LOG_WARN("%s: unknown %s=%s; using cuda\n",
            __func__, GGML_CUDA_RMS_NORM_QEMU_MODE_ENV, value);
    return GGML_CUDA_RMS_NORM_QEMU_MODE_CUDA;
}

ggml_cuda_rms_norm_qemu_mode ggml_cuda_rms_norm_qemu_get_mode() {
    static const ggml_cuda_rms_norm_qemu_mode mode =
            parse_mode(std::getenv(GGML_CUDA_RMS_NORM_QEMU_MODE_ENV));
    return mode;
}

bool ggml_cuda_rms_norm_qemu_enabled() {
    return ggml_cuda_rms_norm_qemu_get_mode() != GGML_CUDA_RMS_NORM_QEMU_MODE_CUDA;
}

static std::string endpoint() {
    const char * value = std::getenv(GGML_CUDA_RMS_NORM_QEMU_ENDPOINT_ENV);
    return value != nullptr && value[0] != '\0' ? value :
            GGML_CUDA_RMS_NORM_QEMU_DEFAULT_ENDPOINT;
}

static int timeout_ms() {
    const char * value = std::getenv(GGML_CUDA_RMS_NORM_QEMU_TIMEOUT_ENV);
    if (value == nullptr || value[0] == '\0') {
        return 300000;
    }
    const long parsed = std::strtol(value, nullptr, 10);
    return parsed > 0 && parsed <= INT_MAX ? (int) parsed : 300000;
}

static std::string artifact_path() {
    const char * value = std::getenv(GGML_CUDA_RMS_NORM_QEMU_ARTIFACT_ENV);
    return value != nullptr && value[0] != '\0' ? value :
            GGML_CUDA_RMS_NORM_QEMU_DEFAULT_ARTIFACT;
}

static std::string mismatch_log_path() {
    const char * value = std::getenv(GGML_CUDA_RMS_NORM_QEMU_MISMATCH_LOG_ENV);
    return value != nullptr && value[0] != '\0' ? value :
            GGML_CUDA_RMS_NORM_QEMU_DEFAULT_MISMATCH_LOG;
}

static bool timing_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv(GGML_CUDA_RMS_NORM_QEMU_TIMING_ENV);
        return value != nullptr && value[0] != '\0' &&
                std::strcmp(value, "0") != 0 &&
                std::strcmp(value, "false") != 0 &&
                std::strcmp(value, "off") != 0;
    }();
    return enabled;
}

static void log_mode_once(ggml_cuda_rms_norm_qemu_mode mode) {
    static std::atomic<bool> logged(false);
    if (mode == GGML_CUDA_RMS_NORM_QEMU_MODE_CUDA || logged.exchange(true)) {
        return;
    }
    if (mode == GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU_CUDA) {
        GGML_LOG_INFO(
                "%s: %s=qemu_cuda enabled; bit-exact FP32-compute/BF16-I/O RMS_NORM "
                "stays on CUDA device, "
                "ZMQ/D2H/H2D are not used, timing=%s\n",
                __func__, GGML_CUDA_RMS_NORM_QEMU_MODE_ENV,
                timing_enabled() ? "on" : "off");
        return;
    }
    if (mode == GGML_CUDA_RMS_NORM_QEMU_MODE_COMPARE) {
        const std::string rpc_endpoint = endpoint();
        const std::string artifact = artifact_path();
        const std::string mismatch = mismatch_log_path();
        GGML_LOG_INFO(
                "%s: %s=compare enabled; downstream=llama CUDA, endpoint=%s, timing=%s, "
                "comparison artifact=%s, mismatch log=%s\n",
                __func__, GGML_CUDA_RMS_NORM_QEMU_MODE_ENV, rpc_endpoint.c_str(),
                timing_enabled() ? "on" : "off", artifact.c_str(), mismatch.c_str());
        return;
    }
    const std::string rpc_endpoint = endpoint();
    GGML_LOG_INFO(
            "%s: %s=qemu enabled; FP32 RVV compute/BF16 I/O RMS_NORM "
            "endpoint=%s, timing=%s\n",
            __func__, GGML_CUDA_RMS_NORM_QEMU_MODE_ENV, rpc_endpoint.c_str(),
            timing_enabled() ? "on" : "off");
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

struct comparison_metrics {
    double mse = 0.0;
    double rmse = 0.0;
    double max_abs = 0.0;
};

static comparison_metrics compare_values(
        const std::vector<float> & left,
        const std::vector<float> & right) {
    GGML_ASSERT(left.size() == right.size());
    double sum_squared = 0.0;
    double max_abs = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        const double difference = (double) left[index] - (double) right[index];
        sum_squared += difference * difference;
        max_abs = std::max(max_abs, std::abs(difference));
    }
    const double mse = left.empty() ? 0.0 : sum_squared / (double) left.size();
    return {mse, std::sqrt(mse), max_abs};
}

struct bit_comparison_metrics {
    size_t mismatches = 0;
    size_t first_mismatch = 0;
};

static bit_comparison_metrics compare_bits(
        const std::vector<uint16_t> & left,
        const std::vector<uint16_t> & right) {
    GGML_ASSERT(left.size() == right.size());
    bit_comparison_metrics result;
    for (size_t index = 0; index < left.size(); ++index) {
        if (left[index] != right[index]) {
            if (result.mismatches == 0) {
                result.first_mismatch = index;
            }
            ++result.mismatches;
        }
    }
    return result;
}

static float bf16_to_float(uint16_t value) {
    const uint32_t bits = (uint32_t) value << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static std::vector<float> bf16_to_float(const std::vector<uint16_t> & values) {
    std::vector<float> result(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        result[index] = bf16_to_float(values[index]);
    }
    return result;
}

static void write_bf16_array(std::ofstream & output, const std::vector<uint16_t> & values) {
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

static void write_comparison_artifact(
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_qemu_params & params,
        const comparison_metrics & llama_qemu,
        const comparison_metrics & qemu_qemu_cuda,
        const bit_comparison_metrics & bit_metrics,
        uint64_t request_id,
        uint64_t qemu_elapsed_ns,
        float llama_cuda_ms,
        float qemu_cuda_ms) {
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
    const char * tensor_name = dst_tensor->name[0] != '\0' ? dst_tensor->name : "(unnamed)";
    output << "{\"op\":\"" << json_escape(ggml_op_desc(dst_tensor))
           << "\",\"dst\":\"" << json_escape(tensor_name)
           << "\",\"request\":" << request_id
           << ",\"shape\":[" << params.ncols << ',' << params.nrows << ','
           << params.nchannels << ',' << params.nsamples << ']'
           << ",\"eps\":" << params.eps
           << ",\"llama_qemu_mse\":" << llama_qemu.mse
           << ",\"llama_qemu_rmse\":" << llama_qemu.rmse
           << ",\"llama_qemu_max_abs\":" << llama_qemu.max_abs
           << ",\"qemu_qemu_cuda_mse\":" << qemu_qemu_cuda.mse
           << ",\"qemu_qemu_cuda_rmse\":" << qemu_qemu_cuda.rmse
           << ",\"qemu_qemu_cuda_max_abs\":" << qemu_qemu_cuda.max_abs
           << ",\"qemu_qemu_cuda_bit_mismatches\":" << bit_metrics.mismatches
           << ",\"qemu_elapsed_ms\":" << (double) qemu_elapsed_ns / 1.0e6
           << ",\"llama_cuda_ms\":" << llama_cuda_ms
           << ",\"qemu_cuda_ms\":" << qemu_cuda_ms
           << "}\n";
}

static void write_mismatch_log(
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_qemu_params & params,
        uint64_t request_id,
        const bit_comparison_metrics & metrics,
        const std::vector<uint16_t> & input,
        const std::vector<uint16_t> & qemu,
        const std::vector<uint16_t> & qemu_cuda) {
    if (metrics.mismatches == 0) {
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
    const char * tensor_name = dst_tensor->name[0] != '\0' ? dst_tensor->name : "(unnamed)";
    output << "{\"op\":\"" << json_escape(ggml_op_desc(dst_tensor))
           << "\",\"dst\":\"" << json_escape(tensor_name)
           << "\",\"request\":" << request_id
           << ",\"shape\":[" << params.ncols << ',' << params.nrows << ','
           << params.nchannels << ',' << params.nsamples << ']'
           << ",\"eps\":" << params.eps
           << ",\"mismatches\":" << metrics.mismatches
           << ",\"first_mismatch\":" << metrics.first_mismatch
           << ",\"input_bf16\":";
    write_bf16_array(output, input);
    output << ",\"qemu_output_bf16\":";
    write_bf16_array(output, qemu);
    output << ",\"qemu_cuda_output_bf16\":";
    write_bf16_array(output, qemu_cuda);
    output << "}\n";
    GGML_LOG_ERROR(
            "QEMU_CUDA_RMS_NORM_BIT_MISMATCH request=%llu dst=%s mismatches=%zu first=%zu log=%s\n",
            (unsigned long long) request_id, tensor_name, metrics.mismatches,
            metrics.first_mismatch, path.string().c_str());
}

static uint64_t elapsed_ns(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static float event_elapsed_ms(cudaEvent_t start, cudaEvent_t finish) {
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, finish));
    return milliseconds;
}

struct qemu_cuda_timing_values {
    float preprocess_ms = 0.0f;
    float operator_ms = 0.0f;
    float output_ms = 0.0f;
    float total_ms = 0.0f;
};

static void log_qemu_cuda_timing(
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_qemu_params & params,
        const qemu_cuda_timing_values & timing,
        ggml_cuda_rms_norm_qemu_mode mode) {
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
    const char * tensor_name = dst_tensor->name[0] != '\0' ? dst_tensor->name : "(unnamed)";
    GGML_LOG_INFO(
            "QEMU_CUDA_RMS_NORM_TIMING mode=%s dst=%s shape=%dx%dx%dx%d "
            "preprocess_ms=%.3f operator_ms=%.3f output_ms=%.3f total_ms=%.3f "
            "calls=%llu average_total_ms=%.3f\n",
            ggml_cuda_rms_norm_qemu_mode_name(mode), tensor_name,
            params.ncols, params.nrows, params.nchannels, params.nsamples,
            timing.preprocess_ms, timing.operator_ms, timing.output_ms, timing.total_ms,
            (unsigned long long) current_calls, average_ms);
}

static void log_llama_cuda_timing(
        const ggml_tensor * dst_tensor,
        size_t elements,
        float milliseconds) {
    if (!timing_enabled()) {
        return;
    }
    const char * tensor_name = dst_tensor->name[0] != '\0' ? dst_tensor->name : "(unnamed)";
    GGML_LOG_INFO(
            "LLAMA_CUDA_RMS_NORM_TIMING mode=compare dst=%s elements=%zu total_ms=%.3f\n",
            tensor_name, elements, milliseconds);
}

struct qemu_rms_norm_result {
    std::vector<uint16_t> input_values;
    std::vector<uint16_t> values;
    uint64_t request_id = 0;
    uint64_t elapsed_ns = 0;
    uint64_t d2h_ns = 0;
    uint64_t rpc_roundtrip_ns = 0;
    uint64_t external_total_ns = 0;
};

static void log_qemu_timing(
        const ggml_tensor * dst_tensor,
        const qemu_rms_norm_result & result,
        size_t elements,
        uint64_t return_copy_ns,
        ggml_cuda_rms_norm_qemu_mode mode) {
    if (!timing_enabled()) {
        return;
    }
    const char * tensor_name = dst_tensor->name[0] != '\0' ? dst_tensor->name : "(unnamed)";
    GGML_LOG_INFO(
            "RVV_RMS_NORM_TIMING request=%llu mode=%s dst=%s elements=%zu "
            "d2h_ms=%.3f rpc_roundtrip_ms=%.3f daemon_request_ms=%.3f "
            "return_copy_ms=%.3f total_ms=%.3f\n",
            (unsigned long long) result.request_id,
            ggml_cuda_rms_norm_qemu_mode_name(mode), tensor_name, elements,
            (double) result.d2h_ns / 1.0e6,
            (double) result.rpc_roundtrip_ns / 1.0e6,
            (double) result.elapsed_ns / 1.0e6,
            (double) return_copy_ns / 1.0e6,
            (double) (result.external_total_ns + return_copy_ns) / 1.0e6);
}

static void stage_bf16_to_host(
        const uint16_t * input,
        size_t elements,
        cudaStream_t stream,
        std::vector<uint16_t> * output) {
    output->resize(elements);
    if (elements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(output->data(), input, elements * sizeof(uint16_t),
                cudaMemcpyDeviceToHost, stream));
    }
}

#if defined(GGML_CUDA_RMS_NORM_QEMU)

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

static void * rpc_socket() {
    static void * socket = [] {
        void * context = zmq_ctx_new();
        if (context == nullptr) {
            GGML_ABORT("%s: zmq_ctx_new failed\n", __func__);
        }
        void * created = zmq_socket(context, ZMQ_REQ);
        if (created == nullptr) {
            GGML_ABORT("%s: zmq_socket failed: %s\n", __func__, zmq_strerror(zmq_errno()));
        }
        const int timeout = timeout_ms();
        zmq_setsockopt(created, ZMQ_RCVTIMEO, &timeout, sizeof(timeout));
        zmq_setsockopt(created, ZMQ_SNDTIMEO, &timeout, sizeof(timeout));
        const std::string rpc_endpoint = endpoint();
        if (zmq_connect(created, rpc_endpoint.c_str()) != 0) {
            GGML_ABORT("%s: failed to connect %s: %s\n", __func__,
                    rpc_endpoint.c_str(), zmq_strerror(zmq_errno()));
        }
        return created;
    }();
    return socket;
}

static qemu_rms_norm_result call_qemu_rpc(
        const ggml_cuda_rms_norm_qemu_params & params,
        std::vector<uint16_t> input) {
    static std::mutex rpc_mutex;
    static std::atomic<uint64_t> next_request_id(1);
    std::lock_guard<std::mutex> lock(rpc_mutex);

    rms_norm_rpc_request_v1 request = {};
    request.magic = RMS_NORM_RPC_MAGIC;
    request.version = RMS_NORM_RPC_VERSION;
    request.header_bytes = sizeof(request);
    request.request_id = next_request_id.fetch_add(1);
    request.flags = RMS_NORM_RPC_REQUEST_CANONICAL_DENSE;
    request.src_type = RMS_NORM_RPC_DTYPE_BF16;
    request.dst_type = RMS_NORM_RPC_DTYPE_BF16;
    request.ncols = params.ncols;
    request.nrows = params.nrows;
    request.nchannels = params.nchannels;
    request.nsamples = params.nsamples;
    request.eps = params.eps;
    request.src0_bytes = input.size() * sizeof(uint16_t);
    request.dst_bytes = request.src0_bytes;

    void * socket = rpc_socket();
    send_frame(socket, &request, sizeof(request), true);
    send_frame(socket, input.data(), request.src0_bytes, false);

    bool more = false;
    const std::vector<unsigned char> response_frame = receive_frame(socket, &more);
    if (!more || response_frame.size() != sizeof(rms_norm_rpc_response_v1)) {
        GGML_ABORT("%s: invalid QEMU RPC response header\n", __func__);
    }
    rms_norm_rpc_response_v1 response = {};
    std::memcpy(&response, response_frame.data(), sizeof(response));
    const std::vector<unsigned char> output_frame = receive_frame(socket, &more);
    if (more || response.magic != RMS_NORM_RPC_MAGIC ||
            response.version != RMS_NORM_RPC_VERSION ||
            response.header_bytes != sizeof(response) ||
            response.request_id != request.request_id) {
        GGML_ABORT("%s: malformed QEMU RPC response\n", __func__);
    }
    if (response.status != RMS_NORM_RPC_STATUS_OK) {
        GGML_ABORT("%s: QEMU RPC failed: status=%u error=%u\n",
                __func__, response.status, response.error_code);
    }
    if (response.output_bytes != request.dst_bytes ||
            output_frame.size() != request.dst_bytes) {
        GGML_ABORT("%s: QEMU RPC output size mismatch: expected=%zu response=%llu frame=%zu\n",
                __func__, (size_t) request.dst_bytes,
                (unsigned long long) response.output_bytes, output_frame.size());
    }

    qemu_rms_norm_result result;
    result.input_values = std::move(input);
    result.values.resize(result.input_values.size());
    if (!output_frame.empty()) {
        std::memcpy(result.values.data(), output_frame.data(), output_frame.size());
    }
    result.request_id = response.request_id;
    result.elapsed_ns = response.elapsed_ns;
    return result;
}

#endif

static qemu_rms_norm_result ggml_qemu_op_rms_norm(
        const ggml_cuda_rms_norm_qemu_params & params,
        const uint16_t * input_bf16,
        size_t elements,
        cudaStream_t stream) {
#if !defined(GGML_CUDA_RMS_NORM_QEMU)
    GGML_UNUSED(params);
    GGML_UNUSED(input_bf16);
    GGML_UNUSED(elements);
    GGML_UNUSED(stream);
    GGML_ABORT("%s: llama.cpp was built without GGML_CUDA_RMS_NORM_QEMU=ON\n", __func__);
#else
    const auto external_start = std::chrono::steady_clock::now();
    std::vector<uint16_t> input;
    stage_bf16_to_host(input_bf16, elements, stream, &input);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto d2h_done = std::chrono::steady_clock::now();
    qemu_rms_norm_result result = call_qemu_rpc(params, std::move(input));
    const auto rpc_done = std::chrono::steady_clock::now();
    result.d2h_ns = elapsed_ns(external_start, d2h_done);
    result.rpc_roundtrip_ns = elapsed_ns(d2h_done, rpc_done);
    result.external_total_ns = elapsed_ns(external_start, rpc_done);
    return result;
#endif
}

static void run_qemu_cuda_only(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_qemu_params & params,
        size_t elements) {
    ggml_cuda_pool_alloc<uint16_t> input_bf16(ctx.pool(), elements);
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

    ggml_cuda_rms_norm_qemu_cuda_preprocess(params, input_bf16.get(), stream);
    if (timing_enabled()) {
        CUDA_CHECK(cudaEventRecord(preprocess_event, stream));
    }
    ggml_cuda_rms_norm_qemu_cuda_run_bf16(
            params, input_bf16.get(), output_bf16.get(), stream);
    if (timing_enabled()) {
        CUDA_CHECK(cudaEventRecord(operator_event, stream));
    }
    ggml_cuda_rms_norm_qemu_cuda_output_to_f32(
            output_bf16.get(), params.dst, elements, stream);

    if (timing_enabled()) {
        CUDA_CHECK(cudaEventRecord(finish_event, stream));
        CUDA_CHECK(cudaEventSynchronize(finish_event));
        const qemu_cuda_timing_values timing = {
            event_elapsed_ms(start_event, preprocess_event),
            event_elapsed_ms(preprocess_event, operator_event),
            event_elapsed_ms(operator_event, finish_event),
            event_elapsed_ms(start_event, finish_event),
        };
        log_qemu_cuda_timing(
                dst_tensor, params, timing, GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU_CUDA);
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(preprocess_event));
        CUDA_CHECK(cudaEventDestroy(operator_event));
        CUDA_CHECK(cudaEventDestroy(finish_event));
    }
}

static void run_qemu_only(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_qemu_params & params,
        size_t elements) {
    ggml_cuda_pool_alloc<uint16_t> input_bf16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> output_bf16(ctx.pool(), elements);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_rms_norm_qemu_cuda_preprocess(params, input_bf16.get(), stream);
    qemu_rms_norm_result qemu = ggml_qemu_op_rms_norm(
            params, input_bf16.get(), elements, stream);
    const auto copy_start = std::chrono::steady_clock::now();
    if (elements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(output_bf16.get(), qemu.values.data(),
                elements * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        ggml_cuda_rms_norm_qemu_cuda_output_to_f32(
                output_bf16.get(), params.dst, elements, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const uint64_t return_copy_ns = elapsed_ns(copy_start, std::chrono::steady_clock::now());
    log_qemu_timing(
            dst_tensor, qemu, elements, return_copy_ns, GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU);
}

static void run_compare(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_qemu_params & params,
        ggml_cuda_rms_norm_launch_fn cuda_launch,
        size_t elements) {
    const size_t f32_bytes = elements * sizeof(float);
    const size_t bf16_bytes = elements * sizeof(uint16_t);
    ggml_cuda_pool_alloc<float> cuda_dst(ctx.pool(), elements);
    ggml_cuda_pool_alloc<float> qemu_cuda_dst(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> input_bf16(ctx.pool(), elements);
    ggml_cuda_pool_alloc<uint16_t> qemu_cuda_output_bf16(ctx.pool(), elements);

    ggml_cuda_rms_norm_qemu_params cuda_params = params;
    cuda_params.dst = cuda_dst.get();

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
    std::vector<float> cuda_host(elements);
    if (f32_bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(cuda_host.data(), cuda_dst.get(), f32_bytes,
                cudaMemcpyDeviceToHost, cuda_stream));
    }

    CUDA_CHECK(cudaEventRecord(qemu_cuda_start_event, qemu_cuda_stream));
    ggml_cuda_rms_norm_qemu_cuda_preprocess(params, input_bf16.get(), qemu_cuda_stream);
    CUDA_CHECK(cudaEventRecord(qemu_cuda_preprocess_event, qemu_cuda_stream));
    CUDA_CHECK(cudaEventRecord(preprocessed_event, qemu_cuda_stream));
    CUDA_CHECK(cudaStreamWaitEvent(qemu_stream, preprocessed_event, 0));
    ggml_cuda_rms_norm_qemu_cuda_run_bf16(
            params, input_bf16.get(), qemu_cuda_output_bf16.get(), qemu_cuda_stream);
    CUDA_CHECK(cudaEventRecord(qemu_cuda_operator_event, qemu_cuda_stream));
    ggml_cuda_rms_norm_qemu_cuda_output_to_f32(
            qemu_cuda_output_bf16.get(), qemu_cuda_dst.get(), elements, qemu_cuda_stream);
    CUDA_CHECK(cudaEventRecord(qemu_cuda_finish_event, qemu_cuda_stream));
    std::vector<uint16_t> qemu_cuda_host(elements);
    if (bf16_bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(qemu_cuda_host.data(), qemu_cuda_output_bf16.get(), bf16_bytes,
                cudaMemcpyDeviceToHost, qemu_cuda_stream));
    }

    qemu_rms_norm_result qemu_result;
    std::thread qemu_thread([&params, input = input_bf16.get(), elements,
            qemu_stream, &qemu_result, device = ctx.device] {
        ggml_cuda_set_device(device);
        qemu_result = ggml_qemu_op_rms_norm(params, input, elements, qemu_stream);
    });

    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(qemu_cuda_stream));
    qemu_thread.join();

    const float llama_cuda_ms = event_elapsed_ms(cuda_start_event, cuda_finish_event);
    const qemu_cuda_timing_values qemu_cuda_timing = {
        event_elapsed_ms(qemu_cuda_start_event, qemu_cuda_preprocess_event),
        event_elapsed_ms(qemu_cuda_preprocess_event, qemu_cuda_operator_event),
        event_elapsed_ms(qemu_cuda_operator_event, qemu_cuda_finish_event),
        event_elapsed_ms(qemu_cuda_start_event, qemu_cuda_finish_event),
    };
    const std::vector<float> qemu_float = bf16_to_float(qemu_result.values);
    const std::vector<float> qemu_cuda_float = bf16_to_float(qemu_cuda_host);
    const comparison_metrics llama_qemu = compare_values(cuda_host, qemu_float);
    const comparison_metrics qemu_qemu_cuda = compare_values(qemu_float, qemu_cuda_float);
    const bit_comparison_metrics bit_metrics = compare_bits(qemu_result.values, qemu_cuda_host);
    write_comparison_artifact(
            dst_tensor, params, llama_qemu, qemu_qemu_cuda, bit_metrics,
            qemu_result.request_id, qemu_result.elapsed_ns,
            llama_cuda_ms, qemu_cuda_timing.total_ms);
    write_mismatch_log(
            dst_tensor, params, qemu_result.request_id, bit_metrics,
            qemu_result.input_values, qemu_result.values, qemu_cuda_host);

    if (f32_bytes != 0) {
        CUDA_CHECK(cudaMemcpyAsync(params.dst, cuda_dst.get(), f32_bytes,
                cudaMemcpyDeviceToDevice, main_stream));
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
    }

    log_llama_cuda_timing(dst_tensor, elements, llama_cuda_ms);
    log_qemu_timing(
            dst_tensor, qemu_result, elements, 0, GGML_CUDA_RMS_NORM_QEMU_MODE_COMPARE);
    log_qemu_cuda_timing(
            dst_tensor, params, qemu_cuda_timing, GGML_CUDA_RMS_NORM_QEMU_MODE_COMPARE);

    CUDA_CHECK(cudaEventDestroy(barrier_event));
    CUDA_CHECK(cudaEventDestroy(preprocessed_event));
    CUDA_CHECK(cudaEventDestroy(cuda_start_event));
    CUDA_CHECK(cudaEventDestroy(cuda_finish_event));
    CUDA_CHECK(cudaEventDestroy(qemu_cuda_start_event));
    CUDA_CHECK(cudaEventDestroy(qemu_cuda_preprocess_event));
    CUDA_CHECK(cudaEventDestroy(qemu_cuda_operator_event));
    CUDA_CHECK(cudaEventDestroy(qemu_cuda_finish_event));
}

void ggml_cuda_rms_norm_qemu_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_qemu_params & params,
        ggml_cuda_rms_norm_launch_fn cuda_launch) {
    const ggml_cuda_rms_norm_qemu_mode mode = ggml_cuda_rms_norm_qemu_get_mode();
    log_mode_once(mode);

    const size_t elements = (size_t) params.ncols * (size_t) params.nrows *
            (size_t) params.nchannels * (size_t) params.nsamples;
    GGML_ASSERT(elements == (size_t) ggml_nelements(dst_tensor));
    switch (mode) {
        case GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU:
            run_qemu_only(ctx, dst_tensor, params, elements);
            return;
        case GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU_CUDA:
            run_qemu_cuda_only(ctx, dst_tensor, params, elements);
            return;
        case GGML_CUDA_RMS_NORM_QEMU_MODE_COMPARE:
            run_compare(ctx, dst_tensor, params, cuda_launch, elements);
            return;
        case GGML_CUDA_RMS_NORM_QEMU_MODE_CUDA:
            GGML_ABORT("%s: CUDA-only mode must use the original RMS_NORM dispatch\n", __func__);
    }
}
