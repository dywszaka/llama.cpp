#include "rms-norm-cim.cuh"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static const char * GGML_CUDA_RMS_NORM_CIM_MODE_ENV = "GGML_CUDA_RMS_NORM_CIM_MODE";
static const char * GGML_CUDA_RMS_NORM_CIM_ARTIFACT_ENV = "GGML_CUDA_RMS_NORM_CIM_ARTIFACT";
static const char * GGML_CUDA_RMS_NORM_CIM_DEFAULT_ARTIFACT = "experiments/rms-norm-cim-compare.jsonl";

static const char * ggml_cuda_rms_norm_cim_mode_name(ggml_cuda_rms_norm_cim_mode mode) {
    switch (mode) {
        case GGML_CUDA_RMS_NORM_CIM_MODE_CUDA:         return "cuda";
        case GGML_CUDA_RMS_NORM_CIM_MODE_CIM:          return "cim";
        case GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CUDA: return "compare_cuda";
        case GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CIM:  return "compare_cim";
    }
    return "cuda";
}

static ggml_cuda_rms_norm_cim_mode parse_mode(const char * env) {
    if (env == nullptr || env[0] == '\0' || std::strcmp(env, "cuda") == 0) {
        return GGML_CUDA_RMS_NORM_CIM_MODE_CUDA;
    }
    if (std::strcmp(env, "cim") == 0) {
        return GGML_CUDA_RMS_NORM_CIM_MODE_CIM;
    }
    if (std::strcmp(env, "compare_cuda") == 0) {
        return GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CUDA;
    }
    if (std::strcmp(env, "compare_cim") == 0) {
        return GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CIM;
    }
    return GGML_CUDA_RMS_NORM_CIM_MODE_CUDA;
}

ggml_cuda_rms_norm_cim_mode ggml_cuda_rms_norm_cim_get_mode() {
    static const ggml_cuda_rms_norm_cim_mode mode = parse_mode(std::getenv(GGML_CUDA_RMS_NORM_CIM_MODE_ENV));
    return mode;
}

bool ggml_cuda_rms_norm_cim_enabled() {
    return ggml_cuda_rms_norm_cim_get_mode() != GGML_CUDA_RMS_NORM_CIM_MODE_CUDA;
}

static std::string artifact_path() {
    const char * env = std::getenv(GGML_CUDA_RMS_NORM_CIM_ARTIFACT_ENV);
    return env != nullptr && env[0] != '\0' ? std::string(env) : std::string(GGML_CUDA_RMS_NORM_CIM_DEFAULT_ARTIFACT);
}

static bool is_compare_mode(ggml_cuda_rms_norm_cim_mode mode) {
    return mode == GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CUDA || mode == GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CIM;
}

static void log_mode_once(ggml_cuda_rms_norm_cim_mode mode) {
    static std::atomic<bool> logged(false);
    if (mode == GGML_CUDA_RMS_NORM_CIM_MODE_CUDA || logged.exchange(true)) {
        return;
    }

    const std::string path = artifact_path();
    GGML_LOG_INFO(
            "%s: %s=%s -> experimental CUDA/CIM RMS_NORM mode enabled; CIM is an RPC/IO placeholder returning zero-filled output%s%s\n",
            __func__,
            GGML_CUDA_RMS_NORM_CIM_MODE_ENV,
            ggml_cuda_rms_norm_cim_mode_name(mode),
            is_compare_mode(mode) ? ", RMSE artifact=" : "",
            is_compare_mode(mode) ? path.c_str() : "");
}

static std::string json_escape(const char * value) {
    std::string escaped;
    for (const char * p = value; *p != '\0'; ++p) {
        switch (*p) {
            case '\\': escaped += "\\\\"; break;
            case '"':  escaped += "\\\""; break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:   escaped += *p;      break;
        }
    }
    return escaped;
}

static void write_rmse_artifact(const ggml_tensor * dst_tensor, double rmse) {
    const std::filesystem::path path(artifact_path());
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            GGML_LOG_WARN("%s: failed to create artifact directory %s: %s\n", __func__, parent.string().c_str(), ec.message().c_str());
            return;
        }
    }

    std::ofstream out(path, std::ios::app);
    if (!out) {
        GGML_LOG_WARN("%s: failed to open artifact %s\n", __func__, path.string().c_str());
        return;
    }

    const char * tensor_name = dst_tensor->name[0] != '\0' ? dst_tensor->name : "(unnamed)";
    out << "{\"op\":\"" << json_escape(ggml_op_desc(dst_tensor))
        << "\",\"dst\":\"" << json_escape(tensor_name)
        << "\",\"rmse\":" << rmse
        << "}\n";
}

struct cim_rpc_stub_result {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready = false;
    std::vector<float> values;
};

static void ggml_cim_op_rms_norm(
        const ggml_cuda_rms_norm_cim_params & params,
        size_t src_nbytes,
        size_t dst_nelements,
        cudaStream_t stream,
        cim_rpc_stub_result * result) {
    std::vector<unsigned char> request(src_nbytes);
    if (src_nbytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(request.data(), params.src0, src_nbytes, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> response(dst_nelements, 0.0f);
    GGML_UNUSED(request);
    GGML_UNUSED(params);

    {
        std::lock_guard<std::mutex> lock(result->mutex);
        result->values = std::move(response);
        result->ready = true;
    }
    result->cv.notify_one();
}

static void wait_for_cim_result(cim_rpc_stub_result & result) {
    std::unique_lock<std::mutex> lock(result.mutex);
    result.cv.wait(lock, [&result] { return result.ready; });
}

static double rmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    if (a.empty()) {
        return 0.0;
    }

    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = (double) a[i] - (double) b[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / (double) a.size());
}

void ggml_cuda_rms_norm_cim_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_cim_params & params,
        ggml_cuda_rms_norm_launch_fn cuda_launch) {
    const ggml_cuda_rms_norm_cim_mode mode = ggml_cuda_rms_norm_cim_get_mode();
    log_mode_once(mode);

    const size_t dst_nelements = (size_t) params.ncols * (size_t) params.nrows * (size_t) params.nchannels * (size_t) params.nsamples;
    const size_t dst_nbytes = dst_nelements * sizeof(float);
    const size_t src_nbytes = ggml_nbytes(dst_tensor->src[0]);
    cudaStream_t main_stream = ctx.stream();

    if (mode == GGML_CUDA_RMS_NORM_CIM_MODE_CIM) {
        cim_rpc_stub_result cim_result;
        ggml_cim_op_rms_norm(params, src_nbytes, dst_nelements, main_stream, &cim_result);
        wait_for_cim_result(cim_result);
        if (dst_nbytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(params.dst, cim_result.values.data(), dst_nbytes, cudaMemcpyHostToDevice, main_stream));
            CUDA_CHECK(cudaStreamSynchronize(main_stream));
        }
        return;
    }

    GGML_ASSERT(is_compare_mode(mode));

    ggml_cuda_pool_alloc<float> cuda_dst(ctx.pool(), dst_nelements);
    cudaStream_t cuda_stream = ctx.stream(ctx.device, 1);
    cudaStream_t cim_stream = ctx.stream(ctx.device, 2);
    cudaEvent_t start_event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(start_event, main_stream));
    CUDA_CHECK(cudaStreamWaitEvent(cuda_stream, start_event, 0));
    CUDA_CHECK(cudaStreamWaitEvent(cim_stream, start_event, 0));

    cim_rpc_stub_result cim_result;
    std::thread cim_thread([&params, src_nbytes, dst_nelements, cim_stream, &cim_result, device = ctx.device] {
        ggml_cuda_set_device(device);
        ggml_cim_op_rms_norm(params, src_nbytes, dst_nelements, cim_stream, &cim_result);
    });

    cuda_launch(
            params.src0, cuda_dst.get(), params.ncols, params.nrows, params.nchannels, params.nsamples,
            params.stride_row, params.stride_channel, params.stride_sample, params.eps, cuda_stream);

    std::vector<float> cuda_host(dst_nelements);
    if (dst_nbytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(cuda_host.data(), cuda_dst.get(), dst_nbytes, cudaMemcpyDeviceToHost, cuda_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));

    wait_for_cim_result(cim_result);
    cim_thread.join();

    const double cmp_rmse = rmse(cuda_host, cim_result.values);
    write_rmse_artifact(dst_tensor, cmp_rmse);

    if (mode == GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CUDA) {
        if (dst_nbytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(params.dst, cuda_dst.get(), dst_nbytes, cudaMemcpyDeviceToDevice, main_stream));
            CUDA_CHECK(cudaStreamSynchronize(main_stream));
        }
    } else {
        if (dst_nbytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(params.dst, cim_result.values.data(), dst_nbytes, cudaMemcpyHostToDevice, main_stream));
            CUDA_CHECK(cudaStreamSynchronize(main_stream));
        }
    }

    CUDA_CHECK(cudaEventDestroy(start_event));
}
