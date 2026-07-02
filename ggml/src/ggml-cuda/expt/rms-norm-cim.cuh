#pragma once

#include "../common.cuh"

enum ggml_cuda_rms_norm_cim_mode {
    GGML_CUDA_RMS_NORM_CIM_MODE_CUDA = 0,
    GGML_CUDA_RMS_NORM_CIM_MODE_CIM,
    GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CUDA,
    GGML_CUDA_RMS_NORM_CIM_MODE_COMPARE_CIM,
};

struct ggml_cuda_rms_norm_cim_params {
    const float * src0;
    float * dst;
    int ncols;
    int nrows;
    int nchannels;
    int nsamples;
    int64_t stride_row;
    int64_t stride_channel;
    int64_t stride_sample;
    float eps;
};

using ggml_cuda_rms_norm_launch_fn = void (*)(
        const float * src0,
        float * dst,
        int ncols,
        int nrows,
        int nchannels,
        int nsamples,
        int64_t stride_row,
        int64_t stride_channel,
        int64_t stride_sample,
        float eps,
        cudaStream_t stream);

ggml_cuda_rms_norm_cim_mode ggml_cuda_rms_norm_cim_get_mode();
bool ggml_cuda_rms_norm_cim_enabled();

void ggml_cuda_rms_norm_cim_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_cim_params & params,
        ggml_cuda_rms_norm_launch_fn cuda_launch);
