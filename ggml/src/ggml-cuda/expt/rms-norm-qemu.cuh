#pragma once

#include "../common.cuh"

enum ggml_cuda_rms_norm_qemu_mode {
    GGML_CUDA_RMS_NORM_QEMU_MODE_CUDA = 0,
    GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU,
    GGML_CUDA_RMS_NORM_QEMU_MODE_QEMU_CUDA,
    GGML_CUDA_RMS_NORM_QEMU_MODE_COMPARE,
};

struct ggml_cuda_rms_norm_qemu_params {
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
        const ggml_cuda_rms_norm_qemu_params & params,
        cudaStream_t stream);

ggml_cuda_rms_norm_qemu_mode ggml_cuda_rms_norm_qemu_get_mode();
bool ggml_cuda_rms_norm_qemu_enabled();

void ggml_cuda_rms_norm_qemu_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rms_norm_qemu_params & params,
        ggml_cuda_rms_norm_launch_fn cuda_launch);
