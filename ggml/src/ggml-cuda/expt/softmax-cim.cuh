#pragma once

#include "../common.cuh"

enum ggml_cuda_soft_max_cim_mode {
    GGML_CUDA_SOFT_MAX_CIM_MODE_CUDA = 0,
    GGML_CUDA_SOFT_MAX_CIM_MODE_CIM,
    GGML_CUDA_SOFT_MAX_CIM_MODE_COMPARE_CUDA,
    GGML_CUDA_SOFT_MAX_CIM_MODE_COMPARE_CIM,
};

enum ggml_cuda_soft_max_mask_type {
    GGML_CUDA_SOFT_MAX_MASK_NONE = 0,
    GGML_CUDA_SOFT_MAX_MASK_F16,
    GGML_CUDA_SOFT_MAX_MASK_F32,
};

struct ggml_cuda_soft_max_cim_params {
    const float * src0;
    const void * src1;
    const float * src2;
    float * dst;

    ggml_cuda_soft_max_mask_type mask_type;

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;
    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

using ggml_cuda_soft_max_launch_fn = void (*)(
        const ggml_cuda_soft_max_cim_params & params,
        cudaStream_t stream);

ggml_cuda_soft_max_cim_mode ggml_cuda_soft_max_cim_get_mode();
bool ggml_cuda_soft_max_cim_enabled();

void ggml_cuda_soft_max_cim_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_soft_max_cim_params & params,
        ggml_cuda_soft_max_launch_fn cuda_launch);
