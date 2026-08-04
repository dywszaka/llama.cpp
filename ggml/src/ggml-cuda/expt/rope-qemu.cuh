#pragma once

#include "../common.cuh"

enum ggml_cuda_rope_qemu_mode {
    GGML_CUDA_ROPE_QEMU_MODE_CUDA = 0,
    GGML_CUDA_ROPE_QEMU_MODE_QEMU,
    GGML_CUDA_ROPE_QEMU_MODE_QEMU_CUDA,
    GGML_CUDA_ROPE_QEMU_MODE_COMPARE,
};

struct ggml_cuda_rope_qemu_params {
    const void * src0;
    const int32_t * positions;
    const float * freq_factors;
    void * dst;
    int64_t ne[4];
    int64_t s0[4];
    int64_t sd[4];
    ggml_type src0_type;
    ggml_type dst_type;
    int n_dims;
    int mode;
    int n_ctx_orig;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int sections[4];
    bool forward;
};

using ggml_cuda_rope_launch_fn = void (*)(
        const ggml_cuda_rope_qemu_params & params,
        cudaStream_t stream);

ggml_cuda_rope_qemu_mode ggml_cuda_rope_qemu_get_mode();
bool ggml_cuda_rope_qemu_enabled();
bool ggml_cuda_rope_qemu_supported(const ggml_cuda_rope_qemu_params & params);

void ggml_cuda_rope_qemu_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_rope_qemu_params & params,
        ggml_cuda_rope_launch_fn cuda_launch);
