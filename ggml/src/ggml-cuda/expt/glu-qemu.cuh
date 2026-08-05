#pragma once

#include "../common.cuh"

enum ggml_cuda_glu_qemu_mode {
    GGML_CUDA_GLU_QEMU_MODE_CUDA = 0,
    GGML_CUDA_GLU_QEMU_MODE_QEMU,
    GGML_CUDA_GLU_QEMU_MODE_QEMU_CUDA,
    GGML_CUDA_GLU_QEMU_MODE_COMPARE,
};

struct ggml_cuda_glu_qemu_params {
    const ggml_tensor * src0_tensor;
    const ggml_tensor * src1_tensor;
    const void * src0;
    const void * src1;
    void * dst;
    int64_t ne[4];
    int64_t ne1[4];
    int64_t s0[4];
    int64_t s1[4];
    int64_t sd[4];
    ggml_type src0_type;
    ggml_type src1_type;
    ggml_type dst_type;
};

using ggml_cuda_glu_launch_fn = void (*)(
        const ggml_cuda_glu_qemu_params & params,
        cudaStream_t stream);

ggml_cuda_glu_qemu_mode ggml_cuda_glu_qemu_get_mode();
bool ggml_cuda_glu_qemu_enabled();

void ggml_cuda_glu_qemu_run(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_tensor,
        const ggml_cuda_glu_qemu_params & params,
        ggml_cuda_glu_launch_fn cuda_launch);
