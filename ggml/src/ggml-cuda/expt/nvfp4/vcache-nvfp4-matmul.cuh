#pragma once

#include "../../common.cuh"
#include "nvfp4-common.cuh"

bool ggml_cuda_mul_mat_vcache_nvfp4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

// Detached legacy fallback. Kept buildable for experiments, but intentionally not called by the main V-cache path.
bool ggml_cuda_mul_mat_vcache_nvfp4_fallback(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
