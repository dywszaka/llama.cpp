#pragma once

#include "../../common.cuh"
#include "nvfp4-common.cuh"

bool ggml_cuda_mul_mat_vcache_nvfp4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

bool ggml_cuda_mul_mat_vcache_nvfp4_qemu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

bool ggml_cuda_nvfp4_vcache_batched_enabled();

bool ggml_cuda_nvfp4_vcache_parallel_lt_enabled();

bool ggml_cuda_mul_mat_vcache_nvfp4_batched(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
