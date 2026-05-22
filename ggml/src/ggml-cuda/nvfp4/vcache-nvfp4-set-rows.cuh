#pragma once

#include "../set-rows.cuh"
#include "nvfp4-common.cuh"

bool ggml_cuda_is_nvfp4_vcache_transposed_set_rows(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst);

bool ggml_cuda_nvfp4_vcache_fast_update_enabled();

void ggml_cuda_op_set_rows_nvfp4_vcache(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * src0,
        const ggml_tensor * src1);
