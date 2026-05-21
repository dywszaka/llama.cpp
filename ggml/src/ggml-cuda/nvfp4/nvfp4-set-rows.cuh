#pragma once

#include "../set-rows.cuh"
#include "nvfp4-common.cuh"

void ggml_cuda_set_rows_nvfp4(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst);

void ggml_cuda_set_rows_nvfp4_8(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst);
