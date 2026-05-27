#pragma once

#include "../../set-rows.cuh"

bool ggml_cuda_fp8_e4m3_e8m0_32_e4m2_experiment_enabled();

void ggml_cuda_set_rows_fp8_e4m3_e8m0_32(
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst,
        bool e4m2_experiment);

void ggml_cuda_set_rows_fp8_e4m3_e8m0_16(
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst);
