#pragma once

#include "../../common.cuh"

void ggml_cuda_fp8_log_e4m3_e8m0_32_e4m2_cpy_once(
        const char * path,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        bool enabled);

void ggml_cuda_fp8_log_e4m3_e8m0_32_e4m2_set_rows_once(
        const ggml_tensor * dst,
        bool enabled);
