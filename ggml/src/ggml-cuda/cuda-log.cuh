#pragma once

#include "common.cuh"

struct block_q8_1_mmq;

void ggml_cuda_log_fattn_tensor_brief_once(
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V,
        const ggml_tensor * dst);

void ggml_cuda_log_mul_mat_kqvp_once(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst);

void ggml_cuda_log_fp8_e4m3_e8m0_32_e4m2_cpy_once(
        const char * path,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        bool enabled);

void ggml_cuda_log_fp8_e4m3_e8m0_32_e4m2_set_rows_once(
        const ggml_tensor * dst,
        bool enabled);

void ggml_cuda_log_nvfp4_vcache_fast_update_once(bool enabled);

void ggml_cuda_log_nvfp4_block(
        const block_nvfp4 & block,
        const ggml_tensor * dst);

void ggml_cuda_log_f32_first4(
        const char * label,
        const float vals[4],
        const ggml_tensor * dst);

void ggml_cuda_log_block_q8_1_mmq(
        const block_q8_1_mmq & block,
        ggml_type type_x,
        const ggml_tensor * dst);
