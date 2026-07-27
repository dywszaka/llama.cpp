#pragma once

#define GGML_COMMON_DECL_CUDA
#include "../../../ggml-common.h"

#include <cuda_runtime.h>

#include <cstdint>

bool ggml_cuda_nvfp4_bf16_quant_enabled();
bool ggml_cuda_nvfp4_bf16_quant_trunc_nn_enabled();
bool ggml_cuda_nvfp4_bf16_quant_bf16_internal_enabled();
bool ggml_cuda_nvfp4_bf16_quant_bf16_block_scale_enabled();
bool ggml_cuda_nvfp4_trunc_bf16_input_enabled();

void ggml_cuda_nvfp4_abs_max_rows_f32(
        const float * src,
        float * amax_rows,
        int64_t ne00,
        int64_t ne01,
        int64_t s01,
        bool truncate_bf16_input,
        cudaStream_t stream);

void ggml_cuda_nvfp4_abs_max_tensor_f32(
        const float * src,
        float * amax,
        int64_t ne00,
        int64_t ne01,
        int64_t s01,
        bool truncate_bf16_input,
        cudaStream_t stream);

void ggml_cuda_nvfp4_prepare_dynamic_input_scales(
        const float * amax_rows,
        float * input_scales,
        float * global_scales,
        int64_t nrows,
        float out_scale,
        bool per_tensor_scale,
        cudaStream_t stream);

void ggml_cuda_nvfp4_quantize_rows_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        float global_scale,
        bool truncate_bf16_input,
        cudaStream_t stream);

void ggml_cuda_nvfp4_quantize_rows_dynamic_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * amax_rows,
        bool per_tensor_scale,
        bool truncate_bf16_input,
        cudaStream_t stream);

void ggml_cuda_nvfp4_quantize_rows_scales_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * global_scales,
        bool per_tensor_scale,
        bool truncate_bf16_input,
        cudaStream_t stream);

void ggml_cuda_nvfp4_quantize_rows_bf16_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * global_scales,
        bool per_tensor_scale,
        bool bf16_internal_arith,
        bool bf16_block_scale,
        cudaStream_t stream);

void ggml_cuda_nvfp4_quantize_rows_dynamic_bf16_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * amax_rows,
        bool per_tensor_scale,
        bool bf16_internal_arith,
        bool bf16_block_scale,
        cudaStream_t stream);
