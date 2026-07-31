#pragma once

#include "../../common.cuh"
#include "nvfp4-common.cuh"
#include "../../../ggml-quants.h"

struct ggml_cuda_nvfp4_native_scratch {
    ggml_cuda_pool_alloc<block_nvfp4> src1_q_nvfp4;
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_data;
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_scale;
    ggml_cuda_pool_alloc<float> dynamic_amax_rows;
    ggml_cuda_pool_alloc<float> dynamic_input_scales;
    ggml_cuda_pool_alloc<uint8_t> src0_repacked_data_tmp;
    ggml_cuda_pool_alloc<uint8_t> src0_repacked_scale_tmp;
    ggml_cuda_pool_alloc<float> dst_padded;

    explicit ggml_cuda_nvfp4_native_scratch(ggml_cuda_pool & pool) :
        src1_q_nvfp4(pool),
        src1_repacked_data(pool),
        src1_repacked_scale(pool),
        dynamic_amax_rows(pool),
        dynamic_input_scales(pool),
        src0_repacked_data_tmp(pool),
        src0_repacked_scale_tmp(pool),
        dst_padded(pool) {
    }
};

bool ggml_cuda_nvfp4_native_no_fallback_enabled();

bool ggml_cuda_mul_mat_nvfp4_native(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

bool ggml_cuda_mul_mat_nvfp4_native_device_weight_scale(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        const float * device_weight_scale,
        bool reciprocal_weight_scale);

bool ggml_cuda_mul_mat_nvfp4_native_device_weight_scale_stream(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        const float * device_weight_scale,
        bool reciprocal_weight_scale,
        cudaStream_t stream,
        ggml_cuda_nvfp4_native_scratch * scratch = nullptr);
