#pragma once

#include "../../common.cuh"
#include "nvfp4-common.cuh"

bool ggml_cuda_nvfp4_native_no_fallback_enabled();
bool ggml_cuda_nvfp4_native_pad_k_enabled();

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
