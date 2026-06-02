#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

struct ggml_backend_cuda_context;
struct ggml_tensor;

static constexpr float GGML_CUDA_NVFP4_KCACHE_OUTLIER_GLOBAL_SCALE_MAX = 1344.0f;
static constexpr float GGML_CUDA_NVFP4_KCACHE_OUTLIER_THRESHOLD = 16.0f;

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_kcache_outlier_global_scale_from_amax(float amax) {
    return (amax > 0.0f && isfinite(amax)) ? (GGML_CUDA_NVFP4_KCACHE_OUTLIER_GLOBAL_SCALE_MAX / amax) : 0.0f;
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_kcache_outlier_k_global_scale(
        float row_amax,
        float threshold,
        bool tensor_scale_enabled) {
    const float amax = tensor_scale_enabled ? threshold : row_amax;
    return ggml_cuda_nvfp4_kcache_outlier_global_scale_from_amax(amax);
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_kcache_outlier_k_input_scale(
        float row_amax,
        float threshold,
        bool tensor_scale_enabled) {
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_k_global_scale(row_amax, threshold, tensor_scale_enabled);
    return (global_scale != 0.0f && isfinite(global_scale)) ? (1.0f / global_scale) : 0.0f;
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_kcache_outlier_q_global_scale(float amax) {
    return ggml_cuda_nvfp4_kcache_outlier_global_scale_from_amax(amax);
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_kcache_outlier_q_input_scale(float amax, float out_scale) {
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_q_global_scale(amax);
    return (global_scale != 0.0f && isfinite(global_scale)) ? (out_scale / global_scale) : 0.0f;
}

void ggml_cuda_nvfp4_kcache_outlier_extract(
        const char * target,
        const float * src,
        const int64_t * dst_rows,
        int32_t * counts,
        int32_t * offsets,
        int32_t * cursor,
        int32_t * indices,
        float * values,
        float * residual_amax,
        int64_t ne00,
        int64_t ne01,
        int64_t src_stride,
        int64_t dst_rows_stride,
        int64_t sidecar_rows,
        int64_t compact_capacity,
        float threshold,
        cudaStream_t stream);

void ggml_cuda_nvfp4_kcache_outlier_apply_correction(
        const int32_t * counts,
        const int32_t * offsets,
        const int32_t * indices,
        const float * values,
        const float * q,
        float * kq,
        const float * k_scale,
        int64_t head_dim,
        int64_t kv_len,
        int64_t q_len,
        int64_t q_heads,
        int64_t kv_heads,
        int64_t q_head,
        int64_t compact_capacity,
        int64_t q_nb0_f32,
        int64_t q_nb1_f32,
        int64_t kq_nb0_f32,
        int64_t kq_nb1_f32,
        cudaStream_t stream);
