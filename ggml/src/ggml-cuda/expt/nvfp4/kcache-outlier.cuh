#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

struct ggml_backend_cuda_context;
struct ggml_tensor;

bool ggml_cuda_nvfp4_kcache_outlier_enabled();
bool ggml_cuda_nvfp4_kcache_outlier_log_enabled();
float ggml_cuda_nvfp4_kcache_outlier_threshold();
int64_t ggml_cuda_nvfp4_kcache_outlier_max();

bool ggml_cuda_f16_kcache_outlier_enabled();
bool ggml_cuda_f16_kcache_outlier_log_enabled();
float ggml_cuda_f16_kcache_outlier_threshold();
int64_t ggml_cuda_f16_kcache_outlier_max();

void ggml_cuda_nvfp4_kcache_outlier_extract(
        const float * src,
        const int64_t * dst_rows,
        int32_t * counts,
        int32_t * indices,
        float * values,
        float * residual_amax,
        int64_t ne00,
        int64_t ne01,
        int64_t src_stride,
        int64_t dst_rows_stride,
        int64_t sidecar_rows,
        int64_t max_outliers,
        float threshold,
        cudaStream_t stream);

void ggml_cuda_nvfp4_kcache_outlier_apply_correction(
        const int32_t * counts,
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
        int64_t max_outliers,
        int64_t q_nb0_f32,
        int64_t q_nb1_f32,
        int64_t kq_nb0_f32,
        int64_t kq_nb1_f32,
        cudaStream_t stream);

void ggml_cuda_f16_kcache_outlier_set_rows(
        const float * src,
        const int64_t * dst_rows,
        half * dst,
        int32_t * counts,
        int32_t * indices,
        float * values,
        int64_t ne00,
        int64_t ne01,
        int64_t src_stride,
        int64_t dst_rows_stride,
        int64_t dst_stride,
        int64_t sidecar_rows,
        int64_t max_outliers,
        float threshold,
        cudaStream_t stream);

void ggml_cuda_f16_kcache_outlier_apply_correction(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);
