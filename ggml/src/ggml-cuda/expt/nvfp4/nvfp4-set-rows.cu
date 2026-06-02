#include "nvfp4-set-rows.cuh"
#include "kcache-outlier.cuh"

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_nvfp4_set_rows(float x) {
    uint8_t best_index = 0;
    float best_err = fabsf((float) kvalues_nvfp4[0] - x);

#pragma unroll
    for (int i = 1; i < 16; ++i) {
        const float err = fabsf((float) kvalues_nvfp4[i] - x);
        if (err < best_err) {
            best_err = err;
            best_index = (uint8_t) i;
        }
    }

    return best_index;
}

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_e4m3_set_rows(float x) {
    uint8_t best_index = 0;
    float best_err = INFINITY;

    for (int i = 0; i < 256; ++i) {
        const float v = ggml_cuda_e4m3_to_fp32((uint8_t) i);
        if (!isfinite(v)) {
            continue;
        }

        const float err = fabsf(v - x);
        if (err < best_err) {
            best_err = err;
            best_index = (uint8_t) i;
        }
    }

    return best_index;
}

static __global__ void k_abs_max_f32_rows(
        const float * __restrict__ src0,
        float * __restrict__ amax,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t s01) {
    const int64_t row = blockIdx.x;
    if (row >= ne01) {
        return;
    }

    float local_max = 0.0f;
    const int64_t row_off = row * s01;
    for (int64_t i = threadIdx.x; i < ne00; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(src0[row_off + i]));
    }

    __shared__ float shared_max[CUDA_SET_ROWS_BLOCK_SIZE];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        amax[row] = shared_max[0];
    }
}

static __global__ void k_set_rows_nvfp4(
        const float * __restrict__ src0, const int64_t * __restrict__ src1, block_nvfp4 * __restrict__ dst,
        const int64_t ne00, const int64_t ne01,
        const int64_t s01,
        const int64_t s10,
        const int64_t s1,
        const float * __restrict__ amax_rows,
        const float threshold,
        const bool use_threshold_global_scale,
        const bool zero_outliers) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const int64_t row_off = (int64_t) i1 * s01;
    const float raw_xi = (lane_active && k0 < ne00) ? src0[row_off + k0] : 0.0f;
    const float xi = (zero_outliers && fabsf(raw_xi) > threshold) ? 0.0f : raw_xi;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    const int64_t dst_row = *(src1 + i1*s10);
    block_nvfp4 * dst_row_ptr = dst + dst_row*s1 / sizeof(block_nvfp4);

    float scale_f = 0.0f;
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_k_global_scale(amax_rows[i1], threshold, use_threshold_global_scale);
    if (lane == 0) {
        const float scale = (global_scale != 0.0f) ? (global_scale * (vmax / GGML_CUDA_NVFP4_FP4_MAX)) : 0.0f;
        const uint8_t scale_q = ggml_cuda_best_index_e4m3_set_rows(scale);
        dst_row_ptr[ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4_set_rows(xi * inv_scale);
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        dst_row_ptr[ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static __global__ void k_set_rows_nvfp4_8(
        const float * __restrict__ src0, const int64_t * __restrict__ src1, block_nvfp4_8 * __restrict__ dst,
        const int64_t ne00, const int64_t ne01,
        const int64_t s01,
        const int64_t s10,
        const int64_t s1,
        const float * __restrict__ amax_rows,
        const float threshold,
        const bool use_threshold_global_scale,
        const bool zero_outliers) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4_8;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4_8 + lane;

    const int64_t row_off = (int64_t) i1 * s01;
    const float raw_xi = (lane_active && k0 < ne00) ? src0[row_off + k0] : 0.0f;
    const float xi = (zero_outliers && fabsf(raw_xi) > threshold) ? 0.0f : raw_xi;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    const int64_t dst_row = *(src1 + i1*s10);
    block_nvfp4_8 * dst_row_ptr = dst + dst_row*s1 / sizeof(block_nvfp4_8);

    float scale_f = 0.0f;
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_k_global_scale(amax_rows[i1], threshold, use_threshold_global_scale);
    if (lane == 0) {
        const float scale = (global_scale != 0.0f) ? (global_scale * (vmax / GGML_CUDA_NVFP4_FP4_MAX)) : 0.0f;
        const uint8_t scale_q = ggml_cuda_best_index_e4m3_set_rows(scale);
        dst_row_ptr[ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4_set_rows(xi * inv_scale);
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        dst_row_ptr[ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static __global__ void k_set_rows_scale(
        const int64_t * __restrict__ src1,
        float * __restrict__ scale,
        const int64_t ne10,
        const int64_t s10,
        const float * __restrict__ amax_rows,
        const float threshold,
        const bool use_threshold_global_scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne10) {
        return;
    }

    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_k_global_scale(amax_rows[i], threshold, use_threshold_global_scale);
    const float input_scale = (global_scale != 0.0f && isfinite(global_scale)) ? (1.0f / global_scale) : 0.0f;
    const int64_t dst_row = *(src1 + i*s10);
    scale[dst_row] = input_scale;
}

static void ggml_cuda_set_rows_nvfp4_common(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst,
        bool qk8) {
    GGML_ASSERT(p.ne02 == 1 && p.ne03 == 1);
    GGML_ASSERT(p.ne10 == p.ne01 && p.ne11 == 1 && p.ne12 == 1 && p.ne13 == 1);
    GGML_ASSERT(p.ne00 % (qk8 ? QK_NVFP4_8 : QK_NVFP4) == 0);

    const ggml_tensor * scale_tensor = ggml_tensor_get_nvfp4_scale(dst);
    GGML_ASSERT(scale_tensor != nullptr);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->data != nullptr);
    const ggml_tensor * outlier_counts = ggml_tensor_get_nvfp4_kcache_outlier_counts(dst);
    const ggml_tensor * outlier_offsets = ggml_tensor_get_nvfp4_kcache_outlier_offsets(dst);
    const ggml_tensor * outlier_cursor = ggml_tensor_get_nvfp4_kcache_outlier_cursor(dst);
    const ggml_tensor * outlier_indices = ggml_tensor_get_nvfp4_kcache_outlier_indices(dst);
    const ggml_tensor * outlier_values = ggml_tensor_get_nvfp4_kcache_outlier_values(dst);
    const bool use_outliers =
            outlier_counts != nullptr &&
            outlier_offsets != nullptr &&
            outlier_cursor != nullptr &&
            outlier_indices != nullptr &&
            outlier_values != nullptr;
    const bool use_tensor_scale = use_outliers;
    const float outlier_threshold = p.kcache_outlier_threshold > 0.0f
            ? p.kcache_outlier_threshold
            : GGML_CUDA_NVFP4_KCACHE_OUTLIER_THRESHOLD;
    if (use_outliers) {
        GGML_ASSERT(outlier_counts->type == GGML_TYPE_I32);
        GGML_ASSERT(outlier_indices->type == GGML_TYPE_I32);
        GGML_ASSERT(outlier_values->type == GGML_TYPE_F32);
        GGML_ASSERT(outlier_counts->data != nullptr);
        GGML_ASSERT(outlier_indices->data != nullptr);
        GGML_ASSERT(outlier_values->data != nullptr);
        GGML_ASSERT(outlier_offsets->type == GGML_TYPE_I32);
        GGML_ASSERT(outlier_offsets->data != nullptr);
        GGML_ASSERT(outlier_cursor->type == GGML_TYPE_I32);
        GGML_ASSERT(outlier_cursor->data != nullptr);
    }

    ggml_cuda_pool_alloc<float> amax_d(ctx.pool(), (size_t) p.ne01);
    if (p.ne01 > 0) {
        if (use_outliers) {
            ggml_cuda_nvfp4_kcache_outlier_extract(
                    dst->name,
                    p.src0_d,
                    p.src1_d,
                    (int32_t *) outlier_counts->data,
                    (int32_t *) outlier_offsets->data,
                    (int32_t *) outlier_cursor->data,
                    (int32_t *) outlier_indices->data,
                    (float *) outlier_values->data,
                    amax_d.get(),
                    p.ne00,
                    p.ne01,
                    p.nb01/sizeof(float),
                    p.nb10/sizeof(int64_t),
                    outlier_counts->ne[0],
                    outlier_indices->ne[0],
                    outlier_threshold,
                    p.stream);
        } else {
            k_abs_max_f32_rows<<<(int) p.ne01, CUDA_SET_ROWS_BLOCK_SIZE, 0, p.stream>>>(
                    p.src0_d, amax_d.get(),
                    p.ne00, p.ne01,
                    p.nb01/sizeof(float));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (p.ne01 > 0) {
        if (qk8) {
            const dim3 block_size(WARP_SIZE);
            const dim3 grid_size((uint32_t) (p.ne00 / QK_NVFP4_8), (uint32_t) p.ne01, 1);
            k_set_rows_nvfp4_8<<<grid_size, block_size, 0, p.stream>>>(
                    p.src0_d, p.src1_d, (block_nvfp4_8 *) dst->data,
                    p.ne00, p.ne01,
                    p.nb01/sizeof(float),
                    p.nb10/sizeof(int64_t),
                    p.nb1,
                    amax_d.get(),
                    outlier_threshold,
                    use_tensor_scale,
                    use_outliers);
        } else {
            const dim3 block_size(QK_NVFP4);
            const dim3 grid_size((uint32_t) (p.ne00 / QK_NVFP4), (uint32_t) p.ne01, 1);
            k_set_rows_nvfp4<<<grid_size, block_size, 0, p.stream>>>(
                    p.src0_d, p.src1_d, (block_nvfp4 *) dst->data,
                    p.ne00, p.ne01,
                    p.nb01/sizeof(float),
                    p.nb10/sizeof(int64_t),
                    p.nb1,
                    amax_d.get(),
                    outlier_threshold,
                    use_tensor_scale,
                    use_outliers);
        }
        CUDA_CHECK(cudaGetLastError());

        const int scale_blocks = (int) ((p.ne10 + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE);
        k_set_rows_scale<<<scale_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, p.stream>>>(
                p.src1_d,
                (float *) scale_tensor->data,
                p.ne10,
                p.nb10/sizeof(int64_t),
                amax_d.get(),
                outlier_threshold,
                use_tensor_scale);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_set_rows_nvfp4(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst) {
    ggml_cuda_set_rows_nvfp4_common(ctx, p, dst, false);
}

void ggml_cuda_set_rows_nvfp4_8(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst) {
    ggml_cuda_set_rows_nvfp4_common(ctx, p, dst, true);
}
