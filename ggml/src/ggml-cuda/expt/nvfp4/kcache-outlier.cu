#include "kcache-outlier.cuh"

#include "../../common.cuh"
#include "nvfp4-log.cuh"
#include "ggml-impl.h"

#include <cmath>
#include <cstdlib>

namespace {

static bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static __global__ void k_reset_outlier_rows(
        const int64_t * __restrict__ dst_rows,
        int32_t * __restrict__ counts,
        int32_t * __restrict__ offsets,
        const int64_t ne01,
        const int64_t sidecar_rows,
        const int64_t dst_rows_stride) {
    const int64_t row = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ne01) {
        return;
    }

    const int64_t dst_row = dst_rows[row * dst_rows_stride];
    if (dst_row < 0 || dst_row >= sidecar_rows) {
        return;
    }

    counts[dst_row] = 0;
    if (offsets != nullptr) {
        offsets[dst_row] = -1;
    }
}

static __global__ void k_count_outliers(
        const float * __restrict__ src,
        const int64_t * __restrict__ dst_rows,
        int32_t * __restrict__ counts,
        float * __restrict__ residual_amax,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t src_stride,
        const int64_t dst_rows_stride,
        const int64_t sidecar_rows,
        const float threshold) {
    const int64_t row = blockIdx.x;
    if (row >= ne01) {
        return;
    }

    const int64_t dst_row = dst_rows[row * dst_rows_stride];
    if (dst_row < 0 || dst_row >= sidecar_rows) {
        if (residual_amax != nullptr) {
            residual_amax[row] = 0.0f;
        }
        return;
    }

    float local_amax = 0.0f;
    int32_t local_count = 0;
    const int64_t src_off = row * src_stride;
    for (int64_t col = threadIdx.x; col < ne00; col += blockDim.x) {
        const float v = src[src_off + col];
        const float av = fabsf(v);
        if (av > threshold) {
            ++local_count;
        } else {
            local_amax = fmaxf(local_amax, av);
        }
    }

    __shared__ float shared_amax[256];
    __shared__ int32_t shared_count[256];
    shared_amax[threadIdx.x] = local_amax;
    shared_count[threadIdx.x] = local_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_amax[threadIdx.x] = fmaxf(shared_amax[threadIdx.x], shared_amax[threadIdx.x + stride]);
            shared_count[threadIdx.x] += shared_count[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        counts[dst_row] = shared_count[0];
        if (residual_amax != nullptr) {
            residual_amax[row] = shared_amax[0] > 0.0f ? shared_amax[0] : (shared_count[0] > 0 ? 1.0f : 0.0f);
        }
    }
}

static __global__ void k_assign_compact_offsets(
        const int64_t * __restrict__ dst_rows,
        const int32_t * __restrict__ counts,
        int32_t * __restrict__ offsets,
        int32_t * __restrict__ cursor,
        const int64_t ne01,
        const int64_t dst_rows_stride,
        const int64_t sidecar_rows,
        const int64_t compact_capacity) {
    const int64_t row = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ne01) {
        return;
    }

    const int64_t dst_row = dst_rows[row * dst_rows_stride];
    if (dst_row < 0 || dst_row >= sidecar_rows) {
        return;
    }

    const int32_t count = counts[dst_row];
    if (count <= 0) {
        offsets[dst_row] = -1;
        return;
    }

    atomicAdd(cursor, count);
    const int64_t row_capacity = compact_capacity / sidecar_rows;
    const int64_t offset = dst_row * row_capacity;
    offsets[dst_row] = row_capacity > 0 && count <= row_capacity && offset + count <= compact_capacity
            ? (int32_t) offset
            : -1;
}

static __global__ void k_fill_compact_outliers(
        const float * __restrict__ src,
        const int64_t * __restrict__ dst_rows,
        int32_t * __restrict__ counts,
        const int32_t * __restrict__ offsets,
        int32_t * __restrict__ indices,
        float * __restrict__ values,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t src_stride,
        const int64_t dst_rows_stride,
        const int64_t sidecar_rows,
        const int64_t compact_capacity,
        const float threshold) {
    const int64_t row = blockIdx.y;
    const int64_t col = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ne01 || col >= ne00) {
        return;
    }

    const int64_t dst_row = dst_rows[row * dst_rows_stride];
    if (dst_row < 0 || dst_row >= sidecar_rows) {
        return;
    }

    const float v = src[row * src_stride + col];
    if (fabsf(v) <= threshold) {
        return;
    }

    const int32_t offset = offsets[dst_row];
    if (offset < 0) {
        return;
    }

    const int32_t slot = atomicAdd(counts + dst_row, 1);
    const int64_t entry = (int64_t) offset + slot;
    if (entry >= 0 && entry < compact_capacity) {
        indices[entry] = (int32_t) col;
        values[entry] = v;
    }
}

static __global__ void k_fill_compact_outliers_deterministic(
        const float * __restrict__ src,
        const int64_t * __restrict__ dst_rows,
        int32_t * __restrict__ counts,
        const int32_t * __restrict__ offsets,
        int32_t * __restrict__ indices,
        float * __restrict__ values,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t src_stride,
        const int64_t dst_rows_stride,
        const int64_t sidecar_rows,
        const int64_t compact_capacity,
        const float threshold) {
    const int64_t row = blockIdx.x;
    if (row >= ne01) {
        return;
    }

    const int64_t dst_row = dst_rows[row * dst_rows_stride];
    if (dst_row < 0 || dst_row >= sidecar_rows) {
        return;
    }

    const int32_t offset = offsets[dst_row];
    if (offset < 0) {
        return;
    }

    int32_t slot = 0;
    const int64_t src_off = row * src_stride;
    for (int64_t col = 0; col < ne00; ++col) {
        const float v = src[src_off + col];
        if (fabsf(v) <= threshold) {
            continue;
        }

        const int64_t entry = (int64_t) offset + slot;
        if (entry >= 0 && entry < compact_capacity) {
            indices[entry] = (int32_t) col;
            values[entry] = v;
        }
        ++slot;
    }
    counts[dst_row] = slot;
}

static __global__ void k_apply_outlier_correction(
        const int32_t * __restrict__ counts,
        const int32_t * __restrict__ offsets,
        const int32_t * __restrict__ indices,
        const float * __restrict__ values,
        const float * __restrict__ q,
        float * __restrict__ kq,
        const float * __restrict__ k_scale,
        const int64_t head_dim,
        const int64_t kv_len,
        const int64_t q_len,
        const int64_t q_heads,
        const int64_t kv_heads,
        const int64_t q_head,
        const int64_t compact_capacity,
        const int64_t q_nb0_f32,
        const int64_t q_nb1_f32,
        const int64_t kq_nb0_f32,
        const int64_t kq_nb1_f32) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = kv_len * q_len;
    if (idx >= total || q_heads <= 0 || kv_heads <= 0) {
        return;
    }

    const int64_t kv_pos = idx % kv_len;
    const int64_t q_pos  = idx / kv_len;
    const int64_t gqa = q_heads / kv_heads;
    const int64_t kv_head = gqa > 0 ? (q_head / gqa) : q_head;
    const int64_t head_begin = kv_head * head_dim;
    const int64_t head_end = head_begin + head_dim;

    const int32_t count = counts[kv_pos];
    const int64_t row_offset = (int64_t) offsets[kv_pos];
    if (row_offset < 0) {
        return;
    }
    const int64_t row_limit = compact_capacity;
    const int64_t row_capacity = row_limit - row_offset;
    const int64_t n = min((int64_t) count, row_capacity);
    float corr = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t entry = row_offset + i;
        if (entry < 0 || entry >= row_limit) {
            continue;
        }

        const int64_t global_dim = indices[entry];
        if (global_dim < head_begin || global_dim >= head_end) {
            continue;
        }

        const int64_t local_dim = global_dim - head_begin;
        const float qv = q[q_pos * q_nb1_f32 + local_dim * q_nb0_f32];
        corr += values[entry] * qv;
    }

    if (k_scale != nullptr) {
        const float scale = k_scale[kv_pos];
        if (scale > 0.0f && isfinite(scale)) {
            corr /= scale;
        }
    }

    kq[kv_pos * kq_nb0_f32 + q_pos * kq_nb1_f32] += corr;
}

} // namespace

bool ggml_cuda_nvfp4_kcache_outlier_deterministic_fill_enabled() {
    return env_flag_enabled("LLAMA_NVFP4_KCACHE_OUTLIER_DETERMINISTIC_FILL");
}

bool ggml_cuda_nvfp4_kcache_outlier_no_correction_enabled() {
    return env_flag_enabled("LLAMA_NVFP4_KCACHE_OUTLIER_NO_CORRECTION");
}

bool ggml_cuda_nvfp4_kcache_outlier_fingerprint_enabled() {
    return env_flag_enabled("LLAMA_NVFP4_KCACHE_OUTLIER_FINGERPRINT");
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
        cudaStream_t stream) {
    if (ne01 <= 0) {
        return;
    }
    GGML_ASSERT(offsets != nullptr);
    GGML_ASSERT(cursor != nullptr);
    GGML_ASSERT(compact_capacity > 0);

    const int block_size = 256;
    const int reset_grid = (int) ((ne01 + block_size - 1) / block_size);
    k_reset_outlier_rows<<<reset_grid, block_size, 0, stream>>>(
            dst_rows, counts, offsets, ne01, sidecar_rows, dst_rows_stride);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(cursor, 0, sizeof(int32_t), stream));
    k_count_outliers<<<(int) ne01, block_size, 0, stream>>>(
            src, dst_rows, counts, residual_amax,
            ne00, ne01, src_stride, dst_rows_stride, sidecar_rows, threshold);
    CUDA_CHECK(cudaGetLastError());

    k_assign_compact_offsets<<<reset_grid, block_size, 0, stream>>>(
            dst_rows, counts, offsets, cursor,
            ne01, dst_rows_stride, sidecar_rows, compact_capacity);
    CUDA_CHECK(cudaGetLastError());

    k_reset_outlier_rows<<<reset_grid, block_size, 0, stream>>>(
            dst_rows, counts, nullptr, ne01, sidecar_rows, dst_rows_stride);
    CUDA_CHECK(cudaGetLastError());

    if (ggml_cuda_nvfp4_kcache_outlier_deterministic_fill_enabled()) {
        k_fill_compact_outliers_deterministic<<<(int) ne01, 1, 0, stream>>>(
                src, dst_rows, counts, offsets, indices, values,
                ne00, ne01, src_stride, dst_rows_stride, sidecar_rows, compact_capacity, threshold);
    } else {
        const dim3 block(block_size);
        const dim3 grid((uint32_t) ((ne00 + block_size - 1) / block_size), (uint32_t) ne01, 1);
        k_fill_compact_outliers<<<grid, block, 0, stream>>>(
                src, dst_rows, counts, offsets, indices, values,
                ne00, ne01, src_stride, dst_rows_stride, sidecar_rows, compact_capacity, threshold);
    }
    CUDA_CHECK(cudaGetLastError());

    if (ggml_cuda_nvfp4_kcache_outlier_fingerprint_enabled()) {
        ggml_cuda_nvfp4_log_kcache_outlier_fingerprint(
                __func__, target, src, dst_rows, counts, offsets, cursor, indices, values, residual_amax,
                ne00, ne01, src_stride, dst_rows_stride, sidecar_rows, compact_capacity, threshold, stream);
    }

    if (ggml_cuda_nvfp4_log_can_copy_from_stream(stream)) {
#ifndef NDEBUG
        ggml_cuda_nvfp4_log_kcache_outlier_counts(
                __func__, target, dst_rows, counts, offsets, cursor,
                ne01, dst_rows_stride, sidecar_rows, compact_capacity, compact_capacity, threshold, stream);
#else
        ggml_cuda_nvfp4_log_kcache_outlier_overflow_if_any(
                __func__, target, dst_rows, counts, offsets, cursor,
                ne01, dst_rows_stride, sidecar_rows, compact_capacity, threshold, stream);
#endif
    }
}

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
        cudaStream_t stream) {
    const int64_t total = kv_len * q_len;
    if (total <= 0 || compact_capacity <= 0 || ggml_cuda_nvfp4_kcache_outlier_no_correction_enabled()) {
        return;
    }
    GGML_ASSERT(offsets != nullptr);

    const int block_size = 256;
    const int grid_size = (int) ((total + block_size - 1) / block_size);
    k_apply_outlier_correction<<<grid_size, block_size, 0, stream>>>(
            counts, offsets, indices, values, q, kq, k_scale,
            head_dim, kv_len, q_len, q_heads, kv_heads, q_head, compact_capacity,
            q_nb0_f32, q_nb1_f32, kq_nb0_f32, kq_nb1_f32);
    CUDA_CHECK(cudaGetLastError());
}
