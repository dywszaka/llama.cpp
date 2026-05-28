#include "kcache-outlier.cuh"

#include "../../common.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

static bool env_flag_enabled(const char * name) {
    const char * value = getenv(name);
    if (value == nullptr) {
        return false;
    }

    return atoi(value) != 0;
}

static float env_float_or_default(const char * name, float default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const float parsed = strtof(value, &end);
    if (end == value || !std::isfinite(parsed)) {
        return default_value;
    }

    return parsed;
}

static int64_t env_i64_or_default(const char * name, int64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const long long parsed = strtoll(value, &end, 10);
    if (end == value || parsed <= 0) {
        return default_value;
    }

    return parsed;
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

static __global__ void k_extract_outliers(
        const float * __restrict__ src,
        const int64_t * __restrict__ dst_rows,
        int32_t * __restrict__ counts,
        int32_t * __restrict__ indices,
        float * __restrict__ values,
        float * __restrict__ residual_amax,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t src_stride,
        const int64_t dst_rows_stride,
        const int64_t sidecar_rows,
        const int64_t max_outliers,
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
    const int64_t src_off = row * src_stride;
    for (int64_t col = threadIdx.x; col < ne00; col += blockDim.x) {
        const float v = src[src_off + col];
        const float av = fabsf(v);
        if (av > threshold) {
            const int32_t slot = atomicAdd(counts + dst_row, 1);
            if ((int64_t) slot < max_outliers) {
                indices[dst_row * max_outliers + slot] = (int32_t) col;
                values [dst_row * max_outliers + slot] = v;
            }
        } else {
            local_amax = fmaxf(local_amax, av);
        }
    }

    __shared__ float shared[256];
    shared[threadIdx.x] = local_amax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        residual_amax[row] = shared[0] > 0.0f ? shared[0] : (counts[dst_row] > 0 ? 1.0f : 0.0f);
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

    const int32_t offset = atomicAdd(cursor, count);
    offsets[dst_row] = offset < compact_capacity ? offset : -1;
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
        const int64_t max_outliers,
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
    const int64_t row_offset = offsets != nullptr ? (int64_t) offsets[kv_pos] : kv_pos * max_outliers;
    const int64_t row_limit = offsets != nullptr ? compact_capacity : row_offset + max_outliers;
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

static __global__ void k_f16_set_rows_outliers(
        const float * __restrict__ src,
        const int64_t * __restrict__ dst_rows,
        half * __restrict__ dst,
        int32_t * __restrict__ counts,
        int32_t * __restrict__ indices,
        float * __restrict__ values,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t src_stride,
        const int64_t dst_rows_stride,
        const int64_t dst_stride,
        const int64_t sidecar_rows,
        const int64_t max_outliers,
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
    if (fabsf(v) > threshold) {
        const int32_t slot = atomicAdd(counts + dst_row, 1);
        if ((int64_t) slot < max_outliers) {
            indices[dst_row * max_outliers + slot] = (int32_t) col;
            values [dst_row * max_outliers + slot] = v;
        }
        dst[dst_row * dst_stride + col] = __float2half(0.0f);
    } else {
        dst[dst_row * dst_stride + col] = __float2half(v);
    }
}

static __global__ void k_f16_set_rows_compact_outliers(
        const float * __restrict__ src,
        const int64_t * __restrict__ dst_rows,
        half * __restrict__ dst,
        int32_t * __restrict__ counts,
        const int32_t * __restrict__ offsets,
        int32_t * __restrict__ indices,
        float * __restrict__ values,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t src_stride,
        const int64_t dst_rows_stride,
        const int64_t dst_stride,
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
    if (fabsf(v) > threshold) {
        const int32_t offset = offsets[dst_row];
        if (offset >= 0) {
            const int32_t slot = atomicAdd(counts + dst_row, 1);
            const int64_t entry = (int64_t) offset + slot;
            if (entry >= 0 && entry < compact_capacity) {
                indices[entry] = (int32_t) col;
                values[entry] = v;
            }
        }
        dst[dst_row * dst_stride + col] = __float2half(0.0f);
    } else {
        dst[dst_row * dst_stride + col] = __float2half(v);
    }
}

static bool can_copy_counts_for_log(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const cudaError_t err = cudaStreamIsCapturing(stream, &status);
    return err == cudaSuccess && status == cudaStreamCaptureStatusNone;
}

} // namespace

bool ggml_cuda_nvfp4_kcache_outlier_enabled() {
    static const bool value = env_flag_enabled("LLAMA_NVFP4_KCACHE_OUTLIER");
    return value;
}

bool ggml_cuda_nvfp4_kcache_outlier_log_enabled() {
    static const bool value = env_flag_enabled("LLAMA_NVFP4_KCACHE_OUTLIER_LOG");
    return value;
}

bool ggml_cuda_nvfp4_kcache_outlier_tensor_scale_enabled() {
    return env_flag_enabled("LLAMA_NVFP4_KCACHE_OUTLIER_TENSOR_SCALE");
}

float ggml_cuda_nvfp4_kcache_outlier_threshold() {
    static const float value = env_float_or_default("LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD", 16.0f);
    return value;
}

int64_t ggml_cuda_nvfp4_kcache_outlier_max() {
    static const int64_t value = env_i64_or_default("LLAMA_NVFP4_KCACHE_OUTLIER_MAX", 32);
    return value;
}

bool ggml_cuda_f16_kcache_outlier_enabled() {
    static const bool value = env_flag_enabled("LLAMA_F16_KCACHE_OUTLIER");
    return value;
}

bool ggml_cuda_f16_kcache_outlier_log_enabled() {
    static const bool value = env_flag_enabled("LLAMA_F16_KCACHE_OUTLIER_LOG");
    return value;
}

float ggml_cuda_f16_kcache_outlier_threshold() {
    static const float value = env_float_or_default("LLAMA_F16_KCACHE_OUTLIER_THRESHOLD", 16.0f);
    return value;
}

int64_t ggml_cuda_f16_kcache_outlier_max() {
    static const int64_t value = env_i64_or_default("LLAMA_F16_KCACHE_OUTLIER_MAX", 32);
    return value;
}

void ggml_cuda_nvfp4_kcache_outlier_extract(
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
        int64_t max_outliers,
        int64_t compact_capacity,
        float threshold,
        cudaStream_t stream) {
    if (ne01 <= 0) {
        return;
    }

    const int block_size = 256;
    const int reset_grid = (int) ((ne01 + block_size - 1) / block_size);
    k_reset_outlier_rows<<<reset_grid, block_size, 0, stream>>>(
            dst_rows, counts, offsets, ne01, sidecar_rows, dst_rows_stride);
    CUDA_CHECK(cudaGetLastError());

    if (offsets != nullptr) {
        GGML_ASSERT(cursor != nullptr);
        GGML_ASSERT(compact_capacity > 0);
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

        const dim3 block(block_size);
        const dim3 grid((uint32_t) ((ne00 + block_size - 1) / block_size), (uint32_t) ne01, 1);
        k_fill_compact_outliers<<<grid, block, 0, stream>>>(
                src, dst_rows, counts, offsets, indices, values,
                ne00, ne01, src_stride, dst_rows_stride, sidecar_rows, compact_capacity, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else {
        k_extract_outliers<<<(int) ne01, block_size, 0, stream>>>(
                src, dst_rows, counts, indices, values, residual_amax,
                ne00, ne01, src_stride, dst_rows_stride, sidecar_rows, max_outliers, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    if (ggml_cuda_nvfp4_kcache_outlier_log_enabled() && can_copy_counts_for_log(stream)) {
        std::vector<int64_t> dst_rows_h((size_t) ne01);
        std::vector<int32_t> counts_h((size_t) sidecar_rows);
        CUDA_CHECK(cudaMemcpyAsync(dst_rows_h.data(), dst_rows, (size_t) ne01 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(counts_h.data(), counts, (size_t) sidecar_rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int64_t total = 0;
        int32_t max_count = 0;
        int64_t overflow_rows = 0;
        for (int64_t i = 0; i < ne01; ++i) {
            const int64_t dst_row = dst_rows_h[(size_t) i];
            if (dst_row < 0 || dst_row >= sidecar_rows) {
                continue;
            }
            const int32_t c = counts_h[(size_t) dst_row];
            total += c;
            max_count = std::max(max_count, c);
            overflow_rows += offsets != nullptr ? 0 : (c > max_outliers ? 1 : 0);
        }

        GGML_LOG_INFO(
                "%s: rows=%lld threshold=%g stored_max=%lld compact_capacity=%lld total_outliers=%lld max_row_outliers=%d overflow_rows=%lld\n",
                __func__,
                (long long) ne01,
                (double) threshold,
                (long long) max_outliers,
                (long long) compact_capacity,
                (long long) total,
                max_count,
                (long long) overflow_rows);
    }
}

void ggml_cuda_f16_kcache_outlier_set_rows(
        const float * src,
        const int64_t * dst_rows,
        half * dst,
        int32_t * counts,
        int32_t * offsets,
        int32_t * cursor,
        int32_t * indices,
        float * values,
        int64_t ne00,
        int64_t ne01,
        int64_t src_stride,
        int64_t dst_rows_stride,
        int64_t dst_stride,
        int64_t sidecar_rows,
        int64_t max_outliers,
        int64_t compact_capacity,
        float threshold,
        cudaStream_t stream) {
    if (ne01 <= 0) {
        return;
    }

    const int block_size = 256;
    const int reset_grid = (int) ((ne01 + block_size - 1) / block_size);
    k_reset_outlier_rows<<<reset_grid, block_size, 0, stream>>>(
            dst_rows, counts, offsets, ne01, sidecar_rows, dst_rows_stride);
    CUDA_CHECK(cudaGetLastError());

    const dim3 block(block_size);
    const dim3 grid((uint32_t) ((ne00 + block_size - 1) / block_size), (uint32_t) ne01, 1);
    if (offsets != nullptr) {
        GGML_ASSERT(cursor != nullptr);
        GGML_ASSERT(compact_capacity > 0);
        k_count_outliers<<<(int) ne01, block_size, 0, stream>>>(
                src, dst_rows, counts, nullptr,
                ne00, ne01, src_stride, dst_rows_stride, sidecar_rows, threshold);
        CUDA_CHECK(cudaGetLastError());

        k_assign_compact_offsets<<<reset_grid, block_size, 0, stream>>>(
                dst_rows, counts, offsets, cursor,
                ne01, dst_rows_stride, sidecar_rows, compact_capacity);
        CUDA_CHECK(cudaGetLastError());

        k_reset_outlier_rows<<<reset_grid, block_size, 0, stream>>>(
                dst_rows, counts, nullptr, ne01, sidecar_rows, dst_rows_stride);
        CUDA_CHECK(cudaGetLastError());

        k_f16_set_rows_compact_outliers<<<grid, block, 0, stream>>>(
                src, dst_rows, dst, counts, offsets, indices, values,
                ne00, ne01, src_stride, dst_rows_stride, dst_stride,
                sidecar_rows, compact_capacity, threshold);
        CUDA_CHECK(cudaGetLastError());
    } else {
        k_f16_set_rows_outliers<<<grid, block, 0, stream>>>(
                src, dst_rows, dst, counts, indices, values,
                ne00, ne01, src_stride, dst_rows_stride, dst_stride,
                sidecar_rows, max_outliers, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    if (ggml_cuda_f16_kcache_outlier_log_enabled() && can_copy_counts_for_log(stream)) {
        std::vector<int64_t> dst_rows_h((size_t) ne01);
        std::vector<int32_t> counts_h((size_t) sidecar_rows);
        CUDA_CHECK(cudaMemcpyAsync(dst_rows_h.data(), dst_rows, (size_t) ne01 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(counts_h.data(), counts, (size_t) sidecar_rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        int64_t total = 0;
        int32_t max_count = 0;
        int64_t overflow_rows = 0;
        for (int64_t i = 0; i < ne01; ++i) {
            const int64_t dst_row = dst_rows_h[(size_t) i];
            if (dst_row < 0 || dst_row >= sidecar_rows) {
                continue;
            }
            const int32_t c = counts_h[(size_t) dst_row];
            total += c;
            max_count = std::max(max_count, c);
            overflow_rows += offsets != nullptr ? 0 : (c > max_outliers ? 1 : 0);
        }

        GGML_LOG_INFO(
                "%s: rows=%lld threshold=%g stored_max=%lld compact_capacity=%lld total_outliers=%lld max_row_outliers=%d overflow_rows=%lld\n",
                __func__,
                (long long) ne01,
                (double) threshold,
                (long long) max_outliers,
                (long long) compact_capacity,
                (long long) total,
                max_count,
                (long long) overflow_rows);
    }
}

void ggml_cuda_f16_kcache_outlier_apply_correction(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    const ggml_tensor * outlier_counts = ggml_tensor_get_nvfp4_kcache_outlier_counts(src0);
    const ggml_tensor * outlier_offsets = ggml_tensor_get_nvfp4_kcache_outlier_offsets(src0);
    const ggml_tensor * outlier_indices = ggml_tensor_get_nvfp4_kcache_outlier_indices(src0);
    const ggml_tensor * outlier_values = ggml_tensor_get_nvfp4_kcache_outlier_values(src0);
    if (outlier_counts == nullptr || outlier_indices == nullptr || outlier_values == nullptr) {
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[2] % src0->ne[2] == 0);
    GGML_ASSERT(src1->ne[3] % src0->ne[3] == 0);

    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];
    cudaStream_t stream = ctx.stream();

    for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
            const int64_t stream_id = i3 / r3;
            const char * q_base = (const char *) src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
            char * dst_base = (char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3];
            ggml_cuda_nvfp4_kcache_outlier_apply_correction(
                    (const int32_t *) ((const char *) outlier_counts->data + stream_id * outlier_counts->nb[3]),
                    outlier_offsets != nullptr ? (const int32_t *) ((const char *) outlier_offsets->data + stream_id * outlier_offsets->nb[3]) : nullptr,
                    (const int32_t *) ((const char *) outlier_indices->data + stream_id * outlier_indices->nb[3]),
                    (const float *)   ((const char *) outlier_values->data  + stream_id * outlier_values->nb[3]),
                    (const float *) q_base,
                    (float *) dst_base,
                    nullptr,
                    src0->ne[0],
                    src0->ne[1],
                    src1->ne[1],
                    src1->ne[2],
                    src0->ne[2],
                    i2,
                    outlier_indices->ne[0],
                    outlier_indices->ne[0],
                    src1->nb[0] / (int64_t) sizeof(float),
                    src1->nb[1] / (int64_t) sizeof(float),
                    dst->nb[0] / (int64_t) sizeof(float),
                    dst->nb[1] / (int64_t) sizeof(float),
                    stream);
        }
    }

    GGML_UNUSED(r2);
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
        int64_t max_outliers,
        int64_t compact_capacity,
        int64_t q_nb0_f32,
        int64_t q_nb1_f32,
        int64_t kq_nb0_f32,
        int64_t kq_nb1_f32,
        cudaStream_t stream) {
    const int64_t total = kv_len * q_len;
    if (total <= 0 || max_outliers <= 0) {
        return;
    }

    const int block_size = 256;
    const int grid_size = (int) ((total + block_size - 1) / block_size);
    k_apply_outlier_correction<<<grid_size, block_size, 0, stream>>>(
            counts, offsets, indices, values, q, kq, k_scale,
            head_dim, kv_len, q_len, q_heads, kv_heads, q_head, max_outliers, compact_capacity,
            q_nb0_f32, q_nb1_f32, kq_nb0_f32, kq_nb1_f32);
    CUDA_CHECK(cudaGetLastError());
}
