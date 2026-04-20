#include "nvfp4-matmul.cuh"

#include "ggml-backend.h"
#include "../ggml-quants.h"

#include <cuda_fp8.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static constexpr float GGML_CUDA_NVFP4_FP4_MAX = 6.0f;
static constexpr float GGML_CUDA_NVFP4_E4M3_HALF_MAX = 224.0f;
static constexpr float GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX = GGML_CUDA_NVFP4_FP4_MAX * GGML_CUDA_NVFP4_E4M3_HALF_MAX;

#if defined(CUBLAS_VERSION)
#define GGML_CUDA_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VERSION >= 130000)
#elif defined(CUBLAS_VER_MAJOR)
#define GGML_CUDA_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VER_MAJOR >= 13)
#else
#define GGML_CUDA_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS 0
#endif

template<typename T>
static bool ggml_cuda_read_scalar_tensor(const ggml_tensor * t, T & out) {
    if (t == nullptr || !ggml_is_scalar(t) || t->data == nullptr) {
        return false;
    }

    if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(&out, t->data, sizeof(T));
        return true;
    }

    ggml_backend_tensor_get(t, &out, 0, sizeof(T));
    return true;
}

static bool ggml_cuda_fetch_input_scale_f32(const ggml_tensor * scale, float & out) {
    if (scale == nullptr || !ggml_is_scalar(scale)) {
        return false;
    }

    switch (scale->type) {
        case GGML_TYPE_F32:
            return ggml_cuda_read_scalar_tensor(scale, out);
        case GGML_TYPE_F16: {
            ggml_fp16_t v = 0;
            if (!ggml_cuda_read_scalar_tensor(scale, v)) {
                return false;
            }
            out = ggml_fp16_to_fp32(v);
            return true;
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t v = { 0 };
            if (!ggml_cuda_read_scalar_tensor(scale, v)) {
                return false;
            }
            out = ggml_bf16_to_fp32(v);
            return true;
        }
        default:
            return false;
    }
}

static __device__ __forceinline__ void ggml_cuda_atomic_max_f32(float * addr, float value) {
    int * addr_i = (int *) addr;
    int old = *addr_i;

    while (__int_as_float(old) < value) {
        const int assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}

static __global__ void ggml_cuda_nvfp4_abs_max_rows_f32(
        const float * __restrict__ src0,
        float * __restrict__ amax_rows,
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
        const float v = fabsf(src0[row_off + i]);
        local_max = fmaxf(local_max, v);
    }

    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        amax_rows[row] = shared_max[0];
    }
}

static __global__ void ggml_cuda_nvfp4_prepare_dynamic_input_scales_kernel(
        const float * __restrict__ amax_rows,
        float * __restrict__ input_scales,
        const int64_t nrows,
        const float out_scale) {
    const int64_t row = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }

    const float amax_f = amax_rows[row];
    const float global_scale = (amax_f > 0.0f && isfinite(amax_f)) ? (GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / amax_f) : 0.0f;
    input_scales[row] = (global_scale != 0.0f) ? (out_scale / global_scale) : 0.0f;
}

static __global__ void ggml_cuda_nvfp4_apply_column_scales_kernel(
        float * __restrict__ dst,
        const float * __restrict__ column_scales,
        const int64_t m,
        const int64_t n,
        const int64_t ld) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = m * n;
    if (idx >= total) {
        return;
    }

    const int64_t col = idx / m;
    const int64_t row = idx - col * m;
    dst[col * ld + row] *= column_scales[col];
}

static bool ggml_cuda_nvfp4_native_debug_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_NATIVE_DEBUG");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_cuda_nvfp4_native_no_fallback_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_cuda_nvfp4_scale_linear_layout_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_SCALE_LINEAR_LAYOUT");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_cuda_nvfp4_native_validate_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_NATIVE_VALIDATE");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_cuda_nvfp4_native_row_split_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_NATIVE_ROW_SPLIT");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static const char * ggml_cuda_nvfp4_scale_channel_attr_diag() {
#if GGML_CUDA_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS
    return "enabled via cublas version gate (CUBLAS_VERSION>=130000)";
#else
    return "disabled by cublas version gate (need CUBLAS_VERSION>=130000)";
#endif
}

static int ggml_cuda_nvfp4_build_cudart_version() {
#ifdef CUDART_VERSION
    return CUDART_VERSION;
#else
    return -1;
#endif
}

static int ggml_cuda_nvfp4_build_cublas_version() {
#ifdef CUBLAS_VERSION
    return CUBLAS_VERSION;
#elif defined(CUBLAS_VER_MAJOR) && defined(CUBLAS_VER_MINOR) && defined(CUBLAS_VER_PATCH)
    return CUBLAS_VER_MAJOR * 1000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH;
#else
    return -1;
#endif
}

static void ggml_cuda_nvfp4_build_cublas_triplet(int * major, int * minor, int * patch) {
    if (major) {
#ifdef CUBLAS_VER_MAJOR
        *major = CUBLAS_VER_MAJOR;
#else
        *major = -1;
#endif
    }
    if (minor) {
#ifdef CUBLAS_VER_MINOR
        *minor = CUBLAS_VER_MINOR;
#else
        *minor = -1;
#endif
    }
    if (patch) {
#ifdef CUBLAS_VER_PATCH
        *patch = CUBLAS_VER_PATCH;
#else
        *patch = -1;
#endif
    }
}

static float ggml_cuda_nvfp4_input_global_scale(
        const ggml_tensor * dst,
        bool * used_dynamic_scale = nullptr) {
    const ggml_tensor * scale = ggml_mul_mat_get_nvfp4_input_scale(dst);
    if (scale == nullptr) {
        if (used_dynamic_scale != nullptr) {
            *used_dynamic_scale = true;
        }
        return 1.0f;
    }

    if (used_dynamic_scale != nullptr) {
        *used_dynamic_scale = false;
    }

    float input_scale = 0.0f;
    if (!ggml_cuda_fetch_input_scale_f32(scale, input_scale) || input_scale == 0.0f || !std::isfinite(input_scale)) {
        static std::atomic<bool> logged(false);
        if (ggml_cuda_nvfp4_native_debug_enabled() || !logged.exchange(true)) {
            GGML_LOG_WARN("%s: invalid NVFP4 input scale for %s, fallback global_scale=1.0\n",
                    __func__, ggml_get_name(dst));
        }
        return 1.0f;
    }

    return 1.0f / input_scale;
}

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_nvfp4(float x) {
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

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_e4m3(float x) {
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

static __global__ void quantize_row_nvfp4_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        int64_t ne00,
        int64_t s01,
        float global_scale) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const int64_t row_off = (int64_t) i1 * s01;
    const float xi = (lane_active && k0 < ne00) ? x[row_off + k0] : 0.0f;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    float scale_f = 0.0f;
    if (lane == 0) {
        const float scale = global_scale * (vmax / 6.0f);
        const uint8_t scale_q = ggml_cuda_best_index_e4m3(scale);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4(xi * inv_scale);
    // Warp shuffles require all lanes in the mask to execute the intrinsic.
    // Compute neighbor nibble unconditionally, then only even active lanes store.
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static __global__ void quantize_row_nvfp4_dynamic_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ amax_rows) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;
    const float xi = (lane_active && k0 < ne00) ? x[(int64_t) i1 * s01 + k0] : 0.0f;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    float scale_f = 0.0f;
    const float amax_f = amax_rows[i1];
    const float global_scale = (amax_f > 0.0f && isfinite(amax_f)) ? (GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / amax_f) : 0.0f;
    if (lane == 0) {
        const float scale = (global_scale != 0.0f) ? (global_scale * (vmax / GGML_CUDA_NVFP4_FP4_MAX)) : 0.0f;
        const uint8_t scale_q = ggml_cuda_best_index_e4m3(scale);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4(xi * inv_scale);
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static void quantize_row_nvfp4_cuda(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        float global_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);

    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01, global_scale);
}

static void quantize_row_nvfp4_dynamic_cuda(
        const float * x,
        block_nvfp4 * y,
        const int64_t ne00,
        const int64_t s01,
        const int64_t ne01,
        const float * amax_rows,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);

    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);

    quantize_row_nvfp4_dynamic_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01, amax_rows);
}

struct ggml_cuda_nvfp4_split_matrix {
    const void * data = nullptr;
    const void * scale = nullptr;
    size_t data_nbytes = 0;
    size_t scale_nbytes = 0;
    int64_t scale_inner_padded = 0;
    int64_t scale_outer_padded = 0;
};

static inline int64_t ggml_cuda_pad_i64(int64_t x, int64_t a) {
    GGML_ASSERT(a > 0);
    return ((x + a - 1) / a) * a;
}

static __host__ __device__ __forceinline__ int64_t ggml_cuda_nvfp4_scale_tiled_index(
        int64_t outer,
        int64_t inner,
        int64_t n_inner_padded) {
    // cuBLASLt VEC16_UE4M3 scale tiling: [outer, inner] -> 128x4 tiled order.
    const int64_t outer_tile = outer / 128;
    const int64_t outer_in_tile = outer % 128;
    const int64_t inner_tile = inner / 4;
    const int64_t inner_in_tile = inner % 4;

    const int64_t tiles_per_outer_block = n_inner_padded / 4;
    const int64_t tile_base = (outer_tile * tiles_per_outer_block + inner_tile) * 512;
    const int64_t tile_offset = (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile;
    return tile_base + tile_offset;
}

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_lt_scale_from_ggml_scale_byte(uint8_t ggml_e) {
    const float scale_f = ggml_cuda_e4m3_to_fp32(ggml_e);
    if (!(scale_f > 0.0f) || !isfinite(scale_f)) {
        return 0;
    }

    return (uint8_t) __nv_cvt_float_to_fp8(scale_f, __NV_SATFINITE, __NV_E4M3);
}

static __global__ void ggml_cuda_nvfp4_split_blocks_kernel(
        const block_nvfp4 * __restrict__ in,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        int64_t nblk_k,
        int64_t n_outer_valid,
        int64_t row_data_bytes,
        int64_t n_inner_padded,
        int32_t linear_scale_layout) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = nblk_k * n_outer_valid;
    if (idx >= total) {
        return;
    }

    const int64_t outer = idx / nblk_k;
    const int64_t inner = idx % nblk_k;

    const block_nvfp4 v = in[idx];
    uint8_t * data_dst = out_data + outer * row_data_bytes + inner * (QK_NVFP4 / 2);
#pragma unroll
    for (int i = 0; i < QK_NVFP4 / 2; ++i) {
        data_dst[i] = v.qs[i];
    }

    const int64_t scale_idx = linear_scale_layout ?
            (outer * n_inner_padded + inner) :
            ggml_cuda_nvfp4_scale_tiled_index(outer, inner, n_inner_padded);
    out_scale[scale_idx] = ggml_cuda_nvfp4_lt_scale_from_ggml_scale_byte(v.e);
}

static void ggml_cuda_nvfp4_split_blocks_cuda(
        const block_nvfp4 * in,
        uint8_t * out_data,
        uint8_t * out_scale,
        int64_t ne_k,
        int64_t n_outer_valid,
        int64_t n_outer_alloc,
        int64_t * scale_inner_padded,
        int64_t * scale_outer_padded,
        size_t * data_nbytes,
        size_t * scale_nbytes,
        bool linear_scale_layout,
        cudaStream_t stream) {
    GGML_ASSERT(ne_k % QK_NVFP4 == 0);
    GGML_ASSERT(n_outer_valid >= 0 && n_outer_alloc >= n_outer_valid);

    const int64_t nblk_k = ne_k / QK_NVFP4;
    const int64_t row_data_bytes = ne_k / 2;
    const int64_t inner_padded = ggml_cuda_pad_i64(nblk_k, 4);
    const int64_t outer_padded = ggml_cuda_pad_i64(n_outer_alloc, 128);

    const size_t dn = (size_t) n_outer_alloc * (size_t) row_data_bytes;
    const size_t sn = (size_t) outer_padded * (size_t) inner_padded;

    if (scale_inner_padded) {
        *scale_inner_padded = inner_padded;
    }
    if (scale_outer_padded) {
        *scale_outer_padded = outer_padded;
    }
    if (data_nbytes) {
        *data_nbytes = dn;
    }
    if (scale_nbytes) {
        *scale_nbytes = sn;
    }

    CUDA_CHECK(cudaMemsetAsync(out_data, 0, dn, stream));
    CUDA_CHECK(cudaMemsetAsync(out_scale, 0, sn, stream));

    const int64_t total = nblk_k * n_outer_valid;
    if (total > 0) {
        const int block_size = 256;
        const int grid_size = (int) ((total + block_size - 1) / block_size);
        ggml_cuda_nvfp4_split_blocks_kernel<<<grid_size, block_size, 0, stream>>>(
                in, out_data, out_scale, nblk_k, n_outer_valid, row_data_bytes, inner_padded,
                linear_scale_layout ? 1 : 0);
        CUDA_CHECK(cudaGetLastError());
    }
}

static bool ggml_cuda_nvfp4_cache_key_match(
        const ggml_cuda_nvfp4_cache_entry & entry,
        const ggml_tensor * src0) {
    if (entry.src0_data != src0->data) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (entry.ne[i] != src0->ne[i] || entry.nb[i] != src0->nb[i]) {
            return false;
        }
    }

    return true;
}

#if GGML_CUDA_HAS_CUBLASLT
static bool ggml_cuda_nvfp4_get_repacked_src0(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        bool linear_scale_layout,
        cudaStream_t stream,
        ggml_cuda_pool_alloc<uint8_t> & transient_data,
        ggml_cuda_pool_alloc<uint8_t> & transient_scale,
        ggml_cuda_nvfp4_split_matrix & out) {
    const bool cacheable = src0->buffer != nullptr &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    auto & cache = ctx.nvfp4_repack_cache[ctx.device];
    if (cacheable) {
        for (const ggml_cuda_nvfp4_cache_entry & entry : cache) {
            if (ggml_cuda_nvfp4_cache_key_match(entry, src0) &&
                entry.data_repacked != nullptr &&
                entry.scale_repacked != nullptr) {
                out.data = entry.data_repacked;
                out.scale = entry.scale_repacked;
                out.data_nbytes = entry.data_nbytes;
                out.scale_nbytes = entry.scale_nbytes;
                out.scale_inner_padded = entry.scale_inner_padded;
                out.scale_outer_padded = entry.scale_outer_padded;
                return true;
            }
        }
    }

    const int64_t ne_k = src0->ne[0];
    const int64_t n_outer = src0->ne[1];
    if (ne_k % QK_NVFP4 != 0) {
        return false;
    }

    const size_t data_nbytes = (size_t) n_outer * (size_t) ne_k / 2;
    const size_t scale_nbytes = (size_t) ggml_cuda_pad_i64(n_outer, 128) * (size_t) ggml_cuda_pad_i64(ne_k / QK_NVFP4, 4);

    void * data_repacked = nullptr;
    void * scale_repacked = nullptr;

    if (cacheable) {
        cudaError_t err = cudaMalloc(&data_repacked, data_nbytes);
        if (err != cudaSuccess) {
            static std::atomic<bool> logged(false);
            if (ggml_cuda_nvfp4_native_debug_enabled() || !logged.exchange(true)) {
                GGML_LOG_WARN("%s: cudaMalloc failed for repacked src0 data (%zu bytes): %s\n",
                        __func__, data_nbytes, cudaGetErrorString(err));
            }
            return false;
        }

        err = cudaMalloc(&scale_repacked, scale_nbytes);
        if (err != cudaSuccess) {
            static std::atomic<bool> logged(false);
            if (ggml_cuda_nvfp4_native_debug_enabled() || !logged.exchange(true)) {
                GGML_LOG_WARN("%s: cudaMalloc failed for repacked src0 scale (%zu bytes): %s\n",
                        __func__, scale_nbytes, cudaGetErrorString(err));
            }
            cudaFree(data_repacked);
            return false;
        }
    } else {
        data_repacked = transient_data.alloc(ctx.pool(), data_nbytes);
        scale_repacked = transient_scale.alloc(ctx.pool(), scale_nbytes);
    }

    int64_t scale_inner_padded = 0;
    int64_t scale_outer_padded = 0;
    size_t data_nbytes_built = 0;
    size_t scale_nbytes_built = 0;
    ggml_cuda_nvfp4_split_blocks_cuda(
            (const block_nvfp4 *) src0->data,
            (uint8_t *) data_repacked,
            (uint8_t *) scale_repacked,
            ne_k,
            n_outer,
            n_outer,
            &scale_inner_padded,
            &scale_outer_padded,
            &data_nbytes_built,
            &scale_nbytes_built,
            linear_scale_layout,
            stream);

    if (cacheable) {
        ggml_cuda_nvfp4_cache_entry entry = {};
        entry.src0_data = src0->data;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            entry.ne[i] = src0->ne[i];
            entry.nb[i] = src0->nb[i];
        }
        entry.data_repacked = data_repacked;
        entry.scale_repacked = scale_repacked;
        entry.data_nbytes = data_nbytes_built;
        entry.scale_nbytes = scale_nbytes_built;
        entry.scale_inner_padded = scale_inner_padded;
        entry.scale_outer_padded = scale_outer_padded;
        cache.push_back(entry);
    }

    out.data = data_repacked;
    out.scale = scale_repacked;
    out.data_nbytes = data_nbytes_built;
    out.scale_nbytes = scale_nbytes_built;
    out.scale_inner_padded = scale_inner_padded;
    out.scale_outer_padded = scale_outer_padded;
    return true;
}
#endif

static ggml_tensor ggml_cuda_nvfp4_make_matrix_slice(
        const ggml_tensor * src,
        const int64_t i2,
        const int64_t i3) {
    ggml_tensor slice = *src;
    slice.data = (char *) src->data + i2 * src->nb[2] + i3 * src->nb[3];
    slice.ne[2] = 1;
    slice.ne[3] = 1;
    slice.nb[2] = slice.nb[1] * slice.ne[1];
    slice.nb[3] = slice.nb[2] * slice.ne[2];
    return slice;
}

static void ggml_cuda_nvfp4_materialize_contiguous_matrix(
        ggml_backend_cuda_context & ctx,
        ggml_tensor & slice,
        ggml_cuda_pool_alloc<char> & storage,
        cudaStream_t stream) {
    if (!ggml_is_transposed(&slice) && ggml_is_contiguous(&slice)) {
        return;
    }

    const size_t row_bytes = ggml_row_size(slice.type, slice.ne[0]);
    const size_t total_bytes = row_bytes * (size_t) slice.ne[1];
    storage.alloc(ctx.pool(), total_bytes);

    CUDA_CHECK(cudaMemcpy2DAsync(
            storage.get(),
            row_bytes,
            slice.data,
            slice.nb[1],
            row_bytes,
            slice.ne[1],
            cudaMemcpyDeviceToDevice,
            stream));

    slice.data = storage.get();
    slice.buffer = nullptr;
    slice.nb[0] = ggml_type_size(slice.type);
    slice.nb[1] = row_bytes;
    slice.nb[2] = row_bytes * slice.ne[1];
    slice.nb[3] = slice.nb[2] * slice.ne[2];
}

} // namespace

bool ggml_cuda_mul_mat_nvfp4_native(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if GGML_CUDA_HAS_CUBLASLT && GGML_CUDA_HAS_FP4 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(dst  != nullptr);

    const bool debug_enabled = ggml_cuda_nvfp4_native_debug_enabled();
    const bool no_fallback = ggml_cuda_nvfp4_native_no_fallback_enabled();
    const bool linear_scale_layout = ggml_cuda_nvfp4_scale_linear_layout_enabled();
    const bool validate_enabled = ggml_cuda_nvfp4_native_validate_enabled();
    const bool verbose_skip = debug_enabled || no_fallback;
    auto log_skip = [&](const char * reason) {
        if (verbose_skip) {
            GGML_LOG_WARN(
                    "%s: skip native NVFP4 path: %s | src0=%s src1=%s dst=%s "
                    "src0_type=%s src1_type=%s dst_type=%s "
                    "src0_ne=[%lld,%lld,%lld,%lld] src1_ne=[%lld,%lld,%lld,%lld] dst_ne=[%lld,%lld,%lld,%lld]\n",
                    __func__, reason,
                    ggml_get_name(src0), ggml_get_name(src1), ggml_get_name(dst),
                    ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type),
                    (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
                    (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
                    (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3]);
        }
    };

    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        log_skip("unsupported tensor types");
        return false;
    }

    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        if (!ggml_is_contiguous(dst)) {
            log_skip("batched path requires contiguous dst tensor");
            return false;
        }

        if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
            log_skip("batched tensor shape broadcast is not supported");
            return false;
        }

        const int64_t r2 = src1->ne[2] / src0->ne[2];
        const int64_t r3 = src1->ne[3] / src0->ne[3];
        cudaStream_t stream = ctx.stream();

        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                ggml_tensor src0_slice = ggml_cuda_nvfp4_make_matrix_slice(src0, i2 / r2, i3 / r3);
                ggml_tensor src1_slice = ggml_cuda_nvfp4_make_matrix_slice(src1, i2, i3);
                ggml_tensor dst_slice  = ggml_cuda_nvfp4_make_matrix_slice(dst,  i2, i3);
                ggml_cuda_pool_alloc<char> src0_contig(ctx.pool());
                ggml_cuda_pool_alloc<char> src1_contig(ctx.pool());

                ggml_cuda_nvfp4_materialize_contiguous_matrix(ctx, src0_slice, src0_contig, stream);
                ggml_cuda_nvfp4_materialize_contiguous_matrix(ctx, src1_slice, src1_contig, stream);

                if (!ggml_cuda_mul_mat_nvfp4_native(ctx, &src0_slice, &src1_slice, &dst_slice)) {
                    return false;
                }
            }
        }

        return true;
    }

    if (ggml_is_transposed(src0) || ggml_is_transposed(src1) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        log_skip("requires contiguous non-transposed tensors");
        return false;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne01 = src0->ne[1];
    if (src0->ne[0] != ne10 || dst->ne[0] != ne01 || dst->ne[1] != ne11) {
        log_skip("incompatible matrix dimensions");
        return false;
    }

#ifndef NDEBUG
    {
        // NVFP4 in ggml stores one extra E4M3 scale byte per 16 values.
        // CUDA_R_4F_E2M1 describes plain packed FP4 (2 values per byte) without in-band scale bytes.
        const size_t nvfp4_row_bytes = ((size_t) ne10 / QK_NVFP4) * sizeof(block_nvfp4);
        const size_t fp4_row_bytes   = (size_t) ne10 / 2;
        static std::atomic<bool> logged(false);
        if (nvfp4_row_bytes != fp4_row_bytes && (debug_enabled || !logged.exchange(true))) {
            GGML_LOG_WARN(
                    "%s: NVFP4 layout diagnostic for %s: row_bytes(nvfp4_block)=%zu vs row_bytes(CUDA_R_4F_E2M1)=%zu "
                    "(k=%lld, block_size=%zu). Splitting into separate data/scale channels for cuBLASLt.\n",
                    __func__, ggml_get_name(dst),
                    nvfp4_row_bytes, fp4_row_bytes,
                    (long long) ne10, sizeof(block_nvfp4));
        }
    }
#endif

    // cuBLASLt native FP4 matmul is restrictive on GEMM dimensions.
    // Keep static matrix dimensions (M/K) aligned and pad dynamic token dimension (N) when needed.
    if ((ne01 % 16) != 0 || (ne10 % 16) != 0) {
        log_skip("native FP4 requires M/K to be multiples of 16");
        return false;
    }
    const bool row_split_mode = ggml_cuda_nvfp4_native_row_split_enabled() && ne11 > 1;
    const int64_t ne11_padded = (ne11 + 15) & ~15LL;
    const int64_t lt_n = row_split_mode ? 16 : ne11_padded;
    const bool pad_n = ne11_padded != ne11;

    if (ne10 % QK_NVFP4 != 0) {
        log_skip("K dimension is not divisible by QK_NVFP4");
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t nblk_k = ne10 / QK_NVFP4;
    const int64_t scale_inner_padded = ggml_cuda_pad_i64(nblk_k, 4);
    const int64_t scale_outer_padded_b = ggml_cuda_pad_i64(ne11_padded, 128);

    float out_scale = 1.0f;
    if (const ggml_tensor * scale = ggml_mul_mat_get_nvfp4_weight_scale(dst)) {
        float scale_val = 0.0f;
        if (ggml_cuda_fetch_input_scale_f32(scale, scale_val) && std::isfinite(scale_val)) {
            out_scale = scale_val;
        }
    }

    ggml_cuda_pool_alloc<block_nvfp4> src1_q_nvfp4(ctx.pool(), (size_t) nblk_k * (size_t) ne11);
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_data(ctx.pool(), (size_t) ne11_padded * (size_t) ne10 / 2);
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_scale(ctx.pool(), (size_t) scale_outer_padded_b * (size_t) scale_inner_padded);
    ggml_cuda_pool_alloc<float> dynamic_amax_rows(ctx.pool(), (size_t) std::max<int64_t>(ne11, 1));
    ggml_cuda_pool_alloc<float> dynamic_input_scales(ctx.pool(), (size_t) std::max<int64_t>(ne11, 1));

    bool used_dynamic_scale = ggml_mul_mat_get_nvfp4_input_scale(dst) == nullptr;
    float global_scale = ggml_cuda_nvfp4_input_global_scale(dst, &used_dynamic_scale);
    if (used_dynamic_scale) {
        const int block_size = 256;
        ggml_cuda_nvfp4_abs_max_rows_f32<<<(int) ne11, block_size, 0, stream>>>(
                (const float *) src1->data,
                dynamic_amax_rows.get(),
                ne10,
                ne11,
                src1->nb[1] / (int64_t) sizeof(float));
        CUDA_CHECK(cudaGetLastError());

        const int scale_grid = (int) ((ne11 + block_size - 1) / block_size);
        ggml_cuda_nvfp4_prepare_dynamic_input_scales_kernel<<<scale_grid, block_size, 0, stream>>>(
                dynamic_amax_rows.get(),
                dynamic_input_scales.get(),
                ne11,
                out_scale);
        CUDA_CHECK(cudaGetLastError());

        quantize_row_nvfp4_dynamic_cuda(
                (const float *) src1->data, src1_q_nvfp4.get(),
                ne10, src1->nb[1] / (int64_t) sizeof(float), ne11,
                dynamic_amax_rows.get(), stream);
    } else {
        quantize_row_nvfp4_cuda(
                (const float *) src1->data, src1_q_nvfp4.get(),
                ne10, src1->nb[1] / (int64_t) sizeof(float), ne11,
                global_scale, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_nvfp4_split_blocks_cuda(
            src1_q_nvfp4.get(),
            src1_repacked_data.get(),
            src1_repacked_scale.get(),
            ne10,
            ne11,
            ne11_padded,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            linear_scale_layout,
            stream);

    ggml_cuda_nvfp4_split_matrix src0_repacked = {};
    ggml_cuda_pool_alloc<uint8_t> src0_repacked_data_tmp(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> src0_repacked_scale_tmp(ctx.pool());
    if (!ggml_cuda_nvfp4_get_repacked_src0(
                ctx, src0, linear_scale_layout, stream,
                src0_repacked_data_tmp, src0_repacked_scale_tmp,
                src0_repacked)) {
        static std::atomic<bool> logged(false);
        if (debug_enabled || !logged.exchange(true)) {
            GGML_LOG_WARN("%s: failed to prepare repacked src0 channels for %s\n", __func__, ggml_get_name(dst));
        }
        return false;
    }

    if (debug_enabled) {
        GGML_LOG_INFO(
                "%s: native channel layout for %s: "
                "A[data=%zu,scale=%zu,scale_shape=(outer=%lld,inner=%lld)] "
                "B[data=%zu,scale=%zu,scale_shape=(outer=%lld,inner=%lld)]\n",
                __func__, ggml_get_name(dst),
                src0_repacked.data_nbytes,
                src0_repacked.scale_nbytes,
                (long long) src0_repacked.scale_outer_padded,
                (long long) src0_repacked.scale_inner_padded,
                (size_t) ne11_padded * (size_t) ne10 / 2,
                (size_t) scale_outer_padded_b * (size_t) scale_inner_padded,
                (long long) scale_outer_padded_b,
                (long long) scale_inner_padded);
        GGML_LOG_INFO("%s: scale layout mode for %s: %s\n",
                __func__, ggml_get_name(dst), linear_scale_layout ? "linear" : "tiled-128x4");
        GGML_LOG_INFO("%s: alpha mode for %s: out_scale/global_scale (%s)\n",
                __func__, ggml_get_name(dst), used_dynamic_scale ? "dynamic-rhs" : "bound-scale");

        // Compare first-row source scale bytes against repacked channel bytes.
        // This helps verify channel split/indexing before Lt matmul.
        const int64_t dump_blocks = std::min<int64_t>(nblk_k, 4);
        if (dump_blocks > 0) {
            std::vector<block_nvfp4> a_blocks((size_t) dump_blocks);
            std::vector<block_nvfp4> b_blocks((size_t) dump_blocks);
            std::vector<uint8_t> a_scale_src((size_t) dump_blocks, 0);
            std::vector<uint8_t> b_scale_src((size_t) dump_blocks, 0);
            std::vector<uint8_t> a_scale_repacked((size_t) dump_blocks, 0);
            std::vector<uint8_t> b_scale_repacked((size_t) dump_blocks, 0);

            CUDA_CHECK(cudaMemcpyAsync(
                    a_blocks.data(),
                    (const char *) src0->data,
                    (size_t) dump_blocks * sizeof(block_nvfp4),
                    cudaMemcpyDeviceToHost,
                    stream));
            CUDA_CHECK(cudaMemcpyAsync(
                    b_blocks.data(),
                    (const char *) src1_q_nvfp4.get(),
                    (size_t) dump_blocks * sizeof(block_nvfp4),
                    cudaMemcpyDeviceToHost,
                    stream));

            for (int64_t i = 0; i < dump_blocks; ++i) {
                const int64_t scale_idx = linear_scale_layout ?
                        i : ggml_cuda_nvfp4_scale_tiled_index(0, i, scale_inner_padded);
                CUDA_CHECK(cudaMemcpyAsync(
                        &a_scale_repacked[(size_t) i],
                        (const uint8_t *) src0_repacked.scale + scale_idx,
                        sizeof(uint8_t),
                        cudaMemcpyDeviceToHost,
                        stream));
                CUDA_CHECK(cudaMemcpyAsync(
                        &b_scale_repacked[(size_t) i],
                        (const uint8_t *) src1_repacked_scale.get() + scale_idx,
                        sizeof(uint8_t),
                        cudaMemcpyDeviceToHost,
                        stream));
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int64_t i = 0; i < dump_blocks; ++i) {
                a_scale_src[(size_t) i] = a_blocks[(size_t) i].e;
                b_scale_src[(size_t) i] = b_blocks[(size_t) i].e;
            }

            GGML_LOG_INFO(
                    "%s: scale-byte probe %s row0 first4 A[src=%u,%u,%u,%u repacked=%u,%u,%u,%u] "
                    "B[src=%u,%u,%u,%u repacked=%u,%u,%u,%u]\n",
                    __func__,
                    ggml_get_name(dst),
                    (unsigned) (dump_blocks > 0 ? a_scale_src[0] : 0),
                    (unsigned) (dump_blocks > 1 ? a_scale_src[1] : 0),
                    (unsigned) (dump_blocks > 2 ? a_scale_src[2] : 0),
                    (unsigned) (dump_blocks > 3 ? a_scale_src[3] : 0),
                    (unsigned) (dump_blocks > 0 ? a_scale_repacked[0] : 0),
                    (unsigned) (dump_blocks > 1 ? a_scale_repacked[1] : 0),
                    (unsigned) (dump_blocks > 2 ? a_scale_repacked[2] : 0),
                    (unsigned) (dump_blocks > 3 ? a_scale_repacked[3] : 0),
                    (unsigned) (dump_blocks > 0 ? b_scale_src[0] : 0),
                    (unsigned) (dump_blocks > 1 ? b_scale_src[1] : 0),
                    (unsigned) (dump_blocks > 2 ? b_scale_src[2] : 0),
                    (unsigned) (dump_blocks > 3 ? b_scale_src[3] : 0),
                    (unsigned) (dump_blocks > 0 ? b_scale_repacked[0] : 0),
                    (unsigned) (dump_blocks > 1 ? b_scale_repacked[1] : 0),
                    (unsigned) (dump_blocks > 2 ? b_scale_repacked[2] : 0),
                    (unsigned) (dump_blocks > 3 ? b_scale_repacked[3] : 0));
        }

        // One-time deep probe: sample multiple outer/inner coordinates to validate
        // both scale channel and packed data channel mapping.
        static std::atomic<bool> deep_probe_logged(false);
        const bool run_deep_probe = !deep_probe_logged.exchange(true);
        if (run_deep_probe) {
            auto probe_matrix = [&](const char * tag,
                                    const block_nvfp4 * src_blocks,
                                    const void * repacked_data,
                                    const void * repacked_scale,
                                    int64_t outer_valid,
                                    int64_t scale_inner_padded_local) {
                if (outer_valid <= 0 || nblk_k <= 0) {
                    return;
                }

                std::vector<int64_t> outer_samples;
                auto push_unique_outer = [&](int64_t v) {
                    if (v < 0 || v >= outer_valid) {
                        return;
                    }
                    for (int64_t x : outer_samples) {
                        if (x == v) {
                            return;
                        }
                    }
                    if (outer_samples.size() < 4) {
                        outer_samples.push_back(v);
                    }
                };

                push_unique_outer(0);
                push_unique_outer(1);
                push_unique_outer(31);
                push_unique_outer(32);
                push_unique_outer(127);
                push_unique_outer(128);
                push_unique_outer(outer_valid - 1);
                if (outer_samples.size() < 4) {
                    push_unique_outer((outer_valid - 1) / 2);
                }

                std::vector<int64_t> inner_samples;
                auto push_unique_inner = [&](int64_t v) {
                    if (v < 0 || v >= nblk_k) {
                        return;
                    }
                    for (int64_t x : inner_samples) {
                        if (x == v) {
                            return;
                        }
                    }
                    if (inner_samples.size() < 4) {
                        inner_samples.push_back(v);
                    }
                };

                push_unique_inner(0);
                push_unique_inner(1);
                push_unique_inner(3);
                push_unique_inner(4);
                push_unique_inner(nblk_k / 2);
                push_unique_inner(nblk_k - 1);
                if (inner_samples.size() < 4) {
                    push_unique_inner(2);
                }

                const int64_t row_data_bytes = ne10 / 2;
                int samples = 0;
                int scale_mismatch = 0;
                int data_mismatch = 0;

                for (int64_t outer : outer_samples) {
                    for (int64_t inner : inner_samples) {
                        const int64_t src_idx = outer * nblk_k + inner;
                        const int64_t scale_idx = linear_scale_layout
                                ? (outer * scale_inner_padded_local + inner)
                                : ggml_cuda_nvfp4_scale_tiled_index(outer, inner, scale_inner_padded_local);
                        const int64_t data_off = outer * row_data_bytes + inner * (QK_NVFP4 / 2);

                        block_nvfp4 src_block = {};
                        uint8_t rep_scale = 0;
                        uint8_t rep_qs[QK_NVFP4 / 2] = { 0 };

                        CUDA_CHECK(cudaMemcpyAsync(
                                &src_block,
                                src_blocks + src_idx,
                                sizeof(src_block),
                                cudaMemcpyDeviceToHost,
                                stream));
                        CUDA_CHECK(cudaMemcpyAsync(
                                &rep_scale,
                                (const uint8_t *) repacked_scale + scale_idx,
                                sizeof(rep_scale),
                                cudaMemcpyDeviceToHost,
                                stream));
                        CUDA_CHECK(cudaMemcpyAsync(
                                rep_qs,
                                (const uint8_t *) repacked_data + data_off,
                                sizeof(rep_qs),
                                cudaMemcpyDeviceToHost,
                                stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        const bool scale_ok = (src_block.e == rep_scale);
                        const bool data_ok = (memcmp(src_block.qs, rep_qs, sizeof(rep_qs)) == 0);
                        samples += 1;
                        if (!scale_ok) {
                            scale_mismatch += 1;
                        }
                        if (!data_ok) {
                            data_mismatch += 1;
                        }

                        GGML_LOG_INFO(
                                "%s: deep-probe %s %s o=%lld i=%lld "
                                "scale[src=%u rep=%u ok=%d idx=%lld] "
                                "data[src0=%u src7=%u rep0=%u rep7=%u ok=%d off=%lld]\n",
                                __func__,
                                ggml_get_name(dst),
                                tag,
                                (long long) outer,
                                (long long) inner,
                                (unsigned) src_block.e,
                                (unsigned) rep_scale,
                                scale_ok ? 1 : 0,
                                (long long) scale_idx,
                                (unsigned) src_block.qs[0],
                                (unsigned) src_block.qs[QK_NVFP4 / 2 - 1],
                                (unsigned) rep_qs[0],
                                (unsigned) rep_qs[QK_NVFP4 / 2 - 1],
                                data_ok ? 1 : 0,
                                (long long) data_off);
                    }
                }

                GGML_LOG_INFO(
                        "%s: deep-probe summary %s %s samples=%d scale_mismatch=%d data_mismatch=%d\n",
                        __func__,
                        ggml_get_name(dst),
                        tag,
                        samples,
                        scale_mismatch,
                        data_mismatch);
            };

            probe_matrix(
                    "A",
                    (const block_nvfp4 *) src0->data,
                    src0_repacked.data,
                    src0_repacked.scale,
                    ne01,
                    src0_repacked.scale_inner_padded);
            probe_matrix(
                    "B",
                    (const block_nvfp4 *) src1_q_nvfp4.get(),
                    (const void *) src1_repacked_data.get(),
                    (const void *) src1_repacked_scale.get(),
                    ne11,
                    scale_inner_padded);
        }
    }

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    const bool use_temp_dst = pad_n || row_split_mode;
    ggml_cuda_pool_alloc<float> dst_padded(ctx.pool(), use_temp_dst ? (size_t) ne01 * (size_t) lt_n : 1);
    void * dst_data = use_temp_dst ? (void *) dst_padded.get() : dst->data;

    const char * stage = "matmul_desc_create";
    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cudaDataType_t scale_type = CUDA_R_32F;
        stage = "matmul_desc_set_scale_type";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type));
    }

    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    stage = "matmul_desc_set_transa";
    st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "matmul_desc_set_transb";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    }

#if GGML_CUDA_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        stage = "matmul_desc_set_a_scale_mode";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        stage = "matmul_desc_set_b_scale_mode";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const void * a_scale_ptr = src0_repacked.scale;
        stage = "matmul_desc_set_a_scale_ptr";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const void * b_scale_ptr = (const void *) src1_repacked_scale.get();
        stage = "matmul_desc_set_b_scale_ptr";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));
    }
#else
    static std::atomic<bool> logged(false);
    if (verbose_skip || !logged.exchange(true)) {
        int runtime_version = 0;
        int driver_version = 0;
        int cublas_runtime_version = 0;
        int cublas_build_major = -1;
        int cublas_build_minor = -1;
        int cublas_build_patch = -1;
        (void) cudaRuntimeGetVersion(&runtime_version);
        (void) cudaDriverGetVersion(&driver_version);
        (void) cublasGetVersion(ctx.cublas_handle(), &cublas_runtime_version);
        ggml_cuda_nvfp4_build_cublas_triplet(&cublas_build_major, &cublas_build_minor, &cublas_build_patch);
        GGML_LOG_WARN(
                "%s: native FP4 scale-channel attrs unavailable at compile-time for %s: %s "
                "(build_cudart=%d runtime=%d driver=%d build_cublas=%d[%d.%d.%d] runtime_cublas=%d)\n",
                __func__,
                ggml_get_name(dst),
                ggml_cuda_nvfp4_scale_channel_attr_diag(),
                ggml_cuda_nvfp4_build_cudart_version(),
                runtime_version,
                driver_version,
                ggml_cuda_nvfp4_build_cublas_version(),
                cublas_build_major,
                cublas_build_minor,
                cublas_build_patch,
                cublas_runtime_version);
    }
    log_skip("toolkit lacks cublasLt FP4 scale-channel attributes");
    if (op_desc != nullptr) {
        cublasLtMatmulDescDestroy(op_desc);
    }
    return false;
#endif

    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_a";
        st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) ne10, (uint64_t) ne01, (int64_t) ne10);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        stage = "layout_set_order_a";
        st = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_b";
        st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) ne10, (uint64_t) lt_n, (int64_t) ne10);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        stage = "layout_set_order_b";
        st = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_c";
        st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, (uint64_t) ne01, (uint64_t) lt_n, (int64_t) ne01);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        stage = "layout_set_order_c";
        st = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }

    // cuBLASLt FP4 path applies the channel scale directly to packed FP4 values.
    // For statically-bound activations we recover the missing 1/global_scale in alpha.
    // For dynamic RHS quantization we apply per-column input_scale after matmul.
    const float matmul_alpha = used_dynamic_scale ? 1.0f : ((global_scale != 0.0f) ? (out_scale / global_scale) : out_scale);

    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = row_split_mode ? "matmul_row_split" : "matmul";
        const float alpha = matmul_alpha;
        const float beta  = 0.0f;
        if (row_split_mode) {
            const char * dst_name = ggml_get_name(dst);
            std::vector<uint8_t> saved_full_scale_row;
            std::vector<uint8_t> saved_full_data_row;
            bool saved_full_q_row = false;
            static std::atomic<bool> row_split_logged(false);
            if (debug_enabled || !row_split_logged.exchange(true)) {
                GGML_LOG_WARN(
                        "%s: row-split active for %s ne11=%lld lt_n=%lld pad_n=%d alpha=%g scale_layout=%s\n",
                        __func__,
                        dst_name != nullptr ? dst_name : "(unnamed)",
                        (long long) ne11,
                        (long long) lt_n,
                        pad_n ? 1 : 0,
                        (double) alpha,
                        linear_scale_layout ? "linear" : "tiled-128x4");
            }
            if (dst_name != nullptr &&
                    strcmp(dst_name, "Qcur-scaled-0") == 0 &&
                    ne11 > 14) {
                const int64_t saved_row = 14;
                const int64_t row_data_bytes = ne10 / 2;
                saved_full_scale_row.resize((size_t) nblk_k);
                saved_full_data_row.resize((size_t) nblk_k * (QK_NVFP4 / 2));
                for (int64_t inner = 0; inner < nblk_k; ++inner) {
                    const int64_t full_scale_idx = linear_scale_layout
                            ? (saved_row * scale_inner_padded + inner)
                            : ggml_cuda_nvfp4_scale_tiled_index(saved_row, inner, scale_inner_padded);
                    const int64_t full_data_off = saved_row * row_data_bytes + inner * (QK_NVFP4 / 2);
                    CUDA_CHECK(cudaMemcpyAsync(
                            &saved_full_scale_row[(size_t) inner],
                            (const uint8_t *) src1_repacked_scale.get() + full_scale_idx,
                            sizeof(saved_full_scale_row[(size_t) inner]),
                            cudaMemcpyDeviceToHost,
                            stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                            saved_full_data_row.data() + inner * (QK_NVFP4 / 2),
                            (const uint8_t *) src1_repacked_data.get() + full_data_off,
                            QK_NVFP4 / 2,
                            cudaMemcpyDeviceToHost,
                            stream));
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
                saved_full_q_row = true;
            }
            for (int64_t row = 0; row < ne11 && st == CUBLAS_STATUS_SUCCESS; ++row) {
                ggml_cuda_nvfp4_split_blocks_cuda(
                        src1_q_nvfp4.get() + row * nblk_k,
                        src1_repacked_data.get(),
                        src1_repacked_scale.get(),
                        ne10,
                        1,
                        lt_n,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        linear_scale_layout,
                        stream);

                st = cublasLtMatmul(
                        ctx.cublaslt_handle(),
                        op_desc,
                        &alpha,
                        src0_repacked.data, a_desc,
                        src1_repacked_data.get(), b_desc,
                        &beta,
                        dst_data, c_desc,
                        dst_data, c_desc,
                        nullptr,
                        nullptr, 0,
                        stream);

                if (st == CUBLAS_STATUS_SUCCESS) {
                    if (used_dynamic_scale) {
                        stage = "post_scale_dynamic_rhs_row_split";
                        const int block_size = 256;
                        const int grid_size = (int) ((ne01 + block_size - 1) / block_size);
                        ggml_cuda_nvfp4_apply_column_scales_kernel<<<grid_size, block_size, 0, stream>>>(
                                dst_padded.get(),
                                dynamic_input_scales.get() + row,
                                ne01,
                                1,
                                ne01);
                        CUDA_CHECK(cudaGetLastError());
                    }

                    CUDA_CHECK(cudaMemcpyAsync(
                            (char *) dst->data + row * dst->nb[1],
                            dst_padded.get(),
                            (size_t) ne01 * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream));

                    if (dst_name != nullptr &&
                            strcmp(dst_name, "Qcur-scaled-0") == 0 &&
                            row == ne11 - 1 &&
                            ne01 > 3793) {
                        int scale_mismatch = 0;
                        int data_mismatch = 0;
                        uint8_t split_scale_8 = 0;
                        uint8_t saved_scale_8 = 0;
                        uint8_t split_scale_12 = 0;
                        uint8_t saved_scale_12 = 0;
                        uint8_t split_scale_60 = 0;
                        uint8_t saved_scale_60 = 0;

                        for (int64_t inner = 0; inner < nblk_k; ++inner) {
                            const int64_t split_scale_idx = linear_scale_layout
                                    ? inner
                                    : ggml_cuda_nvfp4_scale_tiled_index(0, inner, scale_inner_padded);
                            const int64_t split_data_off = inner * (QK_NVFP4 / 2);

                            uint8_t split_scale = 0;
                            uint8_t saved_scale = saved_full_q_row ? saved_full_scale_row[(size_t) inner] : 0;
                            uint8_t split_qs[QK_NVFP4 / 2] = { 0 };

                            CUDA_CHECK(cudaMemcpyAsync(
                                    &split_scale,
                                    (const uint8_t *) src1_repacked_scale.get() + split_scale_idx,
                                    sizeof(split_scale),
                                    cudaMemcpyDeviceToHost,
                                    stream));
                            CUDA_CHECK(cudaMemcpyAsync(
                                    split_qs,
                                    (const uint8_t *) src1_repacked_data.get() + split_data_off,
                                    sizeof(split_qs),
                                    cudaMemcpyDeviceToHost,
                                    stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));

                            if (split_scale != saved_scale) {
                                ++scale_mismatch;
                            }
                            if (saved_full_q_row &&
                                    memcmp(
                                        split_qs,
                                        saved_full_data_row.data() + inner * (QK_NVFP4 / 2),
                                        sizeof(split_qs)) != 0) {
                                ++data_mismatch;
                            }

                            if (inner == 8) {
                                split_scale_8 = split_scale;
                                saved_scale_8 = saved_scale;
                            } else if (inner == 12) {
                                split_scale_12 = split_scale;
                                saved_scale_12 = saved_scale;
                            } else if (inner == 60) {
                                split_scale_60 = split_scale;
                                saved_scale_60 = saved_scale;
                            }
                        }

                        float probe_v = 0.0f;
                        CUDA_CHECK(cudaMemcpyAsync(
                                &probe_v,
                                dst_padded.get() + 3793,
                                sizeof(probe_v),
                                cudaMemcpyDeviceToHost,
                                stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        GGML_LOG_WARN(
                                "%s: row-split input-compare %s r=%lld scale_mismatch=%d/%lld data_mismatch=%d/%lld "
                                "ib8=%u/%u ib12=%u/%u ib60=%u/%u\n",
                                __func__,
                                dst_name,
                                (long long) row,
                                scale_mismatch,
                                (long long) nblk_k,
                                data_mismatch,
                                (long long) nblk_k,
                                (unsigned) split_scale_8,
                                (unsigned) saved_scale_8,
                                (unsigned) split_scale_12,
                                (unsigned) saved_scale_12,
                                (unsigned) split_scale_60,
                                (unsigned) saved_scale_60);
                        GGML_LOG_WARN(
                                "%s: row-split probe %s r=%lld c=3793 temp_out=%g\n",
                                __func__,
                                dst_name,
                                (long long) row,
                                (double) probe_v);
                    }
                }
            }
        } else {
            st = cublasLtMatmul(
                    ctx.cublaslt_handle(),
                    op_desc,
                    &alpha,
                    src0_repacked.data, a_desc,
                    src1_repacked_data.get(), b_desc,
                    &beta,
                    dst_data, c_desc,
                    dst_data, c_desc,
                    nullptr,
                    nullptr, 0,
                    stream);
        }
    }

    if (st == CUBLAS_STATUS_SUCCESS && used_dynamic_scale && !row_split_mode && ne11 > 0) {
        stage = "post_scale_dynamic_rhs";
        const int block_size = 256;
        const int64_t total = ne01 * ne11;
        const int grid_size = (int) ((total + block_size - 1) / block_size);
        ggml_cuda_nvfp4_apply_column_scales_kernel<<<grid_size, block_size, 0, stream>>>(
                (float *) dst_data,
                dynamic_input_scales.get(),
                ne01,
                ne11,
                ne01);
        CUDA_CHECK(cudaGetLastError());
    }

    if (st == CUBLAS_STATUS_SUCCESS && pad_n && !row_split_mode) {
        CUDA_CHECK(cudaMemcpyAsync(
                dst->data, dst_padded.get(),
                (size_t) ne01 * (size_t) ne11 * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
    }


    if (st == CUBLAS_STATUS_SUCCESS && validate_enabled && !used_dynamic_scale && !row_split_mode && ne10 % QK_NVFP4 == 0 && ne01 > 0 && ne11 > 0) {
        static std::atomic<bool> logged(false);
        if (debug_enabled || !logged.exchange(true)) {
            const int64_t nblk = ne10 / QK_NVFP4;
            std::vector<block_nvfp4> w_row((size_t) nblk);
            std::vector<float> x_row((size_t) ne10);
            std::vector<block_nvfp4> x_q((size_t) nblk);
            std::vector<block_nvfp4> x_q_gpu((size_t) nblk);
            std::vector<float> x_roundtrip((size_t) ne10);
            std::vector<float> w_deq((size_t) ne10);
            std::vector<float> x_roundtrip_no_scale((size_t) ne10);
            std::vector<float> w_no_scale((size_t) ne10);
            std::vector<float> x_no_scale((size_t) ne10);

            const char * w_row_ptr = (const char *) src0->data + 0 * src0->nb[1];
            const char * x_row_ptr = (const char *) src1->data + 0 * src1->nb[1];
            CUDA_CHECK(cudaMemcpyAsync(w_row.data(), w_row_ptr, (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(x_row.data(), x_row_ptr, (size_t) ne10 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(x_q_gpu.data(), src1_q_nvfp4.get(), (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            quantize_row_nvfp4_ref(x_row.data(), x_q.data(), ne10, global_scale);
            dequantize_row_nvfp4(x_q.data(), x_roundtrip.data(), ne10, global_scale);
            dequantize_row_nvfp4(w_row.data(), w_deq.data(), ne10, 1.0f);

            int q_mismatch = 0;
            for (int64_t ib = 0; ib < nblk; ++ib) {
                if (x_q[(size_t) ib].e != x_q_gpu[(size_t) ib].e ||
                    memcmp(x_q[(size_t) ib].qs, x_q_gpu[(size_t) ib].qs, QK_NVFP4/2) != 0) {
                    q_mismatch += 1;
                }
            }
            if (q_mismatch > 0) {
                GGML_LOG_WARN(
                        "%s: src1 quant mismatch for %s row0 blocks: mismatches=%d/%lld "
                        "cpu_block0[e=%u qs0=%u qs7=%u] gpu_block0[e=%u qs0=%u qs7=%u]\n",
                        __func__,
                        ggml_get_name(dst),
                        q_mismatch,
                        (long long) nblk,
                        (unsigned) x_q[0].e,
                        (unsigned) x_q[0].qs[0],
                        (unsigned) x_q[0].qs[QK_NVFP4/2 - 1],
                        (unsigned) x_q_gpu[0].e,
                        (unsigned) x_q_gpu[0].qs[0],
                        (unsigned) x_q_gpu[0].qs[QK_NVFP4/2 - 1]);
            }

            quantize_row_nvfp4_ref(x_row.data(), x_q.data(), ne10, 1.0f);
            dequantize_row_nvfp4(x_q.data(), x_roundtrip_no_scale.data(), ne10, 1.0f);

            auto dequantize_row_nvfp4_no_scale = [](const block_nvfp4 * q, float * out, int64_t k) {
                GGML_ASSERT(k % QK_NVFP4 == 0);
                const int64_t nb = k / QK_NVFP4;
                for (int64_t ib = 0; ib < nb; ++ib) {
                    for (int j = 0; j < QK_NVFP4/2; ++j) {
                        const uint8_t packed = q[ib].qs[j];
                        out[ib*QK_NVFP4 + 2*j + 0] = (float) kvalues_nvfp4[packed & 0x0F];
                        out[ib*QK_NVFP4 + 2*j + 1] = (float) kvalues_nvfp4[packed >> 4];
                    }
                }
            };

            dequantize_row_nvfp4_no_scale(w_row.data(), w_no_scale.data(), ne10);
            dequantize_row_nvfp4_no_scale(x_q.data(), x_no_scale.data(), ne10);

            auto compute_refs_for_col = [&](int64_t out_col, double & ref_out, double & ref_no_a_scale_out, double & ref_no_b_scale_out, double & ref_no_ab_scale_out) {
                const char * w_col_ptr = (const char *) src0->data + out_col * src0->nb[1];
                CUDA_CHECK(cudaMemcpyAsync(w_row.data(), w_col_ptr, (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                dequantize_row_nvfp4(w_row.data(), w_deq.data(), ne10, 1.0f);
                dequantize_row_nvfp4_no_scale(w_row.data(), w_no_scale.data(), ne10);

                ref_out = 0.0;
                ref_no_a_scale_out = 0.0;
                ref_no_b_scale_out = 0.0;
                ref_no_ab_scale_out = 0.0;

                for (int64_t k = 0; k < ne10; ++k) {
                    ref_out             += (double) w_deq[(size_t) k]      * (double) x_roundtrip[(size_t) k];
                    ref_no_a_scale_out  += (double) w_no_scale[(size_t) k] * (double) x_roundtrip[(size_t) k];
                    ref_no_b_scale_out  += (double) w_deq[(size_t) k]      * (double) x_roundtrip_no_scale[(size_t) k];
                    ref_no_ab_scale_out += (double) w_no_scale[(size_t) k] * (double) x_no_scale[(size_t) k];
                }

                ref_out             *= (double) out_scale;
                ref_no_a_scale_out  *= (double) out_scale;
                ref_no_b_scale_out  *= (double) out_scale;
                ref_no_ab_scale_out *= (double) out_scale;
            };


            double ref = 0.0;
            double ref_no_a_scale = 0.0;
            double ref_no_b_scale = 0.0;
            double ref_no_ab_scale = 0.0;
            compute_refs_for_col(0, ref, ref_no_a_scale, ref_no_b_scale, ref_no_ab_scale);

            float native_v = 0.0f;
            const char * out_ptr = (const char *) dst_data + 0 * ne01 * (int64_t) sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(&native_v, out_ptr, sizeof(native_v), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const double abs_err = fabs((double) native_v - ref);
            const double rel_err = abs_err / (fabs(ref) + 1e-9);
            const double abs_err_no_a_scale = fabs((double) native_v - ref_no_a_scale);
            const double rel_err_no_a_scale = abs_err_no_a_scale / (fabs(ref_no_a_scale) + 1e-9);
            const double abs_err_no_b_scale = fabs((double) native_v - ref_no_b_scale);
            const double rel_err_no_b_scale = abs_err_no_b_scale / (fabs(ref_no_b_scale) + 1e-9);
            const double abs_err_no_ab_scale = fabs((double) native_v - ref_no_ab_scale);
            const double rel_err_no_ab_scale = abs_err_no_ab_scale / (fabs(ref_no_ab_scale) + 1e-9);
            GGML_LOG_WARN(
                    "%s: native validate %s out[0,0]: native=%g ref=%g abs=%g rel=%g "
                    "(global_scale=%g out_scale=%g alpha=%g alpha_mode=out_scale/global_scale scale_layout=%s)\n",
                    __func__,
                    ggml_get_name(dst),
                    (double) native_v,
                    ref,
                    abs_err,
                    rel_err,
                    (double) global_scale,
                    (double) out_scale,
                    (double) matmul_alpha,
                    linear_scale_layout ? "linear" : "tiled-128x4");
            GGML_LOG_WARN(
                    "%s: native validate %s alt-ref out[0,0]: "
                    "no_a_scale=%g abs=%g rel=%g | "
                    "no_b_scale=%g abs=%g rel=%g | "
                    "no_ab_scale=%g abs=%g rel=%g\n",
                    __func__,
                    ggml_get_name(dst),
                    ref_no_a_scale, abs_err_no_a_scale, rel_err_no_a_scale,
                    ref_no_b_scale, abs_err_no_b_scale, rel_err_no_b_scale,
                    ref_no_ab_scale, abs_err_no_ab_scale, rel_err_no_ab_scale);

            std::vector<int64_t> sample_cols;
            auto append_sample_col = [&](int64_t col) {
                if (col < 0 || col >= ne01) {
                    return;
                }
                for (int64_t existing : sample_cols) {
                    if (existing == col) {
                        return;
                    }
                }
                sample_cols.push_back(col);
            };
            append_sample_col(0);
            append_sample_col(1);
            append_sample_col(127);
            append_sample_col(ne01 / 2);
            append_sample_col(ne01 - 1);

            char sample_buf[1024];
            sample_buf[0] = '\0';
            int sample_off = 0;
            double max_sample_abs_err = abs_err;
            int64_t max_sample_col = 0;

            for (const int64_t out_col : sample_cols) {
                double sample_ref = ref;
                double sample_ref_no_a_scale = ref_no_a_scale;
                double sample_ref_no_b_scale = ref_no_b_scale;
                double sample_ref_no_ab_scale = ref_no_ab_scale;
                (void) sample_ref_no_a_scale;
                (void) sample_ref_no_b_scale;
                (void) sample_ref_no_ab_scale;
                float sample_native_v = native_v;

                if (out_col != 0) {
                    compute_refs_for_col(out_col, sample_ref, sample_ref_no_a_scale, sample_ref_no_b_scale, sample_ref_no_ab_scale);
                    const char * sample_out_ptr = (const char *) dst_data + out_col * (int64_t) sizeof(float);
                    CUDA_CHECK(cudaMemcpyAsync(&sample_native_v, sample_out_ptr, sizeof(sample_native_v), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }

                const double sample_abs_err = fabs((double) sample_native_v - sample_ref);
                if (sample_abs_err > max_sample_abs_err) {
                    max_sample_abs_err = sample_abs_err;
                    max_sample_col = out_col;
                }

                if (sample_off > 0 && sample_off < (int) sizeof(sample_buf)) {
                    sample_off += std::snprintf(sample_buf + sample_off, sizeof(sample_buf) - sample_off, " | ");
                }
                if (sample_off >= 0 && sample_off < (int) sizeof(sample_buf)) {
                    sample_off += std::snprintf(
                            sample_buf + sample_off,
                            sizeof(sample_buf) - sample_off,
                            "c%lld native=%g ref=%g abs=%g",
                            (long long) out_col,
                            (double) sample_native_v,
                            sample_ref,
                            sample_abs_err);
                }
            }

            GGML_LOG_WARN(
                    "%s: native validate %s sampled row0: %s (max_abs=%g at_col=%lld)\n",
                    __func__,
                    ggml_get_name(dst),
                    sample_buf,
                    max_sample_abs_err,
                    (long long) max_sample_col);

            enum scale_probe_mode {
                SCALE_PROBE_CURRENT_TILED = 0,
                SCALE_PROBE_LINEAR = 1,
                SCALE_PROBE_TRANSPOSED_LINEAR = 2,
                SCALE_PROBE_TRANSPOSED_TILED = 3,
            };

            auto scale_probe_mode_name = [&](scale_probe_mode mode) {
                switch (mode) {
                    case SCALE_PROBE_CURRENT_TILED:     return "cur";
                    case SCALE_PROBE_LINEAR:            return "lin";
                    case SCALE_PROBE_TRANSPOSED_LINEAR: return "tlin";
                    case SCALE_PROBE_TRANSPOSED_TILED:  return "ttile";
                    default:                            return "unknown";
                }
            };

            auto load_src0_repacked_col = [&](int64_t out_col, scale_probe_mode mode, std::vector<block_nvfp4> & out_blocks) {
                out_blocks.resize((size_t) nblk);
                const int64_t row_data_bytes = ne10 / 2;
                const int64_t transposed_inner_padded = ggml_cuda_pad_i64(ne01, 4);
                for (int64_t inner = 0; inner < nblk; ++inner) {
                    int64_t scale_idx = 0;
                    switch (mode) {
                        case SCALE_PROBE_CURRENT_TILED:
                            scale_idx = linear_scale_layout
                                    ? (out_col * src0_repacked.scale_inner_padded + inner)
                                    : ggml_cuda_nvfp4_scale_tiled_index(out_col, inner, src0_repacked.scale_inner_padded);
                            break;
                        case SCALE_PROBE_LINEAR:
                            scale_idx = out_col * src0_repacked.scale_inner_padded + inner;
                            break;
                        case SCALE_PROBE_TRANSPOSED_LINEAR:
                            scale_idx = inner * src0_repacked.scale_outer_padded + out_col;
                            break;
                        case SCALE_PROBE_TRANSPOSED_TILED:
                            scale_idx = ggml_cuda_nvfp4_scale_tiled_index(inner, out_col, transposed_inner_padded);
                            break;
                    }
                    const int64_t data_off = out_col * row_data_bytes + inner * (QK_NVFP4 / 2);

                    block_nvfp4 block = {};
                    CUDA_CHECK(cudaMemcpyAsync(
                            &block.e,
                            (const uint8_t *) src0_repacked.scale + scale_idx,
                            sizeof(block.e),
                            cudaMemcpyDeviceToHost,
                            stream));
                    CUDA_CHECK(cudaMemcpyAsync(
                            block.qs,
                            (const uint8_t *) src0_repacked.data + data_off,
                            sizeof(block.qs),
                            cudaMemcpyDeviceToHost,
                            stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    out_blocks[(size_t) inner] = block;
                }
            };

            auto load_src0_repacked_scale_col = [&](int64_t out_col, scale_probe_mode mode, std::vector<uint8_t> & out_scales) {
                out_scales.resize((size_t) nblk);
                const int64_t transposed_inner_padded = ggml_cuda_pad_i64(ne01, 4);
                for (int64_t inner = 0; inner < nblk; ++inner) {
                    int64_t scale_idx = 0;
                    switch (mode) {
                        case SCALE_PROBE_CURRENT_TILED:
                            scale_idx = linear_scale_layout
                                    ? (out_col * src0_repacked.scale_inner_padded + inner)
                                    : ggml_cuda_nvfp4_scale_tiled_index(out_col, inner, src0_repacked.scale_inner_padded);
                            break;
                        case SCALE_PROBE_LINEAR:
                            scale_idx = out_col * src0_repacked.scale_inner_padded + inner;
                            break;
                        case SCALE_PROBE_TRANSPOSED_LINEAR:
                            scale_idx = inner * src0_repacked.scale_outer_padded + out_col;
                            break;
                        case SCALE_PROBE_TRANSPOSED_TILED:
                            scale_idx = ggml_cuda_nvfp4_scale_tiled_index(inner, out_col, transposed_inner_padded);
                            break;
                    }
                    CUDA_CHECK(cudaMemcpyAsync(
                            &out_scales[(size_t) inner],
                            (const uint8_t *) src0_repacked.scale + scale_idx,
                            sizeof(out_scales[(size_t) inner]),
                            cudaMemcpyDeviceToHost,
                            stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            };

            auto compute_row_roundtrip = [&](int64_t row, std::vector<float> & x_src_row, std::vector<float> & x_roundtrip_row) {
                x_src_row.resize((size_t) ne10);
                x_roundtrip_row.resize((size_t) ne10);
                const char * row_ptr = (const char *) src1->data + row * src1->nb[1];
                CUDA_CHECK(cudaMemcpyAsync(x_src_row.data(), row_ptr, (size_t) ne10 * sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<block_nvfp4> q_tmp((size_t) nblk);
                quantize_row_nvfp4_ref(x_src_row.data(), q_tmp.data(), ne10, global_scale);
                dequantize_row_nvfp4(q_tmp.data(), x_roundtrip_row.data(), ne10, global_scale);
            };

            auto dot_host = [&](const std::vector<float> & a, const std::vector<float> & b) {
                double acc = 0.0;
                for (int64_t i = 0; i < ne10; ++i) {
                    acc += (double) a[(size_t) i] * (double) b[(size_t) i];
                }
                return acc;
            };

            auto run_src0_focus_probe = [&](const char * probe_tag, const std::vector<int64_t> & cols) {
                std::vector<int64_t> rows;
                auto push_row = [&](int64_t row) {
                    if (row < 0 || row >= ne11) {
                        return;
                    }
                    for (int64_t existing : rows) {
                        if (existing == row) {
                            return;
                        }
                    }
                    rows.push_back(row);
                };
                push_row(0);
                push_row(1);
                push_row(ne11 - 1);

                std::vector<block_nvfp4> w_src_blocks((size_t) nblk);
                std::vector<block_nvfp4> w_rep_blocks_cur;
                std::vector<block_nvfp4> w_rep_blocks_lin;
                std::vector<block_nvfp4> w_rep_blocks_tlin;
                std::vector<block_nvfp4> w_rep_blocks_ttile;
                std::vector<float> w_src_deq((size_t) ne10);
                std::vector<float> w_rep_cur_deq((size_t) ne10);
                std::vector<float> w_rep_lin_deq((size_t) ne10);
                std::vector<float> w_rep_tlin_deq((size_t) ne10);
                std::vector<float> w_rep_ttile_deq((size_t) ne10);
                std::vector<uint8_t> group_scale_bytes;
                std::vector<block_nvfp4> w_rep_group_blocks;
                std::vector<float> w_rep_group_deq((size_t) ne10);
                std::vector<float> x_src_row;
                std::vector<float> x_roundtrip_row;

                for (int64_t out_col : cols) {
                    if (out_col < 0 || out_col >= ne01) {
                        continue;
                    }

                    const char * w_col_ptr = (const char *) src0->data + out_col * src0->nb[1];
                    CUDA_CHECK(cudaMemcpyAsync(w_src_blocks.data(), w_col_ptr, (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    load_src0_repacked_col(out_col, SCALE_PROBE_CURRENT_TILED, w_rep_blocks_cur);
                    load_src0_repacked_col(out_col, SCALE_PROBE_LINEAR, w_rep_blocks_lin);
                    load_src0_repacked_col(out_col, SCALE_PROBE_TRANSPOSED_LINEAR, w_rep_blocks_tlin);
                    load_src0_repacked_col(out_col, SCALE_PROBE_TRANSPOSED_TILED, w_rep_blocks_ttile);

                    dequantize_row_nvfp4(w_src_blocks.data(), w_src_deq.data(), ne10, 1.0f);
                    dequantize_row_nvfp4(w_rep_blocks_cur.data(),   w_rep_cur_deq.data(),   ne10, 1.0f);
                    dequantize_row_nvfp4(w_rep_blocks_lin.data(),   w_rep_lin_deq.data(),   ne10, 1.0f);
                    dequantize_row_nvfp4(w_rep_blocks_tlin.data(),  w_rep_tlin_deq.data(),  ne10, 1.0f);
                    dequantize_row_nvfp4(w_rep_blocks_ttile.data(), w_rep_ttile_deq.data(), ne10, 1.0f);

                    auto weight_max_abs_vs_src = [&](const std::vector<float> & rep) {
                        double out = 0.0;
                        for (int64_t i = 0; i < ne10; ++i) {
                            const double d = fabs((double) rep[(size_t) i] - (double) w_src_deq[(size_t) i]);
                            if (d > out) {
                                out = d;
                            }
                        }
                        return out;
                    };

                    const double weight_max_abs_cur   = weight_max_abs_vs_src(w_rep_cur_deq);
                    const double weight_max_abs_lin   = weight_max_abs_vs_src(w_rep_lin_deq);
                    const double weight_max_abs_tlin  = weight_max_abs_vs_src(w_rep_tlin_deq);
                    const double weight_max_abs_ttile = weight_max_abs_vs_src(w_rep_ttile_deq);

                    std::vector<double> group_weight_max_abs(4, -1.0);
                    std::vector<std::vector<float>> group_deq(4);
                    const int64_t tile_base_col = (out_col / 128) * 128;
                    const int64_t tile_pos = out_col % 32;
                    const int cur_group = (int) ((out_col % 128) / 32);
                    for (int group = 0; group < 4; ++group) {
                        const int64_t scale_col = tile_base_col + group * 32 + tile_pos;
                        if (scale_col < 0 || scale_col >= ne01) {
                            continue;
                        }
                        load_src0_repacked_scale_col(scale_col, SCALE_PROBE_CURRENT_TILED, group_scale_bytes);
                        w_rep_group_blocks = w_rep_blocks_cur;
                        for (int64_t inner = 0; inner < nblk; ++inner) {
                            w_rep_group_blocks[(size_t) inner].e = group_scale_bytes[(size_t) inner];
                        }
                        group_deq[group].resize((size_t) ne10);
                        dequantize_row_nvfp4(w_rep_group_blocks.data(), group_deq[group].data(), ne10, 1.0f);
                        group_weight_max_abs[group] = weight_max_abs_vs_src(group_deq[group]);
                    }

                    auto ref_from = [&](const std::vector<float> & wv, const std::vector<float> & xv) {
                        return dot_host(wv, xv) * (double) out_scale;
                    };

                    for (int64_t row : rows) {
                        compute_row_roundtrip(row, x_src_row, x_roundtrip_row);
                        const double ref_src   = ref_from(w_src_deq,       x_roundtrip_row);
                        const double ref_cur   = ref_from(w_rep_cur_deq,   x_roundtrip_row);
                        const double ref_lin   = ref_from(w_rep_lin_deq,   x_roundtrip_row);
                        const double ref_tlin  = ref_from(w_rep_tlin_deq,  x_roundtrip_row);
                        const double ref_ttile = ref_from(w_rep_ttile_deq, x_roundtrip_row);

                        float actual_v = 0.0f;
                        const char * probe_out_ptr = (const char *) dst_data + (row * ne01 + out_col) * (int64_t) sizeof(float);
                        CUDA_CHECK(cudaMemcpyAsync(&actual_v, probe_out_ptr, sizeof(actual_v), cudaMemcpyDeviceToHost, stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        GGML_LOG_WARN(
                                "%s: src0-focus %s %s r=%lld c=%lld actual=%g "
                                "ref_src=%g abs_src=%g | "
                                "%s=%g abs=%g w=%g | "
                                "%s=%g abs=%g w=%g | "
                                "%s=%g abs=%g w=%g | "
                                "%s=%g abs=%g w=%g\n",
                                __func__,
                                ggml_get_name(dst),
                                probe_tag,
                                (long long) row,
                                (long long) out_col,
                                (double) actual_v,
                                ref_src,
                                fabs((double) actual_v - ref_src),
                                scale_probe_mode_name(SCALE_PROBE_CURRENT_TILED),
                                ref_cur,
                                fabs((double) actual_v - ref_cur),
                                weight_max_abs_cur,
                                scale_probe_mode_name(SCALE_PROBE_LINEAR),
                                ref_lin,
                                fabs((double) actual_v - ref_lin),
                                weight_max_abs_lin,
                                scale_probe_mode_name(SCALE_PROBE_TRANSPOSED_LINEAR),
                                ref_tlin,
                                fabs((double) actual_v - ref_tlin),
                                weight_max_abs_tlin,
                                scale_probe_mode_name(SCALE_PROBE_TRANSPOSED_TILED),
                                ref_ttile,
                                fabs((double) actual_v - ref_ttile),
                                weight_max_abs_ttile);

                        char group_buf[512];
                        group_buf[0] = '\0';
                        int group_off = 0;
                        for (int group = 0; group < 4; ++group) {
                            if (group_deq[group].empty()) {
                                continue;
                            }
                            const double ref_group = ref_from(group_deq[group], x_roundtrip_row);
                            group_off += std::snprintf(
                                    group_buf + group_off,
                                    sizeof(group_buf) - group_off,
                                    "%sg%d%s=%g abs=%g w=%g",
                                    group_off > 0 ? " | " : "",
                                    group,
                                    group == cur_group ? "*" : "",
                                    ref_group,
                                    fabs((double) actual_v - ref_group),
                                    group_weight_max_abs[group]);
                            if (group_off >= (int) sizeof(group_buf)) {
                                break;
                            }
                        }
                        GGML_LOG_WARN(
                                "%s: src0-focus-groups %s %s r=%lld c=%lld cur_group=%d pos=%lld %s\n",
                                __func__,
                                ggml_get_name(dst),
                                probe_tag,
                                (long long) row,
                                (long long) out_col,
                                cur_group,
                                (long long) tile_pos,
                                group_buf);
                    }
                }
            };

            auto run_block_focus_probe = [&](const char * probe_tag, int64_t row, int64_t out_col) {
                if (row < 0 || row >= ne11 || out_col < 0 || out_col >= ne01) {
                    return;
                }

                std::vector<block_nvfp4> w_src_blocks((size_t) nblk);
                std::vector<float> x_src_row;
                std::vector<float> x_roundtrip_row;
                std::vector<float> w_block_scaled((size_t) QK_NVFP4);
                std::vector<float> w_block_no_scale((size_t) QK_NVFP4);
                std::vector<float> w_block_alt((size_t) QK_NVFP4);

                const char * w_col_ptr = (const char *) src0->data + out_col * src0->nb[1];
                CUDA_CHECK(cudaMemcpyAsync(w_src_blocks.data(), w_col_ptr, (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                compute_row_roundtrip(row, x_src_row, x_roundtrip_row);

                float actual_v = 0.0f;
                const char * probe_out_ptr = (const char *) dst_data + (row * ne01 + out_col) * (int64_t) sizeof(float);
                CUDA_CHECK(cudaMemcpyAsync(&actual_v, probe_out_ptr, sizeof(actual_v), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                std::vector<double> block_ref((size_t) nblk, 0.0);
                std::vector<double> block_no_a((size_t) nblk, 0.0);
                std::vector<double> block_missing_a((size_t) nblk, 0.0);
                double ref_total = 0.0;
                const int cur_group = (int) ((out_col % 128) / 32);
                const int64_t tile_base_col = (out_col / 128) * 128;
                const int64_t tile_pos = out_col % 32;
                std::vector<uint8_t> group_scale_bytes[4];
                std::vector<uint8_t> cur_col_scale_bytes;
                for (int group = 0; group < 4; ++group) {
                    const int64_t scale_col = tile_base_col + group * 32 + tile_pos;
                    if (scale_col >= 0 && scale_col < ne01) {
                        load_src0_repacked_scale_col(scale_col, SCALE_PROBE_CURRENT_TILED, group_scale_bytes[group]);
                    }
                }
                load_src0_repacked_scale_col(out_col, SCALE_PROBE_CURRENT_TILED, cur_col_scale_bytes);

                for (int64_t ib = 0; ib < nblk; ++ib) {
                    dequantize_row_nvfp4(&w_src_blocks[(size_t) ib], w_block_scaled.data(), QK_NVFP4, 1.0f);
                    dequantize_row_nvfp4_no_scale(&w_src_blocks[(size_t) ib], w_block_no_scale.data(), QK_NVFP4);

                    double acc_ref = 0.0;
                    double acc_no_a = 0.0;
                    const int64_t k0 = ib * QK_NVFP4;
                    for (int64_t j = 0; j < QK_NVFP4; ++j) {
                        const double xv = (double) x_roundtrip_row[(size_t) (k0 + j)];
                        acc_ref  += (double) w_block_scaled[(size_t) j] * xv;
                        acc_no_a += (double) w_block_no_scale[(size_t) j] * xv;
                    }

                    block_ref[(size_t) ib] = acc_ref * (double) out_scale;
                    block_no_a[(size_t) ib] = acc_no_a * (double) out_scale;
                    block_missing_a[(size_t) ib] = block_ref[(size_t) ib] - block_no_a[(size_t) ib];
                    ref_total += block_ref[(size_t) ib];
                }

                const double deficit = ref_total - (double) actual_v;

                std::vector<int64_t> top_ref_idx;
                std::vector<int64_t> top_missing_a_idx;
                std::vector<int64_t> top_pos_ref_idx;
                std::vector<int64_t> top_pos_missing_a_idx;
                auto push_top_abs = [&](std::vector<int64_t> & dst_idx, const std::vector<double> & values, int64_t idx) {
                    const double v_abs = fabs(values[(size_t) idx]);
                    int insert_at = (int) dst_idx.size();
                    for (int i = 0; i < (int) dst_idx.size(); ++i) {
                        if (v_abs > fabs(values[(size_t) dst_idx[(size_t) i]])) {
                            insert_at = i;
                            break;
                        }
                    }
                    if (insert_at < 8) {
                        dst_idx.insert(dst_idx.begin() + insert_at, idx);
                        if (dst_idx.size() > 8) {
                            dst_idx.pop_back();
                        }
                    } else if ((int) dst_idx.size() < 8) {
                        dst_idx.push_back(idx);
                    }
                };
                auto push_top_pos = [&](std::vector<int64_t> & dst_idx, const std::vector<double> & values, int64_t idx) {
                    const double v = values[(size_t) idx];
                    if (v <= 0.0) {
                        return;
                    }
                    int insert_at = (int) dst_idx.size();
                    for (int i = 0; i < (int) dst_idx.size(); ++i) {
                        if (v > values[(size_t) dst_idx[(size_t) i]]) {
                            insert_at = i;
                            break;
                        }
                    }
                    if (insert_at < 8) {
                        dst_idx.insert(dst_idx.begin() + insert_at, idx);
                        if (dst_idx.size() > 8) {
                            dst_idx.pop_back();
                        }
                    } else if ((int) dst_idx.size() < 8) {
                        dst_idx.push_back(idx);
                    }
                };

                for (int64_t ib = 0; ib < nblk; ++ib) {
                    push_top_abs(top_ref_idx, block_ref, ib);
                    push_top_abs(top_missing_a_idx, block_missing_a, ib);
                    push_top_pos(top_pos_ref_idx, block_ref, ib);
                    push_top_pos(top_pos_missing_a_idx, block_missing_a, ib);
                }

                double top_pos_cum = 0.0;
                int top_pos_needed = 0;
                for (int64_t ib : top_pos_ref_idx) {
                    if (top_pos_cum < deficit) {
                        top_pos_cum += block_ref[(size_t) ib];
                        ++top_pos_needed;
                    }
                }

                double top_missing_a_cum[3] = { 0.0, 0.0, 0.0 };
                double top_missing_a_ref[3] = { ref_total, ref_total, ref_total };
                double top_sign_flip_cum[3] = { 0.0, 0.0, 0.0 };
                double top_sign_flip_ref[3] = { ref_total, ref_total, ref_total };
                for (int i = 0; i < 3 && i < (int) top_pos_missing_a_idx.size(); ++i) {
                    top_missing_a_cum[i] = (i > 0 ? top_missing_a_cum[i - 1] : 0.0) + block_missing_a[(size_t) top_pos_missing_a_idx[(size_t) i]];
                    top_missing_a_ref[i] = ref_total - top_missing_a_cum[i];
                }
                for (int i = 0; i < 3 && i < (int) top_pos_ref_idx.size(); ++i) {
                    top_sign_flip_cum[i] = (i > 0 ? top_sign_flip_cum[i - 1] : 0.0) + 2.0 * block_ref[(size_t) top_pos_ref_idx[(size_t) i]];
                    top_sign_flip_ref[i] = ref_total - top_sign_flip_cum[i];
                }

                double top_pos_ref_cum[3] = { 0.0, 0.0, 0.0 };
                double top_pos_no_a_cum[3] = { 0.0, 0.0, 0.0 };
                double top_pos_fit_factor[3] = { NAN, NAN, NAN };
                for (int i = 0; i < 3 && i < (int) top_pos_ref_idx.size(); ++i) {
                    const int64_t ib = top_pos_ref_idx[(size_t) i];
                    top_pos_ref_cum[i] = (i > 0 ? top_pos_ref_cum[i - 1] : 0.0) + block_ref[(size_t) ib];
                    top_pos_no_a_cum[i] = (i > 0 ? top_pos_no_a_cum[i - 1] : 0.0) + block_no_a[(size_t) ib];
                    const double base_without_subset = ref_total - top_pos_ref_cum[i];
                    if (top_pos_ref_cum[i] != 0.0) {
                        top_pos_fit_factor[i] = ((double) actual_v - base_without_subset) / top_pos_ref_cum[i];
                    }
                }

                char top_ref_buf[768];
                char top_missing_a_buf[768];
                char attenuation_buf[768];
                char selective_buf[2048];
                top_ref_buf[0] = '\0';
                top_missing_a_buf[0] = '\0';
                attenuation_buf[0] = '\0';
                selective_buf[0] = '\0';
                int top_ref_off = 0;
                int top_missing_a_off = 0;
                int attenuation_off = 0;
                int selective_off = 0;

                for (int64_t ib : top_ref_idx) {
                    top_ref_off += std::snprintf(
                            top_ref_buf + top_ref_off,
                            sizeof(top_ref_buf) - top_ref_off,
                            "%sib%lld=%g(e=%u)",
                            top_ref_off > 0 ? " | " : "",
                            (long long) ib,
                            block_ref[(size_t) ib],
                            (unsigned) w_src_blocks[(size_t) ib].e);
                    if (top_ref_off >= (int) sizeof(top_ref_buf)) {
                        break;
                    }
                }

                for (int64_t ib : top_missing_a_idx) {
                    top_missing_a_off += std::snprintf(
                            top_missing_a_buf + top_missing_a_off,
                            sizeof(top_missing_a_buf) - top_missing_a_off,
                            "%sib%lld=d%g(ref=%g noA=%g e=%u)",
                            top_missing_a_off > 0 ? " | " : "",
                            (long long) ib,
                            block_missing_a[(size_t) ib],
                            block_ref[(size_t) ib],
                            block_no_a[(size_t) ib],
                            (unsigned) w_src_blocks[(size_t) ib].e);
                    if (top_missing_a_off >= (int) sizeof(top_missing_a_buf)) {
                        break;
                    }
                }

                const int selective_n = std::min(3, (int) top_pos_missing_a_idx.size());
                for (int i = 0; i < selective_n; ++i) {
                    const int64_t ib = top_pos_missing_a_idx[(size_t) i];
                    const double ref_without_block = ref_total - block_ref[(size_t) ib];
                    const double missing_a_out = ref_total - block_missing_a[(size_t) ib];
                    const double missing_a_abs = fabs((double) actual_v - missing_a_out);

                    double best_group_out = ref_total;
                    double best_group_abs = fabs((double) actual_v - best_group_out);
                    int best_group = cur_group;
                    double best_group_block = block_ref[(size_t) ib];

                    for (int group = 0; group < 4; ++group) {
                        if (group == cur_group || group_scale_bytes[group].empty()) {
                            continue;
                        }
                        block_nvfp4 alt_block = w_src_blocks[(size_t) ib];
                        alt_block.e = group_scale_bytes[group][(size_t) ib];
                        dequantize_row_nvfp4(&alt_block, w_block_alt.data(), QK_NVFP4, 1.0f);

                        double alt_block_acc = 0.0;
                        const int64_t k0 = ib * QK_NVFP4;
                        for (int64_t j = 0; j < QK_NVFP4; ++j) {
                            alt_block_acc += (double) w_block_alt[(size_t) j] * (double) x_roundtrip_row[(size_t) (k0 + j)];
                        }
                        const double alt_block_out = alt_block_acc * (double) out_scale;
                        const double alt_total = ref_without_block + alt_block_out;
                        const double alt_abs = fabs((double) actual_v - alt_total);
                        if (alt_abs < best_group_abs) {
                            best_group_abs = alt_abs;
                            best_group_out = alt_total;
                            best_group = group;
                            best_group_block = alt_block_out;
                        }
                    }

                    double best_inner_out = ref_total;
                    double best_inner_abs = fabs((double) actual_v - best_inner_out);
                    int64_t best_inner_src = ib;
                    double best_inner_block = block_ref[(size_t) ib];
                    uint8_t best_inner_e = w_src_blocks[(size_t) ib].e;
                    const int inner_offsets[] = { -8, -4, -2, -1, 1, 2, 4, 8 };
                    for (int delta : inner_offsets) {
                        const int64_t src_ib = ib + delta;
                        if (src_ib < 0 || src_ib >= nblk || cur_col_scale_bytes.empty()) {
                            continue;
                        }
                        block_nvfp4 alt_block = w_src_blocks[(size_t) ib];
                        alt_block.e = cur_col_scale_bytes[(size_t) src_ib];
                        dequantize_row_nvfp4(&alt_block, w_block_alt.data(), QK_NVFP4, 1.0f);

                        double alt_block_acc = 0.0;
                        const int64_t k0 = ib * QK_NVFP4;
                        for (int64_t j = 0; j < QK_NVFP4; ++j) {
                            alt_block_acc += (double) w_block_alt[(size_t) j] * (double) x_roundtrip_row[(size_t) (k0 + j)];
                        }
                        const double alt_block_out = alt_block_acc * (double) out_scale;
                        const double alt_total = ref_without_block + alt_block_out;
                        const double alt_abs = fabs((double) actual_v - alt_total);
                        if (alt_abs < best_inner_abs) {
                            best_inner_abs = alt_abs;
                            best_inner_out = alt_total;
                            best_inner_src = src_ib;
                            best_inner_block = alt_block_out;
                            best_inner_e = alt_block.e;
                        }
                    }

                    double best_e_out = ref_total;
                    double best_e_abs = fabs((double) actual_v - best_e_out);
                    double best_e_block = block_ref[(size_t) ib];
                    uint8_t best_e_byte = w_src_blocks[(size_t) ib].e;
                    for (int e_byte = 0; e_byte < 256; ++e_byte) {
                        block_nvfp4 alt_block = w_src_blocks[(size_t) ib];
                        alt_block.e = (uint8_t) e_byte;
                        dequantize_row_nvfp4(&alt_block, w_block_alt.data(), QK_NVFP4, 1.0f);

                        double alt_block_acc = 0.0;
                        const int64_t k0 = ib * QK_NVFP4;
                        for (int64_t j = 0; j < QK_NVFP4; ++j) {
                            alt_block_acc += (double) w_block_alt[(size_t) j] * (double) x_roundtrip_row[(size_t) (k0 + j)];
                        }
                        const double alt_block_out = alt_block_acc * (double) out_scale;
                        const double alt_total = ref_without_block + alt_block_out;
                        const double alt_abs = fabs((double) actual_v - alt_total);
                        if (alt_abs < best_e_abs) {
                            best_e_abs = alt_abs;
                            best_e_out = alt_total;
                            best_e_block = alt_block_out;
                            best_e_byte = (uint8_t) e_byte;
                        }
                    }

                    selective_off += std::snprintf(
                            selective_buf + selective_off,
                            sizeof(selective_buf) - selective_off,
                            "%sib%lld missA=%g abs=%g best_g%d=%g abs=%g best_inner_ib%lld=%g abs=%g "
                            "best_e=%u out=%g abs=%g ratio=%g block_ref=%g block_g=%g block_inner=%g block_e=%g "
                            "e=%u inner_e=%u",
                            selective_off > 0 ? " | " : "",
                            (long long) ib,
                            missing_a_out,
                            missing_a_abs,
                            best_group,
                            best_group_out,
                            best_group_abs,
                            (long long) best_inner_src,
                            best_inner_out,
                            best_inner_abs,
                            (unsigned) best_e_byte,
                            best_e_out,
                            best_e_abs,
                            block_ref[(size_t) ib] != 0.0 ? (best_e_block / block_ref[(size_t) ib]) : NAN,
                            block_ref[(size_t) ib],
                            best_group_block,
                            best_inner_block,
                            best_e_block,
                            (unsigned) w_src_blocks[(size_t) ib].e,
                            (unsigned) best_inner_e);
                    if (selective_off >= (int) sizeof(selective_buf)) {
                        break;
                    }
                }

                for (int i = 0; i < 3 && i < (int) top_pos_ref_idx.size(); ++i) {
                    attenuation_off += std::snprintf(
                            attenuation_buf + attenuation_off,
                            sizeof(attenuation_buf) - attenuation_off,
                            "%stop%d fit=%g noA_ratio=%g",
                            attenuation_off > 0 ? " | " : "",
                            i + 1,
                            top_pos_fit_factor[i],
                            top_pos_ref_cum[i] != 0.0 ? (top_pos_no_a_cum[i] / top_pos_ref_cum[i]) : NAN);
                    if (attenuation_off >= (int) sizeof(attenuation_buf)) {
                        break;
                    }
                }

                GGML_LOG_WARN(
                        "%s: src0-block-focus %s %s r=%lld c=%lld actual=%g ref=%g deficit=%g "
                        "top_pos_needed=%d top_pos_cum=%g\n",
                        __func__,
                        ggml_get_name(dst),
                        probe_tag,
                        (long long) row,
                        (long long) out_col,
                        (double) actual_v,
                        ref_total,
                        deficit,
                        top_pos_needed,
                        top_pos_cum);
                GGML_LOG_WARN(
                        "%s: src0-block-focus %s %s missingA combos "
                        "top1=%g abs=%g | top2=%g abs=%g | top3=%g abs=%g\n",
                        __func__,
                        ggml_get_name(dst),
                        probe_tag,
                        top_missing_a_ref[0],
                        fabs((double) actual_v - top_missing_a_ref[0]),
                        top_missing_a_ref[1],
                        fabs((double) actual_v - top_missing_a_ref[1]),
                        top_missing_a_ref[2],
                        fabs((double) actual_v - top_missing_a_ref[2]));
                GGML_LOG_WARN(
                        "%s: src0-block-focus %s %s signFlip combos "
                        "top1=%g abs=%g | top2=%g abs=%g | top3=%g abs=%g\n",
                        __func__,
                        ggml_get_name(dst),
                        probe_tag,
                        top_sign_flip_ref[0],
                        fabs((double) actual_v - top_sign_flip_ref[0]),
                        top_sign_flip_ref[1],
                        fabs((double) actual_v - top_sign_flip_ref[1]),
                        top_sign_flip_ref[2],
                        fabs((double) actual_v - top_sign_flip_ref[2]));
                GGML_LOG_WARN(
                        "%s: src0-block-focus %s %s attenuation-fit %s\n",
                        __func__,
                        ggml_get_name(dst),
                        probe_tag,
                        attenuation_buf);
                GGML_LOG_WARN(
                        "%s: src0-block-focus %s %s top_ref=%s\n",
                        __func__,
                        ggml_get_name(dst),
                        probe_tag,
                        top_ref_buf);
                GGML_LOG_WARN(
                        "%s: src0-block-focus %s %s top_missing_a=%s\n",
                        __func__,
                        ggml_get_name(dst),
                        probe_tag,
                        top_missing_a_buf);
                GGML_LOG_WARN(
                        "%s: src0-block-focus %s %s selective=%s\n",
                        __func__,
                        ggml_get_name(dst),
                        probe_tag,
                        selective_buf);
            };

            auto run_single_sign_flip_rows_probe = [&](const char * probe_tag, int64_t out_col, const std::vector<int64_t> & rows) {
                if (out_col < 0 || out_col >= ne01) {
                    return;
                }

                std::vector<block_nvfp4> w_src_blocks((size_t) nblk);
                std::vector<float> x_src_row;
                std::vector<float> x_roundtrip_row;

                const char * w_col_ptr = (const char *) src0->data + out_col * src0->nb[1];
                CUDA_CHECK(cudaMemcpyAsync(w_src_blocks.data(), w_col_ptr, (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                for (int64_t row : rows) {
                    if (row < 0 || row >= ne11) {
                        continue;
                    }

                    compute_row_roundtrip(row, x_src_row, x_roundtrip_row);

                    float actual_v = 0.0f;
                    const char * probe_out_ptr = (const char *) dst_data + (row * ne01 + out_col) * (int64_t) sizeof(float);
                    CUDA_CHECK(cudaMemcpyAsync(&actual_v, probe_out_ptr, sizeof(actual_v), cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    std::vector<double> block_ref((size_t) nblk, 0.0);
                    double ref_total = 0.0;
                    for (int64_t ib = 0; ib < nblk; ++ib) {
                        std::vector<float> w_block_scaled((size_t) QK_NVFP4);
                        dequantize_row_nvfp4(&w_src_blocks[(size_t) ib], w_block_scaled.data(), QK_NVFP4, 1.0f);

                        double acc_ref = 0.0;
                        const int64_t k0 = ib * QK_NVFP4;
                        for (int64_t j = 0; j < QK_NVFP4; ++j) {
                            acc_ref += (double) w_block_scaled[(size_t) j] * (double) x_roundtrip_row[(size_t) (k0 + j)];
                        }
                        block_ref[(size_t) ib] = acc_ref * (double) out_scale;
                        ref_total += block_ref[(size_t) ib];
                    }

                    std::vector<int64_t> top_pos_ref_idx;
                    auto push_top_pos = [&](std::vector<int64_t> & dst_idx, int64_t idx) {
                        const double v = block_ref[(size_t) idx];
                        if (v <= 0.0) {
                            return;
                        }
                        int insert_at = (int) dst_idx.size();
                        for (int i = 0; i < (int) dst_idx.size(); ++i) {
                            if (v > block_ref[(size_t) dst_idx[(size_t) i]]) {
                                insert_at = i;
                                break;
                            }
                        }
                        if (insert_at < 8) {
                            dst_idx.insert(dst_idx.begin() + insert_at, idx);
                            if (dst_idx.size() > 8) {
                                dst_idx.pop_back();
                            }
                        } else if ((int) dst_idx.size() < 8) {
                            dst_idx.push_back(idx);
                        }
                    };
                    for (int64_t ib = 0; ib < nblk; ++ib) {
                        push_top_pos(top_pos_ref_idx, ib);
                    }

                    int64_t best_ib = -1;
                    double best_out = ref_total;
                    double best_abs = fabs((double) actual_v - ref_total);
                    double best_block = 0.0;
                    for (int64_t ib : top_pos_ref_idx) {
                        const double alt_out = ref_total - 2.0 * block_ref[(size_t) ib];
                        const double alt_abs = fabs((double) actual_v - alt_out);
                        if (alt_abs < best_abs) {
                            best_abs = alt_abs;
                            best_out = alt_out;
                            best_ib = ib;
                            best_block = block_ref[(size_t) ib];
                        }
                    }

                    GGML_LOG_WARN(
                            "%s: src0-signflip-rows %s %s r=%lld c=%lld actual=%g ref=%g best_ib=%lld best_out=%g abs=%g block_ref=%g\n",
                            __func__,
                            ggml_get_name(dst),
                            probe_tag,
                            (long long) row,
                            (long long) out_col,
                            (double) actual_v,
                            ref_total,
                            (long long) best_ib,
                            best_out,
                            best_abs,
                            best_block);
                }
            };

            auto run_lt_a_scale_patch_probe = [&](const char * probe_tag, int64_t row, int64_t out_col, const std::vector<int64_t> & ibs, uint8_t patched_e) {
                if (row < 0 || row >= ne11 || out_col < 0 || out_col >= ne01) {
                    return;
                }
                if (ibs.empty()) {
                    return;
                }

                std::vector<block_nvfp4> w_src_blocks((size_t) nblk);
                std::vector<float> x_src_row;
                std::vector<float> x_roundtrip_row;
                std::vector<float> w_block_scaled((size_t) QK_NVFP4);
                std::vector<float> w_block_alt((size_t) QK_NVFP4);

                const char * w_col_ptr = (const char *) src0->data + out_col * src0->nb[1];
                CUDA_CHECK(cudaMemcpyAsync(w_src_blocks.data(), w_col_ptr, (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                compute_row_roundtrip(row, x_src_row, x_roundtrip_row);

                std::vector<double> block_ref((size_t) nblk, 0.0);
                double ref_total = 0.0;
                for (int64_t ib = 0; ib < nblk; ++ib) {
                    dequantize_row_nvfp4(&w_src_blocks[(size_t) ib], w_block_scaled.data(), QK_NVFP4, 1.0f);
                    double acc_ref = 0.0;
                    const int64_t k0 = ib * QK_NVFP4;
                    for (int64_t j = 0; j < QK_NVFP4; ++j) {
                        acc_ref += (double) w_block_scaled[(size_t) j] * (double) x_roundtrip_row[(size_t) (k0 + j)];
                    }
                    block_ref[(size_t) ib] = acc_ref * (double) out_scale;
                    ref_total += block_ref[(size_t) ib];
                }

                const float alpha = matmul_alpha;
                const float beta = 0.0f;
                ggml_cuda_pool_alloc<float> diag_out(ctx.pool(), (size_t) ne01 * (size_t) lt_n);

                for (int64_t ib : ibs) {
                    if (ib < 0 || ib >= nblk) {
                        continue;
                    }

                    const int64_t scale_idx = linear_scale_layout
                            ? (out_col * src0_repacked.scale_inner_padded + ib)
                            : ggml_cuda_nvfp4_scale_tiled_index(out_col, ib, src0_repacked.scale_inner_padded);

                    uint8_t original_e = 0;
                    auto run_patch_at = [&](int64_t patch_idx, uint8_t patch_byte) -> std::pair<float, cublasStatus_t> {
                        if (patch_idx < 0 || (size_t) patch_idx >= src0_repacked.scale_nbytes) {
                            return { NAN, CUBLAS_STATUS_INVALID_VALUE };
                        }

                        uint8_t saved_e = 0;
                        CUDA_CHECK(cudaMemcpyAsync(
                                &saved_e,
                                (const uint8_t *) src0_repacked.scale + patch_idx,
                                sizeof(saved_e),
                                cudaMemcpyDeviceToHost,
                                stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        CUDA_CHECK(cudaMemcpyAsync(
                                (uint8_t *) src0_repacked.scale + patch_idx,
                                &patch_byte,
                                sizeof(patch_byte),
                                cudaMemcpyHostToDevice,
                                stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        cublasStatus_t patch_st = cublasLtMatmul(
                                ctx.cublaslt_handle(),
                                op_desc,
                                &alpha,
                                src0_repacked.data, a_desc,
                                src1_repacked_data.get(), b_desc,
                                &beta,
                                diag_out.get(), c_desc,
                                diag_out.get(), c_desc,
                                nullptr,
                                nullptr, 0,
                                stream);

                        float patch_out = NAN;
                        if (patch_st == CUBLAS_STATUS_SUCCESS) {
                            const char * probe_ptr = (const char *) diag_out.get() + (row * ne01 + out_col) * (int64_t) sizeof(float);
                            CUDA_CHECK(cudaMemcpyAsync(&patch_out, probe_ptr, sizeof(patch_out), cudaMemcpyDeviceToHost, stream));
                            CUDA_CHECK(cudaStreamSynchronize(stream));
                        }

                        CUDA_CHECK(cudaMemcpyAsync(
                                (uint8_t *) src0_repacked.scale + patch_idx,
                                &saved_e,
                                sizeof(saved_e),
                                cudaMemcpyHostToDevice,
                                stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));

                        return { patch_out, patch_st };
                    };

                    const uint8_t patched_e_ue = (uint8_t) 120;
                    const std::pair<float, cublasStatus_t> neighbor_m1 = run_patch_at(scale_idx - 1, patched_e);
                    const std::pair<float, cublasStatus_t> center = run_patch_at(scale_idx, patched_e);
                    const std::pair<float, cublasStatus_t> center_ue = run_patch_at(scale_idx, patched_e_ue);
                    const std::pair<float, cublasStatus_t> neighbor_p1 = run_patch_at(scale_idx + 1, patched_e);
                    const std::pair<float, cublasStatus_t> center_zero = run_patch_at(scale_idx, (uint8_t) 0);

                    CUDA_CHECK(cudaMemcpyAsync(
                            &original_e,
                            (const uint8_t *) src0_repacked.scale + scale_idx,
                            sizeof(original_e),
                            cudaMemcpyDeviceToHost,
                            stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    const float patched_out = center.first;
                    const cublasStatus_t diag_st = center.second;

                    block_nvfp4 alt_block = w_src_blocks[(size_t) ib];
                    alt_block.e = patched_e;
                    dequantize_row_nvfp4(&alt_block, w_block_alt.data(), QK_NVFP4, 1.0f);
                    double alt_block_acc = 0.0;
                    const int64_t k0 = ib * QK_NVFP4;
                    for (int64_t j = 0; j < QK_NVFP4; ++j) {
                        alt_block_acc += (double) w_block_alt[(size_t) j] * (double) x_roundtrip_row[(size_t) (k0 + j)];
                    }
                    const double patched_block_host = alt_block_acc * (double) out_scale;
                    const double patched_total_host = ref_total - block_ref[(size_t) ib] + patched_block_host;
                    const double implied_ratio = block_ref[(size_t) ib] != 0.0
                            ? (((double) patched_out - (ref_total - block_ref[(size_t) ib])) / block_ref[(size_t) ib])
                            : NAN;
                    const double zero_host_out = ref_total - block_ref[(size_t) ib];
                    const double zero_implied_ratio = block_ref[(size_t) ib] != 0.0
                            ? (((double) center_zero.first - (ref_total - block_ref[(size_t) ib])) / block_ref[(size_t) ib])
                            : NAN;
                    const int64_t quartet_base = (ib / 4) * 4;
                    double quartet_ref_sum = 0.0;
                    double quartet_patch_sum = 0.0;
                    double quartet_zero_sum = 0.0;
                    int quartet_count = 0;
                    uint8_t quartet_e[4] = { 0, 0, 0, 0 };
                    for (int64_t qib = quartet_base; qib < std::min<int64_t>(quartet_base + 4, nblk); ++qib) {
                        quartet_e[quartet_count] = w_src_blocks[(size_t) qib].e;
                        block_nvfp4 q_alt_patch = w_src_blocks[(size_t) qib];
                        q_alt_patch.e = patched_e;
                        dequantize_row_nvfp4(&q_alt_patch, w_block_alt.data(), QK_NVFP4, 1.0f);

                        double q_patch_acc = 0.0;
                        const int64_t qk0 = qib * QK_NVFP4;
                        for (int64_t j = 0; j < QK_NVFP4; ++j) {
                            q_patch_acc += (double) w_block_alt[(size_t) j] * (double) x_roundtrip_row[(size_t) (qk0 + j)];
                        }

                        block_nvfp4 q_alt_zero = w_src_blocks[(size_t) qib];
                        q_alt_zero.e = 0;
                        dequantize_row_nvfp4(&q_alt_zero, w_block_alt.data(), QK_NVFP4, 1.0f);

                        double q_zero_acc = 0.0;
                        for (int64_t j = 0; j < QK_NVFP4; ++j) {
                            q_zero_acc += (double) w_block_alt[(size_t) j] * (double) x_roundtrip_row[(size_t) (qk0 + j)];
                        }

                        quartet_ref_sum += block_ref[(size_t) qib];
                        quartet_patch_sum += q_patch_acc * (double) out_scale;
                        quartet_zero_sum += q_zero_acc * (double) out_scale;
                        ++quartet_count;
                    }
                    const double quartet_patch_host = ref_total - quartet_ref_sum + quartet_patch_sum;
                    const double quartet_zero_host = ref_total - quartet_ref_sum + quartet_zero_sum;
                    const float orig_dec = ggml_cuda_e4m3_to_fp32(original_e);
                    const float patch_dec = ggml_cuda_e4m3_to_fp32(patched_e);
                    const float zero_dec = ggml_cuda_e4m3_to_fp32((uint8_t) 0);
                    const float dec_122 = ggml_cuda_e4m3_to_fp32((uint8_t) 122);
                    const float dec_124 = ggml_cuda_e4m3_to_fp32((uint8_t) 124);
                    const float dec_248 = ggml_cuda_e4m3_to_fp32((uint8_t) 248);

                    GGML_LOG_WARN(
                            "%s: lt-a-scale-patch %s %s r=%lld c=%lld ib=%lld orig_e=%u patched_e=%u "
                            "patched_out=%g host_out=%g implied_ratio=%g "
                            "neighbor=[m1=%g st=%d p1=%g st=%d] "
                            "ue=[patched_e=%u out=%g st=%d] "
                            "zero=[out=%g host=%g implied_ratio=%g st=%d] "
                            "quartet=[base=%lld count=%d e=%u,%u,%u,%u patch_host=%g zero_host=%g] "
                            "decode=[orig=%g patched=%g zero=%g ref122=%g ref124=%g ref248=%g "
                            "half_orig=%g half_patched=%g half_zero=%g half122=%g half124=%g half248=%g] st=%d\n",
                            __func__,
                            ggml_get_name(dst),
                            probe_tag,
                            (long long) row,
                            (long long) out_col,
                            (long long) ib,
                            (unsigned) original_e,
                            (unsigned) patched_e,
                            (double) patched_out,
                            patched_total_host,
                            implied_ratio,
                            (double) neighbor_m1.first,
                            (int) neighbor_m1.second,
                            (double) neighbor_p1.first,
                            (int) neighbor_p1.second,
                            (unsigned) patched_e_ue,
                            (double) center_ue.first,
                            (int) center_ue.second,
                            (double) center_zero.first,
                            zero_host_out,
                            zero_implied_ratio,
                            (int) center_zero.second,
                            (long long) quartet_base,
                            quartet_count,
                            (unsigned) quartet_e[0],
                            (unsigned) quartet_e[1],
                            (unsigned) quartet_e[2],
                            (unsigned) quartet_e[3],
                            quartet_patch_host,
                            quartet_zero_host,
                            (double) orig_dec,
                            (double) patch_dec,
                            (double) zero_dec,
                            (double) dec_122,
                            (double) dec_124,
                            (double) dec_248,
                            (double) (0.5f * orig_dec),
                            (double) (0.5f * patch_dec),
                            (double) (0.5f * zero_dec),
                            (double) (0.5f * dec_122),
                            (double) (0.5f * dec_124),
                            (double) (0.5f * dec_248),
                            (int) diag_st);
                }
            };

            auto run_lt_b_scale_patch_probe = [&](const char * probe_tag, int64_t row, int64_t out_col, const std::vector<int64_t> & ibs, uint8_t patched_e) {
                if (row < 0 || row >= ne11 || out_col < 0 || out_col >= ne01) {
                    return;
                }
                if (ibs.empty()) {
                    return;
                }

                std::vector<block_nvfp4> w_src_blocks((size_t) nblk);
                std::vector<block_nvfp4> x_q_blocks((size_t) nblk);
                std::vector<float> x_src_row;
                std::vector<float> w_block_scaled((size_t) QK_NVFP4);
                std::vector<float> x_block_alt((size_t) QK_NVFP4);

                const char * w_col_ptr = (const char *) src0->data + out_col * src0->nb[1];
                CUDA_CHECK(cudaMemcpyAsync(w_src_blocks.data(), w_col_ptr, (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));

                x_src_row.resize((size_t) ne10);
                const char * row_ptr = (const char *) src1->data + row * src1->nb[1];
                CUDA_CHECK(cudaMemcpyAsync(x_src_row.data(), row_ptr, (size_t) ne10 * sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                quantize_row_nvfp4_ref(x_src_row.data(), x_q_blocks.data(), ne10, global_scale);

                double ref_total = 0.0;
                std::vector<double> block_ref((size_t) nblk, 0.0);
                for (int64_t ib = 0; ib < nblk; ++ib) {
                    dequantize_row_nvfp4(&w_src_blocks[(size_t) ib], w_block_scaled.data(), QK_NVFP4, 1.0f);
                    float x_block[QK_NVFP4];
                    dequantize_row_nvfp4(&x_q_blocks[(size_t) ib], x_block, QK_NVFP4, global_scale);

                    double acc = 0.0;
                    for (int64_t j = 0; j < QK_NVFP4; ++j) {
                        acc += (double) w_block_scaled[(size_t) j] * (double) x_block[(size_t) j];
                    }
                    block_ref[(size_t) ib] = acc * (double) out_scale;
                    ref_total += block_ref[(size_t) ib];
                }

                const float alpha = matmul_alpha;
                const float beta = 0.0f;
                ggml_cuda_pool_alloc<float> diag_out(ctx.pool(), (size_t) ne01 * (size_t) lt_n);

                auto run_b_patch_at = [&](int64_t patch_idx, uint8_t patch_byte) -> std::pair<float, cublasStatus_t> {
                    if (patch_idx < 0 || (size_t) patch_idx >= (size_t) scale_outer_padded_b * (size_t) scale_inner_padded) {
                        return { NAN, CUBLAS_STATUS_INVALID_VALUE };
                    }

                    uint8_t saved_e = 0;
                    CUDA_CHECK(cudaMemcpyAsync(
                            &saved_e,
                            (const uint8_t *) src1_repacked_scale.get() + patch_idx,
                            sizeof(saved_e),
                            cudaMemcpyDeviceToHost,
                            stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    CUDA_CHECK(cudaMemcpyAsync(
                            (uint8_t *) src1_repacked_scale.get() + patch_idx,
                            &patch_byte,
                            sizeof(patch_byte),
                            cudaMemcpyHostToDevice,
                            stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    cublasStatus_t patch_st = cublasLtMatmul(
                            ctx.cublaslt_handle(),
                            op_desc,
                            &alpha,
                            src0_repacked.data, a_desc,
                            src1_repacked_data.get(), b_desc,
                            &beta,
                            diag_out.get(), c_desc,
                            diag_out.get(), c_desc,
                            nullptr,
                            nullptr, 0,
                            stream);

                    float patch_out = NAN;
                    if (patch_st == CUBLAS_STATUS_SUCCESS) {
                        const char * probe_ptr = (const char *) diag_out.get() + (row * ne01 + out_col) * (int64_t) sizeof(float);
                        CUDA_CHECK(cudaMemcpyAsync(&patch_out, probe_ptr, sizeof(patch_out), cudaMemcpyDeviceToHost, stream));
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                    }

                    CUDA_CHECK(cudaMemcpyAsync(
                            (uint8_t *) src1_repacked_scale.get() + patch_idx,
                            &saved_e,
                            sizeof(saved_e),
                            cudaMemcpyHostToDevice,
                            stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    return { patch_out, patch_st };
                };

                for (int64_t ib : ibs) {
                    if (ib < 0 || ib >= nblk) {
                        continue;
                    }

                    const int64_t scale_idx = linear_scale_layout
                            ? (row * scale_inner_padded + ib)
                            : ggml_cuda_nvfp4_scale_tiled_index(row, ib, scale_inner_padded);

                    uint8_t original_e = 0;
                    CUDA_CHECK(cudaMemcpyAsync(
                            &original_e,
                            (const uint8_t *) src1_repacked_scale.get() + scale_idx,
                            sizeof(original_e),
                            cudaMemcpyDeviceToHost,
                            stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));

                    const std::pair<float, cublasStatus_t> center = run_b_patch_at(scale_idx, patched_e);

                    block_nvfp4 x_alt = x_q_blocks[(size_t) ib];
                    x_alt.e = patched_e;
                    dequantize_row_nvfp4(&w_src_blocks[(size_t) ib], w_block_scaled.data(), QK_NVFP4, 1.0f);
                    dequantize_row_nvfp4(&x_alt, x_block_alt.data(), QK_NVFP4, global_scale);

                    double alt_acc = 0.0;
                    for (int64_t j = 0; j < QK_NVFP4; ++j) {
                        alt_acc += (double) w_block_scaled[(size_t) j] * (double) x_block_alt[(size_t) j];
                    }
                    const double patched_block_host = alt_acc * (double) out_scale;
                    const double patched_total_host = ref_total - block_ref[(size_t) ib] + patched_block_host;
                    const double implied_ratio = block_ref[(size_t) ib] != 0.0
                            ? (((double) center.first - (ref_total - block_ref[(size_t) ib])) / block_ref[(size_t) ib])
                            : NAN;

                    GGML_LOG_WARN(
                            "%s: lt-b-scale-patch %s %s r=%lld c=%lld ib=%lld orig_e=%u patched_e=%u "
                            "patched_out=%g host_out=%g implied_ratio=%g st=%d\n",
                            __func__,
                            ggml_get_name(dst),
                            probe_tag,
                            (long long) row,
                            (long long) out_col,
                            (long long) ib,
                            (unsigned) original_e,
                            (unsigned) patched_e,
                            (double) center.first,
                            patched_total_host,
                            implied_ratio,
                            (int) center.second);
                }
            };

            const char * dst_name = ggml_get_name(dst);
            if (dst_name != nullptr) {
                if (strcmp(dst_name, "Qcur-scaled-0") == 0) {
                    run_src0_focus_probe("Q-bad-cols", { 725, 3793 });
                    run_block_focus_probe("Q-proven-bad-point", 14, 3793);
                    run_single_sign_flip_rows_probe("Q-bad-col-signflip", 3793, { 0, 1, 14 });
                    run_lt_a_scale_patch_probe("Q-bad-col", 14, 3793, { 8, 12, 60 }, (uint8_t) 248);
                    run_lt_b_scale_patch_probe("Q-bad-col", 14, 3793, { 12 }, (uint8_t) 248);
                } else if (strcmp(dst_name, "Kcur-scaled-0") == 0) {
                    run_src0_focus_probe("K-bad-cols", { 179, 207, 990 });
                    run_block_focus_probe("K-proven-bad-point", 14, 207);
                }
            }
        }
    }

    if (c_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (op_desc != nullptr) {
        cublasLtMatmulDescDestroy(op_desc);
    }

    if (st != CUBLAS_STATUS_SUCCESS) {
        const cudaError_t cuda_err = cudaPeekAtLastError();
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        int runtime_version = 0;
        int driver_version = 0;
        (void) cudaRuntimeGetVersion(&runtime_version);
        (void) cudaDriverGetVersion(&driver_version);

        const ggml_tensor * in_scale_tensor = ggml_mul_mat_get_nvfp4_input_scale(dst);
        const ggml_tensor * out_scale_tensor = ggml_mul_mat_get_nvfp4_weight_scale(dst);

        const size_t src0_align16 = ((uintptr_t) src0_repacked.data) & 0xF;
        const size_t src1_align16 = ((uintptr_t) src1_repacked_data.get()) & 0xF;
        const size_t dst_align16  = ((uintptr_t) dst->data) & 0xF;

        static std::atomic<bool> logged(false);
        if (debug_enabled || !logged.exchange(true)) {
            GGML_LOG_WARN(
                "%s: cublasLt NVFP4 matmul failed for %s stage=%s status=%d (%s) cuda_err=%d (%s) "
                "device=%d cc=%d runtime=%d driver=%d stream=%p "
                "A=[k=%lld,n=%lld,ld=%lld,type=CUDA_R_4F_E2M1] "
                "B=[k=%lld,m=%lld(padded=%lld),ld=%lld,type=CUDA_R_4F_E2M1] "
                "C/D=[n=%lld,m=%lld(padded=%lld),ld=%lld,type=CUDA_R_32F] alpha=%g beta=0 "
                "global_scale=%g src0_type=%s src1_type=%s dst_type=%s "
                "in_scale_tensor=%p in_scale_type=%s out_scale_tensor=%p out_scale_type=%s "
                "ptr=[src0_data=%p src0_scale=%p src1_data=%p src1_scale=%p dst=%p dst_data=%p] align16=[src0=%zu src1=%zu dst=%zu] "
                "channel_bytes=[src0_data=%zu src0_scale=%zu src1_data=%zu src1_scale=%zu] "
                "src0_nb=[%zu,%zu,%zu,%zu] src1_nb=[%zu,%zu,%zu,%zu] dst_nb=[%zu,%zu,%zu,%zu]\n",
                __func__,
                ggml_get_name(dst),
                stage,
                (int) st,
                cublas_get_error_str(st),
                (int) cuda_err,
                cudaGetErrorString(cuda_err),
                ctx.device,
                cc,
                runtime_version,
                driver_version,
                (void *) stream,
                (long long) ne10, (long long) ne01, (long long) ne10,
                (long long) ne10, (long long) ne11, (long long) ne11_padded, (long long) ne10,
                (long long) ne01, (long long) ne11, (long long) ne11_padded, (long long) ne01,
                (double) out_scale,
                (double) global_scale,
                ggml_type_name(src0->type),
                ggml_type_name(src1->type),
                ggml_type_name(dst->type),
                (const void *) in_scale_tensor,
                in_scale_tensor ? ggml_type_name(in_scale_tensor->type) : "(null)",
                (const void *) out_scale_tensor,
                out_scale_tensor ? ggml_type_name(out_scale_tensor->type) : "(null)",
                src0_repacked.data,
                src0_repacked.scale,
                (const void *) src1_repacked_data.get(),
                (const void *) src1_repacked_scale.get(),
                (const void *) dst->data,
                (const void *) dst_data,
                src0_align16,
                src1_align16,
                dst_align16,
                src0_repacked.data_nbytes,
                src0_repacked.scale_nbytes,
                (size_t) ne11_padded * (size_t) ne10 / 2,
                (size_t) scale_outer_padded_b * (size_t) scale_inner_padded,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);

            if (st == CUBLAS_STATUS_NOT_SUPPORTED) {
                GGML_LOG_WARN(
                        "%s: hint: CUBLAS_STATUS_NOT_SUPPORTED usually means this GPU/toolkit/shape does not support "
                        "the requested FP4 Lt matmul path; fallback kernels will be used.\n",
                        __func__);
            }
            if (st == CUBLAS_STATUS_INVALID_VALUE) {
                GGML_LOG_WARN(
                        "%s: hint: CUBLAS_STATUS_INVALID_VALUE is commonly caused by unsupported FP4 dimension constraints "
                        "or layout limits. Check M=%lld N=%lld (padded=%lld) K=%lld.\n",
                        __func__,
                        (long long) ne01,
                        (long long) ne11,
                        (long long) ne11_padded,
                        (long long) ne10);
            }
        }
        return false;
    }

    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#endif
}
