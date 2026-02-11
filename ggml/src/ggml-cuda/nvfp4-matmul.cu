#include "nvfp4-matmul.cuh"

#include "ggml-backend.h"
#include "../ggml-quants.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

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

static float ggml_cuda_nvfp4_input_global_scale(const ggml_tensor * dst) {
    const ggml_tensor * scale = ggml_mul_mat_get_nvfp4_input_scale(dst);
    if (scale == nullptr) {
        return 1.0f;
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

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_e4m3_half(float x) {
    uint8_t best_index = 0;
    float best_err = INFINITY;

    for (int i = 0; i < 256; ++i) {
        const float v = ggml_cuda_e4m3_to_fp32_half((uint8_t) i);
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
        const uint8_t scale_q = ggml_cuda_best_index_e4m3_half(scale);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4(xi * inv_scale);

    if (lane_active && (lane & 1) == 0) {
        const uint8_t q_hi = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].qs[lane/2] = q | (q_hi << 4);
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
    out_scale[scale_idx] = v.e;
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
        ggml_cuda_nvfp4_split_matrix & out) {
    auto & cache = ctx.nvfp4_repack_cache[ctx.device];
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

    const int64_t ne_k = src0->ne[0];
    const int64_t n_outer = src0->ne[1];
    if (ne_k % QK_NVFP4 != 0) {
        return false;
    }

    const size_t data_nbytes = (size_t) n_outer * (size_t) ne_k / 2;
    const size_t scale_nbytes = (size_t) ggml_cuda_pad_i64(n_outer, 128) * (size_t) ggml_cuda_pad_i64(ne_k / QK_NVFP4, 4);

    void * data_repacked = nullptr;
    void * scale_repacked = nullptr;

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

    out.data = data_repacked;
    out.scale = scale_repacked;
    out.data_nbytes = data_nbytes_built;
    out.scale_nbytes = scale_nbytes_built;
    out.scale_inner_padded = scale_inner_padded;
    out.scale_outer_padded = scale_outer_padded;
    return true;
}
#endif

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

    // This pass intentionally handles only dense, non-batched MUL_MAT.
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        log_skip("batched tensor shape not supported");
        return false;
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

    // cuBLASLt native FP4 matmul is restrictive on GEMM dimensions.
    // Keep static matrix dimensions (M/K) aligned and pad dynamic token dimension (N) when needed.
    if ((ne01 % 16) != 0 || (ne10 % 16) != 0) {
        log_skip("native FP4 requires M/K to be multiples of 16");
        return false;
    }
    const int64_t ne11_padded = (ne11 + 15) & ~15LL;
    const bool pad_n = ne11_padded != ne11;

    if (ne10 % QK_NVFP4 != 0) {
        log_skip("K dimension is not divisible by QK_NVFP4");
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t nblk_k = ne10 / QK_NVFP4;
    const int64_t scale_inner_padded = ggml_cuda_pad_i64(nblk_k, 4);
    const int64_t scale_outer_padded_b = ggml_cuda_pad_i64(ne11_padded, 128);

    ggml_cuda_pool_alloc<block_nvfp4> src1_q_nvfp4(ctx.pool(), (size_t) nblk_k * (size_t) ne11);
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_data(ctx.pool(), (size_t) ne11_padded * (size_t) ne10 / 2);
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_scale(ctx.pool(), (size_t) scale_outer_padded_b * (size_t) scale_inner_padded);

    const float global_scale = ggml_cuda_nvfp4_input_global_scale(dst);
    quantize_row_nvfp4_cuda(
            (const float *) src1->data, src1_q_nvfp4.get(),
            ne10, src1->nb[1] / (int64_t) sizeof(float), ne11,
            global_scale, stream);
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
    if (!ggml_cuda_nvfp4_get_repacked_src0(ctx, src0, linear_scale_layout, stream, src0_repacked)) {
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
        GGML_LOG_INFO("%s: alpha mode for %s: out_scale\n",
                __func__, ggml_get_name(dst));

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
    }

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    ggml_cuda_pool_alloc<float> dst_padded(ctx.pool(), pad_n ? (size_t) ne01 * (size_t) ne11_padded : 1);
    void * dst_data = pad_n ? (void *) dst_padded.get() : dst->data;

    const char * stage = "matmul_desc_create";
    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

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
        stage = "layout_create_b";
        st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) ne10, (uint64_t) ne11_padded, (int64_t) ne10);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_c";
        st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, (uint64_t) ne01, (uint64_t) ne11_padded, (int64_t) ne01);
    }

    float out_scale = 1.0f;
    if (const ggml_tensor * scale = ggml_mul_mat_get_nvfp4_weight_scale(dst)) {
        float scale_val = 0.0f;
        if (ggml_cuda_fetch_input_scale_f32(scale, scale_val) && std::isfinite(scale_val)) {
            out_scale = scale_val;
        }
    }
    const float matmul_alpha = out_scale;

    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "matmul";
        const float alpha = matmul_alpha;
        const float beta  = 0.0f;
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

    if (st == CUBLAS_STATUS_SUCCESS && pad_n) {
        CUDA_CHECK(cudaMemcpyAsync(
                dst->data, dst_padded.get(),
                (size_t) ne01 * (size_t) ne11 * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
    }

    if (st == CUBLAS_STATUS_SUCCESS && validate_enabled && ne10 % QK_NVFP4 == 0 && ne01 > 0 && ne11 > 0) {
        static std::atomic<bool> logged(false);
        if (debug_enabled || !logged.exchange(true)) {
            const int64_t nblk = ne10 / QK_NVFP4;
            std::vector<block_nvfp4> w_row((size_t) nblk);
            std::vector<float> x_row((size_t) ne10);
            std::vector<block_nvfp4> x_q((size_t) nblk);
            std::vector<float> x_roundtrip((size_t) ne10);
            std::vector<float> w_deq((size_t) ne10);
            std::vector<float> x_roundtrip_no_scale((size_t) ne10);
            std::vector<float> w_no_scale((size_t) ne10);
            std::vector<float> x_no_scale((size_t) ne10);

            const char * w_row_ptr = (const char *) src0->data + 0 * src0->nb[1];
            const char * x_row_ptr = (const char *) src1->data + 0 * src1->nb[1];
            CUDA_CHECK(cudaMemcpyAsync(w_row.data(), w_row_ptr, (size_t) nblk * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(x_row.data(), x_row_ptr, (size_t) ne10 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            quantize_row_nvfp4_ref(x_row.data(), x_q.data(), ne10, global_scale);
            dequantize_row_nvfp4(x_q.data(), x_roundtrip.data(), ne10, global_scale);
            dequantize_row_nvfp4(w_row.data(), w_deq.data(), ne10, 1.0f);
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

            double ref = 0.0;
            for (int64_t k = 0; k < ne10; ++k) {
                ref += (double) w_deq[(size_t) k] * (double) x_roundtrip[(size_t) k];
            }
            ref *= (double) matmul_alpha;

            double ref_no_a_scale = 0.0;
            double ref_no_b_scale = 0.0;
            double ref_no_ab_scale = 0.0;
            for (int64_t k = 0; k < ne10; ++k) {
                ref_no_a_scale  += (double) w_no_scale[(size_t) k] * (double) x_roundtrip[(size_t) k];
                ref_no_b_scale  += (double) w_deq[(size_t) k]      * (double) x_roundtrip_no_scale[(size_t) k];
                ref_no_ab_scale += (double) w_no_scale[(size_t) k] * (double) x_no_scale[(size_t) k];
            }
            ref_no_a_scale  *= (double) matmul_alpha;
            ref_no_b_scale  *= (double) matmul_alpha;
            ref_no_ab_scale *= (double) matmul_alpha;

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
                    "(global_scale=%g out_scale=%g alpha=%g alpha_mode=out_scale scale_layout=%s)\n",
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
