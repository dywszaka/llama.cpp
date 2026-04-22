#include "nvfp4-kq.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {

static constexpr const char * GGML_CUDA_NVFP4_8_KQ_CUBLASLT_ENV = "GGML_CUDA_NVFP4_8_KQ_CUBLASLT";
static constexpr float GGML_CUDA_NVFP4_8_FP4_MAX = 6.0f;
static constexpr float GGML_CUDA_NVFP4_8_E4M3_HALF_MAX = 224.0f;
static constexpr float GGML_CUDA_NVFP4_8_GLOBAL_SCALE_MAX =
        GGML_CUDA_NVFP4_8_FP4_MAX * GGML_CUDA_NVFP4_8_E4M3_HALF_MAX;

#if defined(CUBLAS_VERSION)
#define GGML_CUDA_NVFP4_8_KQ_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VERSION >= 130000)
#elif defined(CUBLAS_VER_MAJOR)
#define GGML_CUDA_NVFP4_8_KQ_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VER_MAJOR >= 13)
#else
#define GGML_CUDA_NVFP4_8_KQ_HAS_LT_SCALE_CHANNEL_ATTRS 0
#endif

static inline int64_t ggml_cuda_nvfp4_8_kq_pad_i64(int64_t x, int64_t a) {
    GGML_ASSERT(a > 0);
    return ((x + a - 1) / a) * a;
}

static __host__ __device__ __forceinline__ int64_t ggml_cuda_nvfp4_8_kq_scale_tiled_index(
        int64_t outer,
        int64_t inner,
        int64_t n_inner_padded) {
    const int64_t outer_tile = outer / 128;
    const int64_t outer_in_tile = outer % 128;
    const int64_t inner_tile = inner / 4;
    const int64_t inner_in_tile = inner % 4;

    const int64_t tiles_per_outer_block = n_inner_padded / 4;
    const int64_t tile_base = (outer_tile * tiles_per_outer_block + inner_tile) * 512;
    const int64_t tile_offset = (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile;
    return tile_base + tile_offset;
}

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_8_kq_lt_scale_from_ggml_scale_byte(uint8_t ggml_e) {
    const float scale_f = ggml_cuda_e4m3_to_fp32(ggml_e);
    if (!(scale_f > 0.0f) || !isfinite(scale_f)) {
        return 0;
    }

    return (uint8_t) __nv_cvt_float_to_fp8(scale_f, __NV_SATFINITE, __NV_E4M3);
}

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_nvfp4_8(float x) {
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

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_e4m3_nvfp4_8(float x) {
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

static __global__ void nvfp4_8_q_abs_max_kernel(
        const float * __restrict__ q,
        float * __restrict__ amax,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t nb10,
        const int64_t nb11,
        const int64_t nb12,
        const int64_t nb13) {
    const int64_t row = blockIdx.x;
    const int64_t nrows = ne11 * ne12 * ne13;
    if (row >= nrows) {
        return;
    }

    const int64_t i13 = row / (ne11 * ne12);
    const int64_t rem = row - i13 * ne11 * ne12;
    const int64_t i12 = rem / ne11;
    const int64_t i11 = rem - i12 * ne11;
    const char * base = (const char *) q + i11 * nb11 + i12 * nb12 + i13 * nb13;

    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < ne10; i += blockDim.x) {
        const float v = *(const float *) (base + i * nb10);
        local_max = fmaxf(local_max, fabsf(v));
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
        amax[row] = shared_max[0];
    }
}

static __global__ void nvfp4_8_q_quantize_kernel(
        const float * __restrict__ q,
        const float * __restrict__ amax,
        block_nvfp4_8 * __restrict__ q_blocks,
        float * __restrict__ q_input_scales,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t nb10,
        const int64_t nb11,
        const int64_t nb12,
        const int64_t nb13,
        const int64_t nblk) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4_8;
    const int64_t ib = blockIdx.x;
    const int64_t row = blockIdx.y;
    const int64_t nrows = ne11 * ne12 * ne13;
    if (row >= nrows || ib >= nblk) {
        return;
    }

    const int64_t i13 = row / (ne11 * ne12);
    const int64_t rem = row - i13 * ne11 * ne12;
    const int64_t i12 = rem / ne11;
    const int64_t i11 = rem - i12 * ne11;
    const int64_t k = ib * QK_NVFP4_8 + lane;
    const char * base = (const char *) q + i11 * nb11 + i12 * nb12 + i13 * nb13;

    const float xi = (lane_active && k < ne10) ? *(const float *) (base + k * nb10) : 0.0f;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    const float amax_f = amax[row];
    const float global_scale = (amax_f > 0.0f && isfinite(amax_f)) ? (GGML_CUDA_NVFP4_8_GLOBAL_SCALE_MAX / amax_f) : 0.0f;

    float scale_f = 0.0f;
    if (lane == 0) {
        q_input_scales[row] = (global_scale != 0.0f && isfinite(global_scale)) ? (1.0f / global_scale) : 0.0f;
        const float scale = (global_scale != 0.0f) ? (global_scale * (vmax / GGML_CUDA_NVFP4_8_FP4_MAX)) : 0.0f;
        const uint8_t scale_q = ggml_cuda_best_index_e4m3_nvfp4_8(scale);
        q_blocks[row * nblk + ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t qv = ggml_cuda_best_index_nvfp4_8(xi * inv_scale);
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, qv, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        q_blocks[row * nblk + ib].qs[lane / 2] = qv | (q_peer << 4);
    }
}

static __global__ void nvfp4_8_kq_kernel(
        const char * __restrict__ k,
        const block_nvfp4_8 * __restrict__ q_blocks,
        const float * __restrict__ q_input_scales,
        char * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne03,
        const int64_t nb00,
        const int64_t nb01,
        const int64_t nb02,
        const int64_t nb03,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t nb0,
        const int64_t nb1,
        const int64_t nb2,
        const int64_t nb3,
        const int64_t nblk,
        const int64_t r2,
        const int64_t r3) {
    const int64_t idx = blockIdx.x;
    const int64_t total = ne01 * ne11 * ne12 * ne13;
    if (idx >= total) {
        return;
    }

    const int64_t i3 = idx / (ne01 * ne11 * ne12);
    const int64_t rem3 = idx - i3 * ne01 * ne11 * ne12;
    const int64_t i2 = rem3 / (ne01 * ne11);
    const int64_t rem2 = rem3 - i2 * ne01 * ne11;
    const int64_t i1 = rem2 / ne01;
    const int64_t i0 = rem2 - i1 * ne01;

    const int64_t k_i2 = i2 / r2;
    const int64_t k_i3 = i3 / r3;
    if (k_i2 >= ne02 || k_i3 >= ne03) {
        return;
    }

    const char * k_row = k + i0 * nb01 + k_i2 * nb02 + k_i3 * nb03;
    const int64_t q_row = i1 + i2 * ne11 + i3 * ne11 * ne12;
    const block_nvfp4_8 * q_row_blocks = q_blocks + q_row * nblk;
    const float q_input_scale = q_input_scales[q_row];

    float acc = 0.0f;
    for (int64_t ib = threadIdx.x; ib < nblk; ib += blockDim.x) {
        const block_nvfp4_8 * kb = (const block_nvfp4_8 *) (k_row + ib * nb00);
        const block_nvfp4_8 qb = q_row_blocks[ib];
        const float kd = ggml_cuda_e4m3_to_fp32_half(kb->e);
        const float qd = ggml_cuda_e4m3_to_fp32_half(qb.e) * q_input_scale;

#pragma unroll
        for (int j = 0; j < QK_NVFP4_8 / 2; ++j) {
            const uint8_t k_packed = kb->qs[j];
            const uint8_t q_packed = qb.qs[j];
            acc += kd * qd * (float) kvalues_nvfp4[k_packed & 0x0F] * (float) kvalues_nvfp4[q_packed & 0x0F];
            acc += kd * qd * (float) kvalues_nvfp4[k_packed >> 4]    * (float) kvalues_nvfp4[q_packed >> 4];
        }
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *(float *) (dst + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3) = partial[0];
    }
}

static __global__ void nvfp4_8_k_repack_lt_vec16_kernel(
        const char * __restrict__ k,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        const int64_t ne01,
        const int64_t nb00,
        const int64_t nb01,
        const int64_t nblk8,
        const int64_t row_data_bytes,
        const int64_t scale_inner_padded) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = ne01 * nblk8;
    if (idx >= total) {
        return;
    }

    const int64_t outer = idx / nblk8;
    const int64_t inner = idx - outer * nblk8;
    const block_nvfp4_8 * b = (const block_nvfp4_8 *) (k + outer * nb01 + inner * nb00);
    uint8_t * data_dst = out_data + outer * row_data_bytes + inner * 8;
#pragma unroll
    for (int j = 0; j < QK_NVFP4_8 / 2; ++j) {
        data_dst[j] = b->qs[j];
    }
    const int64_t scale_idx = ggml_cuda_nvfp4_8_kq_scale_tiled_index(outer, inner, scale_inner_padded);
    out_scale[scale_idx] = ggml_cuda_nvfp4_8_kq_lt_scale_from_ggml_scale_byte(b->e);
}

static __global__ void nvfp4_8_split_blocks_lt_vec16_kernel(
        const block_nvfp4_8 * __restrict__ blocks,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        const int64_t nblk8,
        const int64_t n_outer_valid,
        const int64_t row_data_bytes,
        const int64_t scale_inner_padded) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_outer_valid * nblk8;
    if (idx >= total) {
        return;
    }

    const int64_t outer = idx / nblk8;
    const int64_t inner = idx - outer * nblk8;
    const block_nvfp4_8 b = blocks[idx];
    uint8_t * data_dst = out_data + outer * row_data_bytes + inner * 8;
#pragma unroll
    for (int j = 0; j < QK_NVFP4_8 / 2; ++j) {
        data_dst[j] = b.qs[j];
    }
    const int64_t scale_idx = ggml_cuda_nvfp4_8_kq_scale_tiled_index(outer, inner, scale_inner_padded);
    out_scale[scale_idx] = ggml_cuda_nvfp4_8_kq_lt_scale_from_ggml_scale_byte(b.e);
}

static __global__ void nvfp4_8_kq_store_scaled_kernel(
        const float * __restrict__ src,
        const float * __restrict__ column_scales,
        char * __restrict__ dst,
        const int64_t m,
        const int64_t n,
        const int64_t src_ld,
        const int64_t dst_nb0,
        const int64_t dst_nb1) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = m * n;
    if (idx >= total) {
        return;
    }

    const int64_t col = idx / m;
    const int64_t row = idx - col * m;
    *(float *) (dst + row * dst_nb0 + col * dst_nb1) = src[col * src_ld + row] * column_scales[col];
}

}

bool ggml_cuda_mul_mat_nvfp4_8_kq(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_NVFP4_8 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % QK_NVFP4_8 != 0) {
        return false;
    }
    if (dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        return false;
    }
    if (src0->ne[2] <= 0 || src0->ne[3] <= 0 || src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows_q = ne11 * ne12 * ne13;
    const int64_t nblk = ne10 / QK_NVFP4_8;

    ggml_cuda_pool_alloc<float> q_amax(ctx.pool(), (size_t) std::max<int64_t>(nrows_q, 1));
    ggml_cuda_pool_alloc<float> q_input_scales(ctx.pool(), (size_t) std::max<int64_t>(nrows_q, 1));
    ggml_cuda_pool_alloc<block_nvfp4_8> q_blocks(ctx.pool(), (size_t) std::max<int64_t>(nrows_q * nblk, 1));

    if (nrows_q > 0) {
        nvfp4_8_q_abs_max_kernel<<<(uint32_t) nrows_q, 256, 0, stream>>>(
                (const float *) src1->data,
                q_amax.get(),
                ne10, ne11, ne12, ne13,
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
        CUDA_CHECK(cudaGetLastError());

        const dim3 quant_grid((uint32_t) nblk, (uint32_t) nrows_q, 1);
        nvfp4_8_q_quantize_kernel<<<quant_grid, WARP_SIZE, 0, stream>>>(
                (const float *) src1->data,
                q_amax.get(),
                q_blocks.get(),
                q_input_scales.get(),
                ne10, ne11, ne12, ne13,
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                nblk);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];
    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    if (total > 0) {
        nvfp4_8_kq_kernel<<<(uint32_t) total, 256, 0, stream>>>(
                (const char *) src0->data,
                q_blocks.get(),
                q_input_scales.get(),
                (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[1], src1->ne[2], src1->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                nblk, r2, r3);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

bool ggml_cuda_mul_mat_nvfp4_8_kq_cublaslt(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_UNUSED(GGML_CUDA_NVFP4_8_KQ_CUBLASLT_ENV);
#if GGML_CUDA_HAS_CUBLASLT && GGML_CUDA_HAS_FP4 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && GGML_CUDA_NVFP4_8_KQ_HAS_LT_SCALE_CHANNEL_ATTRS
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(dst  != nullptr);

    if (src0->type != GGML_TYPE_NVFP4_8 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % QK_NVFP4_8 != 0) {
        return false;
    }
    if (dst->ne[0] != src0->ne[1] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] || dst->ne[3] != src1->ne[3]) {
        return false;
    }
    if (src0->ne[2] <= 0 || src0->ne[3] <= 0 || src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    if (dst->nb[0] != sizeof(float)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t ne01 = src0->ne[1];
    const int64_t lt_m = ggml_cuda_nvfp4_8_kq_pad_i64(ne01, 16);
    const int64_t nblk8 = ne10 / QK_NVFP4_8;
    // cuBLASLt exposes VEC16 FP4 scale groups; expand each NVFP4_8 block into one 16-lane group
    // with the high 8 lanes zeroed so the original per-8 scale is preserved.
    const int64_t lt_k = nblk8 * 16;
    const int64_t lt_n = ggml_cuda_nvfp4_8_kq_pad_i64(ne11, 16);
    const int64_t scale_inner_padded = ggml_cuda_nvfp4_8_kq_pad_i64(nblk8, 4);
    const int64_t k_scale_outer_padded = ggml_cuda_nvfp4_8_kq_pad_i64(lt_m, 128);
    const int64_t q_scale_outer_padded = ggml_cuda_nvfp4_8_kq_pad_i64(lt_n, 128);
    const int64_t row_data_bytes = lt_k / 2;
    const size_t k_data_nbytes = (size_t) lt_m * (size_t) row_data_bytes;
    const size_t q_data_nbytes = (size_t) lt_n * (size_t) row_data_bytes;
    const size_t k_scale_nbytes = (size_t) k_scale_outer_padded * (size_t) scale_inner_padded;
    const size_t q_scale_nbytes = (size_t) q_scale_outer_padded * (size_t) scale_inner_padded;

    ggml_cuda_pool_alloc<float> q_amax(ctx.pool(), (size_t) std::max<int64_t>(ne11, 1));
    ggml_cuda_pool_alloc<float> q_input_scales(ctx.pool(), (size_t) std::max<int64_t>(ne11, 1));
    ggml_cuda_pool_alloc<block_nvfp4_8> q_blocks(ctx.pool(), (size_t) std::max<int64_t>(ne11 * nblk8, 1));
    ggml_cuda_pool_alloc<uint8_t> k_data(ctx.pool(), k_data_nbytes);
    ggml_cuda_pool_alloc<uint8_t> k_scale(ctx.pool(), k_scale_nbytes);
    ggml_cuda_pool_alloc<uint8_t> q_data(ctx.pool(), q_data_nbytes);
    ggml_cuda_pool_alloc<uint8_t> q_scale(ctx.pool(), q_scale_nbytes);
    ggml_cuda_pool_alloc<float> dst_tmp(ctx.pool(), (size_t) lt_m * (size_t) lt_n);

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cudaDataType_t scale_type = CUDA_R_32F;
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type));
    }
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) lt_k, (uint64_t) lt_m, (int64_t) lt_k);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        st = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) lt_k, (uint64_t) lt_n, (int64_t) lt_k);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        st = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, (uint64_t) lt_m, (uint64_t) lt_n, (int64_t) lt_m);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        st = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }

    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];

    for (int64_t i3 = 0; i3 < ne13 && st == CUBLAS_STATUS_SUCCESS; ++i3) {
        for (int64_t i2 = 0; i2 < ne12 && st == CUBLAS_STATUS_SUCCESS; ++i2) {
            const int64_t k_i2 = i2 / r2;
            const int64_t k_i3 = i3 / r3;
            const char * k_slice = (const char *) src0->data + k_i2 * src0->nb[2] + k_i3 * src0->nb[3];
            const char * q_slice = (const char *) src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
            char * dst_slice = (char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3];

            CUDA_CHECK(cudaMemsetAsync(k_data.get(), 0, k_data_nbytes, stream));
            CUDA_CHECK(cudaMemsetAsync(k_scale.get(), 0, k_scale_nbytes, stream));
            CUDA_CHECK(cudaMemsetAsync(q_data.get(), 0, q_data_nbytes, stream));
            CUDA_CHECK(cudaMemsetAsync(q_scale.get(), 0, q_scale_nbytes, stream));
            CUDA_CHECK(cudaMemsetAsync(dst_tmp.get(), 0, (size_t) lt_m * (size_t) lt_n * sizeof(float), stream));

            const int block_size = 256;
            const int k_grid = (int) ((ne01 * nblk8 + block_size - 1) / block_size);
            nvfp4_8_k_repack_lt_vec16_kernel<<<k_grid, block_size, 0, stream>>>(
                    k_slice,
                    k_data.get(),
                    k_scale.get(),
                    src0->ne[1],
                    src0->nb[0], src0->nb[1],
                    nblk8,
                    row_data_bytes,
                    scale_inner_padded);
            CUDA_CHECK(cudaGetLastError());

            nvfp4_8_q_abs_max_kernel<<<(uint32_t) ne11, block_size, 0, stream>>>(
                    (const float *) q_slice,
                    q_amax.get(),
                    ne10, ne11, 1, 1,
                    src1->nb[0], src1->nb[1], 0, 0);
            CUDA_CHECK(cudaGetLastError());

            const dim3 quant_grid((uint32_t) nblk8, (uint32_t) ne11, 1);
            nvfp4_8_q_quantize_kernel<<<quant_grid, WARP_SIZE, 0, stream>>>(
                    (const float *) q_slice,
                    q_amax.get(),
                    q_blocks.get(),
                    q_input_scales.get(),
                    ne10, ne11, 1, 1,
                    src1->nb[0], src1->nb[1], 0, 0,
                    nblk8);
            CUDA_CHECK(cudaGetLastError());

            const int q_grid = (int) ((ne11 * nblk8 + block_size - 1) / block_size);
            nvfp4_8_split_blocks_lt_vec16_kernel<<<q_grid, block_size, 0, stream>>>(
                    q_blocks.get(),
                    q_data.get(),
                    q_scale.get(),
                    nblk8,
                    ne11,
                    row_data_bytes,
                    scale_inner_padded);
            CUDA_CHECK(cudaGetLastError());

            if (st == CUBLAS_STATUS_SUCCESS) {
                const void * a_scale_ptr = (const void *) k_scale.get();
                st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
            }
            if (st == CUBLAS_STATUS_SUCCESS) {
                const void * b_scale_ptr = (const void *) q_scale.get();
                st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));
            }
            if (st == CUBLAS_STATUS_SUCCESS) {
                const float alpha = 1.0f;
                const float beta = 0.0f;
                st = cublasLtMatmul(
                        ctx.cublaslt_handle(),
                        op_desc,
                        &alpha,
                        k_data.get(), a_desc,
                        q_data.get(), b_desc,
                        &beta,
                        dst_tmp.get(), c_desc,
                        dst_tmp.get(), c_desc,
                        nullptr,
                        nullptr, 0,
                        stream);
            }
            if (st == CUBLAS_STATUS_SUCCESS) {
                const int64_t total = ne01 * ne11;
                const int store_grid = (int) ((total + block_size - 1) / block_size);
                nvfp4_8_kq_store_scaled_kernel<<<store_grid, block_size, 0, stream>>>(
                        dst_tmp.get(),
                        q_input_scales.get(),
                        dst_slice,
                        ne01,
                        ne11,
                        lt_m,
                        dst->nb[0],
                        dst->nb[1]);
                CUDA_CHECK(cudaGetLastError());
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

    return st == CUBLAS_STATUS_SUCCESS;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#endif
}
