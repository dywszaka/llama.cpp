#include "fp8-e8m0-matmul.cuh"

#include <atomic>

namespace {

#if defined(CUBLAS_VERSION)
#define GGML_CUDA_FP8_E8M0_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VERSION >= 120900)
#elif defined(CUBLAS_VER_MAJOR) && defined(CUBLAS_VER_MINOR)
#define GGML_CUDA_FP8_E8M0_HAS_LT_SCALE_CHANNEL_ATTRS ((CUBLAS_VER_MAJOR > 12) || (CUBLAS_VER_MAJOR == 12 && CUBLAS_VER_MINOR >= 9))
#else
#define GGML_CUDA_FP8_E8M0_HAS_LT_SCALE_CHANNEL_ATTRS 0
#endif

static inline int64_t ggml_cuda_fp8_pad_i64(int64_t x, int64_t a) {
    GGML_ASSERT(a > 0);
    return ((x + a - 1) / a) * a;
}

static __host__ __device__ __forceinline__ int64_t ggml_cuda_fp8_e8m0_scale_tiled_index(
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

static __global__ void quantize_row_fp8_e8m0_kernel(
        const float * __restrict__ x,
        block_fp8_e4m3_e8m0_32 * __restrict__ y,
        int64_t ne00,
        int64_t s01) {
    const int64_t ib = blockIdx.x;
    const int64_t i1 = blockIdx.y;
    const int lane = threadIdx.x;

    const int64_t i0 = ib * QK_FP8_E4M3_E8M0_32 + lane;
    const float xi = x[i1 * s01 + i0];
    float vmax = fabsf(xi);

    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 16, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    uint8_t scale_q = 0;
    float scale = 0.0f;
    if (lane == 0) {
        scale_q = ggml_cuda_fp32_to_e8m0_ceil_scale(vmax / 448.0f);
        scale = ggml_cuda_e8m0_to_fp32(scale_q);
        y[i1 * (ne00 / QK_FP8_E4M3_E8M0_32) + ib].e = scale_q;
    }
    scale_q = __shfl_sync(0xFFFFFFFF, scale_q, 0, WARP_SIZE);
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);

    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    y[i1 * (ne00 / QK_FP8_E4M3_E8M0_32) + ib].qs[lane] =
            scale > 0.0f ? __nv_cvt_float_to_fp8(xi * inv_scale, __NV_SATFINITE, __NV_E4M3) : 0;

    GGML_UNUSED(ne00);
}

static void quantize_row_fp8_e8m0_cuda(
        const float * x,
        block_fp8_e4m3_e8m0_32 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_FP8_E4M3_E8M0_32 == 0);

    const dim3 num_blocks((uint32_t) (ne00 / QK_FP8_E4M3_E8M0_32), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_fp8_e8m0_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01);
}

static __global__ void quantize_row_fp8_e8m0_strided_kernel(
        const float * __restrict__ x,
        block_fp8_e4m3_e8m0_32 * __restrict__ y,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t nb10,
        const int64_t nb11,
        const int64_t nb12,
        const int64_t nb13,
        const int64_t nblk) {
    const int64_t ib = blockIdx.x;
    const int64_t row = blockIdx.y;
    const int lane = threadIdx.x;
    const int64_t nrows = ne11 * ne12 * ne13;
    if (row >= nrows || ib >= nblk) {
        return;
    }

    const int64_t i13 = row / (ne11 * ne12);
    const int64_t rem = row - i13 * ne11 * ne12;
    const int64_t i12 = rem / ne11;
    const int64_t i11 = rem - i12 * ne11;
    const int64_t k = ib * QK_FP8_E4M3_E8M0_32 + lane;
    const char * base = (const char *) x + i11 * nb11 + i12 * nb12 + i13 * nb13;

    const float xi = k < ne10 ? *(const float *) (base + k * nb10) : 0.0f;
    float vmax = fabsf(xi);

    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 16, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    uint8_t scale_q = 0;
    float scale = 0.0f;
    if (lane == 0) {
        scale_q = ggml_cuda_fp32_to_e8m0_ceil_scale(vmax / 448.0f);
        scale = ggml_cuda_e8m0_to_fp32(scale_q);
        y[row * nblk + ib].e = scale_q;
    }
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);

    const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    y[row * nblk + ib].qs[lane] =
            scale > 0.0f ? __nv_cvt_float_to_fp8(xi * inv_scale, __NV_SATFINITE, __NV_E4M3) : 0;
}

static __global__ void fp8_e8m0_vp_kernel(
        const char * __restrict__ v,
        const block_fp8_e4m3_e8m0_32 * __restrict__ p_blocks,
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

    const int64_t v_i2 = i2 / r2;
    const int64_t v_i3 = i3 / r3;
    if (v_i2 >= ne02 || v_i3 >= ne03) {
        return;
    }

    const char * v_row = v + i0 * nb01 + v_i2 * nb02 + v_i3 * nb03;
    const int64_t p_row = i1 + i2 * ne11 + i3 * ne11 * ne12;
    const block_fp8_e4m3_e8m0_32 * p_row_blocks = p_blocks + p_row * nblk;

    float acc = 0.0f;
    for (int64_t ib = threadIdx.x; ib < nblk; ib += blockDim.x) {
        const block_fp8_e4m3_e8m0_32 * vb = (const block_fp8_e4m3_e8m0_32 *) (v_row + ib * nb00);
        const block_fp8_e4m3_e8m0_32 pb = p_row_blocks[ib];
        const float vd = vb->e == 0 ? 0.0f : ggml_cuda_e8m0_to_fp32(vb->e);
        const float pd = pb.e == 0 ? 0.0f : ggml_cuda_e8m0_to_fp32(pb.e);

#pragma unroll
        for (int j = 0; j < QK_FP8_E4M3_E8M0_32; ++j) {
            acc += vd * pd * ggml_cuda_e4m3_to_fp32(vb->qs[j]) * ggml_cuda_e4m3_to_fp32(pb.qs[j]);
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

static __global__ void quantize_row_fp8_e8m0_16_kernel(
        const float * __restrict__ x,
        block_fp8_e4m3_e8m0_16 * __restrict__ y,
        int64_t ne00,
        int64_t s01) {
    const int64_t ib = blockIdx.x;
    const int64_t i1 = blockIdx.y;
    const int lane = threadIdx.x;

    const int64_t i0 = ib * QK_FP8_E4M3_E8M0_16 + lane;
    const float xi = lane < QK_FP8_E4M3_E8M0_16 ? x[i1 * s01 + i0] : 0.0f;
    float vmax = fabsf(xi);

    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 16, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    uint8_t scale_q = 0;
    float scale = 0.0f;
    if (lane == 0) {
        scale_q = vmax == 0.0f ? 0 : ggml_cuda_fp32_to_e8m0_ceil_scale(vmax / 448.0f);
        scale = scale_q == 0 ? 0.0f : ggml_cuda_e8m0_to_fp32(scale_q);
        y[i1 * (ne00 / QK_FP8_E4M3_E8M0_16) + ib].e = scale_q;
    }
    scale_q = __shfl_sync(0xFFFFFFFF, scale_q, 0, WARP_SIZE);
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);

    if (lane < QK_FP8_E4M3_E8M0_16) {
        const float inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        y[i1 * (ne00 / QK_FP8_E4M3_E8M0_16) + ib].qs[lane] =
                scale > 0.0f ? __nv_cvt_float_to_fp8(xi * inv_scale, __NV_SATFINITE, __NV_E4M3) : 0;
    }

    GGML_UNUSED(ne00);
}

static void quantize_row_fp8_e8m0_16_cuda(
        const float * x,
        block_fp8_e4m3_e8m0_16 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_FP8_E4M3_E8M0_16 == 0);

    const dim3 num_blocks((uint32_t) (ne00 / QK_FP8_E4M3_E8M0_16), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_fp8_e8m0_16_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01);
}

static __global__ void split_blocks_fp8_e8m0_kernel(
        const block_fp8_e4m3_e8m0_32 * __restrict__ in,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        int64_t nblk_k,
        int64_t n_outer_valid,
        int64_t row_data_bytes,
        int64_t n_inner_padded) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = nblk_k * n_outer_valid;
    if (idx >= total) {
        return;
    }

    const int64_t outer = idx / nblk_k;
    const int64_t inner = idx % nblk_k;

    const block_fp8_e4m3_e8m0_32 v = in[idx];
    uint8_t * data_dst = out_data + outer * row_data_bytes + inner * QK_FP8_E4M3_E8M0_32;
#pragma unroll
    for (int i = 0; i < QK_FP8_E4M3_E8M0_32; ++i) {
        data_dst[i] = v.qs[i];
    }

    const int64_t scale_idx = ggml_cuda_fp8_e8m0_scale_tiled_index(outer, inner, n_inner_padded);
    out_scale[scale_idx] = v.e;
}

static __global__ void split_blocks_fp8_e8m0_16_to_vec32_kernel(
        const block_fp8_e4m3_e8m0_16 * __restrict__ in,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        int64_t nblk_k,
        int64_t n_outer_valid,
        int64_t row_data_bytes,
        int64_t n_inner_padded) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = nblk_k * n_outer_valid;
    if (idx >= total) {
        return;
    }

    const int64_t outer = idx / nblk_k;
    const int64_t inner = idx % nblk_k;

    const block_fp8_e4m3_e8m0_16 v = in[idx];
    uint8_t * data_dst = out_data + outer * row_data_bytes + inner * QK_FP8_E4M3_E8M0_32;
#pragma unroll
    for (int i = 0; i < QK_FP8_E4M3_E8M0_16; ++i) {
        data_dst[i] = v.qs[i];
    }
#pragma unroll
    for (int i = QK_FP8_E4M3_E8M0_16; i < QK_FP8_E4M3_E8M0_32; ++i) {
        data_dst[i] = 0;
    }

    const int64_t scale_idx = ggml_cuda_fp8_e8m0_scale_tiled_index(outer, inner, n_inner_padded);
    out_scale[scale_idx] = v.e;
}

static void split_blocks_fp8_e8m0_cuda(
        const block_fp8_e4m3_e8m0_32 * in,
        uint8_t * out_data,
        uint8_t * out_scale,
        int64_t ne_k,
        int64_t n_outer_valid,
        int64_t n_outer_alloc,
        int64_t * scale_inner_padded,
        int64_t * scale_outer_padded,
        size_t * data_nbytes,
        size_t * scale_nbytes,
        cudaStream_t stream) {
    GGML_ASSERT(ne_k % QK_FP8_E4M3_E8M0_32 == 0);
    GGML_ASSERT(n_outer_valid >= 0 && n_outer_alloc >= n_outer_valid);

    const int64_t nblk_k = ne_k / QK_FP8_E4M3_E8M0_32;
    const int64_t row_data_bytes = ne_k;
    const int64_t inner_padded = ggml_cuda_fp8_pad_i64(nblk_k, 4);
    const int64_t outer_padded = ggml_cuda_fp8_pad_i64(n_outer_alloc, 128);

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
        split_blocks_fp8_e8m0_kernel<<<grid_size, block_size, 0, stream>>>(
                in, out_data, out_scale, nblk_k, n_outer_valid, row_data_bytes, inner_padded);
        CUDA_CHECK(cudaGetLastError());
    }
}

static void split_blocks_fp8_e8m0_16_to_vec32_cuda(
        const block_fp8_e4m3_e8m0_16 * in,
        uint8_t * out_data,
        uint8_t * out_scale,
        int64_t ne_k,
        int64_t n_outer_valid,
        int64_t n_outer_alloc,
        int64_t * scale_inner_padded,
        int64_t * scale_outer_padded,
        size_t * data_nbytes,
        size_t * scale_nbytes,
        cudaStream_t stream) {
    GGML_ASSERT(ne_k % QK_FP8_E4M3_E8M0_16 == 0);
    GGML_ASSERT(n_outer_valid >= 0 && n_outer_alloc >= n_outer_valid);

    const int64_t nblk_k = ne_k / QK_FP8_E4M3_E8M0_16;
    const int64_t row_data_bytes = ne_k * 2;
    const int64_t inner_padded = ggml_cuda_fp8_pad_i64(nblk_k, 4);
    const int64_t outer_padded = ggml_cuda_fp8_pad_i64(n_outer_alloc, 128);

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
        split_blocks_fp8_e8m0_16_to_vec32_kernel<<<grid_size, block_size, 0, stream>>>(
                in, out_data, out_scale, nblk_k, n_outer_valid, row_data_bytes, inner_padded);
        CUDA_CHECK(cudaGetLastError());
    }
}

static ggml_tensor ggml_cuda_fp8_make_matrix_slice(
        const ggml_tensor * src,
        int64_t i2,
        int64_t i3) {
    ggml_tensor slice = *src;
    slice.data = (char *) src->data + i2 * src->nb[2] + i3 * src->nb[3];
    slice.ne[2] = 1;
    slice.ne[3] = 1;
    slice.nb[2] = slice.nb[1] * slice.ne[1];
    slice.nb[3] = slice.nb[2] * slice.ne[2];
    return slice;
}

} // namespace

static bool ggml_cuda_mul_mat_fp8_e8m0_16_native_2d(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if GGML_CUDA_HAS_CUBLASLT && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && GGML_CUDA_FP8_E8M0_HAS_LT_SCALE_CHANNEL_ATTRS
    if (src0->type != GGML_TYPE_FP8_E4M3_E8M0_16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (ggml_is_transposed(src0) || ggml_is_transposed(src1) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t k = src1->ne[0];
    const int64_t n = src1->ne[1];
    const int64_t m = src0->ne[1];
    if (src0->ne[0] != k || dst->ne[0] != m || dst->ne[1] != n) {
        return false;
    }

    if ((k % QK_FP8_E4M3_E8M0_16) != 0) {
        return false;
    }

    if ((m % 16) != 0) {
        return false;
    }

    const int64_t k_expanded = k * 2;
    const int64_t n_padded = ggml_cuda_fp8_pad_i64(n, 16);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<uint8_t> src0_repacked_data(ctx.pool(), (size_t) m * (size_t) k_expanded);
    ggml_cuda_pool_alloc<uint8_t> src0_repacked_scale(ctx.pool(), (size_t) ggml_cuda_fp8_pad_i64(m, 128) * (size_t) ggml_cuda_fp8_pad_i64(k / QK_FP8_E4M3_E8M0_16, 4));
    ggml_cuda_pool_alloc<block_fp8_e4m3_e8m0_16> src1_q(ctx.pool(), (size_t) (k / QK_FP8_E4M3_E8M0_16) * (size_t) n);
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_data(ctx.pool(), (size_t) n_padded * (size_t) k_expanded);
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_scale(ctx.pool(), (size_t) ggml_cuda_fp8_pad_i64(n_padded, 128) * (size_t) ggml_cuda_fp8_pad_i64(k / QK_FP8_E4M3_E8M0_16, 4));
    ggml_cuda_pool_alloc<float> dst_padded(ctx.pool(), (size_t) m * (size_t) n_padded);

    split_blocks_fp8_e8m0_16_to_vec32_cuda(
            (const block_fp8_e4m3_e8m0_16 *) src0->data,
            src0_repacked_data.get(),
            src0_repacked_scale.get(),
            k,
            m,
            m,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            stream);

    quantize_row_fp8_e8m0_16_cuda(
            (const float *) src1->data,
            src1_q.get(),
            k,
            src1->nb[1] / (int64_t) sizeof(float),
            n,
            stream);

    split_blocks_fp8_e8m0_16_to_vec32_cuda(
            src1_q.get(),
            src1_repacked_data.get(),
            src1_repacked_scale.get(),
            k,
            n,
            n_padded,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            stream);

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const void * a_scale_ptr = (const void *) src0_repacked_scale.get();
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const void * b_scale_ptr = (const void *) src1_repacked_scale.get();
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, (uint64_t) k_expanded, (uint64_t) m, (int64_t) k_expanded);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, (uint64_t) k_expanded, (uint64_t) n_padded, (int64_t) k_expanded);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, (uint64_t) m, (uint64_t) n_padded, (int64_t) m);
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        st = cublasLtMatmul(
                ctx.cublaslt_handle(),
                op_desc,
                &alpha,
                src0_repacked_data.get(), a_desc,
                src1_repacked_data.get(), b_desc,
                &beta,
                dst_padded.get(), c_desc,
                dst_padded.get(), c_desc,
                nullptr,
                nullptr, 0,
                stream);
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        CUDA_CHECK(cudaMemcpyAsync(
                dst->data,
                dst_padded.get(),
                (size_t) m * (size_t) n * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
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

bool ggml_cuda_mul_mat_fp8_e8m0_16_native(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(dst  != nullptr);

    if (src0->type != GGML_TYPE_FP8_E4M3_E8M0_16 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        if (!ggml_is_contiguous(dst)) {
            return false;
        }

        if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
            return false;
        }

        const int64_t r2 = src1->ne[2] / src0->ne[2];
        const int64_t r3 = src1->ne[3] / src0->ne[3];
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                ggml_tensor src0_slice = ggml_cuda_fp8_make_matrix_slice(src0, i2 / r2, i3 / r3);
                ggml_tensor src1_slice = ggml_cuda_fp8_make_matrix_slice(src1, i2, i3);
                ggml_tensor dst_slice  = ggml_cuda_fp8_make_matrix_slice(dst,  i2, i3);

                if (!ggml_cuda_mul_mat_fp8_e8m0_16_native_2d(ctx, &src0_slice, &src1_slice, &dst_slice)) {
                    return false;
                }
            }
        }

        return true;
    }

    return ggml_cuda_mul_mat_fp8_e8m0_16_native_2d(ctx, src0, src1, dst);
}

bool ggml_cuda_mul_mat_fp8_e8m0_native(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if GGML_CUDA_HAS_CUBLASLT && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && GGML_CUDA_FP8_E8M0_HAS_LT_SCALE_CHANNEL_ATTRS
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(dst  != nullptr);

    if (src0->type != GGML_TYPE_FP8_E4M3_E8M0_32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        if (!ggml_is_contiguous(dst)) {
            return false;
        }

        if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
            return false;
        }

        const int64_t r2 = src1->ne[2] / src0->ne[2];
        const int64_t r3 = src1->ne[3] / src0->ne[3];
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                ggml_tensor src0_slice = ggml_cuda_fp8_make_matrix_slice(src0, i2 / r2, i3 / r3);
                ggml_tensor src1_slice = ggml_cuda_fp8_make_matrix_slice(src1, i2, i3);
                ggml_tensor dst_slice  = ggml_cuda_fp8_make_matrix_slice(dst,  i2, i3);

                if (!ggml_cuda_mul_mat_fp8_e8m0_native(ctx, &src0_slice, &src1_slice, &dst_slice)) {
                    return false;
                }
            }
        }

        return true;
    }

    if (ggml_is_transposed(src0) || ggml_is_transposed(src1) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t k = src1->ne[0];
    const int64_t n = src1->ne[1];
    const int64_t m = src0->ne[1];
    if (src0->ne[0] != k || dst->ne[0] != m || dst->ne[1] != n) {
        return false;
    }

    if ((k % QK_FP8_E4M3_E8M0_32) != 0 || (m % 16) != 0) {
        return false;
    }

    const int64_t n_padded = ggml_cuda_fp8_pad_i64(n, 16);
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<uint8_t> src0_repacked_data(ctx.pool(), (size_t) m * (size_t) k);
    ggml_cuda_pool_alloc<uint8_t> src0_repacked_scale(ctx.pool(), (size_t) ggml_cuda_fp8_pad_i64(m, 128) * (size_t) ggml_cuda_fp8_pad_i64(k / QK_FP8_E4M3_E8M0_32, 4));
    ggml_cuda_pool_alloc<block_fp8_e4m3_e8m0_32> src1_q(ctx.pool(), (size_t) (k / QK_FP8_E4M3_E8M0_32) * (size_t) n);
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_data(ctx.pool(), (size_t) n_padded * (size_t) k);
    ggml_cuda_pool_alloc<uint8_t> src1_repacked_scale(ctx.pool(), (size_t) ggml_cuda_fp8_pad_i64(n_padded, 128) * (size_t) ggml_cuda_fp8_pad_i64(k / QK_FP8_E4M3_E8M0_32, 4));
    ggml_cuda_pool_alloc<float> dst_padded(ctx.pool(), (size_t) m * (size_t) n_padded);

    int64_t a_scale_inner_padded = 0;
    int64_t a_scale_outer_padded = 0;
    size_t a_data_nbytes = 0;
    size_t a_scale_nbytes = 0;
    split_blocks_fp8_e8m0_cuda(
            (const block_fp8_e4m3_e8m0_32 *) src0->data,
            src0_repacked_data.get(),
            src0_repacked_scale.get(),
            k,
            m,
            m,
            &a_scale_inner_padded,
            &a_scale_outer_padded,
            &a_data_nbytes,
            &a_scale_nbytes,
            stream);

    quantize_row_fp8_e8m0_cuda(
            (const float *) src1->data,
            src1_q.get(),
            k,
            src1->nb[1] / (int64_t) sizeof(float),
            n,
            stream);

    int64_t b_scale_inner_padded = 0;
    int64_t b_scale_outer_padded = 0;
    size_t b_data_nbytes = 0;
    size_t b_scale_nbytes = 0;
    split_blocks_fp8_e8m0_cuda(
            src1_q.get(),
            src1_repacked_data.get(),
            src1_repacked_scale.get(),
            k,
            n,
            n_padded,
            &b_scale_inner_padded,
            &b_scale_outer_padded,
            &b_data_nbytes,
            &b_scale_nbytes,
            stream);

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const void * a_scale_ptr = (const void *) src0_repacked_scale.get();
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const void * b_scale_ptr = (const void *) src1_repacked_scale.get();
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, (uint64_t) k, (uint64_t) m, (int64_t) k);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, (uint64_t) k, (uint64_t) n_padded, (int64_t) k);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, (uint64_t) m, (uint64_t) n_padded, (int64_t) m);
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        st = cublasLtMatmul(
                ctx.cublaslt_handle(),
                op_desc,
                &alpha,
                src0_repacked_data.get(), a_desc,
                src1_repacked_data.get(), b_desc,
                &beta,
                dst_padded.get(), c_desc,
                dst_padded.get(), c_desc,
                nullptr,
                nullptr, 0,
                stream);
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        CUDA_CHECK(cudaMemcpyAsync(
                dst->data,
                dst_padded.get(),
                (size_t) m * (size_t) n * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
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

bool ggml_cuda_mul_mat_fp8_e8m0_vp(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_FP8_E4M3_E8M0_32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[0] != src1->ne[0] || src0->ne[0] % QK_FP8_E4M3_E8M0_32 != 0) {
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
    const int64_t nrows_p = ne11 * ne12 * ne13;
    const int64_t nblk = ne10 / QK_FP8_E4M3_E8M0_32;

    ggml_cuda_pool_alloc<block_fp8_e4m3_e8m0_32> p_blocks(ctx.pool(), (size_t) std::max<int64_t>(nrows_p * nblk, 1));

    if (nrows_p > 0) {
        const dim3 quant_grid((uint32_t) nblk, (uint32_t) nrows_p, 1);
        quantize_row_fp8_e8m0_strided_kernel<<<quant_grid, WARP_SIZE, 0, stream>>>(
                (const float *) src1->data,
                p_blocks.get(),
                ne10, ne11, ne12, ne13,
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                nblk);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t r2 = src1->ne[2] / src0->ne[2];
    const int64_t r3 = src1->ne[3] / src0->ne[3];
    const int64_t total = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    if (total > 0) {
        fp8_e8m0_vp_kernel<<<(uint32_t) total, 256, 0, stream>>>(
                (const char *) src0->data,
                p_blocks.get(),
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
