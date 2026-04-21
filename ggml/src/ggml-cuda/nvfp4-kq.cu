#include "nvfp4-kq.cuh"

#include <algorithm>
#include <cmath>

namespace {

static constexpr float GGML_CUDA_NVFP4_8_FP4_MAX = 6.0f;
static constexpr float GGML_CUDA_NVFP4_8_E4M3_HALF_MAX = 224.0f;
static constexpr float GGML_CUDA_NVFP4_8_GLOBAL_SCALE_MAX =
        GGML_CUDA_NVFP4_8_FP4_MAX * GGML_CUDA_NVFP4_8_E4M3_HALF_MAX;

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
