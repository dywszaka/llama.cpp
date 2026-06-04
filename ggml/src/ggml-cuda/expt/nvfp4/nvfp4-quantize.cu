#include "nvfp4-quantize.cuh"

#include "../../common.cuh"
#include "kcache-outlier.cuh"
#include "nvfp4-common.cuh"

namespace {

static __device__ __forceinline__ float ggml_cuda_trunc_f32_to_bf16_value(float v) {
    union {
        float    f;
        uint32_t u;
    } bits;
    bits.f = v;
    bits.u &= 0xffff0000u;
    return bits.f;
}

static __device__ __forceinline__ float ggml_cuda_abs_f32_bits(float v) {
    union {
        float    f;
        uint32_t u;
    } bits;
    bits.f = v;
    bits.u &= 0x7fffffffu;
    return bits.f;
}

static __device__ __forceinline__ int ggml_cuda_f32_to_i32_bits(float v) {
    union {
        float f;
        int   i;
    } bits;
    bits.f = v;
    return bits.i;
}

static __device__ __forceinline__ float ggml_cuda_i32_to_f32_bits(int v) {
    union {
        int   i;
        float f;
    } bits;
    bits.i = v;
    return bits.f;
}

static __device__ __forceinline__ float ggml_cuda_max_f32_select(float a, float b) {
    return a > b ? a : b;
}

static __device__ __forceinline__ void ggml_cuda_atomic_max_f32(float * addr, float value) {
    int * addr_i = (int *) addr;
    int old = *addr_i;

    while (ggml_cuda_i32_to_f32_bits(old) < value) {
        const int assumed = old;
        old = atomicCAS(addr_i, assumed, ggml_cuda_f32_to_i32_bits(value));
        if (old == assumed) {
            break;
        }
    }
}

static __global__ void ggml_cuda_nvfp4_abs_max_rows_f32_kernel(
        const float * __restrict__ src,
        float * __restrict__ amax_rows,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t s01,
        const bool truncate_bf16_input) {
    const int64_t row = blockIdx.x;
    if (row >= ne01) {
        return;
    }

    float local_max = 0.0f;
    const int64_t row_off = row * s01;
    for (int64_t i = threadIdx.x; i < ne00; i += blockDim.x) {
        const float xi = src[row_off + i];
        const float xq = truncate_bf16_input ? ggml_cuda_trunc_f32_to_bf16_value(xi) : xi;
        local_max = ggml_cuda_max_f32_select(local_max, ggml_cuda_abs_f32_bits(xq));
    }

    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = ggml_cuda_max_f32_select(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        amax_rows[row] = shared_max[0];
    }
}

static __global__ void ggml_cuda_nvfp4_abs_max_tensor_f32_kernel(
        const float * __restrict__ src,
        float * __restrict__ amax,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t s01,
        const bool truncate_bf16_input) {
    float local_max = 0.0f;
    const int64_t total = ne00 * ne01;
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t) gridDim.x * blockDim.x) {
        const int64_t row = idx / ne00;
        const int64_t col = idx - row * ne00;
        const float xi = src[row * s01 + col];
        const float xq = truncate_bf16_input ? ggml_cuda_trunc_f32_to_bf16_value(xi) : xi;
        local_max = ggml_cuda_max_f32_select(local_max, ggml_cuda_abs_f32_bits(xq));
    }

    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = ggml_cuda_max_f32_select(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        ggml_cuda_atomic_max_f32(amax, shared_max[0]);
    }
}

static __global__ void ggml_cuda_nvfp4_prepare_dynamic_input_scales_kernel(
        const float * __restrict__ amax_rows,
        float * __restrict__ input_scales,
        const int64_t nrows,
        const float out_scale,
        const bool per_tensor_scale) {
    const int64_t row = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }

    const float amax_f = per_tensor_scale ? amax_rows[0] : amax_rows[row];
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_q_global_scale(amax_f);
    input_scales[row] = (global_scale != 0.0f) ? (out_scale / global_scale) : 0.0f;
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
        const int64_t ne00,
        const int64_t s01,
        const float global_scale,
        const bool truncate_bf16_input) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const float xi_src = (lane_active && k0 < ne00) ? x[(int64_t) i1 * s01 + k0] : 0.0f;
    const float xi = truncate_bf16_input ? ggml_cuda_trunc_f32_to_bf16_value(xi_src) : xi_src;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    float scale_f = 0.0f;
    if (lane == 0) {
        const float scale = global_scale * (vmax / GGML_CUDA_NVFP4_FP4_MAX);
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

static __global__ void quantize_row_nvfp4_dynamic_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ amax_rows,
        const bool per_tensor_scale,
        const bool truncate_bf16_input) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;
    const float xi_src = (lane_active && k0 < ne00) ? x[(int64_t) i1 * s01 + k0] : 0.0f;
    const float xi = truncate_bf16_input ? ggml_cuda_trunc_f32_to_bf16_value(xi_src) : xi_src;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    float scale_f = 0.0f;
    const float amax_f = per_tensor_scale ? amax_rows[0] : amax_rows[i1];
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_q_global_scale(amax_f);
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

} // namespace

void ggml_cuda_nvfp4_abs_max_rows_f32(
        const float * src,
        float * amax_rows,
        int64_t ne00,
        int64_t ne01,
        int64_t s01,
        bool truncate_bf16_input,
        cudaStream_t stream) {
    ggml_cuda_nvfp4_abs_max_rows_f32_kernel<<<(int) ne01, 256, 0, stream>>>(
            src, amax_rows, ne00, ne01, s01, truncate_bf16_input);
}

void ggml_cuda_nvfp4_abs_max_tensor_f32(
        const float * src,
        float * amax,
        int64_t ne00,
        int64_t ne01,
        int64_t s01,
        bool truncate_bf16_input,
        cudaStream_t stream) {
    const int block_size = 256;
    const int64_t total = ne00 * ne01;
    const int64_t blocks = (total + block_size - 1) / block_size;
    const int grid_size = (int) (blocks < 1024 ? blocks : 1024);
    ggml_cuda_nvfp4_abs_max_tensor_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            src, amax, ne00, ne01, s01, truncate_bf16_input);
}

void ggml_cuda_nvfp4_prepare_dynamic_input_scales(
        const float * amax_rows,
        float * input_scales,
        int64_t nrows,
        float out_scale,
        bool per_tensor_scale,
        cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = (int) ((nrows + block_size - 1) / block_size);
    ggml_cuda_nvfp4_prepare_dynamic_input_scales_kernel<<<grid_size, block_size, 0, stream>>>(
            amax_rows, input_scales, nrows, out_scale, per_tensor_scale);
}

void ggml_cuda_nvfp4_quantize_rows_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        float global_scale,
        bool truncate_bf16_input,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_kernel<<<num_blocks, block_size, 0, stream>>>(
            x, y, ne00, s01, global_scale, truncate_bf16_input);
}

void ggml_cuda_nvfp4_quantize_rows_dynamic_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * amax_rows,
        bool per_tensor_scale,
        bool truncate_bf16_input,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_dynamic_kernel<<<num_blocks, block_size, 0, stream>>>(
            x, y, ne00, s01, amax_rows, per_tensor_scale, truncate_bf16_input);
}
