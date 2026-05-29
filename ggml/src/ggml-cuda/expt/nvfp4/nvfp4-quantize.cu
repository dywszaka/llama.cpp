#include "nvfp4-quantize.cuh"

#include "../../bf16-round.cuh"
#include "../../common.cuh"
#include "kcache-outlier.cuh"
#include "nvfp4-common.cuh"
#include "nvfp4-log.cuh"

#include <cstdlib>

namespace {

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

static __global__ void ggml_cuda_nvfp4_abs_max_rows_f32_kernel(
        const float * __restrict__ src,
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
        local_max = fmaxf(local_max, fabsf(src[row_off + i]));
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

static __global__ void ggml_cuda_nvfp4_abs_max_tensor_f32_kernel(
        const float * __restrict__ src,
        float * __restrict__ amax,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t s01) {
    float local_max = 0.0f;
    const int64_t total = ne00 * ne01;
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t) gridDim.x * blockDim.x) {
        const int64_t row = idx / ne00;
        const int64_t col = idx - row * ne00;
        local_max = fmaxf(local_max, fabsf(src[row * s01 + col]));
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
        const float global_scale) {
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
        const bool per_tensor_scale) {
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

static __host__ __device__ __forceinline__ uint64_t low_bits_mask_u64(uint8_t width) {
    if (width >= 64u) {
        return ~0ull;
    }
    return width == 0u ? 0ull : ((1ull << width) - 1ull);
}

static __host__ __device__ __forceinline__ uint16_t bf16_abs_bits(uint16_t x) {
    return (uint16_t) (x & (uint16_t) low_bits_mask_u64(15u));
}

static __host__ __device__ __forceinline__ uint64_t shift_hw(uint64_t value, uint8_t shift_amt, uint8_t shift_right) {
    const uint64_t value_q = value & low_bits_mask_u64(32u);
    return ((shift_right & 1u) == 0u)
            ? ((value_q << shift_amt) & low_bits_mask_u64(36u))
            : ((value_q >> shift_amt) & low_bits_mask_u64(36u));
}

static __host__ __device__ __forceinline__ uint32_t float_to_ufixed_q_hw(float val, uint8_t frac_bits) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = val;

    const uint32_t sign = bits.u >> 31;
    const uint32_t exponent = (bits.u >> 23) & (uint32_t) low_bits_mask_u64(8u);
    const uint32_t mantissa = bits.u & (uint32_t) low_bits_mask_u64(23u);
    if (sign != 0u || (exponent == 0u && mantissa == 0u)) {
        return 0;
    }
    if (exponent == 0xffu) {
        return mantissa == 0u ? (uint32_t) low_bits_mask_u64(32u) : 0u;
    }

    const uint32_t significand =
            ((exponent == 0u) ? mantissa : ((1u << 23) | mantissa)) &
            (uint32_t) low_bits_mask_u64(24u);
    const uint32_t exponent_unbiased =
            (exponent == 0u) ? 0x82u : ((exponent - 127u) & (uint32_t) low_bits_mask_u64(8u));
    uint32_t exponent_unbiased_ext = exponent_unbiased;
    if ((exponent_unbiased_ext & 0x80u) != 0u) {
        exponent_unbiased_ext |= 0x100u;
    }

    const uint32_t total_shift =
            (exponent_unbiased_ext + (uint32_t) frac_bits + 0x1e9u) &
            (uint32_t) low_bits_mask_u64(9u);
    const uint8_t shift_right = (uint8_t) ((total_shift >> 8) & 1u);
    uint32_t shift_mag = total_shift;
    if (shift_right != 0u) {
        shift_mag = ((~shift_mag) + 1u) & (uint32_t) low_bits_mask_u64(9u);
    }
    return (uint32_t) (shift_hw(significand, (uint8_t) (shift_mag & 0xffu), shift_right) &
                       low_bits_mask_u64(32u));
}

static __host__ __device__ __forceinline__ uint64_t bf16_abs_mul_uq_hw(
        uint16_t abs_bits,
        uint32_t factor_q,
        int factor_frac_bits,
        int out_frac_bits) {
    if (abs_bits == 0u || factor_q == 0u) {
        return 0;
    }

    const uint32_t exponent = (abs_bits >> 7) & (uint32_t) low_bits_mask_u64(8u);
    const uint32_t mantissa = abs_bits & (uint32_t) low_bits_mask_u64(7u);
    const uint32_t significand =
            ((exponent == 0u) ? mantissa : (0x80u | mantissa)) &
            (uint32_t) low_bits_mask_u64(8u);
    const uint32_t exp_mask = (uint32_t) low_bits_mask_u64(9u);
    const uint32_t exp_sign = 1u << 8;
    const uint32_t value_exp2_tc = (exponent == 0u) ? ((0u - 133u) & exp_mask) : ((exponent - 134u) & exp_mask);
    const uint32_t value_exp2_ext = (value_exp2_tc & exp_sign) != 0u ? (value_exp2_tc | ~exp_mask) : value_exp2_tc;
    const uint32_t frac_delta = (uint32_t) (out_frac_bits - factor_frac_bits) & (uint32_t) low_bits_mask_u64(9u);
    const uint32_t total_shift = (value_exp2_ext + frac_delta) & (uint32_t) low_bits_mask_u64(9u);
    const uint8_t shift_right = (uint8_t) ((total_shift >> 8) & 1u);
    uint32_t shift_mag = total_shift;
    if (shift_right != 0u) {
        shift_mag = ((~shift_mag) + 1u) & (uint32_t) low_bits_mask_u64(9u);
    }

    const uint64_t product = (uint64_t) significand * factor_q;
    uint64_t result = 0;
    if ((shift_mag & 0xffu) < 64u) {
        result = shift_right != 0u ? (product >> (shift_mag & 0xffu)) : (product << (shift_mag & 0xffu));
    }
    return result & low_bits_mask_u64(36u);
}

static __host__ __device__ __forceinline__ uint8_t block_scale_msb_hw(uint64_t block_scale_q) {
    uint64_t msb_probe = block_scale_q & low_bits_mask_u64(34u);
    uint8_t msb = 0u;
    if (msb_probe >= (1ull << 32)) { msb_probe >>= 32; msb = (uint8_t) ((msb + 32u) & 0x3fu); }
    if (msb_probe >= (1ull << 16)) { msb_probe >>= 16; msb = (uint8_t) ((msb + 16u) & 0x3fu); }
    if (msb_probe >= (1ull << 8))  { msb_probe >>= 8;  msb = (uint8_t) ((msb + 8u)  & 0x3fu); }
    if (msb_probe >= (1ull << 4))  { msb_probe >>= 4;  msb = (uint8_t) ((msb + 4u)  & 0x3fu); }
    if (msb_probe >= (1ull << 2))  { msb_probe >>= 2;  msb = (uint8_t) ((msb + 2u)  & 0x3fu); }
    if (msb_probe >= (1ull << 1))  { msb = (uint8_t) ((msb + 1u) & 0x3fu); }
    return (uint8_t) (msb & 0x3fu);
}

static __host__ __device__ __forceinline__ uint8_t compute_block_scale_hw(uint16_t block_abs_max_bits, uint32_t global_scale_q) {
    if (block_abs_max_bits == 0u) {
        return 0u;
    }

    uint64_t block_scale_q = bf16_abs_mul_uq_hw(block_abs_max_bits, global_scale_q, 16, 24);
    block_scale_q = ((block_scale_q + 3u) >> 3) + ((block_scale_q + 3u) >> 5) +
                    ((block_scale_q + 3u) >> 7) + ((block_scale_q + 3u) >> 9) +
                    ((block_scale_q + 3u) >> 11) + ((block_scale_q + 3u) >> 13);
    block_scale_q &= low_bits_mask_u64(34u);

    const uint8_t msb = block_scale_msb_hw(block_scale_q);
    const uint8_t exp_field_tc = (uint8_t) ((msb - 24 + 7) & 0x3fu);
    int32_t exp_field = ((uint32_t) exp_field_tc & 0x20u) != 0u ? (int32_t) ((uint32_t) exp_field_tc | ~0x3fu) : (int32_t) exp_field_tc;
    if (exp_field <= 0) {
        uint64_t mant_q = block_scale_q >> (24 - 9);
        mant_q &= 0xffu;
        return mant_q >= 8u ? 0x08u : (uint8_t) mant_q;
    }

    if (exp_field > 15) {
        exp_field = 15;
    }
    const int rshift = 24 + exp_field - 10;
    uint32_t signif_q_rounded = (uint32_t) (block_scale_q & low_bits_mask_u64(5u));
    if (rshift > 0) {
        const uint64_t shifted = (block_scale_q >> rshift) & low_bits_mask_u64(19u);
        const uint64_t half = (1ull << (rshift - 1)) & low_bits_mask_u64(29u);
        const uint64_t mask = low_bits_mask_u64((uint8_t) rshift) & low_bits_mask_u64(29u);
        const uint64_t remainder = block_scale_q & mask;
        const uint64_t round = remainder > half ? 1ull : 0ull;
        signif_q_rounded = (uint32_t) ((shifted + round) & low_bits_mask_u64(5u));
    }
    const uint8_t carry = ((exp_field < 15) && ((signif_q_rounded & (1u << 4)) != 0u)) ? 1u : 0u;
    const int32_t exp_field_norm = (exp_field + (int32_t) carry) & 0xf;
    const uint32_t signif_q_norm = carry != 0u ? 8u : signif_q_rounded;
    const uint32_t signif_q_floor = signif_q_norm < 8u ? 8u : signif_q_norm;
    const uint32_t signif_q_clamped = exp_field_norm >= 15 ? (signif_q_floor > 14u ? 14u : signif_q_floor) : signif_q_floor;
    return (uint8_t) ((exp_field_norm << 3) | ((signif_q_clamped - 8u) & 0x7u));
}

static __host__ __device__ __forceinline__ uint64_t compute_block_scale_half_q(uint8_t scale) {
    const uint32_t scale_exp = (scale >> 3) & 0xfu;
    const uint32_t scale_mant = scale & 0x7u;
    return scale_exp == 0u
            ? (shift_hw(scale_mant, (uint8_t) ((24 - 10) & 0xffu), 0u) & 0xffffffffu)
            : (shift_hw(8u + scale_mant, (uint8_t) ((24 + (int) scale_exp - 11) & 0xffu), 0u) & 0xffffffffu);
}

static __device__ __forceinline__ uint8_t bf16_quant_mag(
        uint16_t abs_bits,
        uint32_t global_scale_q,
        uint64_t block_scale_half_q) {
    const uint64_t target_q = bf16_abs_mul_uq_hw(abs_bits, global_scale_q, 16, 24) & low_bits_mask_u64(36u);
    const uint64_t target_2x_q = target_q << 1;
    if (target_2x_q < block_scale_half_q) {
        return 0u;
    }
    if (target_2x_q < 3ull * block_scale_half_q) {
        return 1u;
    }
    if (target_2x_q < 5ull * block_scale_half_q) {
        return 2u;
    }
    if (target_2x_q < 7ull * block_scale_half_q) {
        return 3u;
    }
    if (target_2x_q < 10ull * block_scale_half_q) {
        return 4u;
    }
    if (target_2x_q < 14ull * block_scale_half_q) {
        return 5u;
    }
    if (target_2x_q < 20ull * block_scale_half_q) {
        return 6u;
    }
    return 7u;
}

static __global__ void quantize_row_nvfp4_bf16_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ global_scales,
        const bool per_tensor_scale) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const float xi = (lane_active && k0 < ne00) ? x[(int64_t) i1 * s01 + k0] : 0.0f;
    const uint16_t bf16 = ggml_cuda_fp32_to_bf16_round_device(xi);
    uint16_t block_abs_max = bf16_abs_bits(bf16);
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 8, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 4, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 2, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 1, WARP_SIZE));
    block_abs_max = __shfl_sync(0xFFFFFFFF, block_abs_max, 0, WARP_SIZE);

    const float global_scale = per_tensor_scale ? global_scales[0] : global_scales[i1];
    const uint32_t global_scale_q = float_to_ufixed_q_hw(global_scale, 16);

    uint8_t scale = 0u;
    uint64_t block_scale_half_q = 0u;
    if (lane == 0) {
        scale = compute_block_scale_hw(block_abs_max, global_scale_q);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale;
        block_scale_half_q = compute_block_scale_half_q(scale);
    }
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);
    block_scale_half_q = __shfl_sync(0xFFFFFFFF, block_scale_half_q, 0, WARP_SIZE);

    uint8_t q = 0u;
    if (scale != 0u) {
        const uint8_t mag = bf16_quant_mag(bf16_abs_bits(bf16), global_scale_q, block_scale_half_q);
        q = mag == 0u ? 0u : (uint8_t) (((bf16 >> 15) & 1u) << 3 | mag);
    }
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static __global__ void quantize_row_nvfp4_bf16_scalar_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float global_scale) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const float xi = (lane_active && k0 < ne00) ? x[(int64_t) i1 * s01 + k0] : 0.0f;
    const uint16_t bf16 = ggml_cuda_fp32_to_bf16_round_device(xi);
    uint16_t block_abs_max = bf16_abs_bits(bf16);
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 8, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 4, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 2, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 1, WARP_SIZE));
    block_abs_max = __shfl_sync(0xFFFFFFFF, block_abs_max, 0, WARP_SIZE);

    const uint32_t global_scale_q = float_to_ufixed_q_hw(global_scale, 16);

    uint8_t scale = 0u;
    uint64_t block_scale_half_q = 0u;
    if (lane == 0) {
        scale = compute_block_scale_hw(block_abs_max, global_scale_q);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale;
        block_scale_half_q = compute_block_scale_half_q(scale);
    }
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);
    block_scale_half_q = __shfl_sync(0xFFFFFFFF, block_scale_half_q, 0, WARP_SIZE);

    uint8_t q = 0u;
    if (scale != 0u) {
        const uint8_t mag = bf16_quant_mag(bf16_abs_bits(bf16), global_scale_q, block_scale_half_q);
        q = mag == 0u ? 0u : (uint8_t) (((bf16 >> 15) & 1u) << 3 | mag);
    }
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static __global__ void quantize_row_nvfp4_dynamic_bf16_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ amax_rows,
        const bool per_tensor_scale) {
    const int lane = threadIdx.x;
    const int i1 = blockIdx.y;
    const float amax_f = per_tensor_scale ? amax_rows[0] : amax_rows[i1];
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_q_global_scale(amax_f);
    __shared__ float scale_shared;
    if (lane == 0) {
        scale_shared = global_scale;
    }
    __syncthreads();

    const int ib = blockIdx.x;
    const bool lane_active = lane < QK_NVFP4;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;
    const float xi = (lane_active && k0 < ne00) ? x[(int64_t) i1 * s01 + k0] : 0.0f;
    const uint16_t bf16 = ggml_cuda_fp32_to_bf16_round_device(xi);
    uint16_t block_abs_max = bf16_abs_bits(bf16);
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 8, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 4, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 2, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 1, WARP_SIZE));
    block_abs_max = __shfl_sync(0xFFFFFFFF, block_abs_max, 0, WARP_SIZE);

    const uint32_t global_scale_q = float_to_ufixed_q_hw(scale_shared, 16);

    uint8_t scale = 0u;
    uint64_t block_scale_half_q = 0u;
    if (lane == 0) {
        scale = compute_block_scale_hw(block_abs_max, global_scale_q);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale;
        block_scale_half_q = compute_block_scale_half_q(scale);
    }
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);
    block_scale_half_q = __shfl_sync(0xFFFFFFFF, block_scale_half_q, 0, WARP_SIZE);

    uint8_t q = 0u;
    if (scale != 0u) {
        const uint8_t mag = bf16_quant_mag(bf16_abs_bits(bf16), global_scale_q, block_scale_half_q);
        q = mag == 0u ? 0u : (uint8_t) (((bf16 >> 15) & 1u) << 3 | mag);
    }
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].qs[lane/2] = q | (q_peer << 4);
    }
}

} // namespace

bool ggml_cuda_nvfp4_bf16_quant_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_BF16_QUANT");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_bf16_quant_once(env, cached != 0);
    }
    return cached != 0;
}

void ggml_cuda_nvfp4_abs_max_rows_f32(
        const float * src,
        float * amax_rows,
        int64_t ne00,
        int64_t ne01,
        int64_t s01,
        cudaStream_t stream) {
    ggml_cuda_nvfp4_abs_max_rows_f32_kernel<<<(int) ne01, 256, 0, stream>>>(src, amax_rows, ne00, ne01, s01);
}

void ggml_cuda_nvfp4_abs_max_tensor_f32(
        const float * src,
        float * amax,
        int64_t ne00,
        int64_t ne01,
        int64_t s01,
        cudaStream_t stream) {
    const int block_size = 256;
    const int64_t total = ne00 * ne01;
    const int64_t blocks = (total + block_size - 1) / block_size;
    const int grid_size = (int) (blocks < 1024 ? blocks : 1024);
    ggml_cuda_nvfp4_abs_max_tensor_f32_kernel<<<grid_size, block_size, 0, stream>>>(src, amax, ne00, ne01, s01);
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
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01, global_scale);
}

void ggml_cuda_nvfp4_quantize_rows_dynamic_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * amax_rows,
        bool per_tensor_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_dynamic_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01, amax_rows, per_tensor_scale);
}

void ggml_cuda_nvfp4_quantize_rows_bf16_f32(
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
    quantize_row_nvfp4_bf16_scalar_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01, global_scale);
}

void ggml_cuda_nvfp4_quantize_rows_bf16_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * global_scales,
        bool per_tensor_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01, global_scales, per_tensor_scale);
}

void ggml_cuda_nvfp4_quantize_rows_dynamic_bf16_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * amax_rows,
        bool per_tensor_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_dynamic_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01, amax_rows, per_tensor_scale);
}
