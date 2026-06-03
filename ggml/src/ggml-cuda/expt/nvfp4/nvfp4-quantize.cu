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

static __host__ __device__ __forceinline__ uint32_t f32_to_u32_bits_hw(float v) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = v;
    return bits.u;
}

static __host__ __device__ __forceinline__ float u32_to_f32_bits_hw(uint32_t v) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = v;
    return bits.f;
}

static __host__ __device__ __forceinline__ float bf16_abs_to_fp32_bits_hw(uint16_t abs_bits) {
    return u32_to_f32_bits_hw((uint32_t) abs_bits << 16);
}

static __host__ __device__ __forceinline__ uint16_t fp32_to_bf16_trunc_bits_hw(float v) {
    return (uint16_t) (f32_to_u32_bits_hw(v) >> 16);
}

static __host__ __device__ __forceinline__ uint16_t bf16_mul_bf16_rne_bits_hw(uint16_t a, uint16_t b) {
    const uint32_t a_sign = (a >> 15) & 1u;
    const uint32_t b_sign = (b >> 15) & 1u;
    const uint32_t a_exp = (a >> 7) & 0xffu;
    const uint32_t b_exp = (b >> 7) & 0xffu;
    const uint32_t a_mant = a & 0x7fu;
    const uint32_t b_mant = b & 0x7fu;

    const bool a_zero = (a & 0x7fffu) == 0u;
    const bool b_zero = (b & 0x7fffu) == 0u;
    const bool a_inf = (a & 0x7fffu) == 0x7f80u;
    const bool b_inf = (b & 0x7fffu) == 0x7f80u;
    const bool a_nan = a_exp == 0xffu && a_mant != 0u;
    const bool b_nan = b_exp == 0xffu && b_mant != 0u;

    const uint32_t sign_out = a_sign ^ b_sign;
    const bool is_nan = a_nan || b_nan || (a_inf && b_zero) || (a_zero && b_inf);
    const bool is_inf = (a_inf || b_inf) && !is_nan;
    const bool is_zero = (a_zero || b_zero) && !is_inf;

    const uint32_t ma = 0x80u | a_mant;
    const uint32_t mb = 0x80u | b_mant;
    const uint32_t product = ma * mb;
    const int32_t exp_sum = (int32_t) a_exp + (int32_t) b_exp - 127;

    const bool product_overflow = (product & 0x8000u) != 0u;
    const int32_t exp_norm = product_overflow ? exp_sum + 1 : exp_sum;
    const uint32_t mant_pre = product_overflow ? ((product >> 8) & 0x7fu) : ((product >> 7) & 0x7fu);
    const bool guard = product_overflow ? (((product >> 7) & 1u) != 0u) : (((product >> 6) & 1u) != 0u);
    const bool sticky = product_overflow ? ((product & 0x7fu) != 0u) : ((product & 0x3fu) != 0u);
    const bool round_up = guard && (sticky || ((mant_pre & 1u) != 0u));
    const uint32_t mant_rnd = mant_pre + (round_up ? 1u : 0u);

    const int32_t exp_rnd = (mant_rnd & 0x80u) ? exp_norm + 1 : exp_norm;
    const uint32_t mant_final = (mant_rnd & 0x80u) ? 0u : (mant_rnd & 0x7fu);

    const uint16_t normal_result = (uint16_t) ((sign_out << 15) | (((uint32_t) exp_rnd & 0xffu) << 7) | mant_final);
    uint16_t result = normal_result;
    result = exp_rnd < 0 ? (uint16_t) (sign_out << 15) : result;
    result = exp_rnd >= 255 ? (uint16_t) ((sign_out << 15) | 0x7f80u) : result;
    result = is_zero ? (uint16_t) (sign_out << 15) : result;
    result = is_inf ? (uint16_t) ((sign_out << 15) | 0x7f80u) : result;
    result = is_nan ? 0x7fc0u : result;
    return result;
}

static __host__ __device__ __forceinline__ uint16_t bf16_mul_bf16_rne_operands_bits_hw(uint16_t a, float b) {
    const uint16_t a_bf16 = bf16_abs_bits(a);
    const uint16_t b_bf16 = fp32_to_bf16_trunc_bits_hw(b);
    return bf16_mul_bf16_rne_bits_hw(a_bf16, b_bf16);
}

static __host__ __device__ __forceinline__ uint32_t clz_u32_hw(uint32_t value) {
    if (value == 0u) {
        return 32u;
    }

    uint32_t count = 0u;
    if ((value & 0xffff0000u) == 0u) {
        count += 16u;
        value <<= 16;
    }
    if ((value & 0xff000000u) == 0u) {
        count += 8u;
        value <<= 8;
    }
    if ((value & 0xf0000000u) == 0u) {
        count += 4u;
        value <<= 4;
    }
    if ((value & 0xc0000000u) == 0u) {
        count += 2u;
        value <<= 2;
    }
    if ((value & 0x80000000u) == 0u) {
        count += 1u;
    }
    return count;
}

static __host__ __device__ __forceinline__ uint16_t bf16_add_pos_trunc_bits_hw(uint16_t a, uint16_t b) {
    a = bf16_abs_bits(a);
    b = bf16_abs_bits(b);

    uint32_t exp_a = (a >> 7) & 0xffu;
    uint32_t exp_b = (b >> 7) & 0xffu;
    const bool a_zero_in = (a & 0x7fffu) == 0u;
    const bool b_zero_in = (b & 0x7fffu) == 0u;
    const bool a_subnormal_in = exp_a == 0u && !a_zero_in;
    const bool b_subnormal_in = exp_b == 0u && !b_zero_in;
    const bool any_non_finite_in = exp_a == 0xffu || exp_b == 0xffu;
    uint32_t sig_a = 0x80u | (a & 0x7fu);
    uint32_t sig_b = 0x80u | (b & 0x7fu);

    if (exp_a < exp_b || (exp_a == exp_b && sig_a < sig_b)) {
        const uint32_t tmp_exp = exp_a;
        const uint32_t tmp_sig = sig_a;
        exp_a = exp_b;
        sig_a = sig_b;
        exp_b = tmp_exp;
        sig_b = tmp_sig;
    }

    constexpr uint32_t guard_bits = 16u;
    const uint32_t shift = exp_a - exp_b;
    const uint32_t sig_a_ext = sig_a << guard_bits;
    const uint32_t sig_b_ext = (shift >= 24u) ? 0u : ((sig_b << guard_bits) >> shift);
    const uint32_t sum_sig = sig_a_ext + sig_b_ext;

    const bool sum_zero = sum_sig == 0u;
    const uint32_t sum_sig_safe = sum_zero ? 1u : sum_sig;
    const int32_t msb_pos = 31 - (int32_t) clz_u32_hw(sum_sig_safe);
    constexpr int32_t target_msb_pos = 7 + (int32_t) guard_bits;
    const int32_t shift_amt = msb_pos - target_msb_pos;
    const int32_t res_exp = (int32_t) exp_a + shift_amt;
    const uint32_t final_sig = (shift_amt < 0) ? (sum_sig_safe << (uint32_t) (-shift_amt)) : (sum_sig_safe >> (uint32_t) shift_amt);
    const uint32_t final_mant = (final_sig >> guard_bits) & 0x7fu;

    const bool overflow = res_exp >= 255;
    const bool underflow = res_exp <= 0;

    const uint16_t result_normal = (uint16_t) (((uint32_t) res_exp << 7) | final_mant);
    if (b_zero_in) {
        return a;
    }
    if (a_zero_in) {
        return b;
    }
    if (a_subnormal_in) {
        return b;
    }
    if (b_subnormal_in) {
        return a;
    }
    if (any_non_finite_in) {
        return 0x7f80u;
    }
    if (underflow || sum_zero) {
        return 0u;
    }
    if (overflow) {
        return 0x7f80u;
    }
    return result_normal;
}

static __host__ __device__ __forceinline__ bool bf16_pos_le_hw(uint16_t a, uint16_t b) {
    return bf16_abs_bits(a) <= bf16_abs_bits(b);
}

static __host__ __device__ __forceinline__ uint16_t bf16_pos_mul2_bits_hw(uint16_t value) {
    value = bf16_abs_bits(value);
    const uint32_t exp = (value >> 7) & 0xffu;
    const uint32_t mant = value & 0x7fu;

    if ((value & 0x7fffu) == 0u) {
        return 0u;
    }
    if (exp == 0xffu) {
        return value;
    }
    if (exp == 0u) {
        return (mant & 0x40u) != 0u ? (uint16_t) ((1u << 7) | ((mant & 0x3fu) << 1)) : (uint16_t) ((mant & 0x3fu) << 1);
    }
    if (exp == 0xfeu) {
        return 0x7f80u;
    }
    return (uint16_t) (((exp + 1u) << 7) | mant);
}

static __host__ __device__ __forceinline__ uint32_t round_right_ties_down(uint32_t value, int shift) {
    constexpr int max_left_shift = 8;
    constexpr int max_right_shift = 24;
    if (shift <= 0) {
        const int lshift = -shift;
        return (lshift > max_left_shift) ? 0xffffffffu : (value << lshift);
    }
    if (shift > max_right_shift) {
        return 0u;
    }

    const uint32_t shifted = value >> shift;
    const uint32_t half = 1u << (shift - 1);
    const uint32_t mask = (1u << shift) - 1u;
    const uint32_t remainder = value & mask;
    return shifted + ((remainder > half) ? 1u : 0u);
}

static __host__ __device__ __forceinline__ uint8_t e4m3_subnormal_from_fp32_bits_hw(uint32_t exponent, uint32_t mantissa) {
    const bool fp32_subnormal = exponent == 0u;
    const int32_t exp_unbiased = fp32_subnormal ? -126 : (int32_t) exponent - 127;
    const uint32_t significand = fp32_subnormal ? mantissa : (0x00800000u | mantissa);
    const uint32_t mant_q = round_right_ties_down(significand, 14 - exp_unbiased);
    return (uint8_t) (mant_q > 15u ? 15u : mant_q);
}

static __host__ __device__ __forceinline__ uint8_t e4m3_scale_from_fp32_bits_hw(float scale) {
    const uint32_t bits = f32_to_u32_bits_hw(scale);
    const uint32_t sign = bits >> 31;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x007fffffu;
    if (sign != 0u || exponent == 0xffu) {
        return 0u;
    }

    if (scale <= 0.0302734375f) {
        return e4m3_subnormal_from_fp32_bits_hw(exponent, mantissa);
    }

    const int32_t exp_unbiased = (int32_t) exponent - 127;
    int32_t exp_field = exp_unbiased + 7;
    const uint32_t significand = 0x00800000u | mantissa;

    uint32_t signif_q = round_right_ties_down(significand, 20);
    if (signif_q >= 16u) {
        signif_q = 8u;
        ++exp_field;
    }

    if (exp_field > 15 || (exp_field == 15 && signif_q >= 15u)) {
        return 0x7Eu;
    }

    return (uint8_t) (((uint32_t) exp_field << 3) | ((signif_q - 8u) & 0x7u));
}

static __host__ __device__ __forceinline__ float compute_block_scale_value_bf16_internal_fp32_blockscale_hw(
        uint16_t block_abs_max_bits,
        float global_scale) {
    if (block_abs_max_bits == 0u) {
        return 0.0f;
    }

    const float block_abs_max = bf16_abs_to_fp32_bits_hw(block_abs_max_bits);
    return block_abs_max * global_scale * 0.1666666716f;
}

static __host__ __device__ __forceinline__ uint16_t fp32_scale_half_to_bf16_bits_hw(float scale) {
    const uint32_t bits = f32_to_u32_bits_hw(scale);
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abs_bits = bits & 0x7fffffffu;
    uint32_t half_abs_bits = 0u;

    if (abs_bits >= 0x01000000u && abs_bits < 0x7f800000u) {
        half_abs_bits = abs_bits - 0x00800000u;
    } else if (abs_bits >= 0x00800000u && abs_bits < 0x01000000u) {
        half_abs_bits = ((abs_bits & 0x007fffffu) | 0x00800000u) >> 1;
    } else if (abs_bits < 0x00800000u) {
        half_abs_bits = abs_bits >> 1;
    } else {
        half_abs_bits = abs_bits;
    }

    return (uint16_t) ((sign | half_abs_bits) >> 16);
}

static __device__ __forceinline__ uint8_t quant_mag_bf16_internal_hw(
        uint16_t abs_bits,
        float global_scale,
        uint16_t block_scale_half_bits) {
    const uint16_t target = bf16_mul_bf16_rne_operands_bits_hw(abs_bits, global_scale);
    const uint16_t target_2x = bf16_pos_mul2_bits_hw(target);
    const uint16_t scale_2x = bf16_pos_mul2_bits_hw(block_scale_half_bits);
    const uint16_t scale_3x = bf16_add_pos_trunc_bits_hw(scale_2x, block_scale_half_bits);
    const uint16_t scale_5x = bf16_add_pos_trunc_bits_hw(scale_3x, scale_2x);
    const uint16_t scale_7x = bf16_add_pos_trunc_bits_hw(scale_5x, scale_2x);
    const uint16_t scale_10x = bf16_pos_mul2_bits_hw(scale_5x);
    const uint16_t scale_14x = bf16_pos_mul2_bits_hw(scale_7x);
    const uint16_t scale_20x = bf16_pos_mul2_bits_hw(scale_10x);

    if (bf16_pos_le_hw(target_2x, block_scale_half_bits)) {
        return 0u;
    }
    if (bf16_pos_le_hw(target_2x, scale_3x)) {
        return 1u;
    }
    if (bf16_pos_le_hw(target_2x, scale_5x)) {
        return 2u;
    }
    if (bf16_pos_le_hw(target_2x, scale_7x)) {
        return 3u;
    }
    if (bf16_pos_le_hw(target_2x, scale_10x)) {
        return 4u;
    }
    if (bf16_pos_le_hw(target_2x, scale_14x)) {
        return 5u;
    }
    if (bf16_pos_le_hw(target_2x, scale_20x)) {
        return 6u;
    }
    return 7u;
}

static __device__ __forceinline__ void quantize_row_nvfp4_bf16_block(
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

    uint8_t scale = 0u;
    uint16_t block_scale_half_bits = 0u;
    if (lane == 0) {
        const float block_scale = compute_block_scale_value_bf16_internal_fp32_blockscale_hw(block_abs_max, global_scale);
        scale = e4m3_scale_from_fp32_bits_hw(block_scale);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale;
        block_scale_half_bits = fp32_scale_half_to_bf16_bits_hw(block_scale);
    }
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);
    block_scale_half_bits = __shfl_sync(0xFFFFFFFF, block_scale_half_bits, 0, WARP_SIZE);

    uint8_t q = 0u;
    if (scale != 0u) {
        const uint8_t mag = quant_mag_bf16_internal_hw(bf16_abs_bits(bf16), global_scale, block_scale_half_bits);
        q = mag == 0u ? 0u : (uint8_t) (((bf16 >> 15) & 1u) << 3 | mag);
    }
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static __global__ void quantize_row_nvfp4_bf16_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ global_scales,
        const bool per_tensor_scale) {
    const int i1 = blockIdx.y;
    const float global_scale = per_tensor_scale ? global_scales[0] : global_scales[i1];
    quantize_row_nvfp4_bf16_block(x, y, ne00, s01, global_scale);
}

static __global__ void quantize_row_nvfp4_dynamic_bf16_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ amax_rows,
        const bool per_tensor_scale) {
    const int i1 = blockIdx.y;
    const float amax_f = per_tensor_scale ? amax_rows[0] : amax_rows[i1];
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_q_global_scale(amax_f);
    quantize_row_nvfp4_bf16_block(x, y, ne00, s01, global_scale);
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
