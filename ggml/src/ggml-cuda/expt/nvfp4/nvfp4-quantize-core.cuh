#pragma once

#include "nvfp4-common.cuh"

#include <cstdint>

static __device__ __forceinline__ float ggml_cuda_nvfp4_core_trunc_f32_to_bf16_value(float v) {
    union {
        float    f;
        uint32_t u;
    } bits;
    bits.f = v;
    bits.u &= 0xffff0000u;
    return bits.f;
}

static __host__ __device__ __forceinline__ uint32_t ggml_cuda_nvfp4_core_f32_to_u32(float v) {
    union {
        float    f;
        uint32_t u;
    } bits;
    bits.f = v;
    return bits.u;
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_core_u32_to_f32(uint32_t v) {
    union {
        uint32_t u;
        float    f;
    } bits;
    bits.u = v;
    return bits.f;
}

static __host__ __device__ __forceinline__ uint16_t ggml_cuda_nvfp4_core_bf16_abs_bits(uint16_t x) {
    return (uint16_t) (x & 0x7fffu);
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_core_bf16_abs_to_f32(uint16_t abs_bits) {
    return ggml_cuda_nvfp4_core_u32_to_f32((uint32_t) abs_bits << 16);
}

static __device__ __forceinline__ uint16_t ggml_cuda_nvfp4_core_fp32_to_bf16_trunc_bits(float v) {
    return (uint16_t) (ggml_cuda_nvfp4_core_f32_to_u32(v) >> 16);
}

static __host__ __device__ __forceinline__ uint32_t ggml_cuda_nvfp4_core_round_shift_right_ties_down_u32(uint32_t value, int shift) {
    constexpr int kMaxLeftShift = 8;
    constexpr int kMaxRightShift = 24;
    if (shift <= 0) {
        const int lshift = -shift;
        return lshift > kMaxLeftShift ? 0xffffffffu : (value << lshift);
    }
    if (shift > kMaxRightShift) {
        return 0u;
    }

    const uint32_t shifted = value >> shift;
    const uint32_t half = 1u << (shift - 1);
    const uint32_t mask = (1u << shift) - 1u;
    const uint32_t remainder = value & mask;
    return shifted + (remainder > half ? 1u : 0u);
}

static __host__ __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_core_e4m3_subnormal_from_fp32_bits(
        uint32_t exponent,
        uint32_t mantissa) {
    const bool fp32_subnormal = exponent == 0u;
    const int32_t exp_unbiased = fp32_subnormal ? -126 : (int32_t) exponent - 127;
    const uint32_t significand = fp32_subnormal ? mantissa : (0x00800000u | mantissa);
    const uint32_t mant_q = ggml_cuda_nvfp4_core_round_shift_right_ties_down_u32(significand, 14 - exp_unbiased);
    return (uint8_t) (mant_q > 15u ? 15u : mant_q);
}

static __host__ __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_core_e4m3_scale_from_fp32(float scale) {
    const uint32_t bits = ggml_cuda_nvfp4_core_f32_to_u32(scale);
    const uint32_t sign = bits >> 31;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x007fffffu;
    if (sign != 0u || exponent == 0xffu) {
        return 0u;
    }

    if (scale <= 0.0302734375f) {
        return ggml_cuda_nvfp4_core_e4m3_subnormal_from_fp32_bits(exponent, mantissa);
    }

    const int32_t exp_unbiased = (int32_t) exponent - 127;
    int32_t exp_field = exp_unbiased + 7;
    const uint32_t significand = 0x00800000u | mantissa;

    uint32_t signif_q = ggml_cuda_nvfp4_core_round_shift_right_ties_down_u32(significand, 20);
    if (signif_q >= 16u) {
        signif_q = 8u;
        ++exp_field;
    }

    if (exp_field > 15 || (exp_field == 15 && signif_q >= 15u)) {
        return 0x7eu;
    }

    return (uint8_t) (((uint32_t) exp_field << 3) | ((signif_q - 8u) & 0x7u));
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_core_e4m3_scale_half_to_fp32(uint8_t scale) {
    const uint32_t scale_abs = scale & 0x7fu;
    const uint32_t scale_exp = (scale_abs >> 3) & 0xfu;
    const uint32_t scale_mant = scale_abs & 0x7u;
    if (scale_exp == 0u) {
        if (scale_mant == 0u) {
            return 0.0f;
        }
        const uint32_t shift = scale_mant >= 4u ? 0u : (scale_mant >= 2u ? 1u : 2u);
        const uint32_t mant_norm = scale_mant << shift;
        const uint32_t exp_bits = (119u - shift) << 23;
        const uint32_t mant_bits = (mant_norm & 0x3u) << 21;
        return ggml_cuda_nvfp4_core_u32_to_f32(exp_bits | mant_bits);
    }
    const uint32_t scale_mant_clamped = scale_exp == 0x0fu && scale_mant == 7u ? 6u : scale_mant;
    return ggml_cuda_nvfp4_core_u32_to_f32(((scale_exp + 119u) << 23) | (scale_mant_clamped << 20));
}

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_core_best_index_e4m3(float x) {
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

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_core_best_index_e2m1(float x) {
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

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_core_quantize_value_f32(
        float x,
        float global_scale,
        float block_scale_half) {
    const float inv_scale = (global_scale != 0.0f && block_scale_half != 0.0f) ? (global_scale / block_scale_half) : 0.0f;
    return ggml_cuda_nvfp4_core_best_index_e2m1(x * inv_scale);
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_core_quantize_block_f32(
        float xi,
        bool lane_active,
        float global_scale,
        block_nvfp4 * out) {
    const int lane = threadIdx.x;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    float block_scale_half = 0.0f;
    if (lane == 0) {
        const float scale = global_scale * (vmax / GGML_CUDA_NVFP4_FP4_MAX);
        const uint8_t scale_q = ggml_cuda_nvfp4_core_best_index_e4m3(scale);
        out->e = scale_q;
        block_scale_half = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    block_scale_half = __shfl_sync(0xFFFFFFFF, block_scale_half, 0, WARP_SIZE);

    const uint8_t q = ggml_cuda_nvfp4_core_quantize_value_f32(xi, global_scale, block_scale_half);
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);
    if (lane_active && (lane & 1) == 0) {
        out->qs[lane / 2] = q | (q_peer << 4);
    }
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_core_trunc_f32_to_bf16_value_hostdev(float v) {
    return ggml_cuda_nvfp4_core_u32_to_f32(ggml_cuda_nvfp4_core_f32_to_u32(v) & 0xffff0000u);
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_core_bf16_mul_trunc(float a, float b) {
    return ggml_cuda_nvfp4_core_trunc_f32_to_bf16_value_hostdev(a * b);
}

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_core_compute_bf16_block_scale(
        uint16_t block_abs_max_bits,
        float global_scale,
        bool bf16_block_scale) {
    if (block_abs_max_bits == 0u) {
        return 0.0f;
    }

    const float block_abs_max = ggml_cuda_nvfp4_core_bf16_abs_to_f32(block_abs_max_bits);
    return bf16_block_scale
            ? ggml_cuda_nvfp4_core_bf16_mul_trunc(
                    ggml_cuda_nvfp4_core_bf16_mul_trunc(
                            block_abs_max,
                            ggml_cuda_nvfp4_core_trunc_f32_to_bf16_value_hostdev(global_scale)),
                    ggml_cuda_nvfp4_core_trunc_f32_to_bf16_value_hostdev(0.1666666716f))
            : block_abs_max * global_scale * 0.1666666716f;
}

static __device__ __forceinline__ uint32_t ggml_cuda_nvfp4_core_clz_u32(uint32_t value) {
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

static __device__ __forceinline__ uint16_t ggml_cuda_nvfp4_core_bf16_mul_bf16_rne_bits(uint16_t a, uint16_t b) {
    const uint32_t a_sign = (a >> 15) & 1u;
    const uint32_t b_sign = (b >> 15) & 1u;
    const uint32_t a_exp  = (a >> 7) & 0xffu;
    const uint32_t b_exp  = (b >> 7) & 0xffu;
    const uint32_t a_mant = a & 0x7fu;
    const uint32_t b_mant = b & 0x7fu;

    const bool a_zero = (a & 0x7fffu) == 0u;
    const bool b_zero = (b & 0x7fffu) == 0u;
    const bool a_inf  = (a & 0x7fffu) == 0x7f80u;
    const bool b_inf  = (b & 0x7fffu) == 0x7f80u;
    const bool a_nan  = a_exp == 0xffu && a_mant != 0u;
    const bool b_nan  = b_exp == 0xffu && b_mant != 0u;

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

    const uint16_t normal_result =
        (uint16_t) ((sign_out << 15) | (((uint32_t) exp_rnd & 0xffu) << 7) | mant_final);
    uint16_t result = normal_result;
    result = exp_rnd < 0 ? (uint16_t) (sign_out << 15) : result;
    result = exp_rnd >= 255 ? (uint16_t) ((sign_out << 15) | 0x7f80u) : result;
    result = is_zero ? (uint16_t) (sign_out << 15) : result;
    result = is_inf ? (uint16_t) ((sign_out << 15) | 0x7f80u) : result;
    result = is_nan ? 0x7fc0u : result;
    return result;
}

static __device__ __forceinline__ uint16_t ggml_cuda_nvfp4_core_bf16_mul_operands_rne_bits(uint16_t a, float b) {
    const uint16_t a_bf16 = ggml_cuda_nvfp4_core_bf16_abs_bits(a);
    const uint16_t b_bf16 = ggml_cuda_nvfp4_core_fp32_to_bf16_trunc_bits(b);
    return ggml_cuda_nvfp4_core_bf16_mul_bf16_rne_bits(a_bf16, b_bf16);
}

static __device__ __forceinline__ bool ggml_cuda_nvfp4_core_bf16_pos_le(uint16_t a, uint16_t b) {
    return ggml_cuda_nvfp4_core_bf16_abs_bits(a) <= ggml_cuda_nvfp4_core_bf16_abs_bits(b);
}

static __device__ __forceinline__ uint16_t ggml_cuda_nvfp4_core_bf16_pos_mul2_bits(uint16_t value) {
    value = ggml_cuda_nvfp4_core_bf16_abs_bits(value);
    const uint32_t exp = (value >> 7) & 0xffu;
    const uint32_t mant = value & 0x7fu;

    if ((value & 0x7fffu) == 0u) {
        return 0u;
    }
    if (exp == 0xffu) {
        return value;
    }
    if (exp == 0u) {
        return (mant & 0x40u) != 0u
                ? (uint16_t) ((1u << 7) | ((mant & 0x3fu) << 1))
                : (uint16_t) ((mant & 0x3fu) << 1);
    }
    if (exp == 0xfeu) {
        return 0x7f80u;
    }
    return (uint16_t) (((exp + 1u) << 7) | mant);
}

static __device__ __forceinline__ uint16_t ggml_cuda_nvfp4_core_bf16_add_pos_trunc_bits(uint16_t a, uint16_t b) {
    a = ggml_cuda_nvfp4_core_bf16_abs_bits(a);
    b = ggml_cuda_nvfp4_core_bf16_abs_bits(b);

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

    constexpr uint32_t kGuardBits = 16u;
    const uint32_t shift = exp_a - exp_b;
    const uint32_t sig_a_ext = sig_a << kGuardBits;
    const uint32_t sig_b_ext = (shift >= 24u) ? 0u : ((sig_b << kGuardBits) >> shift);
    const uint32_t sum_sig = sig_a_ext + sig_b_ext;

    const bool sum_zero = sum_sig == 0u;
    const uint32_t sum_sig_safe = sum_zero ? 1u : sum_sig;
    const int32_t msb_pos = 31 - (int32_t) ggml_cuda_nvfp4_core_clz_u32(sum_sig_safe);
    constexpr int32_t kTargetMsbPos = 7 + (int32_t) kGuardBits;
    const int32_t shift_amt = msb_pos - kTargetMsbPos;
    const int32_t res_exp = (int32_t) exp_a + shift_amt;
    const uint32_t final_sig = (shift_amt < 0) ? (sum_sig_safe << (uint32_t) -shift_amt) : (sum_sig_safe >> (uint32_t) shift_amt);
    const uint32_t final_mant = (final_sig >> kGuardBits) & 0x7fu;

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

static __device__ __forceinline__ uint16_t ggml_cuda_nvfp4_core_fp32_scale_half_to_bf16_bits(float scale) {
    const uint32_t bits = ggml_cuda_nvfp4_core_f32_to_u32(scale);
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

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_core_bf16_quant_mag_trunc_nn(
        uint16_t abs_bits,
        float global_scale,
        float block_scale_half,
        uint16_t block_scale_half_bits,
        bool bf16_internal_arith) {
    const float abs_f = ggml_cuda_nvfp4_core_bf16_abs_to_f32(abs_bits);
    if (bf16_internal_arith) {
        const uint16_t target = ggml_cuda_nvfp4_core_bf16_mul_operands_rne_bits(abs_bits, global_scale);
        const uint16_t target_2x = ggml_cuda_nvfp4_core_bf16_pos_mul2_bits(target);
        const uint16_t scale_2x = ggml_cuda_nvfp4_core_bf16_pos_mul2_bits(block_scale_half_bits);
        const uint16_t scale_3x = ggml_cuda_nvfp4_core_bf16_add_pos_trunc_bits(scale_2x, block_scale_half_bits);
        const uint16_t scale_5x = ggml_cuda_nvfp4_core_bf16_add_pos_trunc_bits(scale_3x, scale_2x);
        const uint16_t scale_7x = ggml_cuda_nvfp4_core_bf16_add_pos_trunc_bits(scale_5x, scale_2x);
        const uint16_t scale_10x = ggml_cuda_nvfp4_core_bf16_pos_mul2_bits(scale_5x);
        const uint16_t scale_14x = ggml_cuda_nvfp4_core_bf16_pos_mul2_bits(scale_7x);
        const uint16_t scale_20x = ggml_cuda_nvfp4_core_bf16_pos_mul2_bits(scale_10x);
        if (ggml_cuda_nvfp4_core_bf16_pos_le(target_2x, block_scale_half_bits)) {
            return 0u;
        }
        if (ggml_cuda_nvfp4_core_bf16_pos_le(target_2x, scale_3x)) {
            return 1u;
        }
        if (ggml_cuda_nvfp4_core_bf16_pos_le(target_2x, scale_5x)) {
            return 2u;
        }
        if (ggml_cuda_nvfp4_core_bf16_pos_le(target_2x, scale_7x)) {
            return 3u;
        }
        if (ggml_cuda_nvfp4_core_bf16_pos_le(target_2x, scale_10x)) {
            return 4u;
        }
        if (ggml_cuda_nvfp4_core_bf16_pos_le(target_2x, scale_14x)) {
            return 5u;
        }
        if (ggml_cuda_nvfp4_core_bf16_pos_le(target_2x, scale_20x)) {
            return 6u;
        }
        return 7u;
    }

    const float target = abs_f * global_scale;
    const float target_2x = target + target;
    const float scale_2x = block_scale_half + block_scale_half;
    const float scale_3x = scale_2x + block_scale_half;
    const float scale_5x = scale_3x + scale_2x;
    const float scale_7x = scale_5x + scale_2x;
    const float scale_10x = scale_5x + scale_5x;
    const float scale_14x = scale_7x + scale_7x;
    const float scale_20x = scale_10x + scale_10x;
    if (target_2x <= block_scale_half) {
        return 0u;
    }
    if (target_2x <= scale_3x) {
        return 1u;
    }
    if (target_2x <= scale_5x) {
        return 2u;
    }
    if (target_2x <= scale_7x) {
        return 3u;
    }
    if (target_2x <= scale_10x) {
        return 4u;
    }
    if (target_2x <= scale_14x) {
        return 5u;
    }
    if (target_2x <= scale_20x) {
        return 6u;
    }
    return 7u;
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_core_quantize_block_bf16_trunc_nn(
        float xi,
        bool lane_active,
        float global_scale,
        bool bf16_internal_arith,
        bool bf16_block_scale,
        block_nvfp4 * out) {
    const int lane = threadIdx.x;

    const uint16_t bf16 = ggml_cuda_nvfp4_core_fp32_to_bf16_trunc_bits(xi);
    const uint16_t bf16_abs = ggml_cuda_nvfp4_core_bf16_abs_bits(bf16);
    uint16_t block_abs_max = bf16_abs;
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 8, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 4, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 2, WARP_SIZE));
    block_abs_max = max(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 1, WARP_SIZE));
    block_abs_max = __shfl_sync(0xFFFFFFFF, block_abs_max, 0, WARP_SIZE);

    uint8_t scale = 0u;
    float block_scale_half_f = 0.0f;
    uint16_t block_scale_half_bits = 0u;
    if (lane == 0) {
        const float block_scale = ggml_cuda_nvfp4_core_compute_bf16_block_scale(
                block_abs_max, global_scale, bf16_block_scale);
        scale = ggml_cuda_nvfp4_core_e4m3_scale_from_fp32(block_scale);
        block_scale_half_f = ggml_cuda_nvfp4_core_e4m3_scale_half_to_fp32(scale);
        block_scale_half_bits = ggml_cuda_nvfp4_core_fp32_scale_half_to_bf16_bits(block_scale);
        out->e = scale;
    }
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);
    block_scale_half_f = __shfl_sync(0xFFFFFFFF, block_scale_half_f, 0, WARP_SIZE);
    block_scale_half_bits = __shfl_sync(0xFFFFFFFF, block_scale_half_bits, 0, WARP_SIZE);

    uint8_t q = 0u;
    if (scale != 0u) {
        const uint8_t mag = ggml_cuda_nvfp4_core_bf16_quant_mag_trunc_nn(
                bf16_abs, global_scale, block_scale_half_f, block_scale_half_bits, bf16_internal_arith);
        q = mag == 0u ? 0u : (uint8_t) (((bf16 >> 15) & 1u) << 3 | mag);
    }
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        out->qs[lane / 2] = q | (q_peer << 4);
    }
}
