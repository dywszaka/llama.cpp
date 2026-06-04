#include "nvfp4-quantize.cuh"

#include "../../common.cuh"
#include "kcache-outlier.cuh"
#include "nvfp4-common.cuh"
#include "nvfp4-log.cuh"

#include <cstdlib>

namespace {

static __host__ __device__ __forceinline__ uint64_t low_bits_mask_u64(uint8_t width) {
    if (width >= 64u) {
        return ~0ull;
    }
    return width == 0u ? 0ull : ((1ull << width) - 1ull);
}

static __host__ __device__ __forceinline__ uint16_t bf16_abs_bits(uint16_t x) {
    return (uint16_t) (x & (uint16_t) low_bits_mask_u64(15u));
}

static __host__ __device__ __forceinline__ uint16_t max_u16_select_hw(uint16_t a, uint16_t b) {
    return a > b ? a : b;
}

static __host__ __device__ __forceinline__ uint32_t f32_to_u32_bits_hw(float v) {
    union {
        float    f;
        uint32_t u;
    } bits;
    bits.f = v;
    return bits.u;
}

static __host__ __device__ __forceinline__ float u32_to_f32_bits_hw(uint32_t v) {
    union {
        uint32_t u;
        float    f;
    } bits;
    bits.u = v;
    return bits.f;
}

static __host__ __device__ __forceinline__ float bf16_abs_to_fp32_bits_hw(uint16_t abs_bits) {
    return u32_to_f32_bits_hw((uint32_t) abs_bits << 16);
}

static __device__ __forceinline__ uint16_t fp32_to_bf16_trunc_bits_hw(float v) {
    return (uint16_t) (f32_to_u32_bits_hw(v) >> 16);
}

static __device__ __forceinline__ uint32_t clz_u32_hw(uint32_t value) {
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

static __host__ __device__ __forceinline__ float trunc_f32_to_bf16_value_bits_hw(float v) {
    return u32_to_f32_bits_hw(f32_to_u32_bits_hw(v) & 0xffff0000u);
}

static __host__ __device__ __forceinline__ float bf16_mul_trunc_hw(float a, float b) {
    return trunc_f32_to_bf16_value_bits_hw(a * b);
}

static __device__ __forceinline__ uint16_t bf16_mul_bf16_rne_bits_hw(uint16_t a, uint16_t b) {
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

static __device__ __forceinline__ uint16_t bf16_mul_operands_rne_bits_hw(uint16_t a, float b) {
    const uint16_t a_bf16 = bf16_abs_bits(a);
    const uint16_t b_bf16 = fp32_to_bf16_trunc_bits_hw(b);
    return bf16_mul_bf16_rne_bits_hw(a_bf16, b_bf16);
}

static __device__ __forceinline__ bool bf16_pos_le_hw(uint16_t a, uint16_t b) {
    return bf16_abs_bits(a) <= bf16_abs_bits(b);
}

static __device__ __forceinline__ uint16_t bf16_pos_mul2_bits_hw(uint16_t value) {
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
        return (mant & 0x40u) != 0u
                ? (uint16_t) ((1u << 7) | ((mant & 0x3fu) << 1))
                : (uint16_t) ((mant & 0x3fu) << 1);
    }
    if (exp == 0xfeu) {
        return 0x7f80u;
    }
    return (uint16_t) (((exp + 1u) << 7) | mant);
}

static __device__ __forceinline__ uint16_t bf16_add_pos_trunc_bits_hw(uint16_t a, uint16_t b) {
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

    constexpr uint32_t kGuardBits = 16u;
    const uint32_t shift = exp_a - exp_b;
    const uint32_t sig_a_ext = sig_a << kGuardBits;
    const uint32_t sig_b_ext = (shift >= 24u) ? 0u : ((sig_b << kGuardBits) >> shift);
    const uint32_t sum_sig = sig_a_ext + sig_b_ext;

    const bool sum_zero = sum_sig == 0u;
    const uint32_t sum_sig_safe = sum_zero ? 1u : sum_sig;
    const int32_t msb_pos = 31 - (int32_t) clz_u32_hw(sum_sig_safe);
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

static __host__ __device__ __forceinline__ uint32_t round_shift_right_ties_down_u32_hw(uint32_t value, int shift) {
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

static __host__ __device__ __forceinline__ uint8_t e4m3_subnormal_from_fp32_bits_hw(
        uint32_t exponent,
        uint32_t mantissa) {
    const bool fp32_subnormal = exponent == 0u;
    const int32_t exp_unbiased = fp32_subnormal ? -126 : (int32_t) exponent - 127;
    const uint32_t significand = fp32_subnormal ? mantissa : (0x00800000u | mantissa);
    const uint32_t mant_q = round_shift_right_ties_down_u32_hw(significand, 14 - exp_unbiased);
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

    // Positive finite low codes decode to exact units of 1/512.
    if (scale <= 0.0302734375f) {
        return e4m3_subnormal_from_fp32_bits_hw(exponent, mantissa);
    }

    const int32_t exp_unbiased = (int32_t) exponent - 127;
    int32_t exp_field = exp_unbiased + 7;
    const uint32_t significand = 0x00800000u | mantissa;

    uint32_t signif_q = round_shift_right_ties_down_u32_hw(significand, 20);
    if (signif_q >= 16u) {
        signif_q = 8u;
        ++exp_field;
    }

    // E4M3FN exp_field 1..15 are finite normal exponents; only mantissa 7 at
    // exp_field 15 is NaN, so clamp overflow to the max finite 0x7e.
    if (exp_field > 15 || (exp_field == 15 && signif_q >= 15u)) {
        return 0x7eu;
    }

    return (uint8_t) (((uint32_t) exp_field << 3) | ((signif_q - 8u) & 0x7u));
}

static __host__ __device__ __forceinline__ float e4m3_scale_half_to_fp32_bits_hw(uint8_t scale) {
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
        return u32_to_f32_bits_hw(exp_bits | mant_bits);
    }
    const uint32_t scale_mant_clamped = scale_exp == 0x0fu && scale_mant == 7u ? 6u : scale_mant;
    return u32_to_f32_bits_hw(((scale_exp + 119u) << 23) | (scale_mant_clamped << 20));
}

static __host__ __device__ __forceinline__ float compute_block_scale_value_trunc_nn_hw(
        uint16_t block_abs_max_bits,
        float global_scale,
        bool bf16_block_scale) {
    if (block_abs_max_bits == 0u) {
        return 0.0f;
    }

    const float block_abs_max = bf16_abs_to_fp32_bits_hw(block_abs_max_bits);
    return bf16_block_scale
            ? bf16_mul_trunc_hw(
                    bf16_mul_trunc_hw(block_abs_max, trunc_f32_to_bf16_value_bits_hw(global_scale)),
                    trunc_f32_to_bf16_value_bits_hw(0.1666666716f))
            : block_abs_max * global_scale * 0.1666666716f;
}

static __device__ __forceinline__ uint16_t fp32_scale_half_to_bf16_bits_hw(float scale) {
    const uint32_t bits = f32_to_u32_bits_hw(scale);
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abs_bits = bits & 0x7fffffffu;
    uint32_t half_abs_bits = 0u;

    if (abs_bits >= 0x01000000u && abs_bits < 0x7f800000u) {
        // Normal FP32 with exponent > 1: divide by two by decrementing exponent.
        half_abs_bits = abs_bits - 0x00800000u;
    } else if (abs_bits >= 0x00800000u && abs_bits < 0x01000000u) {
        // exp == 1 becomes subnormal after /2.
        half_abs_bits = ((abs_bits & 0x007fffffu) | 0x00800000u) >> 1;
    } else if (abs_bits < 0x00800000u) {
        // Zero or subnormal.
        half_abs_bits = abs_bits >> 1;
    } else {
        // Inf/NaN are not expected for scale, but /2 preserves their exponent.
        half_abs_bits = abs_bits;
    }

    return (uint16_t) ((sign | half_abs_bits) >> 16);
}

static __device__ __forceinline__ uint8_t bf16_quant_mag_trunc_nn_hw(
        uint16_t abs_bits,
        float global_scale,
        float block_scale_half,
        uint16_t block_scale_half_bits,
        bool bf16_internal_arith) {
    const float abs_f = bf16_abs_to_fp32_bits_hw(abs_bits);
    if (bf16_internal_arith) {
        const uint16_t target = bf16_mul_operands_rne_bits_hw(abs_bits, global_scale);
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

static __device__ __forceinline__ void quantize_row_nvfp4_bf16_block(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float global_scale,
        const bool bf16_internal_arith,
        const bool bf16_block_scale) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const float xi = (lane_active && k0 < ne00) ? x[(int64_t) i1 * s01 + k0] : 0.0f;
    const uint16_t bf16 = fp32_to_bf16_trunc_bits_hw(xi);
    const uint16_t bf16_abs = bf16_abs_bits(bf16);
    uint16_t block_abs_max = bf16_abs;
    block_abs_max = max_u16_select_hw(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 8, WARP_SIZE));
    block_abs_max = max_u16_select_hw(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 4, WARP_SIZE));
    block_abs_max = max_u16_select_hw(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 2, WARP_SIZE));
    block_abs_max = max_u16_select_hw(block_abs_max, __shfl_xor_sync(0xFFFFFFFF, block_abs_max, 1, WARP_SIZE));
    block_abs_max = __shfl_sync(0xFFFFFFFF, block_abs_max, 0, WARP_SIZE);

    uint8_t scale = 0u;
    float block_scale_half_f = 0.0f;
    uint16_t block_scale_half_bits = 0u;
    if (lane == 0) {
        const float block_scale = compute_block_scale_value_trunc_nn_hw(block_abs_max, global_scale, bf16_block_scale);
        scale = e4m3_scale_from_fp32_bits_hw(block_scale);
        block_scale_half_f = e4m3_scale_half_to_fp32_bits_hw(scale);
        block_scale_half_bits = fp32_scale_half_to_bf16_bits_hw(block_scale);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale;
    }
    scale = __shfl_sync(0xFFFFFFFF, scale, 0, WARP_SIZE);
    block_scale_half_f = __shfl_sync(0xFFFFFFFF, block_scale_half_f, 0, WARP_SIZE);
    block_scale_half_bits = __shfl_sync(0xFFFFFFFF, block_scale_half_bits, 0, WARP_SIZE);

    uint8_t q = 0u;
    if (scale != 0u) {
        const uint8_t mag = bf16_quant_mag_trunc_nn_hw(
                bf16_abs, global_scale, block_scale_half_f, block_scale_half_bits, bf16_internal_arith);
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
        const bool per_tensor_scale,
        const bool bf16_internal_arith,
        const bool bf16_block_scale) {
    const int i1 = blockIdx.y;
    const float global_scale = per_tensor_scale ? global_scales[0] : global_scales[i1];
    quantize_row_nvfp4_bf16_block(x, y, ne00, s01, global_scale, bf16_internal_arith, bf16_block_scale);
}

static __global__ void quantize_row_nvfp4_dynamic_bf16_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ amax_rows,
        const bool per_tensor_scale,
        const bool bf16_internal_arith,
        const bool bf16_block_scale) {
    const int i1 = blockIdx.y;
    const float amax_f = per_tensor_scale ? amax_rows[0] : amax_rows[i1];
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_q_global_scale(amax_f);
    quantize_row_nvfp4_bf16_block(x, y, ne00, s01, global_scale, bf16_internal_arith, bf16_block_scale);
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

bool ggml_cuda_nvfp4_bf16_quant_trunc_nn_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_bf16_quant_trunc_nn_once(env, cached != 0);
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_bf16_quant_bf16_internal_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_BF16_QUANT_BF16_INTERNAL");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_bf16_quant_bf16_internal_once(env, cached != 0);
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_bf16_quant_bf16_block_scale_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_BF16_QUANT_BF16_BLOCK_SCALE");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_bf16_quant_bf16_block_scale_once(env, cached != 0);
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_trunc_bf16_input_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_TRUNC_BF16_INPUT");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_trunc_bf16_input_once(env, cached != 0);
    }
    return cached != 0;
}

void ggml_cuda_nvfp4_quantize_rows_bf16_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * global_scales,
        bool per_tensor_scale,
        bool bf16_internal_arith,
        bool bf16_block_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
            x, y, ne00, s01, global_scales, per_tensor_scale, bf16_internal_arith, bf16_block_scale);
}

void ggml_cuda_nvfp4_quantize_rows_dynamic_bf16_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * amax_rows,
        bool per_tensor_scale,
        bool bf16_internal_arith,
        bool bf16_block_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_dynamic_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
            x, y, ne00, s01, amax_rows, per_tensor_scale, bf16_internal_arith, bf16_block_scale);
}
