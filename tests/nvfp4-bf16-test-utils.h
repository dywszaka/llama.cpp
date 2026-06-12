#pragma once

#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

static inline uint32_t nvfp4_test_f32_to_u32(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static inline float nvfp4_test_u32_to_f32(uint32_t bits) {
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

static inline uint16_t nvfp4_test_fp32_to_bf16_trunc_bits(float v) {
    return (uint16_t) (nvfp4_test_f32_to_u32(v) >> 16);
}

static inline uint16_t nvfp4_test_bf16_abs_bits(uint16_t v) {
    return (uint16_t) (v & 0x7fffu);
}

static inline float nvfp4_test_bf16_abs_to_f32(uint16_t abs_bits) {
    return nvfp4_test_u32_to_f32((uint32_t) abs_bits << 16);
}

static inline uint32_t nvfp4_test_round_shift_right_ties_down(uint32_t value, int shift) {
    if (shift <= 0) {
        const int lshift = -shift;
        return lshift > 8 ? 0xffffffffu : (value << lshift);
    }
    if (shift > 24) {
        return 0u;
    }

    const uint32_t shifted = value >> shift;
    const uint32_t half = 1u << (shift - 1);
    const uint32_t mask = (1u << shift) - 1u;
    const uint32_t remainder = value & mask;
    return shifted + (remainder > half ? 1u : 0u);
}

static inline uint8_t nvfp4_test_e4m3_subnormal_from_fp32(uint32_t exponent, uint32_t mantissa) {
    const bool fp32_subnormal = exponent == 0u;
    const int32_t exp_unbiased = fp32_subnormal ? -126 : (int32_t) exponent - 127;
    const uint32_t significand = fp32_subnormal ? mantissa : (0x00800000u | mantissa);
    const uint32_t mant_q = nvfp4_test_round_shift_right_ties_down(significand, 14 - exp_unbiased);
    return (uint8_t) (mant_q > 15u ? 15u : mant_q);
}

static inline uint8_t nvfp4_test_e4m3_scale_from_fp32(float scale) {
    const uint32_t bits = nvfp4_test_f32_to_u32(scale);
    const uint32_t sign = bits >> 31;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x007fffffu;
    if (sign != 0u || exponent == 0xffu) {
        return 0u;
    }

    if (scale <= 0.0302734375f) {
        return nvfp4_test_e4m3_subnormal_from_fp32(exponent, mantissa);
    }

    const int32_t exp_unbiased = (int32_t) exponent - 127;
    int32_t exp_field = exp_unbiased + 7;
    const uint32_t significand = 0x00800000u | mantissa;

    uint32_t signif_q = nvfp4_test_round_shift_right_ties_down(significand, 20);
    if (signif_q >= 16u) {
        signif_q = 8u;
        ++exp_field;
    }

    if (exp_field > 15 || (exp_field == 15 && signif_q >= 15u)) {
        return 0x7eu;
    }

    return (uint8_t) (((uint32_t) exp_field << 3) | ((signif_q - 8u) & 0x7u));
}

static inline float nvfp4_test_e4m3_scale_half_to_f32(uint8_t scale) {
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
        return nvfp4_test_u32_to_f32(exp_bits | mant_bits);
    }
    const uint32_t scale_mant_clamped = scale_exp == 0x0fu && scale_mant == 7u ? 6u : scale_mant;
    return nvfp4_test_u32_to_f32(((scale_exp + 119u) << 23) | (scale_mant_clamped << 20));
}

static inline void nvfp4_test_quantize_bf16_trunc_nn_block(
        const float * src,
        block_nvfp4 * dst,
        float global_scale) {
    uint16_t bf16[QK_NVFP4] = {};
    uint16_t block_abs_max = 0u;
    for (int i = 0; i < QK_NVFP4; ++i) {
        bf16[i] = nvfp4_test_fp32_to_bf16_trunc_bits(src[i]);
        block_abs_max = std::max(block_abs_max, nvfp4_test_bf16_abs_bits(bf16[i]));
    }

    const float block_abs_max_f = nvfp4_test_bf16_abs_to_f32(block_abs_max);
    const float block_scale = block_abs_max == 0u ? 0.0f : block_abs_max_f * global_scale * 0.1666666716f;
    const uint8_t scale = nvfp4_test_e4m3_scale_from_fp32(block_scale);
    dst->e = scale;
    std::memset(dst->qs, 0, sizeof(dst->qs));
    if (scale == 0u) {
        return;
    }

    const float block_scale_half = nvfp4_test_e4m3_scale_half_to_f32(scale);
    uint8_t q_raw[QK_NVFP4] = {};
    for (int i = 0; i < QK_NVFP4; ++i) {
        const uint16_t bits = bf16[i];
        const uint16_t abs_bits = nvfp4_test_bf16_abs_bits(bits);
        const float target = nvfp4_test_bf16_abs_to_f32(abs_bits) * global_scale;
        const float target_2x = target + target;
        const float scale_2x = block_scale_half + block_scale_half;
        const float scale_3x = scale_2x + block_scale_half;
        const float scale_5x = scale_3x + scale_2x;
        const float scale_7x = scale_5x + scale_2x;
        const float scale_10x = scale_5x + scale_5x;
        const float scale_14x = scale_7x + scale_7x;
        const float scale_20x = scale_10x + scale_10x;
        uint8_t mag = 7u;
        if (target_2x <= block_scale_half) {
            mag = 0u;
        } else if (target_2x <= scale_3x) {
            mag = 1u;
        } else if (target_2x <= scale_5x) {
            mag = 2u;
        } else if (target_2x <= scale_7x) {
            mag = 3u;
        } else if (target_2x <= scale_10x) {
            mag = 4u;
        } else if (target_2x <= scale_14x) {
            mag = 5u;
        } else if (target_2x <= scale_20x) {
            mag = 6u;
        }
        q_raw[i] = mag == 0u ? 0u : (uint8_t) ((((bits >> 15) & 1u) << 3) | mag);
    }

    for (int i = 0; i < QK_NVFP4 / 2; ++i) {
        dst->qs[i] = (uint8_t) (q_raw[2*i] | (q_raw[2*i + 1] << 4));
    }
}

static inline void nvfp4_test_quantize_bf16_trunc_nn_rows(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int64_t rows,
        int64_t k,
        const std::vector<float> & global_scales) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    GGML_ASSERT((int64_t) global_scales.size() == rows);
    const int64_t nblk = k / QK_NVFP4;
    dst.assign((size_t) rows * (size_t) nblk, {});
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t ib = 0; ib < nblk; ++ib) {
            nvfp4_test_quantize_bf16_trunc_nn_block(
                    src.data() + (size_t) r * (size_t) k + (size_t) ib * QK_NVFP4,
                    dst.data() + (size_t) r * (size_t) nblk + (size_t) ib,
                    global_scales[(size_t) r]);
        }
    }
}
