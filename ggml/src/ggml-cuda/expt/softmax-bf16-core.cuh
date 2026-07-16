#pragma once

#include <stdint.h>

/*
 * Deterministic BF16 softmax primitives mirrored from
 * call_softmax/include/softmax_bf16_core.h.
 *
 * The QEMU/RVV firmware and this CUDA experiment intentionally use the same
 * integer rounding points so their BF16 output bits are directly comparable.
 */

#if defined(__CUDACC__)
#define SOFTMAX_BF16_INLINE static __host__ __device__ __forceinline__
#else
#define SOFTMAX_BF16_INLINE static inline
#endif

#define SOFTMAX_BF16_Q31_ONE UINT32_C(0x80000000)
#define SOFTMAX_BF16_EXP_STEP_Q31 UINT32_C(0x7fe00400)

SOFTMAX_BF16_INLINE uint64_t softmax_bf16_round_shift_u64(uint64_t value, unsigned int shift) {
    if (shift == 0) {
        return value;
    }
    if (shift >= 64) {
        return 0;
    }
    const uint64_t quotient = value >> shift;
    const uint64_t remainder = value & ((UINT64_C(1) << shift) - 1);
    const uint64_t halfway = UINT64_C(1) << (shift - 1);
    return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1) != 0));
}

SOFTMAX_BF16_INLINE int32_t softmax_bf16_to_q16(uint16_t bits) {
    const uint32_t sign = bits >> 15;
    const uint32_t exponent = (bits >> 7) & 0xffu;
    const uint32_t fraction = bits & 0x7fu;

    if (exponent == 0) {
        return 0;
    }
    if (exponent == 0xffu) {
        return sign != 0 ? INT32_MIN : INT32_MAX;
    }

    const uint64_t significand = UINT64_C(128) + fraction;
    const int shift = (int) exponent - 118;
    uint64_t magnitude;
    if (shift >= 0) {
        if (shift >= 56 || significand > (UINT64_C(0x80000000) >> shift)) {
            magnitude = UINT64_C(0x80000000);
        } else {
            magnitude = significand << shift;
        }
    } else {
        magnitude = softmax_bf16_round_shift_u64(significand, (unsigned int) -shift);
    }

    if (sign != 0) {
        return magnitude >= UINT64_C(0x80000000) ? INT32_MIN : -(int32_t) magnitude;
    }
    return magnitude > INT32_MAX ? INT32_MAX : (int32_t) magnitude;
}

SOFTMAX_BF16_INLINE uint32_t softmax_bf16_mul_q31(uint32_t left, uint32_t right) {
    const uint64_t product = (uint64_t) left * (uint64_t) right;
    const uint64_t quotient = product >> 31;
    const uint64_t remainder = product & UINT64_C(0x7fffffff);
    const uint64_t halfway = UINT64_C(0x40000000);
    return (uint32_t) (quotient +
            (remainder > halfway || (remainder == halfway && (quotient & 1) != 0)));
}

SOFTMAX_BF16_INLINE uint32_t softmax_bf16_exp_neg_q31(uint64_t delta_q16) {
    const uint64_t steps = (delta_q16 + 32) >> 6;
    if (steps >= 32768) {
        return 0;
    }

    uint32_t result = SOFTMAX_BF16_Q31_ONE;
    uint32_t factor = SOFTMAX_BF16_EXP_STEP_Q31;
    uint32_t power = (uint32_t) steps;
    while (power != 0) {
        if ((power & 1u) != 0) {
            result = softmax_bf16_mul_q31(result, factor);
        }
        power >>= 1;
        if (power != 0) {
            factor = softmax_bf16_mul_q31(factor, factor);
        }
    }
    return result;
}

SOFTMAX_BF16_INLINE uint64_t softmax_bf16_divide_rne_u64(uint64_t numerator, uint64_t denominator) {
    const uint64_t quotient = numerator / denominator;
    const uint64_t remainder = numerator % denominator;
    return quotient + (remainder > denominator - remainder ||
            (remainder == denominator - remainder && (quotient & 1) != 0));
}

SOFTMAX_BF16_INLINE unsigned int softmax_bf16_msb_u64(uint64_t value) {
    unsigned int position = 0;
    while (value > 1) {
        value >>= 1;
        ++position;
    }
    return position;
}

SOFTMAX_BF16_INLINE uint16_t softmax_bf16_q31_to_bits(uint32_t probability_q31) {
    if (probability_q31 == 0) {
        return 0;
    }

    const unsigned int msb = softmax_bf16_msb_u64(probability_q31);
    uint32_t exponent = msb + 96u;
    uint64_t significand;
    if (msb >= 7) {
        const unsigned int shift = msb - 7;
        significand = softmax_bf16_round_shift_u64(probability_q31, shift);
    } else {
        significand = (uint64_t) probability_q31 << (7 - msb);
    }

    if (significand == 256) {
        significand = 128;
        ++exponent;
    }
    return (uint16_t) ((exponent << 7) | ((uint32_t) significand & 0x7fu));
}

SOFTMAX_BF16_INLINE uint16_t softmax_bf16_probability_bits(uint32_t exponent_q31, uint64_t sum_q31) {
    if (exponent_q31 == 0 || sum_q31 == 0) {
        return 0;
    }
    uint64_t probability_q31 = softmax_bf16_divide_rne_u64(
            (uint64_t) exponent_q31 << 31, sum_q31);
    if (probability_q31 > SOFTMAX_BF16_Q31_ONE) {
        probability_q31 = SOFTMAX_BF16_Q31_ONE;
    }
    return softmax_bf16_q31_to_bits((uint32_t) probability_q31);
}

#undef SOFTMAX_BF16_INLINE

