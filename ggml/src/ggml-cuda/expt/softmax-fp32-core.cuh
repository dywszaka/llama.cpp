#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define SOFTMAX_FP32_INLINE static __host__ __device__ __forceinline__
#else
#define SOFTMAX_FP32_INLINE static inline
#endif

SOFTMAX_FP32_INLINE float softmax_fp32_bf16_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } converted = { (uint32_t) value << 16 };
    return converted.f;
}

SOFTMAX_FP32_INLINE uint16_t softmax_fp32_f32_to_bf16_rz(float value) {
    union {
        uint32_t u;
        float f;
    } converted;
    converted.f = value;
    return (uint16_t) (converted.u >> 16);
}

SOFTMAX_FP32_INLINE uint16_t softmax_fp32_f32_to_bf16_rne(float value) {
    union {
        uint32_t u;
        float f;
    } converted;
    converted.f = value;
    const uint32_t upper = converted.u >> 16;
    const uint32_t lower = converted.u & UINT32_C(0xffff);
    return (uint16_t) (upper +
            (lower > UINT32_C(0x8000) ||
             (lower == UINT32_C(0x8000) && (upper & 1u) != 0)));
}

SOFTMAX_FP32_INLINE int64_t softmax_fp32_sign_extend(uint64_t value, int bits) {
    const uint64_t mask = UINT64_C(1) << (bits - 1);
    return (int64_t) ((value ^ mask) - mask);
}

/* Bit-exact mirror of QEMU target/riscv/vector_helper.c:xperia_bf16_expp(). */
SOFTMAX_FP32_INLINE uint16_t softmax_fp32_ni900_exp_bf16(uint16_t input) {
    const uint32_t sign = input >> 15;
    const uint32_t exponent = (input >> 7) & UINT32_C(0xff);
    const uint32_t mantissa = input & UINT32_C(0x7f);

    if (exponent == 0) {
        return UINT16_C(0x3f80);
    }
    if (exponent == UINT32_C(0xff)) {
        if (mantissa != 0) {
            return UINT16_C(0x7fc0);
        }
        return sign != 0 ? UINT16_C(0x0000) : UINT16_C(0x7f80);
    }
    if (input >= UINT16_C(0x42b2) && input <= UINT16_C(0x7f7f)) {
        return UINT16_C(0x7f80);
    }
    if (input >= UINT16_C(0xc386) && input <= UINT16_C(0xff7f)) {
        return UINT16_C(0x0000);
    }

    const uint32_t mantissa_with_one = UINT32_C(0x80) | mantissa;
    const uint32_t mantissa_times_inv_ln2 =
            (mantissa_with_one * UINT32_C(23637)) & UINT32_C(0x7fffff);
    uint32_t shifted;
    if (exponent >= UINT32_C(127)) {
        const uint32_t amount = exponent - UINT32_C(127);
        shifted = amount < 30 ?
                (mantissa_times_inv_ln2 << amount) & UINT32_C(0x3fffffff) : 0;
    } else {
        const uint32_t amount = UINT32_C(127) - exponent;
        shifted = amount < 23 ?
                (mantissa_times_inv_ln2 >> amount) & UINT32_C(0x3fffffff) : 0;
    }

    const uint16_t no_fraction = (shifted >> 13) & 1u ?
            (uint16_t) (((shifted >> 14) + 1u) & UINT32_C(0xffff)) :
            (uint16_t) ((shifted >> 14) & UINT32_C(0xffff));
    const uint16_t unsigned_value = sign != 0 ?
            (uint16_t) (~no_fraction + 1u) : no_fraction;
    const int64_t signed_value = softmax_fp32_sign_extend(unsigned_value, 16);
    const uint32_t normalized_mantissa = (uint32_t) signed_value & UINT32_C(0x7f);
    const int32_t normalized_exponent = (int32_t) softmax_fp32_sign_extend(
            (uint64_t) (softmax_fp32_sign_extend(
                    (uint64_t) (signed_value >> 7), 9) + 127) & UINT64_C(0x1ff), 9);
    if (normalized_exponent >= 255) {
        return sign != 0 ? UINT16_C(0x0000) : UINT16_C(0x7f80);
    }
    if (normalized_exponent <= 0) {
        return UINT16_C(0x0000);
    }

    const uint32_t fraction_msb = (normalized_mantissa >> 6) & 1u;
    const uint32_t add_result =
            (normalized_mantissa + (fraction_msb != 0 ? 278u : 363u)) & UINT32_C(0x1ff);
    const uint32_t doubled_mantissa = (normalized_mantissa << 1) & UINT32_C(0xff);
    const uint32_t first_product = fraction_msb != 0 ?
            ((UINT32_C(0xff) - doubled_mantissa) * 7u) & UINT32_C(0x7ff) :
            (doubled_mantissa * 4u) & UINT32_C(0x7ff);
    const uint32_t second_product =
            ((add_result * first_product) >> 12) & UINT32_C(0xff);
    const uint32_t result_mantissa = fraction_msb != 0 ?
            (UINT32_C(0x7f) - second_product) & UINT32_C(0x7f) :
            second_product & UINT32_C(0x7f);
    return (uint16_t) (((uint32_t) normalized_exponent << 7) | result_mantissa);
}

SOFTMAX_FP32_INLINE uint16_t softmax_fp32_exp_from_delta(float delta) {
    return softmax_fp32_ni900_exp_bf16(
            softmax_fp32_f32_to_bf16_rne(delta));
}

#undef SOFTMAX_FP32_INLINE
