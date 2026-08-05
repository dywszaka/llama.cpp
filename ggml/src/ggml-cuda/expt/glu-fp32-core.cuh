#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define GLU_FP32_INLINE static __host__ __device__ __forceinline__
#else
#define GLU_FP32_INLINE static inline
#endif

GLU_FP32_INLINE float glu_fp32_bf16_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } converted = { (uint32_t) value << 16 };
    return converted.f;
}

GLU_FP32_INLINE uint16_t glu_fp32_f32_to_bf16_rz(float value) {
    union {
        uint32_t u;
        float f;
    } converted;
    converted.f = value;
    return (uint16_t) (converted.u >> 16);
}

GLU_FP32_INLINE uint16_t glu_fp32_f32_to_bf16_rne(float value) {
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

GLU_FP32_INLINE int64_t glu_fp32_sign_extend(uint64_t value, int bits) {
    const uint64_t mask = UINT64_C(1) << (bits - 1);
    return (int64_t) ((value ^ mask) - mask);
}

/* Bit-exact mirror of QEMU target/riscv/vector_helper.c:xperia_bf16_expp(). */
GLU_FP32_INLINE uint16_t glu_fp32_ni900_exp_bf16(uint16_t input) {
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
    const int64_t signed_value = glu_fp32_sign_extend(unsigned_value, 16);
    const uint32_t normalized_mantissa = (uint32_t) signed_value & UINT32_C(0x7f);
    const int32_t normalized_exponent = (int32_t) glu_fp32_sign_extend(
            (uint64_t) (glu_fp32_sign_extend(
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

GLU_FP32_INLINE uint16_t glu_fp32_exp_from_delta(float delta) {
    return glu_fp32_ni900_exp_bf16(
            glu_fp32_f32_to_bf16_rne(delta));
}

GLU_FP32_INLINE uint16_t glu_fp32_swiglu_bits(uint16_t x_bits, uint16_t gate_bits) {
    const uint16_t x_abs = x_bits & UINT16_C(0x7fff);
    const uint16_t gate_abs = gate_bits & UINT16_C(0x7fff);
    const int x_nan = x_abs > UINT16_C(0x7f80);
    const int gate_nan = gate_abs > UINT16_C(0x7f80);
    if (x_nan || gate_nan) {
        return UINT16_C(0x7fc0);
    }
    const float x = glu_fp32_bf16_to_f32(x_bits);
    const float gate = glu_fp32_bf16_to_f32(gate_bits);
    const uint16_t neg_x_bits = glu_fp32_f32_to_bf16_rne(-x);
    const uint16_t exponent_bits = glu_fp32_ni900_exp_bf16(neg_x_bits);
    if (x_abs == UINT16_C(0x7f80)) {
        if ((x_bits & UINT16_C(0x8000)) != 0 || gate_abs == 0) {
            return UINT16_C(0x7fc0);
        }
        return (uint16_t) (UINT16_C(0x7f80) |
                (gate_bits & UINT16_C(0x8000)));
    }
    if (gate_abs == UINT16_C(0x7f80)) {
        if (x_abs == 0 || exponent_bits == UINT16_C(0x7f80)) {
            return UINT16_C(0x7fc0);
        }
        return (uint16_t) (UINT16_C(0x7f80) |
                ((x_bits ^ gate_bits) & UINT16_C(0x8000)));
    }
    if (exponent_bits == UINT16_C(0x7f80)) {
        return (uint16_t) ((x_bits ^ gate_bits) & UINT16_C(0x8000));
    }
    const float exponent = glu_fp32_bf16_to_f32(exponent_bits);
#if defined(__CUDA_ARCH__)
    const float denominator = __fadd_rn(1.0f, exponent);
    const float result = __fmul_rn(__fdiv_rn(x, denominator), gate);
#else
    const float denominator = 1.0f + exponent;
    const float result = (x / denominator) * gate;
#endif
    const uint16_t output = glu_fp32_f32_to_bf16_rne(result);
    return (output & UINT16_C(0x7f80)) == UINT16_C(0x7f80) &&
            (output & UINT16_C(0x007f)) != 0 ? UINT16_C(0x7fc0) : output;
}

#undef GLU_FP32_INLINE
