#pragma once

#include <cstdint>

static __device__ __forceinline__ uint16_t ggml_cuda_fp32_to_bf16_round_device(float v) {
    union {
        float    f;
        uint32_t u;
    } tmp = { v };
    uint32_t fp32_bits = tmp.u;

    const uint32_t sign = (fp32_bits >> 31) & 0x1;
    uint32_t exp = (fp32_bits >> 23) & 0xff;
    uint32_t mant = fp32_bits & 0x007fffff;

    if (exp == 0xff) {
        if (mant != 0) {
            uint16_t bf16_bits = (sign << 15) | 0x7f80u | ((mant >> 16) & 0x3fu);
            if ((bf16_bits & 0x7fu) == 0) {
                bf16_bits |= 0x0040u;
            }
            return bf16_bits;
        }
        return (uint16_t) ((sign << 15) | 0x7f80u);
    }

    if (exp == 0 && mant == 0) {
        return (uint16_t) (sign << 15);
    }

    if (exp == 0) {
        uint32_t shift = 0;
        while ((mant & (1u << 22)) == 0 && shift < 22) {
            mant <<= 1;
            shift++;
        }
        exp = 1 - shift;
        mant &= 0x007fffffu;
    } else {
        mant |= 0x00800000u;
    }

    const uint32_t guard_bit = (mant >> 15) & 0x1;
    const uint32_t round_bit = (mant >> 14) & 0x1;
    const uint32_t sticky = (mant & 0x3fffu) != 0 ? 1 : 0;
    uint32_t bf16_mant = (mant >> 16) & 0x7f;

    if (guard_bit == 1 && (round_bit == 1 || sticky == 1 || (bf16_mant & 0x1) == 1)) {
        bf16_mant += 1;
    }

    if (bf16_mant > 0x7f) {
        bf16_mant = 0;
        exp += 1;
        if (exp > 0xfe) {
            return (uint16_t) ((sign << 15) | 0x7f80u);
        }
    }

    if (exp < 1) {
        return (uint16_t) (sign << 15);
    }

    return (uint16_t) ((sign << 15) | ((exp & 0xff) << 7) | (bf16_mant & 0x7f));
}
