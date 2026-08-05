#pragma once

#include <cstdint>
#include <cstdlib>

static constexpr __host__ __device__ int32_t ggml_cuda_bf16_expp_sign_extend(
        uint32_t value, uint32_t width) {
    const uint32_t sign = 1u << (width - 1u);
    const uint32_t mask = (1u << width) - 1u;
    value &= mask;
    return (value & sign) != 0u ? int32_t(value) - int32_t(1u << width) : int32_t(value);
}

static constexpr __host__ __device__ uint16_t ggml_cuda_bf16_expp_bits(uint16_t input) {
    constexpr uint32_t BIAS = 127u;
    constexpr uint32_t ONE_BF16 = 0x3f80u;
    constexpr uint32_t LN2_FLIP = 23637u;
    constexpr uint32_t ALPHA = 4u;
    constexpr uint32_t BETA = 7u;
    constexpr uint32_t GAMMA1 = 363u;
    constexpr uint32_t GAMMA2 = 278u;

    const uint32_t sign = input >> 15;
    const uint32_t exponent = (input >> 7) & 0xffu;
    const uint32_t mantissa = input & 0x7fu;

    if (exponent == 0u) {
        return uint16_t(ONE_BF16);
    }
    if (exponent == 0xffu) {
        if (mantissa != 0u) {
            return 0x7fc0u;
        }
        return sign != 0u ? 0x0000u : 0x7f80u;
    }

    if (input >= 0x42b2u && input <= 0x7f7fu) {
        return 0x7f80u;
    }
    if (input >= 0xc386u && input <= 0xff7fu) {
        return 0x0000u;
    }

    const uint32_t mant_with_1 = 0x80u | mantissa;
    const uint32_t mant_mul_ln2flip = (mant_with_1 * LN2_FLIP) & 0x7fffffu;

    uint32_t shm = 0u;
    if (exponent >= BIAS) {
        const uint32_t shift_amount = exponent - BIAS;
        shm = shift_amount < 30u ? (mant_mul_ln2flip << shift_amount) & 0x3fffffffu : 0u;
    } else {
        const uint32_t shift_amount = BIAS - exponent;
        shm = shift_amount < 23u ? (mant_mul_ln2flip >> shift_amount) & 0x3fffffffu : 0u;
    }

    uint32_t shm_q7 = ((shm >> 14) + ((shm >> 13) & 1u)) & 0xffffu;
    if (sign != 0u) {
        shm_q7 = (~shm_q7 + 1u) & 0xffffu;
    }

    const uint32_t nm = shm_q7 & 0x7fu;
    const int32_t integer_part =
        ggml_cuda_bf16_expp_sign_extend((shm_q7 >> 7) & 0x1ffu, 9u);
    const int32_t ne_temp = ggml_cuda_bf16_expp_sign_extend(
        uint32_t(integer_part + int32_t(BIAS)) & 0x1ffu, 9u);

    if (ne_temp >= 255) {
        return sign != 0u ? 0x0000u : 0x7f80u;
    }
    if (ne_temp <= 0) {
        return 0x0000u;
    }

    const uint32_t frac_msb = nm >> 6;
    const uint32_t res_add_1 = (nm + (frac_msb != 0u ? GAMMA2 : GAMMA1)) & 0x1ffu;
    const uint32_t mant_mul = (nm << 1) & 0xffu;
    const uint32_t res_mul_1 = frac_msb != 0u
        ? ((0xffu - mant_mul) * BETA) & 0x7ffu
        : (mant_mul * ALPHA) & 0x7ffu;
    const uint32_t res_mul_2 = ((res_add_1 * res_mul_1) >> 12) & 0xffu;
    const uint32_t result_mantissa = frac_msb != 0u
        ? (0x7fu - res_mul_2) & 0x7fu
        : res_mul_2 & 0x7fu;

    return uint16_t((uint32_t(ne_temp) << 7) | result_mantissa);
}

static __device__ __forceinline__ float ggml_cuda_bf16_expp_f32(float input) {
    const uint16_t input_bf16 = uint16_t(__float_as_uint(input) >> 16);
    const uint16_t output_bf16 = ggml_cuda_bf16_expp_bits(input_bf16);
    return __uint_as_float(uint32_t(output_bf16) << 16);
}

static bool ggml_cuda_soft_max_bf16_exp_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("GGML_CUDA_SOFTMAX_BF16_EXP");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}
