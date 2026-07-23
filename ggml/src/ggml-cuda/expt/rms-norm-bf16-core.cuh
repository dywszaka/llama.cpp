#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

static __device__ __forceinline__ uint16_t rms_norm_bf16_from_f32_rz(float value) {
    // RMS_NORM canonical input is the raw high half of the F32 bit pattern.
    return (uint16_t) (__float_as_uint(value) >> 16);
}

static __device__ __forceinline__ uint16_t rms_norm_bf16_from_f32_rne(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t exponent = bits & UINT32_C(0x7f800000);
    const uint32_t mantissa = bits & UINT32_C(0x007fffff);
    if (exponent == UINT32_C(0x7f800000) && mantissa != 0) {
        uint16_t result = (uint16_t) (bits >> 16);
        result |= UINT16_C(0x0040);
        return result;
    }

    const uint32_t upper = bits >> 16;
    const uint32_t lower = bits & UINT32_C(0xffff);
    return (uint16_t) (upper +
            (lower > UINT32_C(0x8000) ||
             (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0)));
}

static __device__ __forceinline__ float rms_norm_bf16_to_f32(uint16_t value) {
    return __uint_as_float((uint32_t) value << 16);
}

static __device__ __forceinline__ uint16_t rms_norm_bf16_add(
        uint16_t left, uint16_t right) {
    return rms_norm_bf16_from_f32_rne(__fadd_rn(
            rms_norm_bf16_to_f32(left), rms_norm_bf16_to_f32(right)));
}

static __device__ __forceinline__ uint16_t rms_norm_bf16_mul(
        uint16_t left, uint16_t right) {
    return rms_norm_bf16_from_f32_rne(__fmul_rn(
            rms_norm_bf16_to_f32(left), rms_norm_bf16_to_f32(right)));
}

static __device__ __forceinline__ uint16_t rms_norm_bf16_fma(
        uint16_t left, uint16_t right, uint16_t accumulator) {
    return rms_norm_bf16_from_f32_rne(__fmaf_rn(
            rms_norm_bf16_to_f32(left),
            rms_norm_bf16_to_f32(right),
            rms_norm_bf16_to_f32(accumulator)));
}

static __device__ __forceinline__ uint16_t rms_norm_bf16_sqrt(uint16_t value) {
    return rms_norm_bf16_from_f32_rne(__fsqrt_rn(rms_norm_bf16_to_f32(value)));
}

static __device__ __forceinline__ uint16_t rms_norm_bf16_div(
        uint16_t numerator, uint16_t denominator) {
    return rms_norm_bf16_from_f32_rne(__fdiv_rn(
            rms_norm_bf16_to_f32(numerator),
            rms_norm_bf16_to_f32(denominator)));
}

/*
 * Bit-exact model of call_rms_norm/src/rms_norm_rvv.c for VLEN=512:
 *
 * - 32 e16 lanes accumulate x*x independently with BF16 fused multiply-add;
 * - vfredusum consumes lane 0 through lane 31 in order with BF16 add;
 * - inverse ncols, eps, mean, sqrt, reciprocal and output scaling are rounded
 *   to BF16 at the same points as the NI900 BF16-mode e16 instructions.
 */
static __global__ void rms_norm_bf16_bitexact_kernel(
        const uint16_t * input,
        uint16_t * output,
        int ncols,
        float eps) {
    constexpr int rvv_lanes = 32;
    const int lane = (int) threadIdx.x;
    if (lane >= rvv_lanes) {
        return;
    }

    const size_t row = (size_t) blockIdx.x;
    const uint16_t * row_input = input + row * (size_t) ncols;
    uint16_t * row_output = output + row * (size_t) ncols;

    uint16_t lane_sum = UINT16_C(0x0000);
    for (int col = lane; col < ncols; col += rvv_lanes) {
        const uint16_t value = row_input[col];
        lane_sum = rms_norm_bf16_fma(value, value, lane_sum);
    }

    __shared__ uint16_t lane_sums[rvv_lanes];
    __shared__ uint16_t scale;
    lane_sums[lane] = lane_sum;
    __syncthreads();

    if (lane == 0) {
        uint16_t sum_squares = UINT16_C(0x0000);
        for (int index = 0; index < rvv_lanes; ++index) {
            sum_squares = rms_norm_bf16_add(sum_squares, lane_sums[index]);
        }

        const uint16_t inverse_cols = rms_norm_bf16_from_f32_rne(
                __fdiv_rn(1.0f, __int2float_rn(ncols)));
        const uint16_t epsilon = rms_norm_bf16_from_f32_rne(eps);
        const uint16_t mean = rms_norm_bf16_mul(sum_squares, inverse_cols);
        const uint16_t mean_with_eps = rms_norm_bf16_add(mean, epsilon);
        const uint16_t root = rms_norm_bf16_sqrt(mean_with_eps);
        scale = rms_norm_bf16_div(UINT16_C(0x3f80), root);
    }
    __syncthreads();

    for (int col = lane; col < ncols; col += rvv_lanes) {
        row_output[col] = rms_norm_bf16_mul(row_input[col], scale);
    }
}
