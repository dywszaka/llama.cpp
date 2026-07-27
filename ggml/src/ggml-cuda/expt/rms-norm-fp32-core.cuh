#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

static __device__ __forceinline__ uint16_t rms_norm_bf16_from_f32_rz(float value) {
    // Canonical RMS_NORM input is the raw high half of the F32 bit pattern.
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

/*
 * CUDA model of call_rms_norm_fp32/src/rms_norm_rvv.c for VLEN=512:
 *
 * - canonical input/output are BF16;
 * - 32 lanes widen BF16 inputs to FP32 and accumulate with FP32 FMA;
 * - lane 0 reduces the 32 FP32 lane sums in ascending lane order;
 * - mean, epsilon, sqrt, reciprocal and output scaling remain FP32;
 * - only the final output conversion rounds FP32 to BF16 RNE.
 */
static __global__ void rms_norm_fp32_bitexact_kernel(
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

    float lane_sum = 0.0f;
    for (int col = lane; col < ncols; col += rvv_lanes) {
        const float value = rms_norm_bf16_to_f32(row_input[col]);
        lane_sum = __fmaf_rn(value, value, lane_sum);
    }

    __shared__ float lane_sums[rvv_lanes];
    __shared__ float scale;
    lane_sums[lane] = lane_sum;
    __syncthreads();

    if (lane == 0) {
        float sum_squares = 0.0f;
        for (int index = 0; index < rvv_lanes; ++index) {
            sum_squares = __fadd_rn(sum_squares, lane_sums[index]);
        }

        const float inverse_cols = __fdiv_rn(1.0f, __int2float_rn(ncols));
        const float mean = __fmul_rn(sum_squares, inverse_cols);
        const float mean_with_eps = __fadd_rn(mean, eps);
        const float root = __fsqrt_rn(mean_with_eps);
        scale = __fdiv_rn(1.0f, root);
    }
    __syncthreads();

    for (int col = lane; col < ncols; col += rvv_lanes) {
        const float value = rms_norm_bf16_to_f32(row_input[col]);
        row_output[col] = rms_norm_bf16_from_f32_rne(
                __fmul_rn(value, scale));
    }
}
