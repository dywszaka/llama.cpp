#pragma once

#include "nvfp4-common.cuh"

#include <cmath>
#include <cstdint>

bool ggml_cuda_nvfp4_fp4mulmat_enabled();
bool ggml_cuda_nvfp4_fp4mulmat_log_enabled();

static __host__ __device__ __forceinline__ float ggml_cuda_nvfp4_fp4mulmat_bf16_trunc_f32(float x) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    v.u &= 0xFFFF0000u;
    return v.f;
}

static constexpr unsigned GGML_CUDA_NVFP4_FP4MULMAT_EXP_BITS = 9;
static constexpr unsigned GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS = 23;

using ggml_cuda_nvfp4_uint128_t = unsigned __int128;

static __device__ __forceinline__ uint64_t ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(unsigned width) {
    return (width == 64) ? ~0ULL : ((1ULL << width) - 1ULL);
}

static __device__ __forceinline__ ggml_cuda_nvfp4_uint128_t ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(unsigned width) {
    return width == 0 ? 0 : (((ggml_cuda_nvfp4_uint128_t) 1 << width) - 1);
}

static __device__ __forceinline__ int64_t ggml_cuda_nvfp4_fp4mulmat_sign_extend(uint32_t value, unsigned width) {
    const unsigned shift = 32 - width;
    const int32_t extended = ((int32_t) (value << shift)) >> shift;
    return (int64_t) extended;
}

static __device__ __forceinline__ int64_t ggml_cuda_nvfp4_fp4mulmat_sign_extend_u128(
        ggml_cuda_nvfp4_uint128_t value,
        unsigned width) {
    ggml_cuda_nvfp4_uint128_t bits = value & ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(width);
    const ggml_cuda_nvfp4_uint128_t sign_bit = (ggml_cuda_nvfp4_uint128_t) 1 << (width - 1);
    __int128 signed_value = (__int128) bits;
    if (bits & sign_bit) {
        signed_value -= (__int128) 1 << width;
    }
    return (int64_t) signed_value;
}

static __device__ __forceinline__ ggml_cuda_nvfp4_uint128_t ggml_cuda_nvfp4_fp4mulmat_signed_arith_shift_bits(
        ggml_cuda_nvfp4_uint128_t value_bits,
        unsigned width,
        unsigned shift) {
    value_bits &= ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(width);
    const bool negative = (value_bits & ((ggml_cuda_nvfp4_uint128_t) 1 << (width - 1))) != 0;
    if (shift >= width) {
        return negative ? ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(width) : 0;
    }

    ggml_cuda_nvfp4_uint128_t shifted = value_bits >> shift;
    if (negative && shift > 0) {
        shifted |= ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(shift) << (width - shift);
    }
    return shifted & ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(width);
}

static __device__ __forceinline__ int64_t ggml_cuda_nvfp4_fp4mulmat_signed_rnd_shift(
        int64_t value,
        unsigned shift,
        unsigned width) {
    const ggml_cuda_nvfp4_uint128_t value_bits = (ggml_cuda_nvfp4_uint128_t) value & ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(width);
    const ggml_cuda_nvfp4_uint128_t shifted_bits = ggml_cuda_nvfp4_fp4mulmat_signed_arith_shift_bits(value_bits, width, shift);

    bool guard = false;
    bool lsb = false;
    bool sticky = false;
    if (shift > 0) {
        if (shift < width) {
            guard = ((value_bits >> (shift - 1)) & 1) != 0;
            lsb = (shifted_bits & 1) != 0;
            sticky = (value_bits & ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(shift - 1)) != 0;
        } else {
            const bool negative = (value_bits & ((ggml_cuda_nvfp4_uint128_t) 1 << (width - 1))) != 0;
            guard = negative;
            lsb = negative;
            sticky = (value_bits & ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(width - 1)) != 0;
        }
    }

    const bool round_enable = guard && (sticky || lsb);
    const ggml_cuda_nvfp4_uint128_t result_bits =
            (shifted_bits + (round_enable ? 1 : 0)) & ggml_cuda_nvfp4_fp4mulmat_bitmask_u128(width);
    return ggml_cuda_nvfp4_fp4mulmat_sign_extend_u128(result_bits, width);
}

struct ggml_cuda_nvfp4_fp4mulmat_accumulator {
    uint16_t exp;
    int64_t mat;
};

static __device__ __forceinline__ void ggml_cuda_nvfp4_fp4mulmat_fp_add(
        uint16_t exp1,
        int64_t mat1,
        uint16_t exp2,
        int64_t mat2,
        uint16_t * result_exp,
        int64_t * result_mat) {
    if (exp1 > exp2) {
        mat2 = ggml_cuda_nvfp4_fp4mulmat_signed_rnd_shift(mat2, exp1 - exp2, GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);
        *result_exp = exp1;
    } else {
        mat1 = ggml_cuda_nvfp4_fp4mulmat_signed_rnd_shift(mat1, exp2 - exp1, GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);
        *result_exp = exp2;
    }

    mat1 = ggml_cuda_nvfp4_fp4mulmat_sign_extend(
            mat1 & ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS),
            GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);
    mat2 = ggml_cuda_nvfp4_fp4mulmat_sign_extend(
            mat2 & ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS),
            GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);

    int64_t sum = mat1 + mat2;
    sum = ggml_cuda_nvfp4_fp4mulmat_sign_extend(
            sum & ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS + 1),
            GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS + 1);
    const int msb = (sum >> GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS) & 0x1;
    const int sec_msb = (sum >> (GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS - 1)) & 0x1;
    const int overflow = (msb != sec_msb);

    if (overflow) {
        *result_mat = (sum >> 1) & ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);
        (*result_exp)++;
    } else {
        *result_mat = sum & ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);
    }
    *result_mat = ggml_cuda_nvfp4_fp4mulmat_sign_extend(*result_mat, GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);
}

struct ggml_cuda_nvfp4_fp4mulmat_act_denorm {
    uint8_t sign[QK_NVFP4];
    uint8_t exp;
    int64_t mat_pos1[QK_NVFP4];
    int64_t mat_neg1[QK_NVFP4];
    int64_t mat_pos2[QK_NVFP4];
    int64_t mat_neg2[QK_NVFP4];
    int64_t mat_pos3[QK_NVFP4];
    int64_t mat_neg3[QK_NVFP4];
};

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_fp4mulmat_nibble(const block_nvfp4 & block, int i) {
    const uint8_t packed = block.qs[i >> 1];
    return (i & 1) ? (packed >> 4) : (packed & 0x0F);
}

static __device__ void ggml_cuda_nvfp4_fp4mulmat_compute_act_denorm(
        const uint8_t act[QK_NVFP4],
        ggml_cuda_nvfp4_fp4mulmat_act_denorm * out) {
    uint8_t exp_max = 0;
    for (int i = 0; i < QK_NVFP4; ++i) {
        const uint8_t exp = (act[i] >> 1) & 0x3;
        if (exp > exp_max) {
            exp_max = exp;
        }
    }
    out->exp = exp_max;

    for (int i = 0; i < QK_NVFP4; ++i) {
        const uint8_t fp4 = act[i];
        const uint8_t sign = (fp4 >> 3) & 0x1;
        const uint8_t exp = (fp4 >> 1) & 0x3;
        const uint8_t mat_raw = fp4 & 0x1;

        const uint8_t mat = (exp != 0) ? ((1 << 1) | mat_raw) : (mat_raw << 1);
        const uint8_t shift_amt = exp_max - exp;
        const uint32_t mat_shifted = ((uint32_t) mat << 3) >> shift_amt;

        const uint32_t mat_mul1 = mat_shifted;
        const uint32_t mat_mul2 = mat_shifted * 2;
        const uint32_t mat_mul3 = mat_shifted * 3;

        const uint32_t mat_pos1 = mat_mul1 & 0x7F;
        const uint32_t mat_pos2 = mat_mul2 & 0x7F;
        const uint32_t mat_pos3 = mat_mul3 & 0x7F;

        const uint32_t mat_neg1 = (~mat_mul1 + 1) & 0xFF;
        const uint32_t mat_neg2 = (~mat_mul2 + 1) & 0xFF;
        const uint32_t mat_neg3 = (~mat_mul3 + 1) & 0xFF;

        out->sign[i] = sign;
        out->mat_pos1[i] = ggml_cuda_nvfp4_fp4mulmat_sign_extend(mat_pos1, 8);
        out->mat_neg1[i] = ggml_cuda_nvfp4_fp4mulmat_sign_extend(mat_neg1, 8);
        out->mat_pos2[i] = ggml_cuda_nvfp4_fp4mulmat_sign_extend(mat_pos2, 8);
        out->mat_neg2[i] = ggml_cuda_nvfp4_fp4mulmat_sign_extend(mat_neg2, 8);
        out->mat_pos3[i] = ggml_cuda_nvfp4_fp4mulmat_sign_extend(mat_pos3, 8);
        out->mat_neg3[i] = ggml_cuda_nvfp4_fp4mulmat_sign_extend(mat_neg3, 8);
    }
}

static __device__ void ggml_cuda_nvfp4_fp4mulmat_compute_product(
        const ggml_cuda_nvfp4_fp4mulmat_act_denorm & act,
        const uint8_t wgt[QK_NVFP4],
        uint8_t * product_exp,
        int64_t * product_mat) {
    int64_t psum[QK_NVFP4];

    for (int i = 0; i < QK_NVFP4; ++i) {
        const uint8_t w = wgt[i];
        const uint8_t act_sign = act.sign[i];
        const uint8_t wgt_sign = (w >> 3) & 0x1;
        const uint8_t wgt_exp = (w >> 1) & 0x3;

        uint8_t mat;
        if (wgt_exp == 0) {
            mat = (w & 0x1) << 1;
        } else {
            mat = 2 | (w & 0x1);
        }

        const uint8_t product_sign = wgt_sign ^ act_sign;
        int64_t fp4_mat_mul;
        switch (mat) {
            case 0: fp4_mat_mul = 0; break;
            case 1: fp4_mat_mul = product_sign ? act.mat_neg1[i] : act.mat_pos1[i]; break;
            case 2: fp4_mat_mul = product_sign ? act.mat_neg2[i] : act.mat_pos2[i]; break;
            case 3: fp4_mat_mul = product_sign ? act.mat_neg3[i] : act.mat_pos3[i]; break;
            default: fp4_mat_mul = 0; break;
        }

        psum[i] = ggml_cuda_nvfp4_fp4mulmat_sign_extend((uint32_t) fp4_mat_mul, 8) << wgt_exp;
    }

    int64_t sum = 0;
    for (int i = 0; i < QK_NVFP4; ++i) {
        sum += psum[i];
    }

    *product_mat = ggml_cuda_nvfp4_fp4mulmat_sign_extend((uint32_t) sum, 15);
    *product_exp = act.exp;
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_fp4mulmat_psum_accumulate(
        uint8_t product_exp,
        int64_t product_mat,
        uint8_t scale_act,
        uint8_t scale_wgt,
        ggml_cuda_nvfp4_fp4mulmat_accumulator * state) {
    const uint8_t scale_act_exp = (scale_act >> 3) & 0xF;
    const uint8_t scale_act_mat = scale_act_exp != 0 ? (1 << 3 | (scale_act & 0x7)) : (scale_act & 0x7) << 1;
    const uint8_t scale_wgt_exp = (scale_wgt >> 3) & 0xF;
    const uint8_t scale_wgt_mat = scale_wgt_exp != 0 ? (1 << 3 | (scale_wgt & 0x7)) : (scale_wgt & 0x7) << 1;

    const uint16_t prod_exp = (product_exp + scale_act_exp + scale_wgt_exp) &
            ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(GGML_CUDA_NVFP4_FP4MULMAT_EXP_BITS);
    const int64_t product_mat_tmp = ggml_cuda_nvfp4_fp4mulmat_sign_extend(product_mat, 15) *
            ggml_cuda_nvfp4_fp4mulmat_sign_extend(scale_wgt_mat, 5) *
            ggml_cuda_nvfp4_fp4mulmat_sign_extend(scale_act_mat, 5);
    const int64_t prod_mat = ggml_cuda_nvfp4_fp4mulmat_sign_extend(
            product_mat_tmp & ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS),
            GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);

    uint16_t acc_exp;
    int64_t acc_mat;
    ggml_cuda_nvfp4_fp4mulmat_fp_add(prod_exp, prod_mat, state->exp, state->mat, &acc_exp, &acc_mat);
    state->exp = acc_exp;
    state->mat = acc_mat;
}

static __device__ __forceinline__ float ggml_cuda_nvfp4_fp4mulmat_accumulator_to_f32(
        ggml_cuda_nvfp4_fp4mulmat_accumulator state) {
    const uint64_t mat_bits = (uint64_t) state.mat & ggml_cuda_nvfp4_fp4mulmat_bitmask_u64(GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);
    const int64_t mat = ggml_cuda_nvfp4_fp4mulmat_sign_extend((uint32_t) mat_bits, GGML_CUDA_NVFP4_FP4MULMAT_ACC_BITS);
    return ldexpf((float) mat, (int) state.exp - 27);
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_fp4mulmat_accumulate_block(
        const block_nvfp4 & w,
        const block_nvfp4 & act,
        ggml_cuda_nvfp4_fp4mulmat_accumulator * state) {
    uint8_t act_nibbles[QK_NVFP4];
    uint8_t wgt_nibbles[QK_NVFP4];

#pragma unroll
    for (int i = 0; i < QK_NVFP4; ++i) {
        act_nibbles[i] = ggml_cuda_nvfp4_fp4mulmat_nibble(act, i);
        wgt_nibbles[i] = ggml_cuda_nvfp4_fp4mulmat_nibble(w, i);
    }

    ggml_cuda_nvfp4_fp4mulmat_act_denorm act_denorm;
    ggml_cuda_nvfp4_fp4mulmat_compute_act_denorm(act_nibbles, &act_denorm);

    uint8_t product_exp;
    int64_t product_mat;
    ggml_cuda_nvfp4_fp4mulmat_compute_product(act_denorm, wgt_nibbles, &product_exp, &product_mat);
    ggml_cuda_nvfp4_fp4mulmat_psum_accumulate(product_exp, product_mat, act.e, w.e, state);
}

static __device__ __forceinline__ float ggml_cuda_nvfp4_fp4mulmat_block_dot_f32(
        const block_nvfp4 & w,
        const block_nvfp4 & act) {
    ggml_cuda_nvfp4_fp4mulmat_accumulator state = { 0, 0 };
    ggml_cuda_nvfp4_fp4mulmat_accumulate_block(w, act, &state);
    return ggml_cuda_nvfp4_fp4mulmat_accumulator_to_f32(state);
}

void ggml_cuda_nvfp4_fp4mulmat_cuda(
        const block_nvfp4 * src0,
        const block_nvfp4 * src1_q,
        const float * dynamic_input_scales,
        void * dst,
        int64_t ne01,
        int64_t ne11,
        int64_t nblk_k,
        int64_t dst_nb0,
        int64_t dst_nb1,
        float static_scale,
        bool used_dynamic_scale,
        cudaStream_t stream);

void ggml_cuda_nvfp4_fp4mulmat_vcache_cuda(
        const block_nvfp4 * v_data,
        const float * v_scale,
        const block_nvfp4 * p_q,
        const float * p_scale,
        float * dst_data,
        int64_t kv_size,
        int64_t rows,
        int64_t cols,
        int64_t kv_heads,
        int64_t q_heads,
        int64_t kv_streams,
        int64_t q_streams,
        int64_t v_nb0,
        int64_t v_nb1,
        int64_t v_nb2,
        int64_t v_nb3,
        int64_t scale_nb0,
        int64_t scale_row_nb,
        int64_t scale_head_nb,
        int64_t scale_stream_nb,
        bool scale_is_global,
        int64_t dst_nb1,
        int64_t dst_nb2,
        int64_t dst_nb3,
        int64_t r2,
        int64_t r3,
        cudaStream_t stream);
