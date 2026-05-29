#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-quants.h"
#include "../ggml/src/ggml-cuda/expt/nvfp4/nvfp4-quantize.cuh"

#include <cuda_runtime.h>
#include <cublas_api.h>
#include <cublasLt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do {                                                                  \
    cudaError_t err__ = (call);                                                                \
    if (err__ != cudaSuccess) {                                                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1);                                                                          \
    }                                                                                          \
} while (0)

#define CUBLASLT_CHECK(call) do {                                                              \
    cublasStatus_t st__ = (call);                                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                                       \
        std::fprintf(stderr, "cuBLASLt error %s:%d: status=%d\n", __FILE__, __LINE__, (int) st__); \
        std::exit(1);                                                                          \
    }                                                                                          \
} while (0)

static inline int64_t pad_i64(int64_t x, int64_t a) {
    return ((x + a - 1) / a) * a;
}

static inline int64_t scale_tiled_index(int64_t outer, int64_t inner, int64_t n_inner_padded) {
    const int64_t outer_tile = outer / 128;
    const int64_t outer_in_tile = outer % 128;
    const int64_t inner_tile = inner / 4;
    const int64_t inner_in_tile = inner % 4;

    const int64_t tiles_per_outer_block = n_inner_padded / 4;
    const int64_t tile_base = (outer_tile * tiles_per_outer_block + inner_tile) * 512;
    const int64_t tile_offset = (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile;
    return tile_base + tile_offset;
}

static void quantize_matrix_nvfp4(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int rows,
        int k,
        float global_scale) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nblk_k = k / QK_NVFP4;
    dst.resize((size_t) rows * (size_t) nblk_k);

    for (int r = 0; r < rows; ++r) {
        quantize_row_nvfp4_ref(
                src.data() + (size_t) r * (size_t) k,
                dst.data() + (size_t) r * (size_t) nblk_k,
                k,
                global_scale);
    }
}

static void dequantize_matrix_nvfp4(
        const std::vector<block_nvfp4> & src,
        std::vector<float> & dst,
        int rows,
        int k,
        float global_scale) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nblk_k = k / QK_NVFP4;
    dst.resize((size_t) rows * (size_t) k);

    for (int r = 0; r < rows; ++r) {
        dequantize_row_nvfp4(
                src.data() + (size_t) r * (size_t) nblk_k,
                dst.data() + (size_t) r * (size_t) k,
                k,
                global_scale);
    }
}

static void fp32_reference_matmul(
        const std::vector<float> & a_deq,
        const std::vector<float> & b_deq,
        std::vector<float> & c_ref,
        int m,
        int n,
        int k) {
    c_ref.assign((size_t) m * (size_t) n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int x = 0; x < k; ++x) {
                acc += a_deq[(size_t) i * (size_t) k + (size_t) x] *
                       b_deq[(size_t) j * (size_t) k + (size_t) x];
            }
            c_ref[(size_t) i * (size_t) n + (size_t) j] = acc;
        }
    }
}

static float compute_dynamic_global_scale(const std::vector<float> & src) {
    float amax = 0.0f;
    for (float v : src) {
        amax = fmaxf(amax, fabsf(v));
    }

    return amax > 0.0f ? (6.0f * 224.0f) / amax : 0.0f;
}

static void compute_dynamic_global_scales_per_row(
        const std::vector<float> & src,
        int rows,
        int k,
        std::vector<float> & global_scales) {
    global_scales.resize((size_t) rows);
    for (int r = 0; r < rows; ++r) {
        const float * row = src.data() + (size_t) r * (size_t) k;
        float amax = 0.0f;
        for (int i = 0; i < k; ++i) {
            amax = fmaxf(amax, fabsf(row[i]));
        }
        global_scales[(size_t) r] = amax > 0.0f ? (6.0f * 224.0f) / amax : 0.0f;
    }
}

static void quantize_matrix_nvfp4_per_row_scale(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int rows,
        int k,
        const std::vector<float> & global_scales) {
    GGML_ASSERT((int) global_scales.size() == rows);
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nblk_k = k / QK_NVFP4;
    dst.resize((size_t) rows * (size_t) nblk_k);

    for (int r = 0; r < rows; ++r) {
        quantize_row_nvfp4_ref(
                src.data() + (size_t) r * (size_t) k,
                dst.data() + (size_t) r * (size_t) nblk_k,
                k,
                global_scales[(size_t) r]);
    }
}

static void dequantize_matrix_nvfp4_per_row_scale(
        const std::vector<block_nvfp4> & src,
        std::vector<float> & dst,
        int rows,
        int k,
        const std::vector<float> & global_scales) {
    GGML_ASSERT((int) global_scales.size() == rows);
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nblk_k = k / QK_NVFP4;
    dst.resize((size_t) rows * (size_t) k);

    for (int r = 0; r < rows; ++r) {
        dequantize_row_nvfp4(
                src.data() + (size_t) r * (size_t) nblk_k,
                dst.data() + (size_t) r * (size_t) k,
                k,
                global_scales[(size_t) r]);
    }
}

static uint64_t host_low_bits_mask_u64(uint8_t width) {
    if (width >= 64u) {
        return ~0ull;
    }
    return width == 0u ? 0ull : ((1ull << width) - 1ull);
}

static uint16_t host_fp32_to_bf16_bits(float x) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = x;
    const uint32_t sign = (bits.u >> 31) & 0x1u;
    uint32_t exp = (bits.u >> 23) & 0xffu;
    uint32_t mant = bits.u & 0x007fffffu;

    if (exp == 0xffu) {
        if (mant != 0u) {
            uint16_t bf16_bits = (uint16_t) ((sign << 15) | 0x7f80u | ((mant >> 16) & 0x3fu));
            if ((bf16_bits & 0x7fu) == 0u) {
                bf16_bits |= 0x0040u;
            }
            return bf16_bits;
        }
        return (uint16_t) ((sign << 15) | 0x7f80u);
    }

    if (exp == 0u && mant == 0u) {
        return (uint16_t) (sign << 15);
    }

    if (exp == 0u) {
        uint32_t shift = 0;
        while ((mant & (1u << 22)) == 0u && shift < 22u) {
            mant <<= 1;
            shift++;
        }
        exp = 1u - shift;
        mant &= 0x007fffffu;
    } else {
        mant |= 0x00800000u;
    }

    const uint32_t guard_bit = (mant >> 15) & 0x1u;
    const uint32_t round_bit = (mant >> 14) & 0x1u;
    const uint32_t sticky = (mant & 0x3fffu) != 0u ? 1u : 0u;
    uint32_t bf16_mant = (mant >> 16) & 0x7fu;

    if (guard_bit == 1u && (round_bit == 1u || sticky == 1u || (bf16_mant & 0x1u) == 1u)) {
        bf16_mant += 1u;
    }

    if (bf16_mant > 0x7fu) {
        bf16_mant = 0u;
        exp += 1u;
        if (exp > 0xfeu) {
            return (uint16_t) ((sign << 15) | 0x7f80u);
        }
    }

    if (exp < 1u) {
        return (uint16_t) (sign << 15);
    }

    return (uint16_t) ((sign << 15) | ((exp & 0xffu) << 7) | (bf16_mant & 0x7fu));
}

static uint16_t host_bf16_abs_bits(uint16_t x) {
    return (uint16_t) (x & (uint16_t) host_low_bits_mask_u64(15u));
}

static uint64_t host_shift_hw(uint64_t value, uint8_t shift_amt, uint8_t shift_right) {
    const uint64_t value_q = value & host_low_bits_mask_u64(32u);
    return ((shift_right & 1u) == 0u)
            ? ((value_q << shift_amt) & host_low_bits_mask_u64(36u))
            : ((value_q >> shift_amt) & host_low_bits_mask_u64(36u));
}

static uint32_t host_float_to_ufixed_q_hw(float val, uint8_t frac_bits) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = val;

    const uint32_t sign = bits.u >> 31;
    const uint32_t exponent = (bits.u >> 23) & (uint32_t) host_low_bits_mask_u64(8u);
    const uint32_t mantissa = bits.u & (uint32_t) host_low_bits_mask_u64(23u);
    if (sign != 0u || (exponent == 0u && mantissa == 0u)) {
        return 0;
    }
    if (exponent == 0xffu) {
        return mantissa == 0u ? (uint32_t) host_low_bits_mask_u64(32u) : 0u;
    }

    const uint32_t significand =
            ((exponent == 0u) ? mantissa : ((1u << 23) | mantissa)) &
            (uint32_t) host_low_bits_mask_u64(24u);
    const uint32_t exponent_unbiased =
            (exponent == 0u) ? 0x82u : ((exponent - 127u) & (uint32_t) host_low_bits_mask_u64(8u));
    uint32_t exponent_unbiased_ext = exponent_unbiased;
    if ((exponent_unbiased_ext & 0x80u) != 0u) {
        exponent_unbiased_ext |= 0x100u;
    }

    const uint32_t total_shift =
            (exponent_unbiased_ext + (uint32_t) frac_bits + 0x1e9u) &
            (uint32_t) host_low_bits_mask_u64(9u);
    const uint8_t shift_right = (uint8_t) ((total_shift >> 8) & 1u);
    uint32_t shift_mag = total_shift;
    if (shift_right != 0u) {
        shift_mag = ((~shift_mag) + 1u) & (uint32_t) host_low_bits_mask_u64(9u);
    }
    return (uint32_t) (host_shift_hw(significand, (uint8_t) (shift_mag & 0xffu), shift_right) &
                       host_low_bits_mask_u64(32u));
}

static uint64_t host_bf16_abs_mul_uq_hw(uint16_t abs_bits, uint32_t factor_q, int factor_frac_bits, int out_frac_bits) {
    if (abs_bits == 0u || factor_q == 0u) {
        return 0;
    }

    const uint32_t exponent = (abs_bits >> 7) & (uint32_t) host_low_bits_mask_u64(8u);
    const uint32_t mantissa = abs_bits & (uint32_t) host_low_bits_mask_u64(7u);
    const uint32_t significand =
            ((exponent == 0u) ? mantissa : (0x80u | mantissa)) &
            (uint32_t) host_low_bits_mask_u64(8u);
    const uint32_t exp_mask = (uint32_t) host_low_bits_mask_u64(9u);
    const uint32_t exp_sign = 1u << 8;
    const uint32_t value_exp2_tc = (exponent == 0u) ? ((0u - 133u) & exp_mask) : ((exponent - 134u) & exp_mask);
    const uint32_t value_exp2_ext = (value_exp2_tc & exp_sign) != 0u ? (value_exp2_tc | ~exp_mask) : value_exp2_tc;
    const uint32_t frac_delta = (uint32_t) (out_frac_bits - factor_frac_bits) & (uint32_t) host_low_bits_mask_u64(9u);
    const uint32_t total_shift = (value_exp2_ext + frac_delta) & (uint32_t) host_low_bits_mask_u64(9u);
    const uint8_t shift_right = (uint8_t) ((total_shift >> 8) & 1u);
    uint32_t shift_mag = total_shift;
    if (shift_right != 0u) {
        shift_mag = ((~shift_mag) + 1u) & (uint32_t) host_low_bits_mask_u64(9u);
    }
    const uint64_t product = (uint64_t) significand * factor_q;
    uint64_t result = 0;
    if ((shift_mag & 0xffu) < 64u) {
        result = shift_right != 0u ? (product >> (shift_mag & 0xffu)) : (product << (shift_mag & 0xffu));
    }
    return result & host_low_bits_mask_u64(36u);
}

static uint8_t host_block_scale_msb_hw(uint64_t block_scale_q) {
    uint64_t msb_probe = block_scale_q & host_low_bits_mask_u64(34u);
    uint8_t msb = 0u;
    if (msb_probe >= (1ull << 32)) { msb_probe >>= 32; msb = (uint8_t) ((msb + 32u) & 0x3fu); }
    if (msb_probe >= (1ull << 16)) { msb_probe >>= 16; msb = (uint8_t) ((msb + 16u) & 0x3fu); }
    if (msb_probe >= (1ull << 8))  { msb_probe >>= 8;  msb = (uint8_t) ((msb + 8u)  & 0x3fu); }
    if (msb_probe >= (1ull << 4))  { msb_probe >>= 4;  msb = (uint8_t) ((msb + 4u)  & 0x3fu); }
    if (msb_probe >= (1ull << 2))  { msb_probe >>= 2;  msb = (uint8_t) ((msb + 2u)  & 0x3fu); }
    if (msb_probe >= (1ull << 1))  { msb = (uint8_t) ((msb + 1u) & 0x3fu); }
    return (uint8_t) (msb & 0x3fu);
}

static uint8_t host_compute_block_scale_hw(uint16_t block_abs_max_bits, uint32_t global_scale_q) {
    if (block_abs_max_bits == 0u) {
        return 0u;
    }

    uint64_t block_scale_q = host_bf16_abs_mul_uq_hw(block_abs_max_bits, global_scale_q, 16, 24);
    block_scale_q = ((block_scale_q + 3u) >> 3) + ((block_scale_q + 3u) >> 5) +
                    ((block_scale_q + 3u) >> 7) + ((block_scale_q + 3u) >> 9) +
                    ((block_scale_q + 3u) >> 11) + ((block_scale_q + 3u) >> 13);
    block_scale_q &= host_low_bits_mask_u64(34u);

    const uint8_t msb = host_block_scale_msb_hw(block_scale_q);
    const uint8_t exp_field_tc = (uint8_t) ((msb - 24 + 7) & 0x3fu);
    int32_t exp_field = ((uint32_t) exp_field_tc & 0x20u) != 0u ? (int32_t) ((uint32_t) exp_field_tc | ~0x3fu) : (int32_t) exp_field_tc;
    if (exp_field <= 0) {
        uint64_t mant_q = block_scale_q >> (24 - 9);
        mant_q &= 0xffu;
        return mant_q >= 8u ? 0x08u : (uint8_t) mant_q;
    }

    if (exp_field > 15) {
        exp_field = 15;
    }
    const int rshift = 24 + exp_field - 10;
    uint32_t signif_q_rounded = (uint32_t) (block_scale_q & host_low_bits_mask_u64(5u));
    if (rshift > 0) {
        const uint64_t shifted = (block_scale_q >> rshift) & host_low_bits_mask_u64(19u);
        const uint64_t half = (1ull << (rshift - 1)) & host_low_bits_mask_u64(29u);
        const uint64_t mask = host_low_bits_mask_u64((uint8_t) rshift) & host_low_bits_mask_u64(29u);
        const uint64_t remainder = block_scale_q & mask;
        const uint64_t round = remainder > half ? 1ull : 0ull;
        signif_q_rounded = (uint32_t) ((shifted + round) & host_low_bits_mask_u64(5u));
    }
    const uint8_t carry = ((exp_field < 15) && ((signif_q_rounded & (1u << 4)) != 0u)) ? 1u : 0u;
    const int32_t exp_field_norm = (exp_field + (int32_t) carry) & 0xf;
    const uint32_t signif_q_norm = carry != 0u ? 8u : signif_q_rounded;
    const uint32_t signif_q_floor = signif_q_norm < 8u ? 8u : signif_q_norm;
    const uint32_t signif_q_clamped = exp_field_norm >= 15 ? (signif_q_floor > 14u ? 14u : signif_q_floor) : signif_q_floor;
    return (uint8_t) ((exp_field_norm << 3) | ((signif_q_clamped - 8u) & 0x7u));
}

static uint64_t host_compute_block_scale_half_q(uint8_t scale) {
    const uint32_t scale_exp = (scale >> 3) & 0xfu;
    const uint32_t scale_mant = scale & 0x7u;
    return scale_exp == 0u
            ? (host_shift_hw(scale_mant, (uint8_t) ((24 - 10) & 0xffu), 0u) & 0xffffffffu)
            : (host_shift_hw(8u + scale_mant, (uint8_t) ((24 + (int) scale_exp - 11) & 0xffu), 0u) & 0xffffffffu);
}

static void host_quantize_bf16_round_nvfp4(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int rows,
        int k,
        const std::vector<float> & global_scales) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    GGML_ASSERT((int) global_scales.size() == rows);
    const int nblk = k / QK_NVFP4;
    dst.assign((size_t) rows * (size_t) nblk, {});
    std::vector<uint16_t> bf16((size_t) rows * (size_t) k);
    for (size_t i = 0; i < bf16.size(); ++i) {
        bf16[i] = host_fp32_to_bf16_bits(src[i]);
    }

    for (int r = 0; r < rows; ++r) {
        const uint32_t global_scale_q = host_float_to_ufixed_q_hw(global_scales[(size_t) r], 16);
        for (int ib = 0; ib < nblk; ++ib) {
            uint16_t block_abs_max = 0;
            for (int j = 0; j < QK_NVFP4; ++j) {
                block_abs_max = std::max(block_abs_max, host_bf16_abs_bits(bf16[(size_t) r * (size_t) k + (size_t) ib * QK_NVFP4 + j]));
            }
            const uint8_t scale = host_compute_block_scale_hw(block_abs_max, global_scale_q);
            block_nvfp4 & out = dst[(size_t) r * (size_t) nblk + (size_t) ib];
            out.e = scale;
            if (scale == 0u) {
                std::memset(out.qs, 0, sizeof(out.qs));
                continue;
            }

            const uint64_t block_scale_half_q = host_compute_block_scale_half_q(scale);
            uint8_t q_raw[QK_NVFP4] = { 0 };
            for (int j = 0; j < QK_NVFP4; ++j) {
                const uint16_t bits = bf16[(size_t) r * (size_t) k + (size_t) ib * QK_NVFP4 + j];
                const uint8_t sign = (uint8_t) ((bits >> 15) & 1u);
                const uint64_t target_q = host_bf16_abs_mul_uq_hw(host_bf16_abs_bits(bits), global_scale_q, 16, 24) &
                                          host_low_bits_mask_u64(36u);
                const uint64_t target_2x_q = target_q << 1;
                uint8_t best_mag = 0u;
                if (target_2x_q < block_scale_half_q) {
                    best_mag = 0u;
                } else if (target_2x_q < 3ull * block_scale_half_q) {
                    best_mag = 1u;
                } else if (target_2x_q < 5ull * block_scale_half_q) {
                    best_mag = 2u;
                } else if (target_2x_q < 7ull * block_scale_half_q) {
                    best_mag = 3u;
                } else if (target_2x_q < 10ull * block_scale_half_q) {
                    best_mag = 4u;
                } else if (target_2x_q < 14ull * block_scale_half_q) {
                    best_mag = 5u;
                } else if (target_2x_q < 20ull * block_scale_half_q) {
                    best_mag = 6u;
                } else {
                    best_mag = 7u;
                }
                q_raw[j] = best_mag == 0u ? 0u : (uint8_t) ((sign << 3) | best_mag);
            }
            for (int j = 0; j < QK_NVFP4 / 2; ++j) {
                out.qs[j] = (uint8_t) ((q_raw[2*j + 1] << 4) | (q_raw[2*j] & 0x0f));
            }
        }
    }
}

static bool test_bf16_round_quant_enabled() {
    const char * env = std::getenv("GGML_CUDA_NVFP4_BF16_QUANT");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static void quantize_matrix_nvfp4_dynamic_ref(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int rows,
        int k,
        const std::vector<float> & global_scales) {
    if (test_bf16_round_quant_enabled()) {
        host_quantize_bf16_round_nvfp4(src, dst, rows, k, global_scales);
    } else {
        quantize_matrix_nvfp4_per_row_scale(src, dst, rows, k, global_scales);
    }
}

static void quantize_matrix_nvfp4_global_ref(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int rows,
        int k,
        float global_scale) {
    if (test_bf16_round_quant_enabled()) {
        host_quantize_bf16_round_nvfp4(src, dst, rows, k, std::vector<float>((size_t) rows, global_scale));
    } else {
        quantize_matrix_nvfp4(src, dst, rows, k, global_scale);
    }
}

static bool run_case_bf16_round_quantizer_bytes() {
    const int rows = 3;
    const int k = 32;
    std::vector<float> src((size_t) rows * (size_t) k);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < k; ++i) {
            const float base = (float) ((i % 17) - 8) * (0.1375f + 0.025f * r);
            src[(size_t) r * (size_t) k + (size_t) i] = (i & 1) ? -base : base;
        }
    }
    src[5] = 0.33325195f;
    src[19] = -1.8759766f;
    src[(size_t) k + 7] = 8.03125f;
    src[(size_t) 2 * k + 11] = -0.00024420023f;
    src[(size_t) 2 * k + 12] = 1.0e-39f;
    src[(size_t) 2 * k + 13] = -1.0e-39f;

    const std::vector<float> global_scales = { 13.5f, 47.25f, 127.0f };
    std::vector<block_nvfp4> expected;
    host_quantize_bf16_round_nvfp4(src, expected, rows, k, global_scales);

    float * d_src = nullptr;
    float * d_scales = nullptr;
    block_nvfp4 * d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, global_scales.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, expected.size() * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, global_scales.data(), global_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, expected.size() * sizeof(block_nvfp4)));

    ggml_cuda_nvfp4_quantize_rows_bf16_f32(d_src, d_dst, k, k, rows, d_scales, false, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<block_nvfp4> got(expected.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, got.size() * sizeof(block_nvfp4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_src));

    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].e != got[i].e || std::memcmp(expected[i].qs, got[i].qs, sizeof(expected[i].qs)) != 0) {
            std::fprintf(stderr, "bf16-round quant mismatch block=%zu expected_e=%u got_e=%u\n",
                    i, (unsigned) expected[i].e, (unsigned) got[i].e);
            ok = false;
        }
    }
    std::printf("bf16-round quantizer bytes | %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static void split_nvfp4_blocks(
        const std::vector<block_nvfp4> & src,
        int64_t k,
        int64_t n_outer_valid,
        int64_t n_outer_alloc,
        bool linear_scale_layout,
        std::vector<uint8_t> & out_data,
        std::vector<uint8_t> & out_scale,
        int64_t & out_scale_inner_padded,
        int64_t & out_scale_outer_padded) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int64_t nblk_k = k / QK_NVFP4;
    const int64_t row_data_bytes = k / 2;
    const int64_t inner_padded = pad_i64(nblk_k, 4);
    const int64_t outer_padded = pad_i64(n_outer_alloc, 128);

    out_data.assign((size_t) n_outer_alloc * (size_t) row_data_bytes, 0);
    out_scale.assign((size_t) inner_padded * (size_t) outer_padded, 0);

    for (int64_t outer = 0; outer < n_outer_valid; ++outer) {
        for (int64_t inner = 0; inner < nblk_k; ++inner) {
            const block_nvfp4 & b = src[(size_t) outer * (size_t) nblk_k + (size_t) inner];

            uint8_t * data_dst = out_data.data() + (size_t) outer * (size_t) row_data_bytes + (size_t) inner * (QK_NVFP4 / 2);
            std::memcpy(data_dst, b.qs, QK_NVFP4 / 2);

            const int64_t sidx = linear_scale_layout
                    ? (outer * inner_padded + inner)
                    : scale_tiled_index(outer, inner, inner_padded);
            out_scale[(size_t) sidx] = b.e;
        }
    }

    out_scale_inner_padded = inner_padded;
    out_scale_outer_padded = outer_padded;
}

static bool run_case(int m, int n, int k, float global_scale_a, float global_scale_b, uint32_t seed) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((n % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a_fp32((size_t) m * (size_t) k);
    std::vector<float> b_fp32((size_t) n * (size_t) k);
    for (float & v : a_fp32) {
        v = dist(rng);
    }
    for (float & v : b_fp32) {
        v = dist(rng);
    }

    std::vector<block_nvfp4> a_nvfp4;
    std::vector<block_nvfp4> b_nvfp4;
    quantize_matrix_nvfp4(a_fp32, a_nvfp4, m, k, global_scale_a);
    quantize_matrix_nvfp4(b_fp32, b_nvfp4, n, k, global_scale_b);

    std::vector<float> a_deq;
    std::vector<float> b_deq;
    dequantize_matrix_nvfp4(a_nvfp4, a_deq, m, k, global_scale_a);
    dequantize_matrix_nvfp4(b_nvfp4, b_deq, n, k, global_scale_b);

    std::vector<float> c_ref;
    fp32_reference_matmul(a_deq, b_deq, c_ref, m, n, k);
    std::vector<float> c_ref_col_major((size_t) m * (size_t) n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            c_ref_col_major[(size_t) col * (size_t) m + (size_t) row] =
                    c_ref[(size_t) row * (size_t) n + (size_t) col];
        }
    }

    std::vector<uint8_t> a_data;
    std::vector<uint8_t> a_scale;
    std::vector<uint8_t> b_data;
    std::vector<uint8_t> b_scale;
    int64_t a_scale_inner = 0;
    int64_t a_scale_outer = 0;
    int64_t b_scale_inner = 0;
    int64_t b_scale_outer = 0;

    const bool linear_scale_layout = false;
    split_nvfp4_blocks(a_nvfp4, k, m, m, linear_scale_layout, a_data, a_scale, a_scale_inner, a_scale_outer);
    split_nvfp4_blocks(b_nvfp4, k, n, n, linear_scale_layout, b_data, b_scale, b_scale_inner, b_scale_outer);

    uint8_t * d_a_data = nullptr;
    uint8_t * d_b_data = nullptr;
    uint8_t * d_a_scale = nullptr;
    uint8_t * d_b_scale = nullptr;
    float * d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_data, a_data.size()));
    CUDA_CHECK(cudaMalloc(&d_b_data, b_data.size()));
    CUDA_CHECK(cudaMalloc(&d_a_scale, a_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_b_scale, b_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_c, (size_t) m * (size_t) n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a_data, a_data.data(), a_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_data, b_data.data(), b_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_scale, a_scale.data(), a_scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_scale, b_scale.data(), b_scale.size(), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt = nullptr;
    CUBLASLT_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_n = CUBLAS_OP_N;
    const cublasOperation_t op_t = CUBLAS_OP_T;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_t, sizeof(op_t)));

#if defined(CUBLAS_VER_MAJOR) && (CUBLAS_VER_MAJOR >= 13)
    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    const void * a_scale_ptr = d_a_scale;
    const void * b_scale_ptr = d_b_scale;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));
#else
    std::fprintf(stderr, "Skip: cuBLASLt FP4 scale-channel attributes are unavailable in this toolkit.\n");
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));
    return true;
#endif

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) m, (uint64_t) k, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) n, (uint64_t) k, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F,     (uint64_t) m, (uint64_t) n, (int64_t) n));

    const cublasLtOrder_t order_row = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));

    const float alpha = 1.0f / (global_scale_a * global_scale_b);
    const float beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(
            lt,
            op_desc,
            &alpha,
            d_a_data, a_desc,
            d_b_data, b_desc,
            &beta,
            d_c, c_desc,
            d_c, c_desc,
            nullptr,
            nullptr, 0,
            nullptr));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> c_gpu((size_t) m * (size_t) n, 0.0f);
    CUDA_CHECK(cudaMemcpy(c_gpu.data(), d_c, (size_t) m * (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));

    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_i = 0;
    int worst_j = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const float ref = c_ref[(size_t) i * (size_t) n + (size_t) j];
            const float got = c_gpu[(size_t) i * (size_t) n + (size_t) j];
            const float abs_err = std::fabs(got - ref);
            const float rel_err = abs_err / (std::fabs(ref) + 1e-6f);
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                worst_i = i;
                worst_j = j;
            }
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
            }
        }
    }

    const float tol_abs = 1e-2f;
    const float tol_rel = 1e-2f;
    const bool ok = max_abs_err <= tol_abs || max_rel_err <= tol_rel;

    std::printf("case m=%d n=%d k=%d gs_a=%.3f gs_b=%.3f | max_abs=%.6g max_rel=%.6g | %s\n",
            m, n, k, global_scale_a, global_scale_b, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  worst element: (%d, %d), ref=%.8f, gpu=%.8f\n",
                worst_i,
                worst_j,
                c_ref[(size_t) worst_i * (size_t) n + (size_t) worst_j],
                c_gpu[(size_t) worst_i * (size_t) n + (size_t) worst_j]);
    }

    return ok;
}

// Reproduces the descriptor/layout strategy currently used by ggml_cuda_mul_mat_nvfp4_native:
// - column-major default layouts (no ORDER_ROW)
// - TRANSA=T, TRANSB=N
// - optional N padding to multiple-of-16
static bool run_case_integration_style(int m, int n, int k, float global_scale_a, float global_scale_b, uint32_t seed) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a_fp32((size_t) m * (size_t) k);
    std::vector<float> b_fp32((size_t) n * (size_t) k);
    for (float & v : a_fp32) {
        v = dist(rng);
    }
    for (float & v : b_fp32) {
        v = dist(rng);
    }

    std::vector<block_nvfp4> a_nvfp4;
    std::vector<block_nvfp4> b_nvfp4;
    quantize_matrix_nvfp4(a_fp32, a_nvfp4, m, k, global_scale_a);
    quantize_matrix_nvfp4(b_fp32, b_nvfp4, n, k, global_scale_b);

    std::vector<float> a_deq;
    std::vector<float> b_deq;
    dequantize_matrix_nvfp4(a_nvfp4, a_deq, m, k, global_scale_a);
    dequantize_matrix_nvfp4(b_nvfp4, b_deq, n, k, global_scale_b);

    std::vector<float> c_ref;
    fp32_reference_matmul(a_deq, b_deq, c_ref, m, n, k);

    const int n_padded = (int) pad_i64(n, 16);

    std::vector<uint8_t> a_data;
    std::vector<uint8_t> a_scale;
    std::vector<uint8_t> b_data;
    std::vector<uint8_t> b_scale;
    int64_t a_scale_inner = 0;
    int64_t a_scale_outer = 0;
    int64_t b_scale_inner = 0;
    int64_t b_scale_outer = 0;

    const bool linear_scale_layout = false;
    split_nvfp4_blocks(a_nvfp4, k, m, m, linear_scale_layout, a_data, a_scale, a_scale_inner, a_scale_outer);
    split_nvfp4_blocks(b_nvfp4, k, n, n_padded, linear_scale_layout, b_data, b_scale, b_scale_inner, b_scale_outer);

    uint8_t * d_a_data = nullptr;
    uint8_t * d_b_data = nullptr;
    uint8_t * d_a_scale = nullptr;
    uint8_t * d_b_scale = nullptr;
    float * d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_data, a_data.size()));
    CUDA_CHECK(cudaMalloc(&d_b_data, b_data.size()));
    CUDA_CHECK(cudaMalloc(&d_a_scale, a_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_b_scale, b_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_c, (size_t) m * (size_t) n_padded * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a_data, a_data.data(), a_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_data, b_data.data(), b_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_scale, a_scale.data(), a_scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_scale, b_scale.data(), b_scale.size(), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt = nullptr;
    CUBLASLT_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

#if defined(CUBLAS_VER_MAJOR) && (CUBLAS_VER_MAJOR >= 13)
    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    const void * a_scale_ptr = d_a_scale;
    const void * b_scale_ptr = d_b_scale;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));
#else
    std::fprintf(stderr, "Skip: cuBLASLt FP4 scale-channel attributes are unavailable in this toolkit.\n");
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));
    return true;
#endif

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) k, (uint64_t) m, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) k, (uint64_t) n_padded, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F,     (uint64_t) m, (uint64_t) n_padded, (int64_t) m));

    const float alpha = 1.0f / (global_scale_a * global_scale_b);
    const float beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(
            lt,
            op_desc,
            &alpha,
            d_a_data, a_desc,
            d_b_data, b_desc,
            &beta,
            d_c, c_desc,
            d_c, c_desc,
            nullptr,
            nullptr, 0,
            nullptr));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> c_gpu_padded((size_t) m * (size_t) n_padded, 0.0f);
    CUDA_CHECK(cudaMemcpy(
            c_gpu_padded.data(),
            d_c,
            (size_t) m * (size_t) n_padded * sizeof(float),
            cudaMemcpyDeviceToHost));

    // c_desc uses default column-major layout with ld=m. Convert to row-major for comparison.
    std::vector<float> c_gpu((size_t) m * (size_t) n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            c_gpu[(size_t) i * (size_t) n + (size_t) j] =
                    c_gpu_padded[(size_t) j * (size_t) m + (size_t) i];
        }
    }
    std::printf("  integration-style probe e00: ref=%.8f raw0=%.8f\n", c_ref[0], c_gpu_padded[0]);

    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_i = 0;
    int worst_j = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const float ref = c_ref[(size_t) i * (size_t) n + (size_t) j];
            const float got = c_gpu[(size_t) i * (size_t) n + (size_t) j];
            const float abs_err = std::fabs(got - ref);
            const float rel_err = abs_err / (std::fabs(ref) + 1e-6f);
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                worst_i = i;
                worst_j = j;
            }
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
            }
        }
    }

    const float tol_abs = 1e-2f;
    const float tol_rel = 1e-2f;
    const bool ok = max_abs_err <= tol_abs || max_rel_err <= tol_rel;

    std::printf("integration-style case m=%d n=%d(kpad=%d) k=%d gs_a=%.3f gs_b=%.3f | max_abs=%.6g max_rel=%.6g | %s\n",
            m, n, n_padded, k, global_scale_a, global_scale_b, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  worst element: (%d, %d), ref=%.8f, gpu=%.8f\n",
                worst_i,
                worst_j,
                c_ref[(size_t) worst_i * (size_t) n + (size_t) worst_j],
                c_gpu[(size_t) worst_i * (size_t) n + (size_t) worst_j]);
    }

    return ok;
}

static bool run_case_integration_style_dynamic_device_alpha(int m, int n, int k, float global_scale_a, float q_amplitude, uint32_t seed) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_b(-q_amplitude, q_amplitude);

    std::vector<float> a_fp32((size_t) m * (size_t) k);
    std::vector<float> b_fp32((size_t) n * (size_t) k);
    for (float & v : a_fp32) {
        v = dist_a(rng);
    }
    for (float & v : b_fp32) {
        v = dist_b(rng);
    }

    std::vector<float> global_scales_b;
    compute_dynamic_global_scales_per_row(b_fp32, n, k, global_scales_b);

    std::vector<block_nvfp4> a_nvfp4;
    std::vector<block_nvfp4> b_nvfp4;
    quantize_matrix_nvfp4(a_fp32, a_nvfp4, m, k, global_scale_a);
    quantize_matrix_nvfp4_dynamic_ref(b_fp32, b_nvfp4, n, k, global_scales_b);

    std::vector<float> a_deq;
    std::vector<float> b_deq;
    dequantize_matrix_nvfp4(a_nvfp4, a_deq, m, k, global_scale_a);
    dequantize_matrix_nvfp4_per_row_scale(b_nvfp4, b_deq, n, k, global_scales_b);

    std::vector<float> c_ref;
    fp32_reference_matmul(a_deq, b_deq, c_ref, m, n, k);

    const int n_padded = (int) pad_i64(n, 16);

    std::vector<uint8_t> a_data;
    std::vector<uint8_t> a_scale;
    std::vector<uint8_t> b_data;
    std::vector<uint8_t> b_scale;
    int64_t a_scale_inner = 0;
    int64_t a_scale_outer = 0;
    int64_t b_scale_inner = 0;
    int64_t b_scale_outer = 0;

    const bool linear_scale_layout = false;
    split_nvfp4_blocks(a_nvfp4, k, m, m, linear_scale_layout, a_data, a_scale, a_scale_inner, a_scale_outer);
    split_nvfp4_blocks(b_nvfp4, k, n, n_padded, linear_scale_layout, b_data, b_scale, b_scale_inner, b_scale_outer);

    uint8_t * d_a_data = nullptr;
    uint8_t * d_b_data = nullptr;
    uint8_t * d_a_scale = nullptr;
    uint8_t * d_b_scale = nullptr;
    float * d_c = nullptr;
    std::vector<float> input_scales_b((size_t) n, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float gs = global_scales_b[(size_t) i];
        input_scales_b[(size_t) i] = gs != 0.0f ? (1.0f / gs) : 0.0f;
    }

    CUDA_CHECK(cudaMalloc(&d_a_data, a_data.size()));
    CUDA_CHECK(cudaMalloc(&d_b_data, b_data.size()));
    CUDA_CHECK(cudaMalloc(&d_a_scale, a_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_b_scale, b_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_c, (size_t) m * (size_t) n_padded * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a_data, a_data.data(), a_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_data, b_data.data(), b_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_scale, a_scale.data(), a_scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_scale, b_scale.data(), b_scale.size(), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt = nullptr;
    CUBLASLT_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

#if defined(CUBLAS_VER_MAJOR) && (CUBLAS_VER_MAJOR >= 13)
    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    const void * a_scale_ptr = d_a_scale;
    const void * b_scale_ptr = d_b_scale;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));
#else
    std::fprintf(stderr, "Skip: cuBLASLt FP4 scale-channel attributes are unavailable in this toolkit.\n");
    return true;
#endif

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) k, (uint64_t) m, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) k, (uint64_t) n_padded, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F,     (uint64_t) m, (uint64_t) n_padded, (int64_t) m));

    const float alpha = 1.0f / global_scale_a;
    const float beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(
            lt,
            op_desc,
            &alpha,
            d_a_data, a_desc,
            d_b_data, b_desc,
            &beta,
            d_c, c_desc,
            d_c, c_desc,
            nullptr,
            nullptr, 0,
            nullptr));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> c_gpu_padded((size_t) m * (size_t) n_padded, 0.0f);
    CUDA_CHECK(cudaMemcpy(
            c_gpu_padded.data(),
            d_c,
            (size_t) m * (size_t) n_padded * sizeof(float),
            cudaMemcpyDeviceToHost));

    std::vector<float> c_gpu((size_t) m * (size_t) n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            c_gpu[(size_t) i * (size_t) n + (size_t) j] =
                    c_gpu_padded[(size_t) j * (size_t) m + (size_t) i] * input_scales_b[(size_t) j];
        }
    }

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    for (size_t i = 0; i < c_ref.size(); ++i) {
        const float abs_err = std::fabs(c_gpu[i] - c_ref[i]);
        const float rel_err = abs_err / (std::fabs(c_ref[i]) + 1e-6f);
        max_abs_err = fmaxf(max_abs_err, abs_err);
        max_rel_err = fmaxf(max_rel_err, rel_err);
    }

    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));

    const bool ok = max_abs_err <= 1e-2f || max_rel_err <= 1e-2f;
    std::printf("integration-style dynamic-per-row-scale m=%d n=%d k=%d q_amp=%.1f | max_abs=%.6g max_rel=%.6g | %s\n",
            m, n, k, q_amplitude, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    return ok;
}

static bool run_case_backend_batched_dynamic_rhs(
        int m,
        int n,
        int k,
        int batch_k,
        int batch_q,
        float global_scale_a,
        float k_amplitude,
        float q_amplitude,
        uint32_t seed) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);
    GGML_ASSERT(batch_q % batch_k == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_k(-k_amplitude, k_amplitude);
    std::uniform_real_distribution<float> dist_q(-q_amplitude, q_amplitude);

    const int q_per_k = batch_q / batch_k;
    const size_t a_slice_elems = (size_t) m * (size_t) k;
    const size_t b_slice_elems = (size_t) n * (size_t) k;
    const size_t c_slice_elems = (size_t) m * (size_t) n;

    std::vector<float> a_fp32((size_t) batch_k * a_slice_elems);
    std::vector<float> b_fp32((size_t) batch_q * b_slice_elems);
    for (float & v : a_fp32) {
        v = dist_k(rng);
    }
    for (float & v : b_fp32) {
        v = dist_q(rng);
    }

    std::vector<block_nvfp4> a_nvfp4((size_t) batch_k * (a_slice_elems / QK_NVFP4));
    std::vector<float> a_deq((size_t) batch_k * a_slice_elems);
    for (int ib = 0; ib < batch_k; ++ib) {
        std::vector<block_nvfp4> a_q_slice;
        std::vector<float> a_deq_slice;
        std::vector<float> a_fp32_slice(
                a_fp32.begin() + (ptrdiff_t) ib * (ptrdiff_t) a_slice_elems,
                a_fp32.begin() + (ptrdiff_t) (ib + 1) * (ptrdiff_t) a_slice_elems);

        quantize_matrix_nvfp4(a_fp32_slice, a_q_slice, m, k, global_scale_a);
        dequantize_matrix_nvfp4(a_q_slice, a_deq_slice, m, k, global_scale_a);

        std::memcpy(
                a_nvfp4.data() + (size_t) ib * (a_slice_elems / QK_NVFP4),
                a_q_slice.data(),
                a_q_slice.size() * sizeof(block_nvfp4));
        std::memcpy(
                a_deq.data() + (size_t) ib * a_slice_elems,
                a_deq_slice.data(),
                a_deq_slice.size() * sizeof(float));
    }

    std::vector<float> c_ref((size_t) batch_q * c_slice_elems, 0.0f);
    for (int ib = 0; ib < batch_q; ++ib) {
        const int ia = ib / q_per_k;
        std::vector<float> b_fp32_slice(
                b_fp32.begin() + (ptrdiff_t) ib * (ptrdiff_t) b_slice_elems,
                b_fp32.begin() + (ptrdiff_t) (ib + 1) * (ptrdiff_t) b_slice_elems);
        std::vector<float> global_scales_b;
        compute_dynamic_global_scales_per_row(b_fp32_slice, n, k, global_scales_b);

        std::vector<block_nvfp4> b_q_slice;
        std::vector<float> b_deq_slice;
        quantize_matrix_nvfp4_dynamic_ref(b_fp32_slice, b_q_slice, n, k, global_scales_b);
        dequantize_matrix_nvfp4_per_row_scale(b_q_slice, b_deq_slice, n, k, global_scales_b);

        std::vector<float> c_slice;
        fp32_reference_matmul(
                std::vector<float>(
                        a_deq.begin() + (ptrdiff_t) ia * (ptrdiff_t) a_slice_elems,
                        a_deq.begin() + (ptrdiff_t) (ia + 1) * (ptrdiff_t) a_slice_elems),
                b_deq_slice,
                c_slice,
                m,
                n,
                k);

        float * c_ref_slice = c_ref.data() + (size_t) ib * c_slice_elems;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                c_ref_slice[(size_t) j * (size_t) m + (size_t) i] = c_slice[(size_t) i * (size_t) n + (size_t) j];
            }
        }
    }

    ggml_init_params params = {
        /* .mem_size   = */ 16u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to init ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to init CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_NVFP4, k, m, batch_k, 1);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,   k, n, batch_q, 1);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(c, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(a, a_nvfp4.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_fp32.data(), 0, b_fp32.size() * sizeof(float));

#if defined(_WIN32)
    _putenv_s("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1");
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
#else
    setenv("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1", 1);
    setenv("GGML_CUDA_TRUNC_ENABLE", "0", 1);
#endif

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend batched dynamic rhs compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> c_gpu(c_ref.size(), 0.0f);
    ggml_backend_tensor_get(c, c_gpu.data(), 0, c_gpu.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    size_t worst_idx = 0;
    for (size_t i = 0; i < c_ref.size(); ++i) {
        const float ref = c_ref[i];
        const float got = c_gpu[i];
        const float abs_err = std::fabs(got - ref);
        const float rel_err = abs_err / (std::fabs(ref) + 1e-6f);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            worst_idx = i;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
    }

    const float tol_abs = 2e-1f;
    const float tol_rel = 5e-2f;
    const bool ok = max_abs_err <= tol_abs || max_rel_err <= tol_rel;

    std::printf(
            "backend-batched case m=%d n=%d k=%d batch_k=%d batch_q=%d k_amp=%.4g q_amp=%.1f | max_abs=%.6g max_rel=%.6g | %s\n",
            m, n, k, batch_k, batch_q, k_amplitude, q_amplitude, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  worst idx=%zu ref=%.8f gpu=%.8f\n", worst_idx, c_ref[worst_idx], c_gpu[worst_idx]);
    }

    return ok;
}

static bool run_case_backend_outlier_dynamic_rhs_tensor_scale(
        int m,
        int n,
        int k,
        float global_scale_a,
        float k_amplitude,
        float q_amplitude,
        bool tensor_scale_enabled,
        uint32_t seed) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_k(-k_amplitude, k_amplitude);
    std::uniform_real_distribution<float> dist_q(-q_amplitude, q_amplitude);

    std::vector<float> a_fp32((size_t) m * (size_t) k);
    std::vector<float> b_fp32((size_t) n * (size_t) k);
    for (float & v : a_fp32) {
        v = dist_k(rng);
    }
    for (float & v : b_fp32) {
        v = dist_q(rng);
    }

    const float global_scale_b = compute_dynamic_global_scale(b_fp32);
    std::vector<float> global_scales_b;
    compute_dynamic_global_scales_per_row(b_fp32, n, k, global_scales_b);

    std::vector<block_nvfp4> a_nvfp4;
    std::vector<block_nvfp4> b_nvfp4;
    quantize_matrix_nvfp4(a_fp32, a_nvfp4, m, k, global_scale_a);
    if (tensor_scale_enabled) {
        quantize_matrix_nvfp4_global_ref(b_fp32, b_nvfp4, n, k, global_scale_b);
    } else {
        quantize_matrix_nvfp4_dynamic_ref(b_fp32, b_nvfp4, n, k, global_scales_b);
    }

    std::vector<float> a_deq;
    std::vector<float> b_deq;
    dequantize_matrix_nvfp4(a_nvfp4, a_deq, m, k, global_scale_a);
    if (tensor_scale_enabled) {
        dequantize_matrix_nvfp4(b_nvfp4, b_deq, n, k, global_scale_b);
    } else {
        dequantize_matrix_nvfp4_per_row_scale(b_nvfp4, b_deq, n, k, global_scales_b);
    }

    std::vector<float> c_ref;
    fp32_reference_matmul(a_deq, b_deq, c_ref, m, n, k);
    std::vector<float> c_ref_col_major((size_t) m * (size_t) n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            c_ref_col_major[(size_t) col * (size_t) m + (size_t) row] =
                    c_ref[(size_t) row * (size_t) n + (size_t) col];
        }
    }

    ggml_init_params params = {
        /* .mem_size   = */ 16u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to init ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to init CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   k, n);
    ggml_tensor * outlier_counts  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m);
    ggml_tensor * outlier_indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, m);
    ggml_tensor * outlier_values  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, m);
    ggml_tensor_set_nvfp4_kcache_outliers(a, outlier_counts, outlier_indices, outlier_values);

    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(c, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> counts((size_t) m, 0);
    std::vector<int32_t> indices((size_t) m, 0);
    std::vector<float> values((size_t) m, 0.0f);
    ggml_backend_tensor_set(a, a_nvfp4.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_fp32.data(), 0, b_fp32.size() * sizeof(float));
    ggml_backend_tensor_set(outlier_counts, counts.data(), 0, counts.size() * sizeof(int32_t));
    ggml_backend_tensor_set(outlier_indices, indices.data(), 0, indices.size() * sizeof(int32_t));
    ggml_backend_tensor_set(outlier_values, values.data(), 0, values.size() * sizeof(float));

#if defined(_WIN32)
    _putenv_s("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1");
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
    _putenv_s("LLAMA_NVFP4_KCACHE_OUTLIER_TENSOR_SCALE", tensor_scale_enabled ? "1" : "0");
#else
    setenv("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1", 1);
    setenv("GGML_CUDA_TRUNC_ENABLE", "0", 1);
    setenv("LLAMA_NVFP4_KCACHE_OUTLIER_TENSOR_SCALE", tensor_scale_enabled ? "1" : "0", 1);
#endif

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend outlier dynamic rhs compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> c_gpu(c_ref_col_major.size(), 0.0f);
    ggml_backend_tensor_get(c, c_gpu.data(), 0, c_gpu.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    size_t worst_idx = 0;
    for (size_t i = 0; i < c_ref_col_major.size(); ++i) {
        const float ref = c_ref_col_major[i];
        const float got = c_gpu[i];
        const float abs_err = std::fabs(got - ref);
        const float rel_err = abs_err / (std::fabs(ref) + 1e-6f);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            worst_idx = i;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
    }

    const float tol_abs = 2e-1f;
    const float tol_rel = 5e-2f;
    const bool ok = max_abs_err <= tol_abs || max_rel_err <= tol_rel;

    std::printf(
            "backend-outlier-dynamic-rhs case tensor_scale=%d m=%d n=%d k=%d k_amp=%.4g q_amp=%.1f | max_abs=%.6g max_rel=%.6g | %s\n",
            tensor_scale_enabled ? 1 : 0, m, n, k, k_amplitude, q_amplitude, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  worst idx=%zu ref=%.8f gpu=%.8f\n", worst_idx, c_ref_col_major[worst_idx], c_gpu[worst_idx]);
    }

    return ok;
}

static bool run_case_backend_outlier_compact_sidecar() {
    const int m = 32;
    const int n = 4;
    const int k = 128;

    std::vector<float> a_fp32((size_t) m * (size_t) k, 0.0f);
    std::vector<float> b_fp32((size_t) n * (size_t) k, 0.0f);
    for (int col = 0; col < k; ++col) {
        b_fp32[(size_t) 0 * k + col] = 0.01f * (float) (col + 1);
        b_fp32[(size_t) 1 * k + col] = 0.02f * (float) (col + 1);
        b_fp32[(size_t) 2 * k + col] = -0.01f * (float) (col + 1);
        b_fp32[(size_t) 3 * k + col] = 0.005f * (float) (col + 1);
    }

    std::vector<block_nvfp4> a_nvfp4;
    quantize_matrix_nvfp4(a_fp32, a_nvfp4, m, k, 1.0f);

    std::vector<float> c_ref_col_major((size_t) m * (size_t) n, 0.0f);
    const int row0 = 0;
    const int row1 = 1;
    const int dim0 = 3;
    const int dim1 = 17;
    const int dim2 = 64;
    const float val0 = 20.0f;
    const float val1 = -18.0f;
    const float val2 = 24.0f;
    for (int q = 0; q < n; ++q) {
        c_ref_col_major[(size_t) q * m + row0] =
                val0 * b_fp32[(size_t) q * k + dim0] +
                val1 * b_fp32[(size_t) q * k + dim1];
        c_ref_col_major[(size_t) q * m + row1] =
                val2 * b_fp32[(size_t) q * k + dim2];
    }

    ggml_init_params params = {
        /* .mem_size   = */ 16u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to init ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to init CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * outlier_counts  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m);
    ggml_tensor * outlier_offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m);
    ggml_tensor * outlier_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ggml_tensor * outlier_values  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor_set_nvfp4_kcache_outliers_compact(a, outlier_counts, outlier_offsets, outlier_indices, outlier_values);

    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(c, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> counts((size_t) m, 0);
    std::vector<int32_t> offsets((size_t) m, 0);
    std::vector<int32_t> indices = { dim0, dim1, dim2 };
    std::vector<float> values = { val0, val1, val2 };
    counts[(size_t) row0] = 2;
    counts[(size_t) row1] = 1;
    offsets[(size_t) row0] = 0;
    offsets[(size_t) row1] = 2;

    ggml_backend_tensor_set(a, a_nvfp4.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_fp32.data(), 0, b_fp32.size() * sizeof(float));
    ggml_backend_tensor_set(outlier_counts, counts.data(), 0, counts.size() * sizeof(int32_t));
    ggml_backend_tensor_set(outlier_offsets, offsets.data(), 0, offsets.size() * sizeof(int32_t));
    ggml_backend_tensor_set(outlier_indices, indices.data(), 0, indices.size() * sizeof(int32_t));
    ggml_backend_tensor_set(outlier_values, values.data(), 0, values.size() * sizeof(float));

#if defined(_WIN32)
    _putenv_s("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1");
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
#else
    setenv("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1", 1);
    setenv("GGML_CUDA_TRUNC_ENABLE", "0", 1);
#endif

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend compact outlier compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> c_gpu(c_ref_col_major.size(), 0.0f);
    ggml_backend_tensor_get(c, c_gpu.data(), 0, c_gpu.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    float max_abs_err = 0.0f;
    size_t worst_idx = 0;
    for (size_t i = 0; i < c_ref_col_major.size(); ++i) {
        const float abs_err = std::fabs(c_gpu[i] - c_ref_col_major[i]);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            worst_idx = i;
        }
    }

    const bool ok = max_abs_err <= 1e-4f;
    std::printf("backend-outlier-compact-sidecar case | max_abs=%.6g | %s\n", max_abs_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  worst idx=%zu ref=%.8f gpu=%.8f\n", worst_idx, c_ref_col_major[worst_idx], c_gpu[worst_idx]);
    }

    return ok;
}

static bool run_case_backend_batched_dynamic_rhs_permuted_lhs(
        int m,
        int n,
        int k,
        int batch_k,
        int batch_q,
        float global_scale_a,
        float q_amplitude,
        uint32_t seed) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);
    GGML_ASSERT(batch_q % batch_k == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_k(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_q(-q_amplitude, q_amplitude);

    const int q_per_k = batch_q / batch_k;
    const size_t a_slice_elems = (size_t) m * (size_t) k;
    const size_t b_slice_elems = (size_t) n * (size_t) k;
    const size_t c_slice_elems = (size_t) m * (size_t) n;

    std::vector<float> a_fp32((size_t) batch_k * a_slice_elems);
    std::vector<float> b_fp32((size_t) batch_q * b_slice_elems);
    for (float & v : a_fp32) {
        v = dist_k(rng);
    }
    for (float & v : b_fp32) {
        v = dist_q(rng);
    }

    std::vector<block_nvfp4> a_nvfp4((size_t) batch_k * (a_slice_elems / QK_NVFP4));
    std::vector<float> a_deq((size_t) batch_k * a_slice_elems);
    for (int ib = 0; ib < batch_k; ++ib) {
        std::vector<block_nvfp4> a_q_slice;
        std::vector<float> a_deq_slice;
        std::vector<float> a_fp32_slice(
                a_fp32.begin() + (ptrdiff_t) ib * (ptrdiff_t) a_slice_elems,
                a_fp32.begin() + (ptrdiff_t) (ib + 1) * (ptrdiff_t) a_slice_elems);

        quantize_matrix_nvfp4(a_fp32_slice, a_q_slice, m, k, global_scale_a);
        dequantize_matrix_nvfp4(a_q_slice, a_deq_slice, m, k, global_scale_a);

        std::memcpy(
                a_nvfp4.data() + (size_t) ib * (a_slice_elems / QK_NVFP4),
                a_q_slice.data(),
                a_q_slice.size() * sizeof(block_nvfp4));
        std::memcpy(
                a_deq.data() + (size_t) ib * a_slice_elems,
                a_deq_slice.data(),
                a_deq_slice.size() * sizeof(float));
    }

    std::vector<float> c_ref((size_t) batch_q * c_slice_elems, 0.0f);
    for (int ib = 0; ib < batch_q; ++ib) {
        const int ia = ib / q_per_k;
        std::vector<float> b_fp32_slice(
                b_fp32.begin() + (ptrdiff_t) ib * (ptrdiff_t) b_slice_elems,
                b_fp32.begin() + (ptrdiff_t) (ib + 1) * (ptrdiff_t) b_slice_elems);
        std::vector<float> global_scales_b;
        compute_dynamic_global_scales_per_row(b_fp32_slice, n, k, global_scales_b);

        std::vector<block_nvfp4> b_q_slice;
        std::vector<float> b_deq_slice;
        quantize_matrix_nvfp4_dynamic_ref(b_fp32_slice, b_q_slice, n, k, global_scales_b);
        dequantize_matrix_nvfp4_per_row_scale(b_q_slice, b_deq_slice, n, k, global_scales_b);

        std::vector<float> c_slice;
        fp32_reference_matmul(
                std::vector<float>(
                        a_deq.begin() + (ptrdiff_t) ia * (ptrdiff_t) a_slice_elems,
                        a_deq.begin() + (ptrdiff_t) (ia + 1) * (ptrdiff_t) a_slice_elems),
                b_deq_slice,
                c_slice,
                m,
                n,
                k);

        float * c_ref_slice = c_ref.data() + (size_t) ib * c_slice_elems;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                c_ref_slice[(size_t) j * (size_t) m + (size_t) i] = c_slice[(size_t) i * (size_t) n + (size_t) j];
            }
        }
    }

    ggml_init_params params = {
        /* .mem_size   = */ 16u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to init ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to init CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    // Build A in a layout that requires a permute view to reach [k, m, batch_k, 1].
    ggml_tensor * a_base = ggml_new_tensor_4d(ctx, GGML_TYPE_NVFP4, k, batch_k, m, 1);
    ggml_tensor * a = ggml_permute(ctx, a_base, 0, 2, 1, 3);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k, n, batch_q, 1);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(c, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, c);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    // Store A into a_base's physical layout [k, batch_k, m, 1].
    std::vector<block_nvfp4> a_nvfp4_base((size_t) batch_k * (a_slice_elems / QK_NVFP4));
    {
        const size_t row_blocks = (size_t) k / QK_NVFP4;
        for (int ib = 0; ib < batch_k; ++ib) {
            for (int row = 0; row < m; ++row) {
                const block_nvfp4 * src_row =
                        a_nvfp4.data() + ((size_t) ib * (size_t) m + (size_t) row) * row_blocks;
                block_nvfp4 * dst_row =
                        a_nvfp4_base.data() + ((size_t) row * (size_t) batch_k + (size_t) ib) * row_blocks;
                std::memcpy(dst_row, src_row, row_blocks * sizeof(block_nvfp4));
            }
        }
    }

    ggml_backend_tensor_set(a_base, a_nvfp4_base.data(), 0, ggml_nbytes(a_base));
    ggml_backend_tensor_set(b, b_fp32.data(), 0, b_fp32.size() * sizeof(float));

#if defined(_WIN32)
    _putenv_s("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1");
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
#else
    setenv("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1", 1);
    setenv("GGML_CUDA_TRUNC_ENABLE", "0", 1);
#endif

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "backend permuted lhs compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> c_gpu(c_ref.size(), 0.0f);
    ggml_backend_tensor_get(c, c_gpu.data(), 0, c_gpu.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    size_t worst_idx = 0;
    for (size_t i = 0; i < c_ref.size(); ++i) {
        const float ref = c_ref[i];
        const float got = c_gpu[i];
        const float abs_err = std::fabs(got - ref);
        const float rel_err = abs_err / (std::fabs(ref) + 1e-6f);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            worst_idx = i;
        }
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
    }

    const float tol_abs = 2e-1f;
    const float tol_rel = 5e-2f;
    const bool ok = max_abs_err <= tol_abs || max_rel_err <= tol_rel;

    std::printf(
            "backend-permuted-lhs case m=%d n=%d k=%d batch_k=%d batch_q=%d q_amp=%.1f | max_abs=%.6g max_rel=%.6g | %s\n",
            m, n, k, batch_k, batch_q, q_amplitude, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  worst idx=%zu ref=%.8f gpu=%.8f\n", worst_idx, c_ref[worst_idx], c_gpu[worst_idx]);
    }

    return ok;
}

int main() {
    int dev_count = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count <= 0) {
        std::printf("test-nvfp4-matmul: SKIP (no CUDA device)\n");
        return 0;
    }

    CUDA_CHECK(cudaSetDevice(0));

    bool ok = true;
    ok = run_case_bf16_round_quantizer_bytes() && ok;
    ok = run_case(64, 64, 128, 1.00f, 1.00f, 1u) && ok;
    ok = run_case(48, 80, 256, 0.75f, 1.25f, 2u) && ok;
    ok = run_case(96, 96, 192, 1.50f, 0.90f, 3u) && ok;
    ok = run_case(256, 256, 128, 1.00f, 1.00f, 4u) && ok;
    ok = run_case(256, 64, 256, 1.00f, 1.00f, 5u) && ok;

    // Mirror current ggml native descriptor path to detect integration mismatch.
    ok = run_case_integration_style(64, 64, 128, 1.00f, 1.00f, 11u) && ok;
    ok = run_case_integration_style(64, 9, 128, 1.00f, 1.00f, 12u) && ok;
    ok = run_case_integration_style(96, 5, 192, 1.50f, 0.90f, 13u) && ok;
    ok = run_case_integration_style_dynamic_device_alpha(32, 16, 128, 1.0f, 96.0f, 20u) && ok;
    ok = run_case_backend_batched_dynamic_rhs(32, 16, 128, 1, 1, 1.0f, 1.0f, 96.0f, 22u) && ok;
    ok = run_case_backend_batched_dynamic_rhs(32, 16, 128, 8, 8, 1.0f, 1.0f, 96.0f, 23u) && ok;
    ok = run_case_backend_batched_dynamic_rhs(32, 16, 128, 8, 32, 1.0f, 1.0f, 96.0f, 21u) && ok;
    ok = run_case_backend_batched_dynamic_rhs(32, 16, 128, 8, 32, 1.0f, 1e-3f, 96.0f, 25u) && ok;
    ok = run_case_backend_outlier_dynamic_rhs_tensor_scale(32, 16, 128, 1.0f, 1.0f, 96.0f, false, 26u) && ok;
    ok = run_case_backend_outlier_dynamic_rhs_tensor_scale(32, 16, 128, 1.0f, 1.0f, 96.0f, true, 27u) && ok;
    ok = run_case_backend_outlier_compact_sidecar() && ok;
    ok = run_case_backend_batched_dynamic_rhs_permuted_lhs(32, 16, 128, 8, 32, 1.0f, 96.0f, 24u) && ok;

    if (!ok) {
        std::fprintf(stderr, "test-nvfp4-matmul: FAILED\n");
        return 1;
    }

    std::printf("test-nvfp4-matmul: all cases passed\n");
    return 0;
}
