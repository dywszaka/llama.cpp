#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-quants.h"
#include "../ggml/src/ggml-impl.h"
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

static float bf16_round_f32(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
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

static bool test_bf16_trunc_nn_enabled() {
    const char * bf16_env = std::getenv("GGML_CUDA_NVFP4_BF16_QUANT");
    const char * env = std::getenv("GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN");
    return bf16_env != nullptr && bf16_env[0] != '\0' && bf16_env[0] != '0' &&
            env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool test_bf16_internal_arith_enabled() {
    const char * env = std::getenv("GGML_CUDA_NVFP4_BF16_QUANT_BF16_INTERNAL");
    return test_bf16_trunc_nn_enabled() &&
            env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool test_bf16_block_scale_enabled() {
    const char * env = std::getenv("GGML_CUDA_NVFP4_BF16_QUANT_BF16_BLOCK_SCALE");
    return test_bf16_internal_arith_enabled() &&
            env != nullptr && env[0] != '\0' && env[0] != '0';
}

static float legacy_e4m3_to_fp32(uint8_t x) {
    const uint32_t sign     = (uint32_t)(x & 0x80) << 24;
    uint32_t exponent = (x >> 3) & 0x0F;
    uint32_t mantissa = x & 0x07;

    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            const int shift = __builtin_clz(mantissa) - 29;
            mantissa <<= shift;
            const uint32_t exp = 127 - 6 - shift;
            bits = sign | (exp << 23) | ((mantissa & 0x7) << 20);
        }
    } else if (exponent == 0x0F && mantissa == 0x7) {
        bits = sign | 0x7F800000 | (1u << 22);
    } else {
        const uint32_t exp = (exponent - 7 + 127) << 23;
        const uint32_t man = mantissa << (23 - 3);
        bits = sign | exp | man;
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static bool run_case_default_e2m1_e4m3_tables() {
    static const float expected_e4m3_high[8] = {
        256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, NAN,
    };
    static const int8_t expected_e2m1_doubled[16] = {
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    };

    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        ok = ok && std::fabs(GGML_E4M3_TO_FP32((uint8_t) i) - legacy_e4m3_to_fp32((uint8_t) i)) == 0.0f;
    }
    for (int i = 0; i < 7; ++i) {
        ok = ok && std::fabs(GGML_E4M3_TO_FP32((uint8_t) (0x78 + i)) - expected_e4m3_high[i]) == 0.0f;
    }
    ok = ok && std::isnan(GGML_E4M3_TO_FP32((uint8_t) 0x7f));

    block_nvfp4 e2m1_probe = {};
    e2m1_probe.e = 0x40; // E4M3 value 2.0, so GGML_E4M3_TO_FP32_HALF(e) == 1.0.
    for (int i = 0; i < QK_NVFP4 / 2; ++i) {
        e2m1_probe.qs[i] = (uint8_t) (i * 2) | (uint8_t) ((i * 2 + 1) << 4);
    }
    float e2m1_deq[QK_NVFP4] = {};
    dequantize_row_nvfp4(&e2m1_probe, e2m1_deq, QK_NVFP4, 1.0f);
    for (int i = 0; i < QK_NVFP4; ++i) {
        ok = ok && std::fabs(e2m1_deq[i] - (float) expected_e2m1_doubled[i]) == 0.0f;
    }

    std::printf("default e2m1/e4m3 tables | %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static float host_trunc_f32_to_bf16_value(float x) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = x;
    bits.u &= 0xffff0000u;
    return bits.f;
}

static float host_effective_bf16_global_scale(float x) {
    return x;
}

static float compute_dynamic_global_scale(const std::vector<float> & src) {
    const bool truncate_bf16_input = test_bf16_trunc_nn_enabled();
    float amax = 0.0f;
    for (float v : src) {
        const float xq = truncate_bf16_input ? host_trunc_f32_to_bf16_value(v) : v;
        amax = fmaxf(amax, fabsf(xq));
    }

    return amax > 0.0f ? (6.0f * 224.0f) / amax : 0.0f;
}

static void compute_dynamic_global_scales_per_row(
        const std::vector<float> & src,
        int rows,
        int k,
        std::vector<float> & global_scales) {
    const bool truncate_bf16_input = test_bf16_trunc_nn_enabled();
    global_scales.resize((size_t) rows);
    for (int r = 0; r < rows; ++r) {
        const float * row = src.data() + (size_t) r * (size_t) k;
        float amax = 0.0f;
        for (int i = 0; i < k; ++i) {
            const float xq = truncate_bf16_input ? host_trunc_f32_to_bf16_value(row[i]) : row[i];
            amax = fmaxf(amax, fabsf(xq));
        }
        global_scales[(size_t) r] = amax > 0.0f ? (6.0f * 224.0f) / amax : 0.0f;
    }
}

static void compute_dynamic_amax_per_row(
        const std::vector<float> & src,
        int rows,
        int k,
        std::vector<float> & amax_rows) {
    const bool truncate_bf16_input = test_bf16_trunc_nn_enabled();
    amax_rows.resize((size_t) rows);
    for (int r = 0; r < rows; ++r) {
        const float * row = src.data() + (size_t) r * (size_t) k;
        float amax = 0.0f;
        for (int i = 0; i < k; ++i) {
            const float xq = truncate_bf16_input ? host_trunc_f32_to_bf16_value(row[i]) : row[i];
            amax = fmaxf(amax, fabsf(xq));
        }
        amax_rows[(size_t) r] = amax;
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
        const float effective_global_scale = host_effective_bf16_global_scale(global_scales[(size_t) r]);
        dequantize_row_nvfp4(
                src.data() + (size_t) r * (size_t) nblk_k,
                dst.data() + (size_t) r * (size_t) k,
                k,
                effective_global_scale);
    }
}

static uint64_t host_low_bits_mask_u64(uint8_t width) {
    if (width >= 64u) {
        return ~0ull;
    }
    return width == 0u ? 0ull : ((1ull << width) - 1ull);
}

static uint16_t host_fp32_to_bf16_trunc_bits(float x) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = x;
    return (uint16_t) (bits.u >> 16);
}

static uint16_t host_bf16_abs_bits(uint16_t x) {
    return (uint16_t) (x & (uint16_t) host_low_bits_mask_u64(15u));
}

static uint32_t host_clz_u32_hw(uint32_t value) {
    if (value == 0u) {
        return 32u;
    }

    uint32_t count = 0u;
    if ((value & 0xffff0000u) == 0u) {
        count += 16u;
        value <<= 16;
    }
    if ((value & 0xff000000u) == 0u) {
        count += 8u;
        value <<= 8;
    }
    if ((value & 0xf0000000u) == 0u) {
        count += 4u;
        value <<= 4;
    }
    if ((value & 0xc0000000u) == 0u) {
        count += 2u;
        value <<= 2;
    }
    if ((value & 0x80000000u) == 0u) {
        count += 1u;
    }
    return count;
}

static uint32_t host_f32_to_u32_bits_hw(float v) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = v;
    return bits.u;
}

static float host_u32_to_f32_bits_hw(uint32_t v) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = v;
    return bits.f;
}

static float host_bf16_abs_to_fp32_bits_hw(uint16_t abs_bits) {
    return host_u32_to_f32_bits_hw((uint32_t) abs_bits << 16);
}

static float host_bf16_mul_trunc_hw(float a, float b) {
    return host_trunc_f32_to_bf16_value(a * b);
}

static uint16_t host_bf16_mul_bf16_rne_bits_hw(uint16_t a, uint16_t b) {
    const uint32_t a_sign = (a >> 15) & 1u;
    const uint32_t b_sign = (b >> 15) & 1u;
    const uint32_t a_exp  = (a >> 7) & 0xffu;
    const uint32_t b_exp  = (b >> 7) & 0xffu;
    const uint32_t a_mant = a & 0x7fu;
    const uint32_t b_mant = b & 0x7fu;

    const bool a_zero = (a & 0x7fffu) == 0u;
    const bool b_zero = (b & 0x7fffu) == 0u;
    const bool a_inf  = (a & 0x7fffu) == 0x7f80u;
    const bool b_inf  = (b & 0x7fffu) == 0x7f80u;
    const bool a_nan  = a_exp == 0xffu && a_mant != 0u;
    const bool b_nan  = b_exp == 0xffu && b_mant != 0u;

    const uint32_t sign_out = a_sign ^ b_sign;
    const bool is_nan = a_nan || b_nan || (a_inf && b_zero) || (a_zero && b_inf);
    const bool is_inf = (a_inf || b_inf) && !is_nan;
    const bool is_zero = (a_zero || b_zero) && !is_inf;

    const uint32_t ma = 0x80u | a_mant;
    const uint32_t mb = 0x80u | b_mant;
    const uint32_t product = ma * mb;
    const int32_t exp_sum = (int32_t) a_exp + (int32_t) b_exp - 127;

    const bool product_overflow = (product & 0x8000u) != 0u;
    const int32_t exp_norm = product_overflow ? exp_sum + 1 : exp_sum;
    const uint32_t mant_pre = product_overflow ? ((product >> 8) & 0x7fu) : ((product >> 7) & 0x7fu);
    const bool guard = product_overflow ? (((product >> 7) & 1u) != 0u) : (((product >> 6) & 1u) != 0u);
    const bool sticky = product_overflow ? ((product & 0x7fu) != 0u) : ((product & 0x3fu) != 0u);
    const bool round_up = guard && (sticky || ((mant_pre & 1u) != 0u));
    const uint32_t mant_rnd = mant_pre + (round_up ? 1u : 0u);

    const int32_t exp_rnd = (mant_rnd & 0x80u) ? exp_norm + 1 : exp_norm;
    const uint32_t mant_final = (mant_rnd & 0x80u) ? 0u : (mant_rnd & 0x7fu);

    const uint16_t normal_result =
        (uint16_t) ((sign_out << 15) | (((uint32_t) exp_rnd & 0xffu) << 7) | mant_final);
    uint16_t result = normal_result;
    result = exp_rnd < 0 ? (uint16_t) (sign_out << 15) : result;
    result = exp_rnd >= 255 ? (uint16_t) ((sign_out << 15) | 0x7f80u) : result;
    result = is_zero ? (uint16_t) (sign_out << 15) : result;
    result = is_inf ? (uint16_t) ((sign_out << 15) | 0x7f80u) : result;
    result = is_nan ? 0x7fc0u : result;
    return result;
}

static uint16_t host_bf16_mul_operands_rne_bits_hw(uint16_t a, float b) {
    const uint16_t a_bf16 = host_bf16_abs_bits(a);
    const uint16_t b_bf16 = host_fp32_to_bf16_trunc_bits(b);
    return host_bf16_mul_bf16_rne_bits_hw(a_bf16, b_bf16);
}

static bool host_bf16_pos_le_hw(uint16_t a, uint16_t b) {
    return host_bf16_abs_bits(a) <= host_bf16_abs_bits(b);
}

static uint16_t host_bf16_pos_mul2_bits_hw(uint16_t value) {
    value = host_bf16_abs_bits(value);
    const uint32_t exp = (value >> 7) & 0xffu;
    const uint32_t mant = value & 0x7fu;

    if ((value & 0x7fffu) == 0u) {
        return 0u;
    }
    if (exp == 0xffu) {
        return value;
    }
    if (exp == 0u) {
        return (mant & 0x40u) != 0u
                ? (uint16_t) ((1u << 7) | ((mant & 0x3fu) << 1))
                : (uint16_t) ((mant & 0x3fu) << 1);
    }
    if (exp == 0xfeu) {
        return 0x7f80u;
    }
    return (uint16_t) (((exp + 1u) << 7) | mant);
}

static uint16_t host_bf16_add_pos_trunc_bits_hw(uint16_t a, uint16_t b) {
    a = host_bf16_abs_bits(a);
    b = host_bf16_abs_bits(b);

    uint32_t exp_a = (a >> 7) & 0xffu;
    uint32_t exp_b = (b >> 7) & 0xffu;
    const bool a_zero_in = (a & 0x7fffu) == 0u;
    const bool b_zero_in = (b & 0x7fffu) == 0u;
    const bool a_subnormal_in = exp_a == 0u && !a_zero_in;
    const bool b_subnormal_in = exp_b == 0u && !b_zero_in;
    const bool any_non_finite_in = exp_a == 0xffu || exp_b == 0xffu;
    uint32_t sig_a = 0x80u | (a & 0x7fu);
    uint32_t sig_b = 0x80u | (b & 0x7fu);

    if (exp_a < exp_b || (exp_a == exp_b && sig_a < sig_b)) {
        const uint32_t tmp_exp = exp_a;
        const uint32_t tmp_sig = sig_a;
        exp_a = exp_b;
        sig_a = sig_b;
        exp_b = tmp_exp;
        sig_b = tmp_sig;
    }

    constexpr uint32_t kGuardBits = 16u;
    const uint32_t shift = exp_a - exp_b;
    const uint32_t sig_a_ext = sig_a << kGuardBits;
    const uint32_t sig_b_ext = (shift >= 24u) ? 0u : ((sig_b << kGuardBits) >> shift);
    const uint32_t sum_sig = sig_a_ext + sig_b_ext;

    const bool sum_zero = sum_sig == 0u;
    const uint32_t sum_sig_safe = sum_zero ? 1u : sum_sig;
    const int32_t msb_pos = 31 - (int32_t) host_clz_u32_hw(sum_sig_safe);
    constexpr int32_t kTargetMsbPos = 7 + (int32_t) kGuardBits;
    const int32_t shift_amt = msb_pos - kTargetMsbPos;
    const int32_t res_exp = (int32_t) exp_a + shift_amt;
    const uint32_t final_sig = (shift_amt < 0) ? (sum_sig_safe << (uint32_t) -shift_amt) : (sum_sig_safe >> (uint32_t) shift_amt);
    const uint32_t final_mant = (final_sig >> kGuardBits) & 0x7fu;

    const bool overflow = res_exp >= 255;
    const bool underflow = res_exp <= 0;

    const uint16_t result_normal = (uint16_t) (((uint32_t) res_exp << 7) | final_mant);
    if (b_zero_in) {
        return a;
    }
    if (a_zero_in) {
        return b;
    }
    if (a_subnormal_in) {
        return b;
    }
    if (b_subnormal_in) {
        return a;
    }
    if (any_non_finite_in) {
        return 0x7f80u;
    }
    if (underflow || sum_zero) {
        return 0u;
    }
    if (overflow) {
        return 0x7f80u;
    }
    return result_normal;
}

static uint32_t host_round_shift_right_ties_down_u32_hw(uint32_t value, int shift) {
    constexpr int kMaxLeftShift = 8;
    constexpr int kMaxRightShift = 24;
    if (shift <= 0) {
        const int lshift = -shift;
        return lshift > kMaxLeftShift ? 0xffffffffu : (value << lshift);
    }
    if (shift > kMaxRightShift) {
        return 0u;
    }

    const uint32_t shifted = value >> shift;
    const uint32_t half = 1u << (shift - 1);
    const uint32_t mask = (1u << shift) - 1u;
    const uint32_t remainder = value & mask;
    return shifted + (remainder > half ? 1u : 0u);
}

static uint8_t host_e4m3_subnormal_from_fp32_bits_hw(uint32_t exponent, uint32_t mantissa) {
    const bool fp32_subnormal = exponent == 0u;
    const int32_t exp_unbiased = fp32_subnormal ? -126 : (int32_t) exponent - 127;
    const uint32_t significand = fp32_subnormal ? mantissa : (0x00800000u | mantissa);
    const uint32_t mant_q = host_round_shift_right_ties_down_u32_hw(significand, 14 - exp_unbiased);
    return (uint8_t) (mant_q > 15u ? 15u : mant_q);
}

static uint8_t host_e4m3_scale_from_fp32_bits_hw(float scale) {
    const uint32_t bits = host_f32_to_u32_bits_hw(scale);
    const uint32_t sign = bits >> 31;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x007fffffu;
    if (sign != 0u || exponent == 0xffu) {
        return 0u;
    }
    if (scale <= 0.0302734375f) {
        return host_e4m3_subnormal_from_fp32_bits_hw(exponent, mantissa);
    }

    const int32_t exp_unbiased = (int32_t) exponent - 127;
    int32_t exp_field = exp_unbiased + 7;
    const uint32_t significand = 0x00800000u | mantissa;

    uint32_t signif_q = host_round_shift_right_ties_down_u32_hw(significand, 20);
    if (signif_q >= 16u) {
        signif_q = 8u;
        ++exp_field;
    }

    if (exp_field > 15 || (exp_field == 15 && signif_q >= 15u)) {
        return 0x7eu;
    }

    return (uint8_t) (((uint32_t) exp_field << 3) | ((signif_q - 8u) & 0x7u));
}

static float host_e4m3_scale_half_to_fp32_bits_hw(uint8_t scale) {
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
        return host_u32_to_f32_bits_hw(exp_bits | mant_bits);
    }
    const uint32_t scale_mant_clamped = scale_exp == 0x0fu && scale_mant == 7u ? 6u : scale_mant;
    return host_u32_to_f32_bits_hw(((scale_exp + 119u) << 23) | (scale_mant_clamped << 20));
}

static float host_compute_block_scale_value_trunc_nn_hw(
        uint16_t block_abs_max_bits,
        float global_scale,
        bool bf16_block_scale) {
    if (block_abs_max_bits == 0u) {
        return 0.0f;
    }

    const float block_abs_max = host_bf16_abs_to_fp32_bits_hw(block_abs_max_bits);
    return bf16_block_scale
            ? host_bf16_mul_trunc_hw(
                    host_bf16_mul_trunc_hw(block_abs_max, host_trunc_f32_to_bf16_value(global_scale)),
                    host_trunc_f32_to_bf16_value(0.1666666716f))
            : block_abs_max * global_scale * 0.1666666716f;
}

static uint16_t host_fp32_scale_half_to_bf16_bits_hw(float scale) {
    const uint32_t bits = host_f32_to_u32_bits_hw(scale);
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t abs_bits = bits & 0x7fffffffu;
    uint32_t half_abs_bits = 0u;

    if (abs_bits >= 0x01000000u && abs_bits < 0x7f800000u) {
        half_abs_bits = abs_bits - 0x00800000u;
    } else if (abs_bits >= 0x00800000u && abs_bits < 0x01000000u) {
        half_abs_bits = ((abs_bits & 0x007fffffu) | 0x00800000u) >> 1;
    } else if (abs_bits < 0x00800000u) {
        half_abs_bits = abs_bits >> 1;
    } else {
        half_abs_bits = abs_bits;
    }

    return (uint16_t) ((sign | half_abs_bits) >> 16);
}

static void host_quantize_bf16_trunc_nn_nvfp4(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int rows,
        int k,
        const std::vector<float> & global_scales) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    GGML_ASSERT((int) global_scales.size() == rows);
    const int nblk = k / QK_NVFP4;
    const bool bf16_internal_arith = test_bf16_internal_arith_enabled();
    const bool bf16_block_scale = test_bf16_block_scale_enabled();
    dst.assign((size_t) rows * (size_t) nblk, {});
    std::vector<uint16_t> bf16((size_t) rows * (size_t) k);
    for (size_t i = 0; i < bf16.size(); ++i) {
        bf16[i] = host_fp32_to_bf16_trunc_bits(src[i]);
    }

    for (int r = 0; r < rows; ++r) {
        for (int ib = 0; ib < nblk; ++ib) {
            const float global_scale = global_scales[(size_t) r];
            uint16_t block_abs_max = 0;
            for (int j = 0; j < QK_NVFP4; ++j) {
                block_abs_max = std::max(block_abs_max, host_bf16_abs_bits(bf16[(size_t) r * (size_t) k + (size_t) ib * QK_NVFP4 + j]));
            }
            const float block_scale = host_compute_block_scale_value_trunc_nn_hw(
                    block_abs_max, global_scale, bf16_block_scale);
            const uint8_t scale = host_e4m3_scale_from_fp32_bits_hw(block_scale);
            block_nvfp4 & out = dst[(size_t) r * (size_t) nblk + (size_t) ib];
            out.e = scale;
            if (scale == 0u) {
                std::memset(out.qs, 0, sizeof(out.qs));
                continue;
            }

            const float block_scale_half_f = host_e4m3_scale_half_to_fp32_bits_hw(scale);
            const uint16_t block_scale_half_bits = host_fp32_scale_half_to_bf16_bits_hw(block_scale);
            uint8_t q_raw[QK_NVFP4] = { 0 };
            for (int j = 0; j < QK_NVFP4; ++j) {
                const uint16_t bits = bf16[(size_t) r * (size_t) k + (size_t) ib * QK_NVFP4 + j];
                const uint8_t sign = (uint8_t) ((bits >> 15) & 1u);
                const uint16_t abs_bits = host_bf16_abs_bits(bits);
                uint8_t best_mag = 0u;
                const float abs_f = host_bf16_abs_to_fp32_bits_hw(abs_bits);
                if (bf16_internal_arith) {
                    const uint16_t target = host_bf16_mul_operands_rne_bits_hw(abs_bits, global_scale);
                    const uint16_t target_2x = host_bf16_pos_mul2_bits_hw(target);
                    const uint16_t scale_2x = host_bf16_pos_mul2_bits_hw(block_scale_half_bits);
                    const uint16_t scale_3x = host_bf16_add_pos_trunc_bits_hw(scale_2x, block_scale_half_bits);
                    const uint16_t scale_5x = host_bf16_add_pos_trunc_bits_hw(scale_3x, scale_2x);
                    const uint16_t scale_7x = host_bf16_add_pos_trunc_bits_hw(scale_5x, scale_2x);
                    const uint16_t scale_10x = host_bf16_pos_mul2_bits_hw(scale_5x);
                    const uint16_t scale_14x = host_bf16_pos_mul2_bits_hw(scale_7x);
                    const uint16_t scale_20x = host_bf16_pos_mul2_bits_hw(scale_10x);
                    if (host_bf16_pos_le_hw(target_2x, block_scale_half_bits)) {
                        best_mag = 0u;
                    } else if (host_bf16_pos_le_hw(target_2x, scale_3x)) {
                        best_mag = 1u;
                    } else if (host_bf16_pos_le_hw(target_2x, scale_5x)) {
                        best_mag = 2u;
                    } else if (host_bf16_pos_le_hw(target_2x, scale_7x)) {
                        best_mag = 3u;
                    } else if (host_bf16_pos_le_hw(target_2x, scale_10x)) {
                        best_mag = 4u;
                    } else if (host_bf16_pos_le_hw(target_2x, scale_14x)) {
                        best_mag = 5u;
                    } else if (host_bf16_pos_le_hw(target_2x, scale_20x)) {
                        best_mag = 6u;
                    } else {
                        best_mag = 7u;
                    }
                } else {
                    const float target = abs_f * global_scale;
                    const float target_2x = target + target;
                    const float scale_2x = block_scale_half_f + block_scale_half_f;
                    const float scale_3x = scale_2x + block_scale_half_f;
                    const float scale_5x = scale_3x + scale_2x;
                    const float scale_7x = scale_5x + scale_2x;
                    const float scale_10x = scale_5x + scale_5x;
                    const float scale_14x = scale_7x + scale_7x;
                    const float scale_20x = scale_10x + scale_10x;
                    if (target_2x <= block_scale_half_f) {
                        best_mag = 0u;
                    } else if (target_2x <= scale_3x) {
                        best_mag = 1u;
                    } else if (target_2x <= scale_5x) {
                        best_mag = 2u;
                    } else if (target_2x <= scale_7x) {
                        best_mag = 3u;
                    } else if (target_2x <= scale_10x) {
                        best_mag = 4u;
                    } else if (target_2x <= scale_14x) {
                        best_mag = 5u;
                    } else if (target_2x <= scale_20x) {
                        best_mag = 6u;
                    } else {
                        best_mag = 7u;
                    }
                }
                q_raw[j] = best_mag == 0u ? 0u : (uint8_t) ((sign << 3) | best_mag);
            }
            for (int j = 0; j < QK_NVFP4 / 2; ++j) {
                out.qs[j] = (uint8_t) ((q_raw[2*j + 1] << 4) | (q_raw[2*j] & 0x0f));
            }
        }
    }
}

static bool test_bf16_trunc_nn_quant_enabled() {
    const char * env = std::getenv("GGML_CUDA_NVFP4_BF16_QUANT");
    return test_bf16_trunc_nn_enabled() &&
            env != nullptr && env[0] != '\0' && env[0] != '0';
}

static void quantize_matrix_nvfp4_dynamic_ref(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int rows,
        int k,
        const std::vector<float> & global_scales) {
    if (test_bf16_trunc_nn_quant_enabled()) {
        host_quantize_bf16_trunc_nn_nvfp4(src, dst, rows, k, global_scales);
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
    if (test_bf16_trunc_nn_quant_enabled()) {
        const float effective_global_scale = host_effective_bf16_global_scale(global_scale);
        host_quantize_bf16_trunc_nn_nvfp4(
                src, dst, rows, k, std::vector<float>((size_t) rows, effective_global_scale));
    } else {
        quantize_matrix_nvfp4(src, dst, rows, k, global_scale);
    }
}

static bool run_case_bf16_trunc_nn_quantizer_bytes() {
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
    host_quantize_bf16_trunc_nn_nvfp4(src, expected, rows, k, global_scales);

    float * d_src = nullptr;
    float * d_scales = nullptr;
    block_nvfp4 * d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, global_scales.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, expected.size() * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, global_scales.data(), global_scales.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, expected.size() * sizeof(block_nvfp4)));

    ggml_cuda_nvfp4_quantize_rows_bf16_f32(
            d_src, d_dst, k, k, rows, d_scales, false,
            test_bf16_internal_arith_enabled(), test_bf16_block_scale_enabled(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<block_nvfp4> got(expected.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, got.size() * sizeof(block_nvfp4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_src));

    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].e != got[i].e || std::memcmp(expected[i].qs, got[i].qs, sizeof(expected[i].qs)) != 0) {
            std::fprintf(stderr, "bf16-trunc-nn quant mismatch block=%zu expected_e=%u got_e=%u\n",
                    i, (unsigned) expected[i].e, (unsigned) got[i].e);
            ok = false;
        }
    }
    std::printf("bf16-trunc-nn quantizer bytes | %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool run_case_bf16_dynamic_quantizer_bytes_seed(uint32_t seed) {
    const int rows = 16;
    const int k = 128;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-96.0f, 96.0f);

    std::vector<float> src((size_t) rows * (size_t) k);
    for (float & v : src) {
        v = dist(rng);
    }

    std::vector<float> global_scales;
    std::vector<float> amax_rows;
    compute_dynamic_global_scales_per_row(src, rows, k, global_scales);
    compute_dynamic_amax_per_row(src, rows, k, amax_rows);

    std::vector<block_nvfp4> expected;
    host_quantize_bf16_trunc_nn_nvfp4(src, expected, rows, k, global_scales);

    float * d_src = nullptr;
    float * d_amax = nullptr;
    block_nvfp4 * d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_amax, amax_rows.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, expected.size() * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_amax, amax_rows.data(), amax_rows.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, expected.size() * sizeof(block_nvfp4)));

    ggml_cuda_nvfp4_quantize_rows_dynamic_bf16_f32(
            d_src, d_dst, k, k, rows, d_amax, false,
            test_bf16_internal_arith_enabled(), test_bf16_block_scale_enabled(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<block_nvfp4> got(expected.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, got.size() * sizeof(block_nvfp4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_amax));
    CUDA_CHECK(cudaFree(d_src));

    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].e != got[i].e || std::memcmp(expected[i].qs, got[i].qs, sizeof(expected[i].qs)) != 0) {
            std::fprintf(stderr, "bf16-dynamic quant mismatch block=%zu expected_e=%u got_e=%u\n",
                    i, (unsigned) expected[i].e, (unsigned) got[i].e);
            ok = false;
            break;
        }
    }
    std::printf("bf16-dynamic quantizer bytes seed=%u | %s\n", seed, ok ? "PASS" : "FAIL");
    return ok;
}

static bool run_case_bf16_dynamic_quantizer_bytes() {
    bool ok = true;
    ok = run_case_bf16_dynamic_quantizer_bytes_seed(23u) && ok;
    ok = run_case_bf16_dynamic_quantizer_bytes_seed(22u) && ok;
    return ok;
}

static bool run_case_bf16_dynamic_quantizer_fast_math_regression(uint32_t seed) {
    const int m = 32;
    const int rows = 16;
    const int k = 128;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_a(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_b(-96.0f, 96.0f);

    std::vector<float> a((size_t) m * (size_t) k);
    std::vector<float> src((size_t) rows * (size_t) k);
    for (float & v : a) {
        v = dist_a(rng);
    }
    for (float & v : src) {
        v = dist_b(rng);
    }

    std::vector<float> global_scales;
    std::vector<float> amax_rows;
    compute_dynamic_global_scales_per_row(src, rows, k, global_scales);
    compute_dynamic_amax_per_row(src, rows, k, amax_rows);

    std::vector<block_nvfp4> expected;
    host_quantize_bf16_trunc_nn_nvfp4(src, expected, rows, k, global_scales);

    float * d_src = nullptr;
    float * d_amax = nullptr;
    block_nvfp4 * d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_amax, amax_rows.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, expected.size() * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_amax, amax_rows.data(), amax_rows.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, expected.size() * sizeof(block_nvfp4)));

    ggml_cuda_nvfp4_quantize_rows_dynamic_bf16_f32(
            d_src, d_dst, k, k, rows, d_amax, false,
            test_bf16_internal_arith_enabled(), test_bf16_block_scale_enabled(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<block_nvfp4> got(expected.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_dst, got.size() * sizeof(block_nvfp4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_amax));
    CUDA_CHECK(cudaFree(d_src));

    bool ok = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].e != got[i].e || std::memcmp(expected[i].qs, got[i].qs, sizeof(expected[i].qs)) != 0) {
            std::fprintf(stderr, "bf16-fast-math-regression dynamic quant mismatch block=%zu expected_e=%u got_e=%u\n",
                    i, (unsigned) expected[i].e, (unsigned) got[i].e);
            ok = false;
            break;
        }
    }
    std::printf("bf16-fast-math-regression dynamic quantizer bytes seed=%u | %s\n", seed, ok ? "PASS" : "FAIL");
    return ok;
}

static bool run_case_bf16_dynamic_amax_rows() {
    const int rows = 16;
    const int k = 128;
    std::mt19937 rng(23u);
    std::uniform_real_distribution<float> dist(-96.0f, 96.0f);

    std::vector<float> src((size_t) rows * (size_t) k);
    for (float & v : src) {
        v = dist(rng);
    }

    std::vector<float> expected;
    compute_dynamic_amax_per_row(src, rows, k, expected);

    float * d_src = nullptr;
    float * d_amax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_amax, expected.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_amax, 0, expected.size() * sizeof(float)));

    ggml_cuda_nvfp4_abs_max_rows_f32(d_src, d_amax, k, rows, k, test_bf16_trunc_nn_enabled(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> got(expected.size(), 0.0f);
    CUDA_CHECK(cudaMemcpy(got.data(), d_amax, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_amax));
    CUDA_CHECK(cudaFree(d_src));

    float max_abs_err = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float err = std::fabs(got[i] - expected[i]);
        if (err > max_abs_err) {
            max_abs_err = err;
            worst = i;
        }
    }
    const bool ok = max_abs_err == 0.0f;
    std::printf("bf16-dynamic amax rows | max_abs=%.8g worst=%zu expected=%.8f got=%.8f | %s\n",
            max_abs_err, worst, expected[worst], got[worst], ok ? "PASS" : "FAIL");
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
        uint32_t seed,
        bool capture_rhs = false) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);
    GGML_ASSERT(batch_q % batch_k == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist_k(-k_amplitude, k_amplitude);
    std::uniform_real_distribution<float> dist_q(-q_amplitude, q_amplitude);

    const int q_per_k = batch_q / batch_k;
    const float a_scale = global_scale_a != 0.0f ? (1.0f / global_scale_a) : 0.0f;
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
    std::vector<block_nvfp4> b_q_ref(capture_rhs ? (size_t) batch_q * (b_slice_elems / QK_NVFP4) : 0);
    std::vector<float> b_final_scale_ref(capture_rhs ? (size_t) batch_q * (size_t) n : 0);
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
        if (capture_rhs) {
            std::memcpy(
                    b_q_ref.data() + (size_t) ib * (b_slice_elems / QK_NVFP4),
                    b_q_slice.data(), b_q_slice.size() * sizeof(block_nvfp4));
            for (int row = 0; row < n; ++row) {
                const float global_scale = global_scales_b[(size_t) row];
                const float final_scale = global_scale != 0.0f ? (1.0f / global_scale) : 0.0f;
                b_final_scale_ref[(size_t) ib * (size_t) n + (size_t) row] =
                        host_trunc_f32_to_bf16_value(final_scale);
            }
        }
        dequantize_matrix_nvfp4_per_row_scale(b_q_slice, b_deq_slice, n, k, global_scales_b);
        for (int row = 0; row < n; ++row) {
            const float b_scale = global_scales_b[(size_t) row] != 0.0f ? (1.0f / global_scales_b[(size_t) row]) : 0.0f;
            const float scale = a_scale * b_scale;
            const float scale_ratio = scale != 0.0f ? (bf16_round_f32(scale) / scale) : 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                b_deq_slice[(size_t) row * (size_t) k + (size_t) kk] *= scale_ratio;
            }
        }

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
    ggml_tensor * b_capture = nullptr;
    ggml_tensor * b_scale_capture = nullptr;
    if (capture_rhs) {
        b_capture = ggml_new_tensor_4d(ctx, GGML_TYPE_NVFP4, k, n, batch_q, 1);
        b_scale_capture = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, batch_q, 1);
        ggml_set_output(b_capture);
        ggml_set_output(b_scale_capture);
        ggml_mul_mat_set_nvfp4_rhs_capture(c, b_capture, b_scale_capture);
    }

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
    _putenv_s("GGML_CUDA_NVFP4_FP4MULMAT", "1");
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
#else
    setenv("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1", 1);
    setenv("GGML_CUDA_NVFP4_FP4MULMAT", "1", 1);
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
    bool capture_ok = true;
    if (capture_rhs) {
        std::vector<block_nvfp4> b_q_gpu(b_q_ref.size());
        std::vector<float> b_final_scale_gpu(b_final_scale_ref.size());
        ggml_backend_tensor_get(b_capture, b_q_gpu.data(), 0, b_q_gpu.size() * sizeof(block_nvfp4));
        ggml_backend_tensor_get(b_scale_capture, b_final_scale_gpu.data(), 0, b_final_scale_gpu.size() * sizeof(float));
        const uint32_t capture_flags = ggml_mul_mat_get_nvfp4_capture_flags(c);
        capture_ok = (capture_flags & GGML_NVFP4_MUL_MAT_CAPTURE_VALID) != 0 &&
                (capture_flags & GGML_NVFP4_MUL_MAT_CAPTURE_FINAL_SCALE) != 0 &&
                std::memcmp(b_q_gpu.data(), b_q_ref.data(), b_q_ref.size() * sizeof(block_nvfp4)) == 0;
        bool captured_bf16_scale = true;
        for (size_t i = 0; i < b_final_scale_ref.size() && capture_ok; ++i) {
            const float tolerance = 1e-6f * std::max(std::fabs(b_final_scale_ref[i]), 1.0f);
            capture_ok = std::fabs(b_final_scale_gpu[i] - b_final_scale_ref[i]) <= tolerance;
            uint32_t scale_bits = 0;
            std::memcpy(&scale_bits, &b_final_scale_gpu[i], sizeof(scale_bits));
            captured_bf16_scale = captured_bf16_scale && ((scale_bits & 0xffffu) == 0);
        }
        capture_ok = capture_ok && captured_bf16_scale;
        if (!capture_ok) {
            std::fprintf(stderr, "NVFP4 RHS capture mismatch flags=0x%x\n", capture_flags);
        }
    }

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
    const bool ok = (max_abs_err <= tol_abs || max_rel_err <= tol_rel) && capture_ok;

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
    const bool outlier_sidecar_uses_tensor_scale = true;
    if (tensor_scale_enabled || outlier_sidecar_uses_tensor_scale) {
        quantize_matrix_nvfp4_global_ref(b_fp32, b_nvfp4, n, k, global_scale_b);
    } else {
        quantize_matrix_nvfp4_dynamic_ref(b_fp32, b_nvfp4, n, k, global_scales_b);
    }

    std::vector<float> a_deq;
    std::vector<float> b_deq;
    dequantize_matrix_nvfp4(a_nvfp4, a_deq, m, k, global_scale_a);
    if (tensor_scale_enabled || outlier_sidecar_uses_tensor_scale) {
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
    ggml_tensor * outlier_offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m);
    ggml_tensor * outlier_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, m);
    ggml_tensor * outlier_values  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m);
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
    std::vector<int32_t> indices((size_t) m, 0);
    std::vector<float> values((size_t) m, 0.0f);
    for (int i = 0; i < m; ++i) {
        offsets[(size_t) i] = i;
    }
    ggml_backend_tensor_set(a, a_nvfp4.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_fp32.data(), 0, b_fp32.size() * sizeof(float));
    ggml_backend_tensor_set(outlier_counts, counts.data(), 0, counts.size() * sizeof(int32_t));
    ggml_backend_tensor_set(outlier_offsets, offsets.data(), 0, offsets.size() * sizeof(int32_t));
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
    const float a_scale = global_scale_a != 0.0f ? (1.0f / global_scale_a) : 0.0f;
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
        for (int row = 0; row < n; ++row) {
            const float b_scale = global_scales_b[(size_t) row] != 0.0f ? (1.0f / global_scales_b[(size_t) row]) : 0.0f;
            const float scale = a_scale * b_scale;
            const float scale_ratio = scale != 0.0f ? (bf16_round_f32(scale) / scale) : 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                b_deq_slice[(size_t) row * (size_t) k + (size_t) kk] *= scale_ratio;
            }
        }

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
    _putenv_s("GGML_CUDA_NVFP4_FP4MULMAT", "1");
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
#else
    setenv("GGML_CUDA_NVFP4_NATIVE_NO_FALLBACK", "1", 1);
    setenv("GGML_CUDA_NVFP4_FP4MULMAT", "1", 1);
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
    ok = run_case_default_e2m1_e4m3_tables() && ok;
    ok = run_case_bf16_trunc_nn_quantizer_bytes() && ok;
    ok = run_case_bf16_dynamic_amax_rows() && ok;
    ok = run_case_bf16_dynamic_quantizer_bytes() && ok;
    ok = run_case_bf16_dynamic_quantizer_fast_math_regression(22u) && ok;
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
    ok = run_case_backend_batched_dynamic_rhs(32, 16, 128, 1, 1, 1.0f, 1.0f, 96.0f, 22u, true) && ok;
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
