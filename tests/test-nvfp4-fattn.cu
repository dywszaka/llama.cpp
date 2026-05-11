#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-quants.h"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static constexpr float FP8_E4M3FN_MAX = 448.0f;
static constexpr float FP4_E2M1_MAX   = 6.0f;

struct scoped_env_var {
    const char * name;
    std::string old_value;
    bool had_value;

    scoped_env_var(const char * name, const char * value) : name(name) {
        const char * old = getenv(name);
        had_value = old != nullptr;
        if (had_value) {
            old_value = old;
        }
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }

    ~scoped_env_var() {
#if defined(_WIN32)
        if (had_value) {
            _putenv_s(name, old_value.c_str());
        } else {
            _putenv_s(name, "");
        }
#else
        if (had_value) {
            setenv(name, old_value.c_str(), 1);
        } else {
            unsetenv(name);
        }
#endif
    }
};

static void disable_cuda_truncation() {
#if defined(_WIN32)
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
    _putenv_s("GGML_CUDA_GRAPHS", "0");
    _putenv_s("GGML_CUDA_NVFP4_FATTN", "1");
    _putenv_s("GGML_CUDA_NVFP4_FATTN_NO_FALLBACK", "1");
#else
    setenv("GGML_CUDA_TRUNC_ENABLE", "0", 1);
    setenv("GGML_CUDA_GRAPHS", "0", 1);
    setenv("GGML_CUDA_NVFP4_FATTN", "1", 1);
    setenv("GGML_CUDA_NVFP4_FATTN_NO_FALLBACK", "1", 1);
#endif
}

static float max_abs(const std::vector<float> & x) {
    float amax = 0.0f;
    for (float v : x) {
        amax = fmaxf(amax, fabsf(v));
    }
    return amax;
}

static float max_abs_row(const std::vector<float> & x, int64_t row, int64_t cols) {
    float amax = 0.0f;
    for (int64_t i = 0; i < cols; ++i) {
        amax = fmaxf(amax, fabsf(x[(size_t) row * (size_t) cols + (size_t) i]));
    }
    return amax;
}

static void nvfp4_quant_dequant_rows(
        const std::vector<float> & src,
        std::vector<float> & dst,
        int64_t rows,
        int64_t cols,
        float global_scale_inv) {
    GGML_ASSERT(cols % QK_NVFP4 == 0);
    const int64_t nblk = cols / QK_NVFP4;
    std::vector<block_nvfp4> q((size_t) rows * (size_t) nblk);
    dst.assign(src.size(), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        quantize_row_nvfp4_ref(src.data() + (size_t) r * (size_t) cols, q.data() + (size_t) r * (size_t) nblk, cols, global_scale_inv);
        dequantize_row_nvfp4(q.data() + (size_t) r * (size_t) nblk, dst.data() + (size_t) r * (size_t) cols, cols, global_scale_inv);
    }
}

static void nvfp4_quant_dequant_rows_with_scales(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & quantized,
        std::vector<float> & scales,
        std::vector<float> & dequantized,
        int64_t rows,
        int64_t cols) {
    GGML_ASSERT(cols % QK_NVFP4 == 0);
    const int64_t nblk = cols / QK_NVFP4;
    quantized.assign((size_t) rows * (size_t) nblk, {});
    scales.assign((size_t) rows, 0.0f);
    dequantized.assign(src.size(), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        const float amax = max_abs_row(src, r, cols);
        const float global_scale = amax > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / amax : 0.0f;
        scales[(size_t) r] = global_scale != 0.0f ? 1.0f / global_scale : 0.0f;
        quantize_row_nvfp4_ref(src.data() + (size_t) r * (size_t) cols, quantized.data() + (size_t) r * (size_t) nblk, cols, global_scale);
        dequantize_row_nvfp4(quantized.data() + (size_t) r * (size_t) nblk, dequantized.data() + (size_t) r * (size_t) cols, cols, global_scale);
    }
}

static float alibi_slope(float max_bias, int64_t head, int64_t n_head) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    uint32_t n_head_log2 = 1;
    while ((int64_t) (n_head_log2 << 1) <= n_head) {
        n_head_log2 <<= 1;
    }

    const float m0 = powf(2.0f, -max_bias / (float) n_head_log2);
    const float m1 = powf(2.0f, -max_bias / 2.0f / (float) n_head_log2);
    const float base = head < (int64_t) n_head_log2 ? m0 : m1;
    const int exph = head < (int64_t) n_head_log2 ? (int) head + 1 : 2 * ((int) head - (int) n_head_log2) + 1;

    return powf(base, exph);
}

static std::vector<float> reference_nvfp4_decode(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> * mask,
        int64_t d,
        int64_t kv_len,
        int64_t visible_kv_len,
        float scale,
        float max_bias = 0.0f,
        int64_t head = 0,
        int64_t n_head = 1) {
    GGML_ASSERT(visible_kv_len > 0 && visible_kv_len <= kv_len);
    std::vector<float> q_mean((size_t) d);
    std::vector<float> q_centered((size_t) d);
    for (int64_t i = 0; i < d; ++i) {
        q_mean[(size_t) i] = q[(size_t) i] / 128.0f;
        q_centered[(size_t) i] = q[(size_t) i] - q_mean[(size_t) i];
    }

    std::vector<float> k_mean((size_t) d, 0.0f);
    for (int64_t t = 0; t < visible_kv_len; ++t) {
        for (int64_t i = 0; i < d; ++i) {
            k_mean[(size_t) i] += k[(size_t) t * (size_t) d + (size_t) i];
        }
    }
    for (float & x : k_mean) {
        x /= (float) visible_kv_len;
    }

    std::vector<float> k_centered((size_t) kv_len * (size_t) d, 0.0f);
    for (int64_t t = 0; t < visible_kv_len; ++t) {
        for (int64_t i = 0; i < d; ++i) {
            k_centered[(size_t) t * (size_t) d + (size_t) i] = k[(size_t) t * (size_t) d + (size_t) i] - k_mean[(size_t) i];
        }
    }

    const float q_gscale_inv = (max_abs(q_centered) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(q_centered) : 0.0f;
    const float k_gscale_inv = (max_abs(k_centered) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(k_centered) : 0.0f;
    std::vector<float> v_for_amax((size_t) visible_kv_len * (size_t) d);
    for (int64_t t = 0; t < visible_kv_len; ++t) {
        for (int64_t i = 0; i < d; ++i) {
            v_for_amax[(size_t) t * (size_t) d + (size_t) i] = v[(size_t) t * (size_t) d + (size_t) i];
        }
    }
    const float v_gscale_inv = (max_abs(v_for_amax) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(v_for_amax) : 0.0f;

    std::vector<float> q_deq;
    nvfp4_quant_dequant_rows(q_centered, q_deq, 1, d, q_gscale_inv);

    std::vector<float> k_deq;
    nvfp4_quant_dequant_rows(k_centered, k_deq, kv_len, d, k_gscale_inv);

    std::vector<float> v_by_dim((size_t) d * (size_t) kv_len, 0.0f);
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t t = 0; t < visible_kv_len; ++t) {
            v_by_dim[(size_t) i * (size_t) kv_len + (size_t) t] = v[(size_t) t * (size_t) d + (size_t) i];
        }
    }
    std::vector<float> v_by_dim_deq;
    nvfp4_quant_dequant_rows(v_by_dim, v_by_dim_deq, d, kv_len, v_gscale_inv);

    std::vector<float> scores((size_t) kv_len);
    float score_max = -INFINITY;
    const float slope = alibi_slope(max_bias, head, n_head);
    for (int64_t t = 0; t < kv_len; ++t) {
        float main = 0.0f;
        float corr = 0.0f;
        for (int64_t i = 0; i < d; ++i) {
            main += q_deq[(size_t) i] * k_deq[(size_t) t * (size_t) d + (size_t) i];
            corr += q_mean[(size_t) i] * k_centered[(size_t) t * (size_t) d + (size_t) i];
        }
        scores[(size_t) t] = (main + corr) * scale;
        if (mask != nullptr) {
            scores[(size_t) t] += slope * (*mask)[(size_t) t];
        }
        score_max = fmaxf(score_max, scores[(size_t) t]);
    }

    std::vector<float> probs((size_t) kv_len);
    float prob_sum = 0.0f;
    for (int64_t t = 0; t < kv_len; ++t) {
        probs[(size_t) t] = expf(scores[(size_t) t] - score_max);
        prob_sum += probs[(size_t) t];
    }
    for (float & p : probs) {
        p /= prob_sum;
    }

    const float row_max = max_abs(probs);
    const float first_level_scale = row_max > 0.0f ? row_max / (FP8_E4M3FN_MAX * FP4_E2M1_MAX) : 0.0f;
    std::vector<float> probs_scaled((size_t) kv_len);
    for (int64_t t = 0; t < kv_len; ++t) {
        probs_scaled[(size_t) t] = first_level_scale > 0.0f ? probs[(size_t) t] / first_level_scale : 0.0f;
    }
    std::vector<float> probs_deq_scaled;
    nvfp4_quant_dequant_rows(probs_scaled, probs_deq_scaled, 1, kv_len, 1.0f);
    for (int64_t t = 0; t < kv_len; ++t) {
        probs_deq_scaled[(size_t) t] *= first_level_scale;
    }

    std::vector<float> out((size_t) d, 0.0f);
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t t = 0; t < kv_len; ++t) {
            out[(size_t) i] += probs_deq_scaled[(size_t) t] * v_by_dim_deq[(size_t) i * (size_t) kv_len + (size_t) t];
        }
    }
    return out;
}

static std::vector<float> reference_nvfp4_prefill(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        int64_t d,
        int64_t q_len,
        int64_t kv_len,
        float scale) {
    std::vector<float> q_centered((size_t) q_len * (size_t) d);
    std::vector<float> q_mean(q_centered.size());
    for (int64_t block = 0; block < (q_len + 127) / 128; ++block) {
        const int64_t block_start = block * 128;
        const int64_t block_end = std::min<int64_t>(block_start + 128, q_len);
        for (int64_t i = 0; i < d; ++i) {
            float mean = 0.0f;
            for (int64_t t = block_start; t < block_end; ++t) {
                mean += q[(size_t) t * (size_t) d + (size_t) i];
            }
            mean /= 128.0f;
            for (int64_t t = block_start; t < block_end; ++t) {
                const size_t idx = (size_t) t * (size_t) d + (size_t) i;
                q_mean[idx] = mean;
                q_centered[idx] = q[idx] - mean;
            }
        }
    }

    std::vector<float> k_centered((size_t) kv_len * (size_t) d);
    for (int64_t i = 0; i < d; ++i) {
        float mean = 0.0f;
        for (int64_t t = 0; t < kv_len; ++t) {
            mean += k[(size_t) t * (size_t) d + (size_t) i];
        }
        mean /= (float) kv_len;
        for (int64_t t = 0; t < kv_len; ++t) {
            const size_t idx = (size_t) t * (size_t) d + (size_t) i;
            k_centered[idx] = k[idx] - mean;
        }
    }

    const float q_gscale_inv = (max_abs(q_centered) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(q_centered) : 0.0f;
    const float k_gscale_inv = (max_abs(k_centered) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(k_centered) : 0.0f;

    std::vector<float> q_deq;
    nvfp4_quant_dequant_rows(q_centered, q_deq, q_len, d, q_gscale_inv);
    std::vector<float> k_deq;
    nvfp4_quant_dequant_rows(k_centered, k_deq, kv_len, d, k_gscale_inv);

    std::vector<float> v_by_dim((size_t) d * (size_t) kv_len);
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t t = 0; t < kv_len; ++t) {
            v_by_dim[(size_t) i * (size_t) kv_len + (size_t) t] = v[(size_t) t * (size_t) d + (size_t) i];
        }
    }
    const float v_gscale_inv = (max_abs(v_by_dim) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(v_by_dim) : 0.0f;
    std::vector<float> v_deq;
    nvfp4_quant_dequant_rows(v_by_dim, v_deq, d, kv_len, v_gscale_inv);

    std::vector<float> out((size_t) q_len * (size_t) d, 0.0f);
    std::vector<float> scores((size_t) kv_len);
    std::vector<float> probs((size_t) kv_len);
    for (int64_t qt = 0; qt < q_len; ++qt) {
        float score_max = -INFINITY;
        for (int64_t kt = 0; kt < kv_len; ++kt) {
            float main = 0.0f;
            float corr = 0.0f;
            for (int64_t i = 0; i < d; ++i) {
                main += q_deq[(size_t) qt * (size_t) d + (size_t) i] * k_deq[(size_t) kt * (size_t) d + (size_t) i];
                corr += q_mean[(size_t) qt * (size_t) d + (size_t) i] * k_centered[(size_t) kt * (size_t) d + (size_t) i];
            }
            scores[(size_t) kt] = (main + corr) * scale;
            score_max = fmaxf(score_max, scores[(size_t) kt]);
        }

        float prob_sum = 0.0f;
        for (int64_t kt = 0; kt < kv_len; ++kt) {
            probs[(size_t) kt] = expf(scores[(size_t) kt] - score_max);
            prob_sum += probs[(size_t) kt];
        }
        for (float & p : probs) {
            p /= prob_sum;
        }

        const float row_max = max_abs(probs);
        const float first_level_scale = row_max > 0.0f ? row_max / (FP8_E4M3FN_MAX * FP4_E2M1_MAX) : 0.0f;
        std::vector<float> probs_scaled((size_t) kv_len);
        for (int64_t kt = 0; kt < kv_len; ++kt) {
            probs_scaled[(size_t) kt] = first_level_scale > 0.0f ? probs[(size_t) kt] / first_level_scale : 0.0f;
        }
        std::vector<float> probs_deq;
        nvfp4_quant_dequant_rows(probs_scaled, probs_deq, 1, kv_len, 1.0f);
        for (float & p : probs_deq) {
            p *= first_level_scale;
        }

        for (int64_t i = 0; i < d; ++i) {
            float acc = 0.0f;
            for (int64_t kt = 0; kt < kv_len; ++kt) {
                acc += probs_deq[(size_t) kt] * v_deq[(size_t) i * (size_t) kv_len + (size_t) kt];
            }
            out[(size_t) qt * (size_t) d + (size_t) i] = acc;
        }
    }

    return out;
}

static std::vector<float> reference_nvfp4_decode_no_k_smooth(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        int64_t d,
        int64_t kv_len,
        float scale) {
    std::vector<float> q_mean((size_t) d);
    std::vector<float> q_centered((size_t) d);
    for (int64_t i = 0; i < d; ++i) {
        q_mean[(size_t) i] = q[(size_t) i] / 128.0f;
        q_centered[(size_t) i] = q[(size_t) i] - q_mean[(size_t) i];
    }

    const float q_gscale_inv = (max_abs(q_centered) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(q_centered) : 0.0f;
    std::vector<float> q_deq;
    nvfp4_quant_dequant_rows(q_centered, q_deq, 1, d, q_gscale_inv);

    const float k_gscale_inv = (max_abs(k) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(k) : 0.0f;
    std::vector<float> k_deq;
    nvfp4_quant_dequant_rows(k, k_deq, kv_len, d, k_gscale_inv);

    std::vector<float> v_by_dim((size_t) d * (size_t) kv_len, 0.0f);
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t t = 0; t < kv_len; ++t) {
            v_by_dim[(size_t) i * (size_t) kv_len + (size_t) t] = v[(size_t) t * (size_t) d + (size_t) i];
        }
    }
    const float v_gscale_inv = (max_abs(v_by_dim) > 0.0f) ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(v_by_dim) : 0.0f;
    std::vector<float> v_by_dim_deq;
    nvfp4_quant_dequant_rows(v_by_dim, v_by_dim_deq, d, kv_len, v_gscale_inv);

    std::vector<float> scores((size_t) kv_len);
    float score_max = -INFINITY;
    for (int64_t t = 0; t < kv_len; ++t) {
        float main = 0.0f;
        float corr = 0.0f;
        for (int64_t i = 0; i < d; ++i) {
            main += q_deq[(size_t) i] * k_deq[(size_t) t * (size_t) d + (size_t) i];
            corr += q_mean[(size_t) i] * k[(size_t) t * (size_t) d + (size_t) i];
        }
        scores[(size_t) t] = (main + corr) * scale;
        score_max = fmaxf(score_max, scores[(size_t) t]);
    }

    std::vector<float> probs((size_t) kv_len);
    float prob_sum = 0.0f;
    for (int64_t t = 0; t < kv_len; ++t) {
        probs[(size_t) t] = expf(scores[(size_t) t] - score_max);
        prob_sum += probs[(size_t) t];
    }
    for (float & p : probs) {
        p /= prob_sum;
    }

    const float row_max = max_abs(probs);
    const float first_level_scale = row_max > 0.0f ? row_max / (FP8_E4M3FN_MAX * FP4_E2M1_MAX) : 0.0f;
    std::vector<float> probs_scaled((size_t) kv_len);
    for (int64_t t = 0; t < kv_len; ++t) {
        probs_scaled[(size_t) t] = first_level_scale > 0.0f ? probs[(size_t) t] / first_level_scale : 0.0f;
    }
    std::vector<float> probs_deq_scaled;
    nvfp4_quant_dequant_rows(probs_scaled, probs_deq_scaled, 1, kv_len, 1.0f);
    for (int64_t t = 0; t < kv_len; ++t) {
        probs_deq_scaled[(size_t) t] *= first_level_scale;
    }

    std::vector<float> out((size_t) d, 0.0f);
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t t = 0; t < kv_len; ++t) {
            out[(size_t) i] += probs_deq_scaled[(size_t) t] * v_by_dim_deq[(size_t) i * (size_t) kv_len + (size_t) t];
        }
    }
    return out;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        diff = fmaxf(diff, fabsf(a[i] - b[i]));
    }
    return diff;
}

static bool test_flash_attn_nvfp4_flags_roundtrip() {
    ggml_init_params params = {
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 1, 1, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 256, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 256, 1, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / 8.0f, 0.0f, 0.0f);

    const int32_t flags =
        GGML_FLASH_ATTN_FLAG_NVFP4_QKVP |
        GGML_FLASH_ATTN_FLAG_NVFP4_P_TWOLEVEL |
        GGML_FLASH_ATTN_FLAG_NVFP4_SMOOTH_QK;
    ggml_flash_attn_ext_set_flags(fa, flags);

    const int32_t got = ggml_flash_attn_ext_get_flags(fa);
    ggml_free(ctx);

    if (got != flags) {
        std::fprintf(stderr, "flags mismatch got=%d expected=%d\n", got, flags);
        return false;
    }

    return true;
}

static bool test_flash_attn_nvfp4_decode_matches_reference() {
    disable_cuda_truncation();

    static constexpr int64_t d = 64;
    static constexpr int64_t q_len = 1;
    static constexpr int64_t kv_len = 256;
    static constexpr int64_t n_head = 1;
    const float scale = 1.0f / sqrtf((float) d);

    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to initialize CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, q_len, n_head, 1);
    ggml_tensor * k_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 2 * kv_len, n_head, 1);
    ggml_tensor * v_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 2 * kv_len, n_head, 1);
    ggml_tensor * k = ggml_view_4d(ctx, k_base, d, kv_len, n_head, 1, 2 * k_base->nb[1], k_base->nb[2], k_base->nb[3], 0);
    ggml_tensor * v = ggml_view_4d(ctx, v_base, d, kv_len, n_head, 1, 2 * v_base->nb[1], v_base->nb[2], v_base->nb[3], 0);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_flash_attn_ext_set_flags(fa,
            GGML_FLASH_ATTN_FLAG_NVFP4_QKVP |
            GGML_FLASH_ATTN_FLAG_NVFP4_P_TWOLEVEL |
            GGML_FLASH_ATTN_FLAG_NVFP4_SMOOTH_QK);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> q_data((size_t) d);
    std::vector<float> k_data((size_t) kv_len * (size_t) d);
    std::vector<float> v_data((size_t) kv_len * (size_t) d);
    for (int64_t i = 0; i < d; ++i) {
        q_data[(size_t) i] = 0.35f * sinf(0.17f * (float) i) + 0.11f * cosf(0.07f * (float) i);
    }
    for (int64_t t = 0; t < kv_len; ++t) {
        for (int64_t i = 0; i < d; ++i) {
            const float x = (float) (t * d + i);
            k_data[(size_t) t * (size_t) d + (size_t) i] = 0.27f * sinf(0.013f * x) + 0.05f * cosf(0.031f * x);
            v_data[(size_t) t * (size_t) d + (size_t) i] = 1.80f * sinf(0.019f * x + 0.3f) + 0.40f * cosf(0.023f * x);
        }
    }

    std::vector<ggml_fp16_t> k_f16(k_data.size());
    std::vector<ggml_fp16_t> v_f16(v_data.size());
    ggml_fp32_to_fp16_row(k_data.data(), k_f16.data(), (int64_t) k_data.size());
    ggml_fp32_to_fp16_row(v_data.data(), v_f16.data(), (int64_t) v_data.size());
    std::vector<float> k_ref(k_data.size());
    std::vector<float> v_ref(v_data.size());
    ggml_fp16_to_fp32_row(k_f16.data(), k_ref.data(), (int64_t) k_ref.size());
    ggml_fp16_to_fp32_row(v_f16.data(), v_ref.data(), (int64_t) v_ref.size());

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    std::vector<ggml_fp16_t> k_base_f16((size_t) 2 * k_f16.size(), ggml_fp32_to_fp16(123.0f));
    std::vector<ggml_fp16_t> v_base_f16((size_t) 2 * v_f16.size(), ggml_fp32_to_fp16(-77.0f));
    for (int64_t t = 0; t < kv_len; ++t) {
        for (int64_t i = 0; i < d; ++i) {
            k_base_f16[((size_t) 2 * (size_t) t) * (size_t) d + (size_t) i] = k_f16[(size_t) t * (size_t) d + (size_t) i];
            v_base_f16[((size_t) 2 * (size_t) t) * (size_t) d + (size_t) i] = v_f16[(size_t) t * (size_t) d + (size_t) i];
        }
    }
    ggml_backend_tensor_set(k_base, k_base_f16.data(), 0, k_base_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v_base, v_base_f16.data(), 0, v_base_f16.size() * sizeof(ggml_fp16_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> got((size_t) ggml_nelements(fa));
    ggml_backend_tensor_get(fa, got.data(), 0, got.size() * sizeof(float));

    const std::vector<float> ref = reference_nvfp4_decode(q_data, k_ref, v_ref, nullptr, d, kv_len, kv_len, scale);
    const float diff = max_abs_diff(ref, got);
    if (diff > 1.0e-4f) {
        std::fprintf(stderr, "decode mismatch max_abs_diff=%f\n", diff);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static bool test_flash_attn_nvfp4_decode_mask_matches_reference() {
    disable_cuda_truncation();

    static constexpr int64_t d = 64;
    static constexpr int64_t q_len = 1;
    static constexpr int64_t kv_len = 256;
    static constexpr int64_t n_head = 1;
    const float scale = 1.0f / sqrtf((float) d);

    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to initialize CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, q_len, n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv_len, n_head, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv_len, n_head, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_len, GGML_PAD(q_len, GGML_KQ_MASK_PAD), 1, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_flash_attn_ext_set_flags(fa,
            GGML_FLASH_ATTN_FLAG_NVFP4_QKVP |
            GGML_FLASH_ATTN_FLAG_NVFP4_P_TWOLEVEL |
            GGML_FLASH_ATTN_FLAG_NVFP4_SMOOTH_QK);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> q_data((size_t) d);
    std::vector<float> k_data((size_t) kv_len * (size_t) d);
    std::vector<float> v_data((size_t) kv_len * (size_t) d);
    static constexpr int64_t visible_kv_len = kv_len / 2;
    std::vector<float> mask_data((size_t) kv_len, 0.0f);
    for (int64_t i = 0; i < d; ++i) {
        q_data[(size_t) i] = 0.31f * sinf(0.19f * (float) i) + 0.07f * cosf(0.03f * (float) i);
    }
    for (int64_t t = 0; t < kv_len; ++t) {
        if (t >= visible_kv_len) {
            mask_data[(size_t) t] = -INFINITY;
        }
        for (int64_t i = 0; i < d; ++i) {
            const float x = (float) (t * d + i);
            if (t < visible_kv_len) {
                k_data[(size_t) t * (size_t) d + (size_t) i] = 0.24f * sinf(0.011f * x) + 0.08f * cosf(0.027f * x);
                v_data[(size_t) t * (size_t) d + (size_t) i] = 1.20f * sinf(0.017f * x + 0.1f) + 0.30f * cosf(0.021f * x);
            } else {
                k_data[(size_t) t * (size_t) d + (size_t) i] = 300.0f + 0.01f * (float) ((t + i) % 7);
                v_data[(size_t) t * (size_t) d + (size_t) i] = -200.0f + 0.01f * (float) ((t + i) % 11);
            }
        }
    }

    std::vector<ggml_fp16_t> k_f16(k_data.size());
    std::vector<ggml_fp16_t> v_f16(v_data.size());
    ggml_fp32_to_fp16_row(k_data.data(), k_f16.data(), (int64_t) k_data.size());
    ggml_fp32_to_fp16_row(v_data.data(), v_f16.data(), (int64_t) v_data.size());
    std::vector<float> k_ref(k_data.size());
    std::vector<float> v_ref(v_data.size());
    ggml_fp16_to_fp32_row(k_f16.data(), k_ref.data(), (int64_t) k_ref.size());
    ggml_fp16_to_fp32_row(v_f16.data(), v_ref.data(), (int64_t) v_ref.size());

    std::vector<ggml_fp16_t> mask_f16((size_t) ggml_nelements(m), ggml_fp32_to_fp16(0.0f));
    for (int64_t t = 0; t < kv_len; ++t) {
        mask_f16[(size_t) t] = ggml_fp32_to_fp16(mask_data[(size_t) t]);
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m, mask_f16.data(), 0, mask_f16.size() * sizeof(ggml_fp16_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "masked graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> got((size_t) ggml_nelements(fa));
    ggml_backend_tensor_get(fa, got.data(), 0, got.size() * sizeof(float));

    const std::vector<float> ref = reference_nvfp4_decode(q_data, k_ref, v_ref, &mask_data, d, kv_len, visible_kv_len, scale);
    const float diff = max_abs_diff(ref, got);
    if (diff > 1.0e-4f) {
        std::fprintf(stderr, "masked decode mismatch max_abs_diff=%f\n", diff);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static bool test_flash_attn_nvfp4_prefill_smoke() {
    disable_cuda_truncation();

    static constexpr int64_t d = 64;
    static constexpr int64_t q_len = 2;
    static constexpr int64_t kv_len = 256;
    static constexpr int64_t n_head = 1;
    const float scale = 1.0f / sqrtf((float) d);

    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to initialize CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, q_len, n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv_len, n_head, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv_len, n_head, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_flash_attn_ext_set_flags(fa,
            GGML_FLASH_ATTN_FLAG_NVFP4_QKVP |
            GGML_FLASH_ATTN_FLAG_NVFP4_P_TWOLEVEL |
            GGML_FLASH_ATTN_FLAG_NVFP4_SMOOTH_QK);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> q_data((size_t) d * (size_t) q_len);
    std::vector<float> k_data((size_t) kv_len * (size_t) d);
    std::vector<float> v_data((size_t) kv_len * (size_t) d);
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = 0.21f * sinf(0.11f * (float) i);
    }
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = 0.17f * cosf(0.007f * (float) i);
        v_data[i] = 0.90f * sinf(0.009f * (float) i + 0.2f);
    }
    std::vector<ggml_fp16_t> k_f16(k_data.size());
    std::vector<ggml_fp16_t> v_f16(v_data.size());
    ggml_fp32_to_fp16_row(k_data.data(), k_f16.data(), (int64_t) k_data.size());
    ggml_fp32_to_fp16_row(v_data.data(), v_f16.data(), (int64_t) v_data.size());

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> got((size_t) ggml_nelements(fa));
    ggml_backend_tensor_get(fa, got.data(), 0, got.size() * sizeof(float));
    if (max_abs(got) == 0.0f) {
        std::fprintf(stderr, "prefill output is all zero\n");
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static bool test_flash_attn_nvfp4_k_cache_no_k_smooth_matches_reference() {
    disable_cuda_truncation();
    scoped_env_var no_k_smooth("GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH", "1");

    static constexpr int64_t d = 64;
    static constexpr int64_t q_len = 1;
    static constexpr int64_t kv_len = 256;
    static constexpr int64_t n_head = 1;
    const float scale = 1.0f / sqrtf((float) d);

    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to initialize CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, q_len, n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_NVFP4, d, kv_len, n_head, 1);
    ggml_tensor * k_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kv_len);
    ggml_tensor_set_nvfp4_scale(k, k_scale);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv_len, n_head, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_flash_attn_ext_set_flags(fa,
            GGML_FLASH_ATTN_FLAG_NVFP4_QKVP |
            GGML_FLASH_ATTN_FLAG_NVFP4_P_TWOLEVEL |
            GGML_FLASH_ATTN_FLAG_NVFP4_SMOOTH_QK);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> q_data((size_t) d);
    std::vector<float> k_data((size_t) kv_len * (size_t) d);
    std::vector<float> v_data((size_t) kv_len * (size_t) d);
    for (int64_t i = 0; i < d; ++i) {
        q_data[(size_t) i] = 0.33f * sinf(0.13f * (float) i) + 0.09f * cosf(0.05f * (float) i);
    }
    for (int64_t t = 0; t < kv_len; ++t) {
        for (int64_t i = 0; i < d; ++i) {
            const float x = (float) (t * d + i);
            k_data[(size_t) t * (size_t) d + (size_t) i] = 0.73f * sinf(0.015f * x) + 0.19f * cosf(0.037f * x);
            v_data[(size_t) t * (size_t) d + (size_t) i] = 1.10f * sinf(0.017f * x + 0.2f) + 0.37f * cosf(0.021f * x);
        }
    }

    std::vector<block_nvfp4> k_q;
    std::vector<float> k_scales;
    std::vector<float> k_ref;
    nvfp4_quant_dequant_rows_with_scales(k_data, k_q, k_scales, k_ref, kv_len, d);

    std::vector<ggml_fp16_t> v_f16(v_data.size());
    ggml_fp32_to_fp16_row(v_data.data(), v_f16.data(), (int64_t) v_data.size());
    std::vector<float> v_ref(v_data.size());
    ggml_fp16_to_fp32_row(v_f16.data(), v_ref.data(), (int64_t) v_ref.size());

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_q.data(), 0, k_q.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_set(k_scale, k_scales.data(), 0, k_scales.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "NVFP4 K cache graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> got((size_t) ggml_nelements(fa));
    ggml_backend_tensor_get(fa, got.data(), 0, got.size() * sizeof(float));

    const std::vector<float> ref = reference_nvfp4_decode_no_k_smooth(q_data, k_ref, v_ref, d, kv_len, scale);
    const float diff = max_abs_diff(ref, got);
    if (diff > 1.0e-4f) {
        std::fprintf(stderr, "NVFP4 K cache no-K-smooth mismatch max_abs_diff=%f\n", diff);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static bool test_flash_attn_nvfp4_prefill_matches_sega3_quantization() {
    disable_cuda_truncation();

    static constexpr int64_t d = 64;
    static constexpr int64_t q_len = 17;
    static constexpr int64_t kv_len = 256;
    static constexpr int64_t n_head = 1;
    const float scale = 1.0f / sqrtf((float) d);

    ggml_init_params params = {
        /* .mem_size   = */ 32 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to initialize CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, q_len, n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv_len, n_head, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kv_len, n_head, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_flash_attn_ext_set_flags(fa,
            GGML_FLASH_ATTN_FLAG_NVFP4_QKVP |
            GGML_FLASH_ATTN_FLAG_NVFP4_P_TWOLEVEL |
            GGML_FLASH_ATTN_FLAG_NVFP4_SMOOTH_QK);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> q_data((size_t) q_len * (size_t) d);
    std::vector<float> k_data((size_t) kv_len * (size_t) d);
    std::vector<float> v_data((size_t) kv_len * (size_t) d);
    for (int64_t t = 0; t < q_len; ++t) {
        const float amp = t == 0 ? 0.03f : (0.20f + 0.017f * (float) t);
        for (int64_t i = 0; i < d; ++i) {
            q_data[(size_t) t * (size_t) d + (size_t) i] =
                    amp * sinf(0.13f * (float) (t*d + i)) + 0.07f * cosf(0.19f * (float) i);
        }
    }
    for (int64_t t = 0; t < kv_len; ++t) {
        for (int64_t i = 0; i < d; ++i) {
            const float x = (float) (t * d + i);
            k_data[(size_t) t * (size_t) d + (size_t) i] = 1.70f * sinf(0.017f * x) + 0.31f * cosf(0.029f * x);
            v_data[(size_t) t * (size_t) d + (size_t) i] = 2.40f * sinf(0.011f * x + 0.2f) + 0.53f * cosf(0.023f * x);
        }
    }

    std::vector<ggml_fp16_t> k_f16(k_data.size());
    std::vector<ggml_fp16_t> v_f16(v_data.size());
    ggml_fp32_to_fp16_row(k_data.data(), k_f16.data(), (int64_t) k_data.size());
    ggml_fp32_to_fp16_row(v_data.data(), v_f16.data(), (int64_t) v_data.size());
    std::vector<float> k_ref(k_data.size());
    std::vector<float> v_ref(v_data.size());
    ggml_fp16_to_fp32_row(k_f16.data(), k_ref.data(), (int64_t) k_ref.size());
    ggml_fp16_to_fp32_row(v_f16.data(), v_ref.data(), (int64_t) v_ref.size());

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_f16.data(), 0, k_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, v_f16.data(), 0, v_f16.size() * sizeof(ggml_fp16_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> got((size_t) ggml_nelements(fa));
    ggml_backend_tensor_get(fa, got.data(), 0, got.size() * sizeof(float));
    std::vector<float> got_by_q((size_t) q_len * (size_t) d);
    for (int64_t qt = 0; qt < q_len; ++qt) {
        for (int64_t i = 0; i < d; ++i) {
            got_by_q[(size_t) qt * (size_t) d + (size_t) i] = got[(size_t) qt * (size_t) d + (size_t) i];
        }
    }

    const std::vector<float> ref = reference_nvfp4_prefill(q_data, k_ref, v_ref, d, q_len, kv_len, scale);
    const float diff = max_abs_diff(ref, got_by_q);
    if (diff > 1.0e-2f) {
        std::fprintf(stderr, "prefill sega3 quantization mismatch max_abs_diff=%f\n", diff);
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

int main() {
    if (!test_flash_attn_nvfp4_flags_roundtrip()) {
        return 1;
    }
    if (!test_flash_attn_nvfp4_decode_matches_reference()) {
        return 1;
    }
    if (!test_flash_attn_nvfp4_decode_mask_matches_reference()) {
        return 1;
    }
    if (!test_flash_attn_nvfp4_prefill_smoke()) {
        return 1;
    }
    if (!test_flash_attn_nvfp4_k_cache_no_k_smooth_matches_reference()) {
        return 1;
    }
    if (!test_flash_attn_nvfp4_prefill_matches_sega3_quantization()) {
        return 1;
    }

    return 0;
}
