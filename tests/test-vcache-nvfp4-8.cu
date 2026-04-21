#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-quants.h"
#include "../ggml/src/ggml-impl.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr float NVFP4_FP4_MAX = 6.0f;
static constexpr float NVFP4_E4M3_MAX = 224.0f;
static constexpr float NVFP4_GLOBAL_SCALE_MAX = NVFP4_FP4_MAX * NVFP4_E4M3_MAX;
static constexpr int8_t NVFP4_VALUES[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

static void disable_cuda_truncation() {
#if defined(_WIN32)
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
#else
    setenv("GGML_CUDA_TRUNC_ENABLE", "0", 1);
#endif
}

static bool almost_equal(float a, float b, float atol, float rtol) {
    const float diff = fabsf(a - b);
    const float tol = atol + rtol * fmaxf(fabsf(a), fabsf(b));
    return diff <= tol;
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double mse_ref_got = 0.0;
    double mse_ref_0 = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = ref[i];
        const double g = got[i];
        mse_ref_got += (r - g) * (r - g);
        mse_ref_0 += r * r;
    }

    return mse_ref_0 > 0.0 ? mse_ref_got / mse_ref_0 : mse_ref_got;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_abs = fmaxf(max_abs, fabsf(a[i] - b[i]));
    }
    return max_abs;
}

static std::vector<float> make_signal(size_t n, float amplitude, float bias, float phase) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        const float x = (float) i;
        out[i] = bias
            + amplitude * sinf(0.043f * x + phase)
            + 0.25f * amplitude * cosf(0.017f * x - 0.7f * phase);
    }
    return out;
}

static uint8_t best_index_nvfp4_ref(float x) {
    uint8_t best_index = 0;
    float best_err = fabsf((float) NVFP4_VALUES[0] - x);
    for (int i = 1; i < 16; ++i) {
        const float err = fabsf((float) NVFP4_VALUES[i] - x);
        if (err < best_err) {
            best_err = err;
            best_index = (uint8_t) i;
        }
    }
    return best_index;
}

static uint8_t best_index_e4m3_ref(float x) {
    uint8_t best_index = 0;
    float best_err = INFINITY;
    for (int i = 0; i < 256; ++i) {
        const float v = GGML_E4M3_TO_FP32((uint8_t) i);
        if (!isfinite(v)) {
            continue;
        }
        const float err = fabsf(v - x);
        if (err < best_err) {
            best_err = err;
            best_index = (uint8_t) i;
        }
    }
    return best_index;
}

static void quantize_row_nvfp4_8_test_ref(
        const float * src,
        block_nvfp4_8 * dst,
        int64_t k,
        float global_scale) {
    if (k % QK_NVFP4_8 != 0) {
        std::fprintf(stderr, "k must be divisible by QK_NVFP4_8\n");
        std::abort();
    }

    const int64_t nb = k / QK_NVFP4_8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        float vmax = 0.0f;
        for (int j = 0; j < QK_NVFP4_8; ++j) {
            vmax = fmaxf(vmax, fabsf(src[ib * QK_NVFP4_8 + j]));
        }

        const float scale = global_scale * (vmax / NVFP4_FP4_MAX);
        const uint8_t scale_q = best_index_e4m3_ref(scale);
        const float scale_f = GGML_E4M3_TO_FP32_HALF(scale_q);
        const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;

        dst[ib].e = scale_q;
        for (int j = 0; j < QK_NVFP4_8 / 2; ++j) {
            const uint8_t q0 = best_index_nvfp4_ref(src[ib * QK_NVFP4_8 + 2*j + 0] * inv_scale);
            const uint8_t q1 = best_index_nvfp4_ref(src[ib * QK_NVFP4_8 + 2*j + 1] * inv_scale);
            dst[ib].qs[j] = q0 | (q1 << 4);
        }
    }
}

static void dequantize_row_nvfp4_8_test_ref(
        const block_nvfp4_8 * src,
        float * dst,
        int64_t k,
        float global_scale) {
    if (k % QK_NVFP4_8 != 0) {
        std::fprintf(stderr, "k must be divisible by QK_NVFP4_8\n");
        std::abort();
    }

    const int64_t nb = k / QK_NVFP4_8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float d = GGML_E4M3_TO_FP32_HALF(src[ib].e);
        const float out_scale = global_scale != 0.0f ? d / global_scale : 0.0f;
        for (int j = 0; j < QK_NVFP4_8 / 2; ++j) {
            const uint8_t packed = src[ib].qs[j];
            dst[ib * QK_NVFP4_8 + 2*j + 0] = (float) NVFP4_VALUES[packed & 0x0F] * out_scale;
            dst[ib * QK_NVFP4_8 + 2*j + 1] = (float) NVFP4_VALUES[packed >> 4] * out_scale;
        }
    }
}

static bool test_set_rows_roundtrip() {
    ggml_init_params params = {
        /* .mem_size   = */ 4 * 1024 * 1024,
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

    constexpr int64_t k = 64;
    constexpr int64_t rows = 3;

    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4_8, k, rows);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows);
    ggml_tensor_set_nvfp4_scale(cache, scale);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, rows);
    ggml_tensor * set = ggml_set_rows(ctx, cache, src, idx);
    ggml_tensor_set_nvfp4_scale(set, scale);
    ggml_tensor * out = ggml_cpy(ctx, cache, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, set);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> src_data = make_signal((size_t) k * (size_t) rows, 5.0f, 0.05f, 0.3f);
    const std::vector<int64_t> idx_data = { 2, 0, 1 };
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "set_rows graph failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> got((size_t) k * (size_t) rows, 0.0f);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));

    std::vector<float> expected(got.size(), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        const float * src_row = src_data.data() + (size_t) r * (size_t) k;
        float amax = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
            amax = fmaxf(amax, fabsf(src_row[i]));
        }
        const float global_scale = amax > 0.0f ? NVFP4_GLOBAL_SCALE_MAX / amax : 0.0f;
        std::vector<block_nvfp4_8> q((size_t) k / QK_NVFP4_8);
        quantize_row_nvfp4_8_test_ref(src_row, q.data(), k, global_scale);
        dequantize_row_nvfp4_8_test_ref(q.data(), expected.data() + (size_t) idx_data[(size_t) r] * (size_t) k, k, global_scale);
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (!almost_equal(expected[i], got[i], 1e-5f, 1e-5f)) {
            std::fprintf(stderr, "set_rows roundtrip mismatch i=%zu expected=%f got=%f\n", i, expected[i], got[i]);
            ggml_backend_buffer_free(buf);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return false;
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static bool run_kq_case(int64_t k, int64_t n_kv, int64_t n_tokens) {
    ggml_init_params params = {
        /* .mem_size   = */ 8 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to init ggml context for KQ\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to init CUDA backend for KQ\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * k_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4_8, k, n_kv);
    ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n_tokens);
    ggml_tensor * kq = ggml_mul_mat(ctx, k_cache, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, kq);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate KQ backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> k_src = make_signal((size_t) k * (size_t) n_kv, 4.0f, 0.02f, 0.15f);
    const std::vector<float> q_src = make_signal((size_t) k * (size_t) n_tokens, 0.9f, -0.01f, 0.45f);
    std::vector<block_nvfp4_8> k_q((size_t) n_kv * (size_t) (k / QK_NVFP4_8));
    std::vector<float> k_deq(k_src.size(), 0.0f);
    std::vector<float> q_deq(q_src.size(), 0.0f);

    for (int64_t row = 0; row < n_kv; ++row) {
        const float * k_row = k_src.data() + (size_t) row * (size_t) k;
        const float global_scale = 1.0f;
        quantize_row_nvfp4_8_test_ref(k_row, k_q.data() + (size_t) row * (size_t) (k / QK_NVFP4_8), k, global_scale);
        dequantize_row_nvfp4_8_test_ref(k_q.data() + (size_t) row * (size_t) (k / QK_NVFP4_8),
                k_deq.data() + (size_t) row * (size_t) k, k, global_scale);
    }

    for (int64_t col = 0; col < n_tokens; ++col) {
        const float * q_col = q_src.data() + (size_t) col * (size_t) k;
        float amax = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
            amax = fmaxf(amax, fabsf(q_col[i]));
        }
        const float global_scale = amax > 0.0f ? NVFP4_GLOBAL_SCALE_MAX / amax : 0.0f;
        std::vector<block_nvfp4_8> q_tmp((size_t) k / QK_NVFP4_8);
        quantize_row_nvfp4_8_test_ref(q_col, q_tmp.data(), k, global_scale);
        dequantize_row_nvfp4_8_test_ref(q_tmp.data(), q_deq.data() + (size_t) col * (size_t) k, k, global_scale);
    }

    ggml_backend_tensor_set(k_cache, k_q.data(), 0, k_q.size() * sizeof(block_nvfp4_8));
    ggml_backend_tensor_set(q, q_src.data(), 0, q_src.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "KQ graph failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> got((size_t) n_kv * (size_t) n_tokens, 0.0f);
    std::vector<float> ref(got.size(), 0.0f);
    ggml_backend_tensor_get(kq, got.data(), 0, got.size() * sizeof(float));

    for (int64_t col = 0; col < n_tokens; ++col) {
        for (int64_t row = 0; row < n_kv; ++row) {
            float acc = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                acc += k_deq[(size_t) row * (size_t) k + (size_t) i] * q_deq[(size_t) col * (size_t) k + (size_t) i];
            }
            ref[(size_t) col * (size_t) n_kv + (size_t) row] = acc;
        }
    }

    const double err_nmse = nmse(ref, got);
    const float err_max_abs = max_abs_diff(ref, got);
    std::fprintf(stderr, "KQ nvfp4_8 k=%lld n_kv=%lld n_tokens=%lld nmse=%g max_abs=%g\n",
            (long long) k, (long long) n_kv, (long long) n_tokens, err_nmse, err_max_abs);
    const bool ok = err_nmse <= 1e-6 || err_max_abs <= 1e-4f;

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok;
}

int main() {
    disable_cuda_truncation();

    int dev_count = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count <= 0) {
        std::puts("test-vcache-nvfp4-8: SKIP (no CUDA device)");
        return 0;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        std::puts("test-vcache-nvfp4-8: SKIP (failed to select CUDA device 0)");
        return 0;
    }

    if (!test_set_rows_roundtrip()) {
        return 1;
    }
    if (!run_kq_case(64, 17, 1)) {
        return 1;
    }
    if (!run_kq_case(128, 33, 5)) {
        return 1;
    }

    std::puts("test-vcache-nvfp4-8: ok");
    return 0;
}
