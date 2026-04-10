#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

static bool test_fp8_row_roundtrip(ggml_type type, float scale) {
    std::vector<float> src = {
        -2.5f, -1.0f, -0.25f, 0.0f,
         0.125f, 0.5f, 1.5f, 3.0f,
    };
    std::vector<uint8_t> q(src.size());
    std::vector<float> dst(src.size(), 0.0f);

    if (type == GGML_TYPE_FP8_E4M3_S3) {
        quantize_row_fp8_e4m3_s3_ref(src.data(), q.data(), (int64_t) src.size());
        dequantize_row_fp8_e4m3_s3(q.data(), dst.data(), (int64_t) src.size());
    } else {
        quantize_row_fp8_e4m3_s5_ref(src.data(), q.data(), (int64_t) src.size());
        dequantize_row_fp8_e4m3_s5(q.data(), dst.data(), (int64_t) src.size());
    }

    for (size_t i = 0; i < src.size(); ++i) {
        const float scaled = src[i] * scale;
        if (fabsf(scaled) > 448.0f) {
            continue;
        }
        if (!almost_equal(src[i], dst[i], 0.08f, 0.15f)) {
            std::fprintf(stderr, "roundtrip mismatch type=%s i=%zu src=%f dst=%f\n",
                    ggml_type_name(type), i, src[i], dst[i]);
            return false;
        }
    }

    return true;
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
            + amplitude * sinf(0.071f * x + phase)
            + 0.35f * amplitude * cosf(0.047f * x - 0.5f * phase);
    }
    return out;
}

static bool set_tensor_from_fp32(ggml_tensor * tensor, const std::vector<float> & src) {
    if ((int64_t) src.size() != ggml_nelements(tensor)) {
        std::fprintf(stderr, "size mismatch for tensor type=%s expected=%lld got=%zu\n",
                ggml_type_name(tensor->type), (long long) ggml_nelements(tensor), src.size());
        return false;
    }

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, src.data(), 0, src.size() * sizeof(float));
        return true;
    }

    std::vector<uint8_t> dst(ggml_nbytes(tensor));
    const int64_t n_per_row = tensor->ne[0];
    const int64_t nrows = ggml_nelements(tensor) / n_per_row;
    ggml_quantize_chunk(tensor->type, src.data(), dst.data(), 0, nrows, n_per_row, nullptr);
    ggml_backend_tensor_set(tensor, dst.data(), 0, dst.size());
    return true;
}

struct flash_attn_case {
    int64_t hs;
    int64_t q_len;
    int64_t kv_len;
    int64_t n_head;
    const char * name;
};

static bool run_flash_attn_case(const flash_attn_case & tc, ggml_type v_type, std::vector<float> & out) {
    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to init ggml context for %s\n", tc.name);
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to init CUDA backend for %s\n", tc.name);
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.hs, tc.q_len, tc.n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.hs, tc.kv_len, tc.n_head, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, v_type,        tc.hs, tc.kv_len, tc.n_head, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float) tc.hs), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors for %s\n", tc.name);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> q_data = make_signal((size_t) ggml_nelements(q), 0.35f, 0.00f, 0.10f);
    const std::vector<float> k_data = make_signal((size_t) ggml_nelements(k), 0.40f, 0.05f, 0.35f);
    const std::vector<float> v_data = make_signal((size_t) ggml_nelements(v), 1.65f, 0.00f, 0.70f);

    if (!set_tensor_from_fp32(q, q_data) || !set_tensor_from_fp32(k, k_data) || !set_tensor_from_fp32(v, v_data)) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "flash_attn compute failed for %s with V=%s: %s\n",
                tc.name, ggml_type_name(v_type), ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    out.assign((size_t) ggml_nelements(fa), 0.0f);
    ggml_backend_tensor_get(fa, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static bool test_flash_attn_vs_f16(const flash_attn_case & tc, ggml_type v_type) {
    std::vector<float> ref;
    std::vector<float> got;

    if (!run_flash_attn_case(tc, GGML_TYPE_F16, ref)) {
        return false;
    }
    if (!run_flash_attn_case(tc, v_type, got)) {
        return false;
    }

    const double err_nmse = nmse(ref, got);
    const float err_max_abs = max_abs_diff(ref, got);

    const bool native_fp8_p_case = tc.q_len == 1;
    const bool is_s5 = v_type == GGML_TYPE_FP8_E4M3_S5;
    const double tol_nmse = native_fp8_p_case ? (is_s5 ? 1.4e-1 : 8e-2) : 5e-2;
    const float tol_max_abs = native_fp8_p_case ? (is_s5 ? 2.5e-1f : 2e-1f) : 1.2e-1f;
    std::fprintf(stderr, "flash_attn case=%s V=%s nmse=%g max_abs=%g\n",
            tc.name, ggml_type_name(v_type), err_nmse, err_max_abs);
    if (err_nmse > tol_nmse && err_max_abs > tol_max_abs) {
        std::fprintf(stderr,
                "flash_attn mismatch case=%s V=%s nmse=%g max_abs=%g\n",
                tc.name, ggml_type_name(v_type), err_nmse, err_max_abs);
        return false;
    }

    return true;
}

int main() {
    disable_cuda_truncation();

    if (!test_fp8_row_roundtrip(GGML_TYPE_FP8_E4M3_S3, 448.0f / 3.0f)) {
        return 1;
    }

    if (!test_fp8_row_roundtrip(GGML_TYPE_FP8_E4M3_S5, 448.0f / 5.0f)) {
        return 1;
    }

    const flash_attn_case native_vec_case = {
        /* .hs     = */ 64,
        /* .q_len  = */ 1,
        /* .kv_len = */ 256,
        /* .n_head = */ 2,
        /* .name   = */ "native-vec-bs1-h64",
    };
    const flash_attn_case fallback_case = {
        /* .hs     = */ 64,
        /* .q_len  = */ 2,
        /* .kv_len = */ 256,
        /* .n_head = */ 2,
        /* .name   = */ "fallback-f16-bs2-h64",
    };

    if (!test_flash_attn_vs_f16(native_vec_case, GGML_TYPE_FP8_E4M3_S3)) {
        return 1;
    }
    if (!test_flash_attn_vs_f16(native_vec_case, GGML_TYPE_FP8_E4M3_S5)) {
        return 1;
    }
    if (!test_flash_attn_vs_f16(fallback_case, GGML_TYPE_FP8_E4M3_S3)) {
        return 1;
    }
    if (!test_flash_attn_vs_f16(fallback_case, GGML_TYPE_FP8_E4M3_S5)) {
        return 1;
    }

    std::puts("test-vcache-fp8-e4m3: ok");
    return 0;
}
