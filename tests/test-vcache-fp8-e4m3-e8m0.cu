#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int32_t GGML_FLASH_ATTN_FLAG_FP8_P_E4M3_E8M0 = 1;

static void disable_cuda_truncation() {
#if defined(_WIN32)
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
    _putenv_s("GGML_CUDA_FP8_E8M0_NATIVE_NO_FALLBACK", "1");
#else
    setenv("GGML_CUDA_TRUNC_ENABLE", "0", 1);
    setenv("GGML_CUDA_FP8_E8M0_NATIVE_NO_FALLBACK", "1", 1);
#endif
}

static bool almost_equal(float a, float b, float atol, float rtol) {
    const float diff = fabsf(a - b);
    const float tol = atol + rtol * fmaxf(fabsf(a), fabsf(b));
    return diff <= tol;
}

static std::vector<float> make_signal(size_t n, float amplitude, float bias, float phase) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        const float x = (float) i;
        out[i] = bias
            + amplitude * sinf(0.051f * x + phase)
            + 0.30f * amplitude * cosf(0.029f * x - 0.5f * phase);
    }
    return out;
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

static bool test_block32_roundtrip() {
    std::vector<float> src = {
        -30.0f, -12.0f, -8.0f, -3.5f, -1.25f, -0.75f, -0.25f, -0.03125f,
         0.0f,   0.03125f, 0.125f, 0.25f, 0.75f, 1.50f, 3.00f, 6.00f,
        -28.0f, -16.0f, -4.5f, -2.0f, -1.0f, -0.5f, -0.125f, -0.0625f,
         0.0625f, 0.125f, 0.5f, 1.0f, 2.0f, 4.5f, 16.0f, 28.0f,
    };

    std::vector<block_fp8_e4m3_e8m0_32> q(src.size() / QK_FP8_E4M3_E8M0_32);
    std::vector<float> dst(src.size(), 0.0f);

    quantize_row_fp8_e4m3_e8m0_32_ref(src.data(), q.data(), (int64_t) src.size());
    dequantize_row_fp8_e4m3_e8m0_32(q.data(), dst.data(), (int64_t) src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        if (!almost_equal(src[i], dst[i], 0.45f, 0.18f)) {
            std::fprintf(stderr, "block32 roundtrip mismatch i=%zu src=%f dst=%f\n", i, src[i], dst[i]);
            return false;
        }
    }

    return true;
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

enum flash_attn_variant {
    FLASH_ATTN_BASELINE = 0,
    FLASH_ATTN_V_ONLY,
    FLASH_ATTN_P_ONLY,
    FLASH_ATTN_V_AND_P,
};

struct mul_mat_case {
    int64_t k;
    int64_t m;
    int64_t n;
    const char * name;
};

struct flash_attn_case {
    int64_t hs;
    int64_t q_len;
    int64_t kv_len;
    int64_t n_head;
    const char * name;
};

static bool run_flash_attn_case(const flash_attn_case & tc, flash_attn_variant variant, std::vector<float> & out) {
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

    const ggml_type v_type = (variant == FLASH_ATTN_V_ONLY || variant == FLASH_ATTN_V_AND_P)
        ? GGML_TYPE_FP8_E4M3_E8M0_32
        : GGML_TYPE_F16;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.hs, tc.q_len, tc.n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.hs, tc.kv_len, tc.n_head, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, v_type,        tc.hs, tc.kv_len, tc.n_head, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float) tc.hs), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);

    const int32_t flags = (variant == FLASH_ATTN_P_ONLY || variant == FLASH_ATTN_V_AND_P)
        ? GGML_FLASH_ATTN_FLAG_FP8_P_E4M3_E8M0
        : 0;
    std::memcpy((int32_t *) fa->op_params + 4, &flags, sizeof(flags));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors for %s\n", tc.name);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> q_data = make_signal((size_t) ggml_nelements(q), 0.30f, 0.00f, 0.10f);
    const std::vector<float> k_data = make_signal((size_t) ggml_nelements(k), 0.35f, 0.02f, 0.35f);
    const std::vector<float> v_data = make_signal((size_t) ggml_nelements(v), 12.0f, 0.0f, 0.70f);

    if (!set_tensor_from_fp32(q, q_data) || !set_tensor_from_fp32(k, k_data) || !set_tensor_from_fp32(v, v_data)) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "flash_attn compute failed for %s variant=%d: %s\n",
                tc.name, (int) variant, ggml_status_to_string(status));
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

static bool test_flash_attn_variant(const flash_attn_case & tc, flash_attn_variant variant, double tol_nmse, float tol_max_abs) {
    std::vector<float> ref;
    std::vector<float> got;

    if (!run_flash_attn_case(tc, FLASH_ATTN_BASELINE, ref)) {
        return false;
    }
    if (!run_flash_attn_case(tc, variant, got)) {
        return false;
    }

    const double err_nmse = nmse(ref, got);
    const float err_max_abs = max_abs_diff(ref, got);
    std::fprintf(stderr, "flash_attn case=%s variant=%d nmse=%g max_abs=%g\n",
            tc.name, (int) variant, err_nmse, err_max_abs);

    if (err_nmse > tol_nmse && err_max_abs > tol_max_abs) {
        std::fprintf(stderr, "flash_attn mismatch case=%s variant=%d nmse=%g max_abs=%g\n",
                tc.name, (int) variant, err_nmse, err_max_abs);
        return false;
    }

    return true;
}

static bool run_mul_mat_case(const mul_mat_case & tc, ggml_type src0_type, std::vector<float> & out) {
    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to init ggml context for mul_mat %s\n", tc.name);
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to init CUDA backend for mul_mat %s\n", tc.name);
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, src0_type, tc.k, tc.m);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tc.k, tc.n);
    ggml_tensor * mm = ggml_mul_mat(ctx, src0, src1);
    ggml_mul_mat_set_prec(mm, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, mm);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors for mul_mat %s\n", tc.name);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> src0_data = make_signal((size_t) ggml_nelements(src0), 8.0f, 0.0f, 0.25f);
    std::vector<float> src1_data((size_t) ggml_nelements(src1), 0.0f);
    for (int64_t col = 0; col < tc.n; ++col) {
        float sum = 0.0f;
        for (int64_t row = 0; row < tc.k; ++row) {
            const float v = 0.01f + fabsf(0.75f * sinf(0.031f * (float) row + 0.27f * (float) col));
            src1_data[(size_t) col * (size_t) tc.k + (size_t) row] = v;
            sum += v;
        }
        for (int64_t row = 0; row < tc.k; ++row) {
            src1_data[(size_t) col * (size_t) tc.k + (size_t) row] /= sum;
        }
    }

    if (!set_tensor_from_fp32(src0, src0_data) || !set_tensor_from_fp32(src1, src1_data)) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "mul_mat compute failed for %s src0_type=%s: %s\n",
                tc.name, ggml_type_name(src0_type), ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    out.assign((size_t) ggml_nelements(mm), 0.0f);
    ggml_backend_tensor_get(mm, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static bool test_mul_mat_variant(const mul_mat_case & tc, double tol_nmse, float tol_max_abs) {
    std::vector<float> ref;
    std::vector<float> got;

    if (!run_mul_mat_case(tc, GGML_TYPE_F32, ref)) {
        return false;
    }
    if (!run_mul_mat_case(tc, GGML_TYPE_FP8_E4M3_E8M0_32, got)) {
        return false;
    }

    const double err_nmse = nmse(ref, got);
    const float err_max_abs = max_abs_diff(ref, got);
    std::fprintf(stderr, "mul_mat case=%s nmse=%g max_abs=%g\n", tc.name, err_nmse, err_max_abs);

    if (err_nmse > tol_nmse && err_max_abs > tol_max_abs) {
        std::fprintf(stderr, "mul_mat mismatch case=%s nmse=%g max_abs=%g\n", tc.name, err_nmse, err_max_abs);
        return false;
    }

    return true;
}

int main() {
    int dev_count = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count <= 0) {
        std::puts("test-vcache-fp8-e4m3-e8m0: SKIP (no CUDA device)");
        return 0;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        std::puts("test-vcache-fp8-e4m3-e8m0: SKIP (failed to select CUDA device 0)");
        return 0;
    }

    disable_cuda_truncation();

    if (!test_block32_roundtrip()) {
        return 1;
    }

    const flash_attn_case decode_case = {
        /* .hs     = */ 128,
        /* .q_len  = */ 1,
        /* .kv_len = */ 256,
        /* .n_head = */ 2,
        /* .name   = */ "decode-bs1-h128",
    };

    if (!test_flash_attn_variant(decode_case, FLASH_ATTN_V_ONLY, 5e-2, 4.5f)) {
        return 1;
    }
    if (!test_flash_attn_variant(decode_case, FLASH_ATTN_P_ONLY, 3e-2, 3.0f)) {
        return 1;
    }
    if (!test_flash_attn_variant(decode_case, FLASH_ATTN_V_AND_P, 8e-2, 5.5f)) {
        return 1;
    }

    const mul_mat_case non_flash_decode_case = {
        /* .k    = */ 256,
        /* .m    = */ 128,
        /* .n    = */ 1,
        /* .name = */ "non-flash-decode-k256-m128-n1",
    };

    if (!test_mul_mat_variant(non_flash_decode_case, 1e-1, 2.0f)) {
        return 1;
    }

    const mul_mat_case non_flash_prefill_case = {
        /* .k    = */ 256,
        /* .m    = */ 128,
        /* .n    = */ 17,
        /* .name = */ "non-flash-prefill-k256-m128-n17",
    };

    if (!test_mul_mat_variant(non_flash_prefill_case, 1e-1, 2.0f)) {
        return 1;
    }

    std::puts("test-vcache-fp8-e4m3-e8m0: ok");
    return 0;
}
