#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void fail(const char * msg) {
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

static void expect(bool cond, const char * msg) {
    if (!cond) {
        fail(msg);
    }
}

static std::vector<float> read_f32_tensor(const ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return out;
}

static std::vector<float> read_f16_tensor_as_f32(const ggml_tensor * t) {
    std::vector<ggml_fp16_t> buf(ggml_nelements(t));
    std::vector<float> out(buf.size());
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
    for (size_t i = 0; i < buf.size(); ++i) {
        out[i] = ggml_fp16_to_fp32(buf[i]);
    }
    return out;
}

static bool nearly_equal(float a, float b, float atol = 1e-3f, float rtol = 1e-3f) {
    const float diff = std::fabs(a - b);
    return diff <= atol + rtol * std::max(std::fabs(a), std::fabs(b));
}

int main() {
#ifdef _WIN32
    _putenv_s("GGML_CUDA_TRUNC_ENABLE", "0");
#else
    setenv("GGML_CUDA_TRUNC_ENABLE", "0", 1);
#endif

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "Skip: GPU backend is unavailable.\n");
        return 0;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 1u << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    expect(ctx != nullptr, "failed to create ggml context");

    constexpr int64_t n_embd = QK_NVFP4;
    constexpr int64_t n_tokens = 2;
    constexpr int64_t n_rows = 4;
    const float global_scale = 0.75f;

    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, n_tokens);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * cache_q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_NVFP4, n_embd, n_rows);

    ggml_tensor * set_rows = ggml_set_rows(ctx.get(), cache_q, src, idx);
    ggml_set_rows_set_nvfp4_scale(set_rows, scale);

    ggml_tensor * out_f32 = ggml_cast(ctx.get(), set_rows, GGML_TYPE_F32);
    ggml_cpy_set_nvfp4_scale(out_f32, scale);

    ggml_tensor * out_f16 = ggml_cast(ctx.get(), set_rows, GGML_TYPE_F16);
    ggml_cpy_set_nvfp4_scale(out_f16, scale);

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    expect(buf != nullptr, "failed to allocate backend buffer");
    ggml_backend_buffer_clear(buf.get(), 0);

    std::vector<float> src_data = {
        0.50f, -0.25f, 0.75f, -1.00f,  0.10f,  0.20f, -0.30f,  0.40f,
       -0.50f,  0.60f, -0.70f,  0.80f, -0.90f,  1.00f, -1.10f,  1.20f,
       -0.15f,  0.35f, -0.55f,  0.75f, -0.95f,  1.15f, -1.35f,  1.55f,
        0.05f, -0.10f, 0.15f, -0.20f,  0.25f, -0.30f,  0.35f, -0.40f,
    };
    const int64_t idx_data[2] = { 1, 3 };

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(idx, idx_data, 0, sizeof(idx_data));
    ggml_backend_tensor_set(scale, &global_scale, 0, sizeof(global_scale));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 8, false);
    ggml_build_forward_expand(gf, out_f32);
    ggml_build_forward_expand(gf, out_f16);

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    expect(status == GGML_STATUS_SUCCESS, "ggml_backend_graph_compute failed");

    std::vector<block_nvfp4> cache_blocks(n_rows);
    ggml_backend_tensor_get(cache_q, cache_blocks.data(), 0, cache_blocks.size() * sizeof(block_nvfp4));

    std::vector<block_nvfp4> expected_blocks(n_rows);
    std::memset(expected_blocks.data(), 0, expected_blocks.size() * sizeof(block_nvfp4));
    quantize_row_nvfp4_ref(src_data.data() + 0 * n_embd, expected_blocks.data() + idx_data[0], n_embd, global_scale);
    quantize_row_nvfp4_ref(src_data.data() + 1 * n_embd, expected_blocks.data() + idx_data[1], n_embd, global_scale);

    if (std::memcmp(cache_blocks.data(), expected_blocks.data(), expected_blocks.size() * sizeof(block_nvfp4)) != 0) {
        fail("NVFP4 set_rows output does not match reference quantization");
    }

    std::vector<float> expected_f32(n_rows * n_embd, 0.0f);
    dequantize_row_nvfp4(expected_blocks.data() + idx_data[0], expected_f32.data() + idx_data[0] * n_embd, n_embd, global_scale);
    dequantize_row_nvfp4(expected_blocks.data() + idx_data[1], expected_f32.data() + idx_data[1] * n_embd, n_embd, global_scale);

    std::vector<float> got_f32 = read_f32_tensor(out_f32);
    std::vector<float> got_f16 = read_f16_tensor_as_f32(out_f16);

    for (size_t i = 0; i < expected_f32.size(); ++i) {
        if (!nearly_equal(got_f32[i], expected_f32[i], 1e-4f, 1e-4f)) {
            std::fprintf(stderr, "F32 mismatch at %zu: got=%f exp=%f\n", i, got_f32[i], expected_f32[i]);
            return 1;
        }
        if (!nearly_equal(got_f16[i], expected_f32[i], 5e-3f, 5e-3f)) {
            std::fprintf(stderr, "F16 mismatch at %zu: got=%f exp=%f\n", i, got_f16[i], expected_f32[i]);
            return 1;
        }
    }

    std::printf("NVFP4 KV cache backend ops test passed.\n");
    ggml_backend_free(backend);
    return 0;
}
