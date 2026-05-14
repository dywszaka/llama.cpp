#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-quants.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool run_case() {
#if defined(_WIN32)
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif

    ggml_init_params params = {
        /* .mem_size   = */ 32 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        ggml_free(ctx);
        return false;
    }

    const int64_t kv_size = 16;
    const int64_t rows = 32;
    const int64_t cols = 3;
    const int64_t blocks = kv_size / 16;

    ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, kv_size, rows);
    ggml_tensor * v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, blocks, rows);
    ggml_tensor * p = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_size, cols);
    ggml_tensor * out = ggml_mul_mat(ctx, v, p);
    ggml_tensor_set_nvfp4_scale(v, v_scale);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> v_ref((size_t) rows * (size_t) kv_size);
    std::vector<block_nvfp4> v_q((size_t) rows * (size_t) blocks);
    std::vector<float> v_s((size_t) rows * (size_t) blocks);
    std::vector<float> v_deq((size_t) rows * (size_t) kv_size);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < kv_size; ++i) {
            v_ref[(size_t) r * (size_t) kv_size + (size_t) i] = sinf(0.07f * (float) (r * kv_size + i));
        }
        float amax = 0.0f;
        for (int64_t i = 0; i < kv_size; ++i) {
            amax = fmaxf(amax, fabsf(v_ref[(size_t) r * (size_t) kv_size + (size_t) i]));
        }
        const float global = amax > 0.0f ? (6.0f * 224.0f / amax) : 0.0f;
        v_s[(size_t) r] = global > 0.0f ? 1.0f / global : 0.0f;
        quantize_row_nvfp4_ref(v_ref.data() + (size_t) r * (size_t) kv_size, v_q.data() + (size_t) r * (size_t) blocks, kv_size, global);
        dequantize_row_nvfp4(v_q.data() + (size_t) r * (size_t) blocks, v_deq.data() + (size_t) r * (size_t) kv_size, kv_size, global);
    }

    std::vector<float> p_host((size_t) kv_size * (size_t) cols);
    for (int64_t c = 0; c < cols; ++c) {
        for (int64_t i = 0; i < kv_size; ++i) {
            p_host[(size_t) c * (size_t) kv_size + (size_t) i] = cosf(0.05f * (float) (c * kv_size + i));
        }
    }

    ggml_backend_tensor_set(v, v_q.data(), 0, v_q.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_set(v_scale, v_s.data(), 0, v_s.size() * sizeof(float));
    ggml_backend_tensor_set(p, p_host.data(), 0, p_host.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> out_host((size_t) rows * (size_t) cols);
    ggml_backend_tensor_get(out, out_host.data(), 0, out_host.size() * sizeof(float));

    for (int64_t c = 0; c < cols; ++c) {
        for (int64_t r = 0; r < rows; ++r) {
            float ref = 0.0f;
            for (int64_t i = 0; i < kv_size; ++i) {
                ref += v_deq[(size_t) r * (size_t) kv_size + (size_t) i] * p_host[(size_t) c * (size_t) kv_size + (size_t) i];
            }
            const float got = out_host[(size_t) c * (size_t) rows + (size_t) r];
            if (fabsf(got - ref) > 0.25f) {
                std::fprintf(stderr, "matmul mismatch r=%lld c=%lld got=%g ref=%g\n",
                        (long long) r, (long long) c, got, ref);
                ggml_backend_buffer_free(buf);
                ggml_backend_free(backend);
                ggml_free(ctx);
                return false;
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        std::puts("test-vcache-nvfp4-matmul: SKIP (no CUDA device)");
        return 0;
    }

    if (!run_case()) {
        return 1;
    }

    std::puts("test-vcache-nvfp4-matmul: ok");
    return 0;
}
