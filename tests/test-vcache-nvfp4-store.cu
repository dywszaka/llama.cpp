#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include "../ggml/src/ggml-quants.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

static bool run_store_case(int64_t head_dim, int64_t n_tokens) {
#if defined(_WIN32)
    _putenv_s("LLAMA_EXPERIMENT_NVFP4_VCACHE", "1");
#else
    setenv("LLAMA_EXPERIMENT_NVFP4_VCACHE", "1", 1);
#endif

    ggml_init_params params = {
        /* .mem_size   = */ 32 * 1024 * 1024,
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

    const int64_t kv_size = 32;
    const int64_t kv_blocks = kv_size / 16;
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, kv_size, head_dim, 1);
    ggml_tensor * v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim * kv_blocks, 1);
    ggml_tensor * v_cur_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, n_tokens);
    ggml_tensor * v_cur = ggml_reshape_2d(ctx, v_cur_3d, head_dim, n_tokens);

    ggml_tensor * v_view = ggml_reshape_2d(ctx, v_cache, 1, head_dim * kv_size);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim * n_tokens);
    ggml_tensor * set = ggml_set_rows(ctx, v_view, ggml_reshape_2d(ctx, v_cur, 1, head_dim * n_tokens), idx);
    ggml_tensor_set_nvfp4_scale(set, v_scale);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, set);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const std::vector<float> src = make_signal((size_t) ggml_nelements(v_cur_3d), 2.0f, 0.1f, 0.3f);
    std::vector<int64_t> idx_data((size_t) (head_dim * n_tokens));
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int64_t j = 0; j < head_dim; ++j) {
            idx_data[(size_t) i * (size_t) head_dim + (size_t) j] = j * kv_size + i;
        }
    }

    ggml_backend_tensor_set(v_cur_3d, src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);

    std::vector<block_nvfp4> cache_host((size_t) head_dim * (size_t) kv_blocks);
    std::vector<float> scale_host((size_t) head_dim * (size_t) kv_blocks);
    ggml_backend_tensor_get(v_cache, cache_host.data(), 0, cache_host.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_get(v_scale, scale_host.data(), 0, scale_host.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }

    for (int64_t row = 0; row < head_dim; ++row) {
        for (int64_t block = 0; block < kv_blocks; ++block) {
            const size_t off = (size_t) row * (size_t) kv_blocks + (size_t) block;
            const float input_scale = scale_host[off];
            const float global_scale = input_scale > 0.0f ? (1.0f / input_scale) : 0.0f;
            float deq[16];
            dequantize_row_nvfp4(&cache_host[off], deq, 16, global_scale);

            float ref_tile[16] = {};
            for (int64_t lane = 0; lane < 16; ++lane) {
                const int64_t token = block * 16 + lane;
                ref_tile[lane] = token < n_tokens ? src[(size_t) token * (size_t) head_dim + (size_t) row] : 0.0f;
            }

            block_nvfp4 ref_q;
            float ref_deq[16];
            float amax = 0.0f;
            for (float v : ref_tile) {
                amax = fmaxf(amax, fabsf(v));
            }
            const float ref_global = amax > 0.0f ? (6.0f * 224.0f / amax) : 0.0f;
            quantize_row_nvfp4_ref(ref_tile, &ref_q, 16, ref_global);
            dequantize_row_nvfp4(&ref_q, ref_deq, 16, ref_global);

            for (int64_t lane = 0; lane < 16; ++lane) {
                const int64_t token = block * 16 + lane;
                const float got = deq[lane];
                const float expect = ref_deq[lane];
                if (fabsf(got - expect) > 1e-5f || cache_host[off].e != ref_q.e || cache_host[off].qs[lane / 2] != ref_q.qs[lane / 2]) {
                    std::fprintf(stderr,
                            "mismatch row=%lld block=%lld lane=%lld token=%lld got=%g expect=%g input_scale=%g global=%g "
                            "cache_e=%u ref_e=%u cache_q=%u ref_q=%u\n",
                            (long long) row, (long long) block, (long long) lane, (long long) token,
                            got, expect, input_scale, global_scale,
                            (unsigned) cache_host[off].e, (unsigned) ref_q.e,
                            (unsigned) cache_host[off].qs[lane / 2], (unsigned) ref_q.qs[lane / 2]);
                    return false;
                }
            }
        }
    }

    return true;
}

int main() {
    int dev_count = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count <= 0) {
        std::puts("test-vcache-nvfp4-store: SKIP (no CUDA device)");
        return 0;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        std::puts("test-vcache-nvfp4-store: SKIP (failed to select CUDA device 0)");
        return 0;
    }

    if (!run_store_case(128, 17)) {
        return 1;
    }

    std::puts("test-vcache-nvfp4-store: ok");
    return 0;
}
