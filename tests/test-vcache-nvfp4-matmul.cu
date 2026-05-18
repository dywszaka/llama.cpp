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

static bool run_permuted_case() {
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
    const int64_t heads = 2;
    const int64_t head_dim = rows / heads;
    const int64_t cols = 3;
    const int64_t blocks = kv_size / 16;

    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_NVFP4, kv_size, heads, head_dim, 1);
    ggml_tensor * v_scale = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, blocks, heads, head_dim, 1);
    ggml_tensor * p = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_size, cols, heads, 1);
    ggml_tensor_set_nvfp4_scale(v, v_scale);

    ggml_tensor * v_perm = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor_set_nvfp4_scale(v_perm, v_scale);
    ggml_tensor * out = ggml_mul_mat(ctx, v_perm, p);

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
    for (int64_t d = 0; d < head_dim; ++d) {
        for (int64_t h = 0; h < heads; ++h) {
            const int64_t phys = d * heads + h;
            for (int64_t i = 0; i < kv_size; ++i) {
                v_ref[(size_t) phys * (size_t) kv_size + (size_t) i] = sinf(0.09f * (float) (phys * kv_size + i));
            }
            float amax = 0.0f;
            for (int64_t i = 0; i < kv_size; ++i) {
                amax = fmaxf(amax, fabsf(v_ref[(size_t) phys * (size_t) kv_size + (size_t) i]));
            }
            const float global = amax > 0.0f ? (6.0f * 224.0f / amax) : 0.0f;
            v_s[(size_t) phys] = global > 0.0f ? 1.0f / global : 0.0f;
            quantize_row_nvfp4_ref(v_ref.data() + (size_t) phys * (size_t) kv_size, v_q.data() + (size_t) phys * (size_t) blocks, kv_size, global);
            dequantize_row_nvfp4(v_q.data() + (size_t) phys * (size_t) blocks, v_deq.data() + (size_t) phys * (size_t) kv_size, kv_size, global);
        }
    }

    std::vector<float> p_host((size_t) kv_size * (size_t) cols * (size_t) heads);
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t c = 0; c < cols; ++c) {
            for (int64_t i = 0; i < kv_size; ++i) {
                const size_t off = (size_t) h * (size_t) cols * (size_t) kv_size + (size_t) c * (size_t) kv_size + (size_t) i;
                p_host[off] = cosf(0.03f * (float) (off + 1));
            }
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

    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t c = 0; c < cols; ++c) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const int64_t phys = d * heads + h;
                float ref = 0.0f;
                for (int64_t i = 0; i < kv_size; ++i) {
                    const size_t p_off = (size_t) h * (size_t) cols * (size_t) kv_size + (size_t) c * (size_t) kv_size + (size_t) i;
                    ref += v_deq[(size_t) phys * (size_t) kv_size + (size_t) i] * p_host[p_off];
                }
                const size_t out_off = (size_t) h * (size_t) cols * (size_t) head_dim + (size_t) c * (size_t) head_dim + (size_t) d;
                const float got = out_host[out_off];
                if (fabsf(got - ref) > 0.25f) {
                    std::fprintf(stderr, "permuted matmul mismatch h=%lld d=%lld c=%lld got=%g ref=%g\n",
                            (long long) h, (long long) d, (long long) c, got, ref);
                    ggml_backend_buffer_free(buf);
                    ggml_backend_free(backend);
                    ggml_free(ctx);
                    return false;
                }
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

static bool run_real_vcache_view_case(int64_t kv_size) {
    ggml_init_params params = {
        /* .mem_size   = */ 64 * 1024 * 1024,
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

    const int64_t head_dim = 128;
    const int64_t kv_heads = 8;
    const int64_t q_heads = 32;
    const int64_t cols = 2;
    const int64_t blocks = kv_size / 16;
    const int64_t n_embd = head_dim * kv_heads;

    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, kv_size, n_embd, 1);
    ggml_tensor * v_scale_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, blocks * n_embd, 1);
    ggml_tensor * v = ggml_view_4d(ctx, v_cache,
            kv_size, kv_heads, head_dim, 1,
            (int64_t) blocks * head_dim * sizeof(block_nvfp4),
            (int64_t) blocks * sizeof(block_nvfp4),
            (int64_t) blocks * n_embd * sizeof(block_nvfp4),
            0);
    ggml_tensor * v_scale = ggml_view_4d(ctx, v_scale_base,
            blocks, kv_heads, head_dim, 1,
            (int64_t) blocks * head_dim * sizeof(float),
            (int64_t) blocks * sizeof(float),
            (int64_t) blocks * n_embd * sizeof(float),
            0);
    ggml_tensor * p = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_size, cols, q_heads, 1);

    ggml_tensor_set_nvfp4_scale(v, v_scale);
    ggml_tensor * v_perm = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor_set_nvfp4_scale(v_perm, v_scale);
    ggml_tensor * out = ggml_mul_mat(ctx, v_perm, p);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> v_ref((size_t) n_embd * (size_t) kv_size);
    std::vector<block_nvfp4> v_q((size_t) n_embd * (size_t) blocks);
    std::vector<float> v_s((size_t) n_embd * (size_t) blocks);
    std::vector<float> v_deq((size_t) n_embd * (size_t) kv_size);
    for (int64_t h = 0; h < kv_heads; ++h) {
        for (int64_t d = 0; d < head_dim; ++d) {
            const int64_t row = h * head_dim + d;
            for (int64_t i = 0; i < kv_size; ++i) {
                const float x = (float) (17 * h + 5 * d + i);
                v_ref[(size_t) row * (size_t) kv_size + (size_t) i] =
                    0.7f * sinf(0.013f * x) + 0.2f * cosf(0.031f * x);
            }
            for (int64_t b = 0; b < blocks; ++b) {
                float amax = 0.0f;
                for (int64_t i = 0; i < 16; ++i) {
                    amax = fmaxf(amax, fabsf(v_ref[(size_t) row * (size_t) kv_size + (size_t) b * 16 + (size_t) i]));
                }
                const float global = amax > 0.0f ? (6.0f * 224.0f / amax) : 0.0f;
                const size_t off = (size_t) row * (size_t) blocks + (size_t) b;
                v_s[off] = global > 0.0f ? 1.0f / global : 0.0f;
                quantize_row_nvfp4_ref(
                        v_ref.data() + (size_t) row * (size_t) kv_size + (size_t) b * 16,
                        v_q.data() + off, 16, global);
                dequantize_row_nvfp4(
                        v_q.data() + off,
                        v_deq.data() + (size_t) row * (size_t) kv_size + (size_t) b * 16,
                        16, global);
            }
        }
    }

    std::vector<float> p_host((size_t) kv_size * (size_t) cols * (size_t) q_heads);
    for (int64_t h = 0; h < q_heads; ++h) {
        for (int64_t c = 0; c < cols; ++c) {
            for (int64_t i = 0; i < kv_size; ++i) {
                const size_t off = (size_t) h * (size_t) cols * (size_t) kv_size + (size_t) c * (size_t) kv_size + (size_t) i;
                p_host[off] = 0.5f * cosf(0.007f * (float) (off + 3)) + 0.1f * sinf(0.019f * (float) (i + 1));
            }
        }
    }

    ggml_backend_tensor_set(v_cache, v_q.data(), 0, v_q.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_set(v_scale_base, v_s.data(), 0, v_s.size() * sizeof(float));
    ggml_backend_tensor_set(p, p_host.data(), 0, p_host.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> out_host((size_t) head_dim * (size_t) cols * (size_t) q_heads);
    ggml_backend_tensor_get(out, out_host.data(), 0, out_host.size() * sizeof(float));

    for (int64_t h = 0; h < q_heads; ++h) {
        const int64_t kv_head = h / (q_heads / kv_heads);
        for (int64_t c = 0; c < cols; ++c) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const int64_t row = kv_head * head_dim + d;
                float ref = 0.0f;
                for (int64_t i = 0; i < kv_size; ++i) {
                    const size_t p_off = (size_t) h * (size_t) cols * (size_t) kv_size + (size_t) c * (size_t) kv_size + (size_t) i;
                    ref += v_deq[(size_t) row * (size_t) kv_size + (size_t) i] * p_host[p_off];
                }
                const size_t out_off = (size_t) h * (size_t) cols * (size_t) head_dim + (size_t) c * (size_t) head_dim + (size_t) d;
                const float got = out_host[out_off];
                if (fabsf(got - ref) > 0.5f) {
                    std::fprintf(stderr, "real vcache matmul mismatch kv_size=%lld h=%lld d=%lld c=%lld got=%g ref=%g\n",
                            (long long) kv_size, (long long) h, (long long) d, (long long) c, got, ref);
                    ggml_backend_buffer_free(buf);
                    ggml_backend_free(backend);
                    ggml_free(ctx);
                    return false;
                }
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

    if (!run_permuted_case()) {
        return 1;
    }

    if (!run_real_vcache_view_case(16)) {
        return 1;
    }

    if (!run_real_vcache_view_case(512)) {
        return 1;
    }

    std::puts("test-vcache-nvfp4-matmul: ok");
    return 0;
}
