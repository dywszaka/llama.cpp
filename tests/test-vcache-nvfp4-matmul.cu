#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-quants.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr int8_t kvalues_nvfp4_test[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static float e4m3_to_f32(uint8_t x) {
    const uint32_t sign     = (uint32_t) (x & 0x80) << 24;
    const uint32_t exponent = (x >> 3) & 0x0F;
    const uint32_t mantissa = x & 0x07;

    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            const int leading = __builtin_clz(mantissa);
            const int shift = leading - 29;
            const uint32_t man = mantissa << shift;
            const uint32_t exp = 127 - 6 - shift;
            bits = sign | (exp << 23) | (man & 0x7) << 20;
        }
    } else if (exponent == 0x0F) {
        bits = mantissa == 0x7 ? (sign | 0x7F800000 | (1u << 22)) : (sign | 0x43E00000);
    } else {
        const uint32_t exp = (exponent - 7 + 127) << 23;
        const uint32_t man = mantissa << (23 - 3);
        bits = sign | exp | man;
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint8_t best_index_e4m3_test(float x) {
    uint8_t best_index = 0;
    float best_err = INFINITY;
    for (int i = 0; i < 256; ++i) {
        const float v = e4m3_to_f32((uint8_t) i);
        if (!std::isfinite(v)) {
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

static float lt_scale_roundtrip(float x) {
    return e4m3_to_f32(best_index_e4m3_test(x));
}

#if defined(CUBLAS_VERSION)
#define TEST_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VERSION >= 130000)
#elif defined(CUBLAS_VER_MAJOR)
#define TEST_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VER_MAJOR >= 13)
#else
#define TEST_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS 0
#endif

static bool expect_lt_reference_for_kv_size(int64_t kv_size) {
    GGML_UNUSED(kv_size);
    return TEST_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS;
}

static void dequantize_v_blocks_like_lt(
        const std::vector<block_nvfp4> & v_q,
        const std::vector<float> & v_s,
        std::vector<float> & v_deq,
        int64_t rows,
        int64_t kv_size,
        int64_t blocks) {
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t b = 0; b < blocks; ++b) {
            const block_nvfp4 & block = v_q[(size_t) row * (size_t) blocks + (size_t) b];
            const float scale = lt_scale_roundtrip(e4m3_to_f32(block.e) * v_s[(size_t) row * (size_t) blocks + (size_t) b]);
            for (int j = 0; j < QK_NVFP4 / 2; ++j) {
                const uint8_t packed = block.qs[j];
                v_deq[(size_t) row * (size_t) kv_size + (size_t) b * QK_NVFP4 + (size_t) 2 * j + 0] =
                    0.5f * scale * (float) kvalues_nvfp4_test[packed & 0x0F];
                v_deq[(size_t) row * (size_t) kv_size + (size_t) b * QK_NVFP4 + (size_t) 2 * j + 1] =
                    0.5f * scale * (float) kvalues_nvfp4_test[packed >> 4];
            }
        }
    }
}

static void dequantize_v_blocks_scalar_like_lt(
        const std::vector<block_nvfp4> & v_q,
        float v_global_scale,
        std::vector<float> & v_deq,
        int64_t rows,
        int64_t kv_size,
        int64_t blocks) {
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t b = 0; b < blocks; ++b) {
            const block_nvfp4 & block = v_q[(size_t) row * (size_t) blocks + (size_t) b];
            const float scale = v_global_scale > 0.0f ? lt_scale_roundtrip(e4m3_to_f32(block.e) / v_global_scale) : 0.0f;
            for (int j = 0; j < QK_NVFP4 / 2; ++j) {
                const uint8_t packed = block.qs[j];
                v_deq[(size_t) row * (size_t) kv_size + (size_t) b * QK_NVFP4 + (size_t) 2 * j + 0] =
                    0.5f * scale * (float) kvalues_nvfp4_test[packed & 0x0F];
                v_deq[(size_t) row * (size_t) kv_size + (size_t) b * QK_NVFP4 + (size_t) 2 * j + 1] =
                    0.5f * scale * (float) kvalues_nvfp4_test[packed >> 4];
            }
        }
    }
}

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
            const float tol = 1.25f;
            if (fabsf(got - ref) > tol) {
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
                const float tol = 1.25f;
                if (fabsf(got - ref) > tol) {
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

static bool run_real_vcache_view_case(int64_t kv_size, bool scalar_global_scale = false) {
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
    ggml_tensor * v_scale_base = scalar_global_scale ?
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1) :
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, blocks * n_embd, 1);
    ggml_tensor * v = ggml_view_4d(ctx, v_cache,
            kv_size, kv_heads, head_dim, 1,
            (int64_t) blocks * head_dim * sizeof(block_nvfp4),
            (int64_t) blocks * sizeof(block_nvfp4),
            (int64_t) blocks * n_embd * sizeof(block_nvfp4),
            0);
    ggml_tensor * v_scale = scalar_global_scale ? v_scale_base :
        ggml_view_4d(ctx, v_scale_base,
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
    std::vector<float> v_deq_lt((size_t) n_embd * (size_t) kv_size);
    const float fixed_amax = 64.0f;
    const float fixed_global = 6.0f * 224.0f / fixed_amax;
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
                const float global = scalar_global_scale ? fixed_global : (amax > 0.0f ? (6.0f * 224.0f / amax) : 0.0f);
                const size_t off = (size_t) row * (size_t) blocks + (size_t) b;
                v_s[off] = scalar_global_scale ? fixed_global : (global > 0.0f ? 1.0f / global : 0.0f);
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
    if (scalar_global_scale) {
        dequantize_v_blocks_scalar_like_lt(v_q, fixed_global, v_deq_lt, n_embd, kv_size, blocks);
    } else {
        dequantize_v_blocks_like_lt(v_q, v_s, v_deq_lt, n_embd, kv_size, blocks);
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
    std::vector<float> p_ref = p_host;
    std::vector<block_nvfp4> p_q((size_t) cols * (size_t) q_heads * (size_t) blocks);
    for (int64_t h = 0; h < q_heads; ++h) {
        for (int64_t c = 0; c < cols; ++c) {
            const size_t row_off = (size_t) h * (size_t) cols * (size_t) kv_size + (size_t) c * (size_t) kv_size;
            float amax = 0.0f;
            for (int64_t i = 0; i < kv_size; ++i) {
                amax = fmaxf(amax, fabsf(p_host[row_off + (size_t) i]));
            }
            const float global = amax > 0.0f ? (6.0f * 224.0f / amax) : 0.0f;
            for (int64_t b = 0; b < blocks; ++b) {
                const size_t q_off = ((size_t) h * (size_t) cols + (size_t) c) * (size_t) blocks + (size_t) b;
                quantize_row_nvfp4_ref(p_host.data() + row_off + (size_t) b * 16, p_q.data() + q_off, 16, global);
                dequantize_row_nvfp4(p_q.data() + q_off, p_ref.data() + row_off + (size_t) b * 16, 16, global);
            }
        }
    }

    ggml_backend_tensor_set(v_cache, v_q.data(), 0, v_q.size() * sizeof(block_nvfp4));
    if (scalar_global_scale) {
        ggml_backend_tensor_set(v_scale_base, &fixed_global, 0, sizeof(float));
    } else {
        ggml_backend_tensor_set(v_scale_base, v_s.data(), 0, v_s.size() * sizeof(float));
    }
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
                    const std::vector<float> & v_ref_deq = expect_lt_reference_for_kv_size(kv_size) ? v_deq_lt : v_deq;
                    ref += v_ref_deq[(size_t) row * (size_t) kv_size + (size_t) i] * p_ref[p_off];
                }
                const size_t out_off = (size_t) h * (size_t) cols * (size_t) head_dim + (size_t) c * (size_t) head_dim + (size_t) d;
                const float got = out_host[out_off];
                const float tol = 0.5f;
                if (fabsf(got - ref) > tol) {
                    std::fprintf(stderr, "real vcache matmul mismatch scalar=%d kv_size=%lld h=%lld d=%lld c=%lld got=%g ref=%g\n",
                            scalar_global_scale ? 1 : 0, (long long) kv_size, (long long) h, (long long) d, (long long) c, got, ref);
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

static bool run_set_rows_then_matmul_case(bool scalar_global_scale) {
    ggml_init_params params = {
        /* .mem_size   = */ 96 * 1024 * 1024,
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

    const int64_t kv_size = 512;
    const int64_t head_dim = 128;
    const int64_t kv_heads = 8;
    const int64_t q_heads = 32;
    const int64_t cols = 2;
    const int64_t blocks = kv_size / QK_NVFP4;
    const int64_t n_embd = head_dim * kv_heads;
    const int64_t n_row_groups = n_embd / QK_NVFP4;

    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, kv_size, n_embd, 1);
    ggml_tensor * v_scale_base = scalar_global_scale ?
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1) :
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, blocks * n_embd, 1);
    ggml_tensor * v_view_set = ggml_reshape_2d(ctx, v_cache, QK_NVFP4, kv_size * n_embd / QK_NVFP4);
    ggml_tensor * v_cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QK_NVFP4, kv_size * n_row_groups);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, kv_size * n_row_groups);
    ggml_tensor * set = ggml_set_rows(ctx, v_view_set, v_cur, v_idxs);
    ggml_tensor_set_nvfp4_scale(set, v_scale_base);

    ggml_tensor * v = ggml_view_4d(ctx, v_cache,
            kv_size, kv_heads, head_dim, 1,
            (int64_t) blocks * head_dim * sizeof(block_nvfp4),
            (int64_t) blocks * sizeof(block_nvfp4),
            (int64_t) blocks * n_embd * sizeof(block_nvfp4),
            0);
    ggml_tensor * v_scale = scalar_global_scale ? v_scale_base :
        ggml_view_4d(ctx, v_scale_base,
                blocks, kv_heads, head_dim, 1,
                (int64_t) blocks * head_dim * sizeof(float),
                (int64_t) blocks * sizeof(float),
                (int64_t) blocks * n_embd * sizeof(float),
                0);
    ggml_tensor_set_nvfp4_scale(v, v_scale);

    ggml_tensor * v_perm = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor_set_nvfp4_scale(v_perm, v_scale);
    ggml_tensor * p = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_size, cols, q_heads, 1);
    ggml_tensor * out = ggml_mul_mat(ctx, v_perm, p);

    ggml_cgraph * gf_set = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf_set, set);

    ggml_cgraph * gf_out = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf_out, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> v_ref((size_t) n_embd * (size_t) kv_size);
    std::vector<float> v_cur_host((size_t) QK_NVFP4 * (size_t) kv_size * (size_t) n_row_groups);
    for (int64_t token = 0; token < kv_size; ++token) {
        for (int64_t row = 0; row < n_embd; ++row) {
            const int64_t h = row / head_dim;
            const int64_t d = row - h * head_dim;
            const float x = (float) (17 * h + 5 * d + token);
            const float v = 0.7f * sinf(0.013f * x) + 0.2f * cosf(0.031f * x);
            v_ref[(size_t) row * (size_t) kv_size + (size_t) token] = v;
            const int64_t row_group = row / QK_NVFP4;
            const int64_t row_in_group = row - row_group * QK_NVFP4;
            const int64_t flat_group = token * n_row_groups + row_group;
            v_cur_host[(size_t) flat_group * QK_NVFP4 + (size_t) row_in_group] = v;
        }
    }

    std::vector<int64_t> idx_host((size_t) kv_size * (size_t) n_row_groups);
    for (int64_t token = 0; token < kv_size; ++token) {
        for (int64_t row_group = 0; row_group < n_row_groups; ++row_group) {
            idx_host[(size_t) token * (size_t) n_row_groups + (size_t) row_group] =
                row_group * QK_NVFP4 * kv_size + token;
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
    std::vector<float> p_ref = p_host;
    std::vector<block_nvfp4> p_q((size_t) cols * (size_t) q_heads * (size_t) blocks);
    for (int64_t h = 0; h < q_heads; ++h) {
        for (int64_t c = 0; c < cols; ++c) {
            const size_t row_off = (size_t) h * (size_t) cols * (size_t) kv_size + (size_t) c * (size_t) kv_size;
            float amax = 0.0f;
            for (int64_t i = 0; i < kv_size; ++i) {
                amax = fmaxf(amax, fabsf(p_host[row_off + (size_t) i]));
            }
            const float global = amax > 0.0f ? (6.0f * 224.0f / amax) : 0.0f;
            for (int64_t b = 0; b < blocks; ++b) {
                const size_t q_off = ((size_t) h * (size_t) cols + (size_t) c) * (size_t) blocks + (size_t) b;
                quantize_row_nvfp4_ref(p_host.data() + row_off + (size_t) b * QK_NVFP4, p_q.data() + q_off, QK_NVFP4, global);
                dequantize_row_nvfp4(p_q.data() + q_off, p_ref.data() + row_off + (size_t) b * QK_NVFP4, QK_NVFP4, global);
            }
        }
    }

    const float fixed_amax = 64.0f;
    const float fixed_global = 6.0f * 224.0f / fixed_amax;
    std::vector<float> zero_scale((size_t) n_embd * (size_t) blocks, 0.0f);

    ggml_backend_tensor_set(v_cur, v_cur_host.data(), 0, v_cur_host.size() * sizeof(float));
    ggml_backend_tensor_set(v_idxs, idx_host.data(), 0, idx_host.size() * sizeof(int64_t));
    if (scalar_global_scale) {
        ggml_backend_tensor_set(v_scale_base, &fixed_global, 0, sizeof(float));
    } else {
        ggml_backend_tensor_set(v_scale_base, zero_scale.data(), 0, zero_scale.size() * sizeof(float));
    }
    ggml_backend_tensor_set(p, p_host.data(), 0, p_host.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf_set) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    if (ggml_backend_graph_compute(backend, gf_out) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<block_nvfp4> v_q((size_t) n_embd * (size_t) blocks);
    std::vector<float> v_s((size_t) n_embd * (size_t) blocks);
    std::vector<float> v_deq((size_t) n_embd * (size_t) kv_size);
    ggml_backend_tensor_get(v_cache, v_q.data(), 0, v_q.size() * sizeof(block_nvfp4));
    if (scalar_global_scale) {
        if (expect_lt_reference_for_kv_size(kv_size)) {
            dequantize_v_blocks_scalar_like_lt(v_q, fixed_global, v_deq, n_embd, kv_size, blocks);
        } else {
            for (int64_t row = 0; row < n_embd; ++row) {
                dequantize_row_nvfp4(v_q.data() + (size_t) row * (size_t) blocks,
                        v_deq.data() + (size_t) row * (size_t) kv_size, kv_size, fixed_global);
            }
        }
    } else {
        ggml_backend_tensor_get(v_scale_base, v_s.data(), 0, v_s.size() * sizeof(float));
        if (expect_lt_reference_for_kv_size(kv_size)) {
            dequantize_v_blocks_like_lt(v_q, v_s, v_deq, n_embd, kv_size, blocks);
        } else {
            for (int64_t row = 0; row < n_embd; ++row) {
                for (int64_t b = 0; b < blocks; ++b) {
                    const size_t off = (size_t) row * (size_t) blocks + (size_t) b;
                    const float global = v_s[off] > 0.0f ? 1.0f / v_s[off] : 0.0f;
                    dequantize_row_nvfp4(v_q.data() + off,
                            v_deq.data() + (size_t) row * (size_t) kv_size + (size_t) b * QK_NVFP4,
                            QK_NVFP4, global);
                }
            }
        }
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
                    ref += v_deq[(size_t) row * (size_t) kv_size + (size_t) i] * p_ref[p_off];
                }
                const size_t out_off = (size_t) h * (size_t) cols * (size_t) head_dim + (size_t) c * (size_t) head_dim + (size_t) d;
                const float got = out_host[out_off];
                if (fabsf(got - ref) > 0.5f) {
                    std::fprintf(stderr, "set_rows matmul mismatch scalar=%d h=%lld d=%lld c=%lld got=%g ref=%g\n",
                            scalar_global_scale ? 1 : 0, (long long) h, (long long) d, (long long) c, got, ref);
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

static bool run_real_vcache_view_benchmark(int64_t kv_size = 512) {
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

    std::vector<block_nvfp4> v_q((size_t) n_embd * (size_t) blocks);
    std::vector<float> v_s((size_t) n_embd * (size_t) blocks);
    std::vector<float> tmp((size_t) kv_size);
    for (int64_t row = 0; row < n_embd; ++row) {
        for (int64_t b = 0; b < blocks; ++b) {
            float amax = 0.0f;
            for (int64_t i = 0; i < 16; ++i) {
                const float x = 0.6f * sinf(0.011f * (float) (row * kv_size + b * 16 + i))
                    + 0.15f * cosf(0.023f * (float) (row + b * 16 + i));
                tmp[(size_t) b * 16 + (size_t) i] = x;
                amax = fmaxf(amax, fabsf(x));
            }
            const float global = amax > 0.0f ? (6.0f * 224.0f / amax) : 0.0f;
            const size_t off = (size_t) row * (size_t) blocks + (size_t) b;
            v_s[off] = global > 0.0f ? 1.0f / global : 0.0f;
            quantize_row_nvfp4_ref(tmp.data() + (size_t) b * 16, v_q.data() + off, 16, global);
        }
    }

    std::vector<float> p_host((size_t) kv_size * (size_t) cols * (size_t) q_heads);
    for (size_t i = 0; i < p_host.size(); ++i) {
        p_host[i] = 0.5f * cosf(0.007f * (float) (i + 3)) + 0.1f * sinf(0.019f * (float) (i + 1));
    }

    ggml_backend_tensor_set(v_cache, v_q.data(), 0, v_q.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_set(v_scale_base, v_s.data(), 0, v_s.size() * sizeof(float));
    ggml_backend_tensor_set(p, p_host.data(), 0, p_host.size() * sizeof(float));

    const int warmup = 10;
    const int iters = 100;
    for (int i = 0; i < warmup; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return false;
        }
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            ggml_backend_buffer_free(buf);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return false;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::printf("test-vcache-nvfp4-matmul: benchmark kv_size=%lld fp4_p=default %.3f us/iter (%d iters)\n",
            (long long) kv_size, elapsed_ms * 1000.0f / (float) iters, iters);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        std::puts("test-vcache-nvfp4-matmul: SKIP (no CUDA device)");
        return 0;
    }

    if (argc > 1 && std::strcmp(argv[1], "--benchmark-only") == 0) {
        const int64_t kv_size = argc > 2 ? std::strtoll(argv[2], nullptr, 10) : 512;
        return run_real_vcache_view_benchmark(kv_size) ? 0 : 1;
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

    if (!run_real_vcache_view_case(512, true)) {
        return 1;
    }

    if (!run_set_rows_then_matmul_case(false)) {
        return 1;
    }

    if (!run_set_rows_then_matmul_case(true)) {
        return 1;
    }

    if (!run_real_vcache_view_case(2048)) {
        return 1;
    }

    if (!run_real_vcache_view_benchmark()) {
        return 1;
    }

    std::puts("test-vcache-nvfp4-matmul: ok");
    return 0;
}
