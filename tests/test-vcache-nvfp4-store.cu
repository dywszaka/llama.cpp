#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>
#include "../ggml/src/ggml-quants.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr float NVFP4_FP4_MAX = 6.0f;
static constexpr int8_t NVFP4_VALUES[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

static float e4m3_to_fp32(uint8_t x) {
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

static float e4m3_to_fp32_half(uint8_t x) {
    return e4m3_to_fp32(x) * 0.5f;
}

static uint8_t best_index_nvfp4_ref(float x) {
    uint8_t best_index = 0;
    float best_err = fabsf((float) NVFP4_VALUES[0] - x);

    for (int i = 1; i < 16; ++i) {
        const float err = fabsf((float) NVFP4_VALUES[i] - x);
        if (err < best_err) {
            best_index = (uint8_t) i;
            best_err = err;
        }
    }

    return best_index;
}

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
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

static bool run_store_case(int64_t head_dim, int64_t n_tokens) {
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

    ggml_tensor * v_view = ggml_reshape_2d(ctx, v_cache, 16, head_dim * kv_size / 16);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim * n_tokens / 16);
    ggml_tensor * set = ggml_set_rows(ctx, v_view, ggml_reshape_2d(ctx, v_cur, 16, head_dim * n_tokens / 16), idx);
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
    std::vector<int64_t> idx_data((size_t) (head_dim * n_tokens / 16));
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int64_t j = 0; j < head_dim; j += 16) {
            idx_data[(size_t) i * (size_t) (head_dim / 16) + (size_t) (j / 16)] = j * kv_size + i;
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

static bool run_scalar_global_scale_store_case() {
    set_env("LLAMA_NVFP4_VCACHE_FAST_UPDATE", "0");

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

    const int64_t head_dim = 16;
    const int64_t kv_size = 16;
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, kv_size, head_dim, 1);
    ggml_tensor * v_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * v_cur_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, kv_size);
    ggml_tensor * v_cur = ggml_reshape_2d(ctx, v_cur_3d, head_dim, kv_size);

    ggml_tensor * v_view = ggml_reshape_2d(ctx, v_cache, 16, head_dim * kv_size / 16);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim);
    ggml_tensor * set = ggml_set_rows(ctx, v_view, ggml_reshape_2d(ctx, v_cur, 16, head_dim), idx);
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

    std::vector<float> src((size_t) head_dim * (size_t) kv_size);
    for (int64_t token = 0; token < kv_size; ++token) {
        for (int64_t row = 0; row < head_dim; ++row) {
            src[(size_t) token * (size_t) head_dim + (size_t) row] =
                    0.05f * (float) (row + 1) - 0.03f * (float) token;
        }
    }
    src[7 * head_dim + 3] = 9.0f;
    src[11 * head_dim + 9] = -12.0f;

    std::vector<int64_t> idx_data((size_t) head_dim);
    for (int64_t token = 0; token < kv_size; ++token) {
        idx_data[(size_t) token] = token;
    }

    const float global_scale = NVFP4_FP4_MAX * 224.0f / 64.0f;
    ggml_backend_tensor_set(v_cur_3d, src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_set(v_scale, &global_scale, 0, sizeof(float));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);

    std::vector<block_nvfp4> cache_host((size_t) head_dim);
    float scale_host = 0.0f;
    ggml_backend_tensor_get(v_cache, cache_host.data(), 0, cache_host.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_get(v_scale, &scale_host, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "scalar global scale graph compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }

    if (fabsf(scale_host - global_scale) > 1e-6f) {
        std::fprintf(stderr, "scalar global scale was modified got=%g expected=%g\n", scale_host, global_scale);
        return false;
    }

    for (int64_t row = 0; row < head_dim; ++row) {
        float ref_tile[16] = {};
        for (int64_t lane = 0; lane < 16; ++lane) {
            ref_tile[lane] = src[(size_t) lane * (size_t) head_dim + (size_t) row];
        }

        float amax = 0.0f;
        for (float v : ref_tile) {
            amax = fmaxf(amax, fabsf(v));
        }
        const float scale_f = global_scale != 0.0f ? global_scale * (amax / NVFP4_FP4_MAX) : 0.0f;

        uint8_t best_e = 0;
        float best_err = INFINITY;
        for (int i = 0; i < 256; ++i) {
            const float v = e4m3_to_fp32((uint8_t) i);
            if (!std::isfinite(v)) {
                continue;
            }
            const float err = fabsf(v - scale_f);
            if (err < best_err) {
                best_err = err;
                best_e = (uint8_t) i;
            }
        }
        const float block_scale = e4m3_to_fp32_half(best_e);
        const float inv_scale = (global_scale != 0.0f && block_scale != 0.0f) ? global_scale / block_scale : 0.0f;

        uint8_t ref_qs[QK_NVFP4 / 2] = {};
        for (int lane = 0; lane < QK_NVFP4; lane += 2) {
            const uint8_t q0 = best_index_nvfp4_ref(ref_tile[lane + 0] * inv_scale);
            const uint8_t q1 = best_index_nvfp4_ref(ref_tile[lane + 1] * inv_scale);
            ref_qs[lane / 2] = q0 | (q1 << 4);
        }

        if (cache_host[(size_t) row].e != best_e ||
                std::memcmp(cache_host[(size_t) row].qs, ref_qs, sizeof(ref_qs)) != 0) {
            std::fprintf(stderr, "scalar global scale quant mismatch row=%lld got_e=%u expected_e=%u\n",
                    (long long) row, (unsigned) cache_host[(size_t) row].e, (unsigned) best_e);
            return false;
        }
    }

    return true;
}

static bool run_scalar_global_scale_multi_stream_store_case() {
    set_env("LLAMA_NVFP4_VCACHE_FAST_UPDATE", "0");

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

    const int64_t head_dim = 16;
    const int64_t kv_size = 16;
    const int64_t n_stream = 2;
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, kv_size, head_dim, n_stream);
    ggml_tensor * v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_stream);
    ggml_tensor * v_cur_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, kv_size);
    ggml_tensor * v_cur = ggml_reshape_2d(ctx, v_cur_3d, head_dim, kv_size);

    ggml_tensor * v_view = ggml_reshape_2d(ctx, v_cache, 16, n_stream * head_dim * kv_size / 16);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim);
    ggml_tensor * set = ggml_set_rows(ctx, v_view, ggml_reshape_2d(ctx, v_cur, 16, head_dim), idx);
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

    std::vector<float> src((size_t) head_dim * (size_t) kv_size);
    for (int64_t token = 0; token < kv_size; ++token) {
        for (int64_t row = 0; row < head_dim; ++row) {
            src[(size_t) token * (size_t) head_dim + (size_t) row] =
                    0.07f * (float) (row + 1) - 0.02f * (float) token;
        }
    }
    src[5 * head_dim + 4] = 7.0f;
    src[12 * head_dim + 10] = -8.0f;

    std::vector<int64_t> idx_data((size_t) head_dim);
    for (int64_t token = 0; token < kv_size; ++token) {
        idx_data[(size_t) token] = head_dim * kv_size + token;
    }

    const float global_scales[2] = {
        NVFP4_FP4_MAX * 224.0f / 32.0f,
        NVFP4_FP4_MAX * 224.0f / 64.0f,
    };
    ggml_backend_tensor_set(v_cur_3d, src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_set(v_scale, global_scales, 0, sizeof(global_scales));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);

    std::vector<block_nvfp4> cache_host((size_t) n_stream * (size_t) head_dim);
    float scale_host[2] = {};
    ggml_backend_tensor_get(v_cache, cache_host.data(), 0, cache_host.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_get(v_scale, scale_host, 0, sizeof(scale_host));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "scalar multi-stream graph compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }

    for (int64_t s = 0; s < n_stream; ++s) {
        if (fabsf(scale_host[s] - global_scales[s]) > 1e-6f) {
            std::fprintf(stderr, "scalar multi-stream scale modified stream=%lld got=%g expected=%g\n",
                    (long long) s, scale_host[s], global_scales[s]);
            return false;
        }
    }

    for (int64_t row = 0; row < head_dim; ++row) {
        float ref_tile[16] = {};
        for (int64_t lane = 0; lane < 16; ++lane) {
            ref_tile[lane] = src[(size_t) lane * (size_t) head_dim + (size_t) row];
        }

        float amax = 0.0f;
        for (float v : ref_tile) {
            amax = fmaxf(amax, fabsf(v));
        }
        const float global_scale = global_scales[1];
        const float scale_f = global_scale != 0.0f ? global_scale * (amax / NVFP4_FP4_MAX) : 0.0f;

        uint8_t best_e = 0;
        float best_err = INFINITY;
        for (int i = 0; i < 256; ++i) {
            const float v = e4m3_to_fp32((uint8_t) i);
            if (!std::isfinite(v)) {
                continue;
            }
            const float err = fabsf(v - scale_f);
            if (err < best_err) {
                best_err = err;
                best_e = (uint8_t) i;
            }
        }

        const block_nvfp4 & block = cache_host[(size_t) head_dim + (size_t) row];
        if (block.e != best_e) {
            std::fprintf(stderr, "scalar multi-stream wrong stream scale row=%lld got_e=%u expected_e=%u\n",
                    (long long) row, (unsigned) block.e, (unsigned) best_e);
            return false;
        }
    }

    return true;
}

static bool compute_store_graph(
        ggml_backend_t backend,
        ggml_cgraph * gf,
        ggml_tensor * v_cur,
        ggml_tensor * idx,
        const std::vector<float> & src,
        const std::vector<int64_t> & idx_data) {
    ggml_backend_tensor_set(v_cur, src.data(), 0, src.size() * sizeof(float));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
        return false;
    }

    return true;
}

static bool run_fast_update_patch_case() {
    set_env("LLAMA_NVFP4_VCACHE_FAST_UPDATE", "1");

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

    const int64_t head_dim = 16;
    const int64_t kv_size = 16;
    const int64_t kv_blocks = kv_size / 16;
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, kv_size, head_dim, 1);
    ggml_tensor * v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim * kv_blocks, 1);

    ggml_tensor * v_view = ggml_reshape_2d(ctx, v_cache, 16, head_dim * kv_size / 16);
    ggml_tensor * v_cur_init_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, 16);
    ggml_tensor * idx_init = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim);
    ggml_tensor * set_init = ggml_set_rows(ctx, v_view,
            ggml_reshape_2d(ctx, ggml_reshape_2d(ctx, v_cur_init_3d, head_dim, 16), 16, head_dim),
            idx_init);
    ggml_tensor_set_nvfp4_scale(set_init, v_scale);

    ggml_tensor * v_cur_patch_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, 1);
    ggml_tensor * idx_patch = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim / 16);
    ggml_tensor * set_patch = ggml_set_rows(ctx, v_view,
            ggml_reshape_2d(ctx, ggml_reshape_2d(ctx, v_cur_patch_3d, head_dim, 1), 16, head_dim / 16),
            idx_patch);
    ggml_tensor_set_nvfp4_scale(set_patch, v_scale);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, set_init);
    ggml_cgraph * gf_patch = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf_patch, set_patch);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<int64_t> idx_data((size_t) head_dim);
    for (int64_t token = 0; token < 16; ++token) {
        for (int64_t j = 0; j < head_dim; j += 16) {
            idx_data[(size_t) token * (size_t) (head_dim / 16) + (size_t) j / 16] = j * kv_size + token;
        }
    }

    std::vector<float> initial((size_t) head_dim * 16);
    for (int64_t token = 0; token < 16; ++token) {
        for (int64_t row = 0; row < head_dim; ++row) {
            initial[(size_t) token * (size_t) head_dim + (size_t) row] = 0.15f + 0.03f * (float) token + 0.01f * (float) row;
        }
    }
    initial[7 * head_dim + 3] = 4.0f;

    if (!compute_store_graph(backend, gf, v_cur_init_3d, idx_init, initial, idx_data)) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<block_nvfp4> before((size_t) head_dim * (size_t) kv_blocks);
    std::vector<float> scale_before((size_t) head_dim * (size_t) kv_blocks);
    ggml_backend_tensor_get(v_cache, before.data(), 0, before.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_get(v_scale, scale_before.data(), 0, scale_before.size() * sizeof(float));

    std::vector<float> patch((size_t) head_dim);
    for (int64_t row = 0; row < head_dim; ++row) {
        patch[(size_t) row] = 0.02f * (float) (row + 1);
    }
    patch[3] = 1.0f;

    std::vector<int64_t> idx_patch_data(1, 5);
    if (!compute_store_graph(backend, gf_patch, v_cur_patch_3d, idx_patch, patch, idx_patch_data)) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<block_nvfp4> after((size_t) head_dim * (size_t) kv_blocks);
    std::vector<float> scale_after((size_t) head_dim * (size_t) kv_blocks);
    ggml_backend_tensor_get(v_cache, after.data(), 0, after.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_get(v_scale, scale_after.data(), 0, scale_after.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    for (int64_t row = 0; row < head_dim; ++row) {
        const size_t off = (size_t) row;
        if (after[off].e != before[off].e || scale_after[off] != scale_before[off]) {
            std::fprintf(stderr, "fast update changed scale at row=%lld before_e=%u after_e=%u before_scale=%g after_scale=%g\n",
                    (long long) row, (unsigned) before[off].e, (unsigned) after[off].e, scale_before[off], scale_after[off]);
            return false;
        }

        const float input_scale = scale_before[off];
        const float global_scale = input_scale > 0.0f ? 1.0f / input_scale : 0.0f;
        const float scale_f = e4m3_to_fp32_half(before[off].e);
        const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? global_scale / scale_f : 0.0f;
        const uint8_t expected_q = best_index_nvfp4_ref(patch[(size_t) row] * inv_scale);
        const uint8_t got_q = (after[off].qs[5 / 2] >> 4) & 0x0F;
        if (got_q != expected_q) {
            std::fprintf(stderr, "fast update patched wrong nibble row=%lld got=%u expected=%u\n",
                    (long long) row, (unsigned) got_q, (unsigned) expected_q);
            return false;
        }

        for (int byte = 0; byte < QK_NVFP4 / 2; ++byte) {
            uint8_t expected_byte = before[off].qs[byte];
            if (byte == 5 / 2) {
                expected_byte = (uint8_t) ((expected_byte & 0x0F) | (expected_q << 4));
            }
            if (after[off].qs[byte] != expected_byte) {
                std::fprintf(stderr, "fast update changed unexpected byte row=%lld byte=%d got=%u expected=%u\n",
                        (long long) row, byte, (unsigned) after[off].qs[byte], (unsigned) expected_byte);
                return false;
            }
        }
    }

    return true;
}

static bool run_fast_update_fallback_case() {
    set_env("LLAMA_NVFP4_VCACHE_FAST_UPDATE", "1");

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

    const int64_t head_dim = 16;
    const int64_t kv_size = 16;
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, kv_size, head_dim, 1);
    ggml_tensor * v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 1);

    ggml_tensor * v_view = ggml_reshape_2d(ctx, v_cache, 16, head_dim * kv_size / 16);
    ggml_tensor * v_cur_init_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, 16);
    ggml_tensor * idx_init = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim);
    ggml_tensor * set_init = ggml_set_rows(ctx, v_view,
            ggml_reshape_2d(ctx, ggml_reshape_2d(ctx, v_cur_init_3d, head_dim, 16), 16, head_dim),
            idx_init);
    ggml_tensor_set_nvfp4_scale(set_init, v_scale);

    ggml_tensor * v_cur_patch_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, 1);
    ggml_tensor * idx_patch = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim / 16);
    ggml_tensor * set_patch = ggml_set_rows(ctx, v_view,
            ggml_reshape_2d(ctx, ggml_reshape_2d(ctx, v_cur_patch_3d, head_dim, 1), 16, head_dim / 16),
            idx_patch);
    ggml_tensor_set_nvfp4_scale(set_patch, v_scale);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, set_init);
    ggml_cgraph * gf_patch = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf_patch, set_patch);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<int64_t> idx_data((size_t) head_dim);
    for (int64_t token = 0; token < 16; ++token) {
        for (int64_t j = 0; j < head_dim; j += 16) {
            idx_data[(size_t) token * (size_t) (head_dim / 16) + (size_t) j / 16] = j * kv_size + token;
        }
    }

    std::vector<float> initial((size_t) head_dim * 16);
    for (int64_t token = 0; token < 16; ++token) {
        for (int64_t row = 0; row < head_dim; ++row) {
            initial[(size_t) token * (size_t) head_dim + (size_t) row] = 0.1f + 0.02f * (float) token + 0.01f * (float) row;
        }
    }

    if (!compute_store_graph(backend, gf, v_cur_init_3d, idx_init, initial, idx_data)) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<block_nvfp4> before((size_t) head_dim);
    std::vector<float> scale_before((size_t) head_dim);
    ggml_backend_tensor_get(v_cache, before.data(), 0, before.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_get(v_scale, scale_before.data(), 0, scale_before.size() * sizeof(float));

    std::vector<float> patch((size_t) head_dim, 0.05f);
    patch[4] = 8.0f;
    std::vector<int64_t> idx_patch_data(1, 6);
    if (!compute_store_graph(backend, gf_patch, v_cur_patch_3d, idx_patch, patch, idx_patch_data)) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<block_nvfp4> after((size_t) head_dim);
    std::vector<float> scale_after((size_t) head_dim);
    ggml_backend_tensor_get(v_cache, after.data(), 0, after.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_get(v_scale, scale_after.data(), 0, scale_after.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (scale_after[4] == scale_before[4]) {
        std::fprintf(stderr, "fast update did not fall back when new value exceeded current block amax\n");
        return false;
    }

    float ref_tile[16];
    const float before_global = scale_before[4] > 0.0f ? 1.0f / scale_before[4] : 0.0f;
    dequantize_row_nvfp4(&before[4], ref_tile, 16, before_global);
    ref_tile[6] = patch[4];
    float amax = 0.0f;
    for (float v : ref_tile) {
        amax = fmaxf(amax, fabsf(v));
    }
    const float ref_global = amax > 0.0f ? (NVFP4_FP4_MAX * 224.0f / amax) : 0.0f;
    block_nvfp4 ref_q;
    quantize_row_nvfp4_ref(ref_tile, &ref_q, 16, ref_global);
    const float ref_input_scale = ref_global > 0.0f ? 1.0f / ref_global : 0.0f;

    if (after[4].e != ref_q.e || fabsf(scale_after[4] - ref_input_scale) > 1e-7f ||
            std::memcmp(after[4].qs, ref_q.qs, sizeof(ref_q.qs)) != 0) {
        std::fprintf(stderr, "fallback re-quant mismatch row=4 got_e=%u ref_e=%u got_scale=%g ref_scale=%g\n",
                (unsigned) after[4].e, (unsigned) ref_q.e, scale_after[4], ref_input_scale);
        return false;
    }

    return true;
}

static bool run_fast_update_benchmark() {
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

    const int64_t head_dim = 1024;
    const int64_t kv_size = 16;
    ggml_tensor * v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_NVFP4, kv_size, head_dim, 1);
    ggml_tensor * v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, 1);

    ggml_tensor * v_view = ggml_reshape_2d(ctx, v_cache, 16, head_dim * kv_size / 16);
    ggml_tensor * v_cur_init_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, 16);
    ggml_tensor * idx_init = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim);
    ggml_tensor * set_init = ggml_set_rows(ctx, v_view,
            ggml_reshape_2d(ctx, ggml_reshape_2d(ctx, v_cur_init_3d, head_dim, 16), 16, head_dim),
            idx_init);
    ggml_tensor_set_nvfp4_scale(set_init, v_scale);

    ggml_tensor * v_cur_patch_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, 1);
    ggml_tensor * idx_patch = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, head_dim / 16);
    ggml_tensor * set_patch = ggml_set_rows(ctx, v_view,
            ggml_reshape_2d(ctx, ggml_reshape_2d(ctx, v_cur_patch_3d, head_dim, 1), 16, head_dim / 16),
            idx_patch);
    ggml_tensor_set_nvfp4_scale(set_patch, v_scale);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf, set_init);
    ggml_cgraph * gf_patch = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(gf_patch, set_patch);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<int64_t> idx_data((size_t) head_dim / 16);
    idx_data.resize((size_t) head_dim);
    for (int64_t token = 0; token < 16; ++token) {
        for (int64_t j = 0; j < head_dim; j += 16) {
            idx_data[(size_t) token * (size_t) (head_dim / 16) + (size_t) j / 16] = j * kv_size + token;
        }
    }

    std::vector<float> initial((size_t) head_dim * 16);
    for (int64_t token = 0; token < 16; ++token) {
        for (int64_t row = 0; row < head_dim; ++row) {
            initial[(size_t) token * (size_t) head_dim + (size_t) row] = 0.2f + 0.01f * (float) (token % 7) + 0.001f * (float) (row % 13);
        }
    }
    if (!compute_store_graph(backend, gf, v_cur_init_3d, idx_init, initial, idx_data)) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> patch((size_t) head_dim);
    for (int64_t row = 0; row < head_dim; ++row) {
        patch[(size_t) row] = 0.05f + 0.0005f * (float) (row % 11);
    }
    std::vector<int64_t> idx_patch_data((size_t) head_dim / 16);
    for (int64_t j = 0; j < head_dim; j += 16) {
        idx_patch_data[(size_t) j / 16] = j * kv_size + 3;
    }

    const int warmup = 20;
    const int iters = 200;
    for (int i = 0; i < warmup; ++i) {
        if (!compute_store_graph(backend, gf_patch, v_cur_patch_3d, idx_patch, patch, idx_patch_data)) {
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
        if (!compute_store_graph(backend, gf_patch, v_cur_patch_3d, idx_patch, patch, idx_patch_data)) {
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

    const char * fast_env = getenv("LLAMA_NVFP4_VCACHE_FAST_UPDATE");
    std::printf("test-vcache-nvfp4-store: benchmark fast_update=%s %.3f us/iter (%d iters, head_dim=%lld)\n",
            fast_env != nullptr ? fast_env : "(unset)",
            elapsed_ms * 1000.0f / (float) iters, iters, (long long) head_dim);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    return true;
}

int main(int argc, char ** argv) {
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

    if (argc > 1 && std::strcmp(argv[1], "--benchmark-only") == 0) {
        return run_fast_update_benchmark() ? 0 : 1;
    }

    set_env("LLAMA_NVFP4_VCACHE_FAST_UPDATE", "1");

    if (!run_store_case(128, 17)) {
        return 1;
    }
    if (!run_scalar_global_scale_store_case()) {
        return 1;
    }
    if (!run_scalar_global_scale_multi_stream_store_case()) {
        return 1;
    }
    if (!run_fast_update_patch_case()) {
        return 1;
    }
    if (!run_fast_update_fallback_case()) {
        return 1;
    }
    if (!run_fast_update_benchmark()) {
        return 1;
    }

    std::puts("test-vcache-nvfp4-store: ok");
    return 0;
}
