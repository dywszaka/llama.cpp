#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-quants.h"
#include "nvfp4-bf16-test-utils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr float NVFP4_GLOBAL_SCALE_MAX = 6.0f * 224.0f;

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static float row_amax(const float * row, int64_t k) {
    float amax = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        amax = fmaxf(amax, fabsf(row[i]));
    }
    return amax;
}

static bool run_set_rows_case(bool bf16_switches, bool use_outliers) {
    if (bf16_switches) {
        set_env("GGML_CUDA_NVFP4_BF16_QUANT", "1");
        set_env("GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN", "1");
    } else {
        set_env("GGML_CUDA_NVFP4_BF16_QUANT", "0");
        set_env("GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN", "0");
    }

    ggml_init_params params = {
        /* .mem_size   = */ 8 * 1024 * 1024,
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

    constexpr int64_t k = 32;
    constexpr int64_t rows = 4;
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, k, rows);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows);
    ggml_tensor_set_nvfp4_scale(cache, scale);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, rows);

    ggml_tensor * outlier_counts = nullptr;
    ggml_tensor * outlier_offsets = nullptr;
    ggml_tensor * outlier_cursor = nullptr;
    ggml_tensor * outlier_indices = nullptr;
    ggml_tensor * outlier_values = nullptr;
    if (use_outliers) {
        outlier_counts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
        outlier_offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
        outlier_cursor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        outlier_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows * 2);
        outlier_values = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows * 2);
        ggml_tensor_set_nvfp4_kcache_outliers_compact(
                cache, outlier_counts, outlier_offsets, outlier_indices, outlier_values);
        ggml_tensor_set_nvfp4_kcache_outlier_cursor(cache, outlier_cursor);
    }

    ggml_tensor * set = ggml_set_rows(ctx, cache, src, idx);
    ggml_tensor_set_nvfp4_scale(set, scale);
    if (use_outliers) {
        ggml_tensor_set_nvfp4_kcache_outliers_compact(
                set, outlier_counts, outlier_offsets, outlier_indices, outlier_values);
        ggml_tensor_set_nvfp4_kcache_outlier_cursor(set, outlier_cursor);
        const float threshold = 1.25f;
        std::memcpy(&set->op_params[0], &threshold, sizeof(threshold));
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, set);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        std::fprintf(stderr, "failed to allocate backend tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> src_data((size_t) k * (size_t) rows);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < k; ++i) {
            const float sign = ((i + r) & 1) ? -1.0f : 1.0f;
            src_data[(size_t) r * (size_t) k + (size_t) i] =
                    sign * (0.33325195f + 0.1375f * (float) ((i % 13) - 6) + 0.025f * (float) r);
        }
    }
    src_data[(size_t) 0 * k + 5] = 1.8759766f;
    src_data[(size_t) 1 * k + 7] = 1.75f;
    src_data[(size_t) 2 * k + 19] = -1.8759766f;
    const std::vector<int64_t> idx_data = { 2, 0, 3, 1 };

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));
    if (use_outliers) {
        std::vector<int32_t> zeros_i32((size_t) rows * 2, 0);
        std::vector<int32_t> offsets((size_t) rows, 0);
        std::vector<float> zeros_f32((size_t) rows * 2, 0.0f);
        for (int64_t r = 0; r < rows; ++r) {
            offsets[(size_t) r] = (int32_t) (r * 2);
        }
        ggml_backend_tensor_set(outlier_counts, zeros_i32.data(), 0, rows * sizeof(int32_t));
        ggml_backend_tensor_set(outlier_offsets, offsets.data(), 0, offsets.size() * sizeof(int32_t));
        ggml_backend_tensor_set(outlier_cursor, zeros_i32.data(), 0, sizeof(int32_t));
        ggml_backend_tensor_set(outlier_indices, zeros_i32.data(), 0, zeros_i32.size() * sizeof(int32_t));
        ggml_backend_tensor_set(outlier_values, zeros_f32.data(), 0, zeros_f32.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "set_rows graph failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<block_nvfp4> got((size_t) rows * (size_t) (k / QK_NVFP4));
    std::vector<float> scale_host((size_t) rows);
    ggml_backend_tensor_get(cache, got.data(), 0, got.size() * sizeof(block_nvfp4));
    ggml_backend_tensor_get(scale, scale_host.data(), 0, scale_host.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    for (int64_t src_row = 0; src_row < rows; ++src_row) {
        std::vector<float> ref_row((size_t) k);
        const float * input_row = src_data.data() + (size_t) src_row * (size_t) k;
        const float amax = row_amax(input_row, k);
        const float threshold = use_outliers ? 1.25f : 0.0f;
        const float global_scale = use_outliers
                ? (threshold > 0.0f ? NVFP4_GLOBAL_SCALE_MAX / threshold : (amax > 0.0f ? NVFP4_GLOBAL_SCALE_MAX / amax : 0.0f))
                : (amax > 0.0f ? NVFP4_GLOBAL_SCALE_MAX / amax : 0.0f);
        for (int64_t i = 0; i < k; ++i) {
            const float raw = input_row[i];
            ref_row[(size_t) i] = (use_outliers && fabsf(raw) > threshold) ? 0.0f : raw;
        }

        std::vector<block_nvfp4> expected;
        if (bf16_switches) {
            nvfp4_test_quantize_bf16_trunc_nn_rows(ref_row, expected, 1, k, std::vector<float>{ global_scale });
        } else {
            expected.resize((size_t) k / QK_NVFP4);
            quantize_row_nvfp4_ref(ref_row.data(), expected.data(), k, global_scale);
        }

        const int64_t dst_row = idx_data[(size_t) src_row];
        const float expected_input_scale = global_scale > 0.0f ? 1.0f / global_scale : 0.0f;
        if (fabsf(scale_host[(size_t) dst_row] - expected_input_scale) > 1e-7f) {
            std::fprintf(stderr, "scale mismatch src_row=%lld dst_row=%lld got=%g expected=%g\n",
                    (long long) src_row, (long long) dst_row, scale_host[(size_t) dst_row], expected_input_scale);
            return false;
        }

        for (int64_t ib = 0; ib < k / QK_NVFP4; ++ib) {
            const block_nvfp4 & block = got[(size_t) dst_row * (size_t) (k / QK_NVFP4) + (size_t) ib];
            const block_nvfp4 & ref = expected[(size_t) ib];
            if (block.e != ref.e || std::memcmp(block.qs, ref.qs, sizeof(ref.qs)) != 0) {
                std::fprintf(stderr,
                        "%s set_rows mismatch use_outliers=%d src_row=%lld dst_row=%lld block=%lld got_e=%u expected_e=%u\n",
                        bf16_switches ? "bf16" : "default",
                        use_outliers ? 1 : 0,
                        (long long) src_row, (long long) dst_row, (long long) ib,
                        (unsigned) block.e, (unsigned) ref.e);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    int dev_count = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count <= 0) {
        std::puts("test-nvfp4-kcache-set-rows: SKIP (no CUDA device)");
        return 0;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        std::puts("test-nvfp4-kcache-set-rows: SKIP (failed to select CUDA device 0)");
        return 0;
    }

    const bool bf16_only = argc > 1 && std::strcmp(argv[1], "--bf16-switch-only") == 0;
    if (bf16_only) {
        if (!run_set_rows_case(true, false)) {
            return 1;
        }
        if (!run_set_rows_case(true, true)) {
            return 1;
        }
        std::puts("test-nvfp4-kcache-set-rows: bf16 cases ok");
        return 0;
    }

    if (!run_set_rows_case(false, false)) {
        return 1;
    }
    if (!run_set_rows_case(false, true)) {
        return 1;
    }

    std::puts("test-nvfp4-kcache-set-rows: ok");
    return 0;
}
