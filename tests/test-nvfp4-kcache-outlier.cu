#include <ggml.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-cuda/expt/nvfp4/kcache-outlier.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do {                                                                  \
    cudaError_t err__ = (call);                                                                \
    if (err__ != cudaSuccess) {                                                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1);                                                                          \
    }                                                                                          \
} while (0)

static bool nearly_equal(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void set_env_var(const char * name, const char * value) {
    if (value == nullptr) {
#if defined(_WIN32)
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    } else {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
}

static bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        cudaGetLastError();
        return false;
    }
    return true;
}

static void test_extract_compact_offsets_and_pool() {
    constexpr int64_t k = 16;
    constexpr int64_t rows = 2;
    constexpr int64_t sidecar_rows = 8;
    constexpr int64_t row_capacity = 2;
    constexpr int64_t compact_capacity = sidecar_rows * row_capacity;
    constexpr float threshold = 16.0f;

    std::vector<float> src((size_t) rows * (size_t) k, 0.0f);
    src[3] = 17.5f;
    src[7] = -19.0f;
    src[(size_t) k + 4] = -16.25f;
    src[(size_t) k + 6] = 23.0f;
    const int64_t idx_h[2] = { 5, 2 };

    float * src_d = nullptr;
    int64_t * idx_d = nullptr;
    int32_t * counts_d = nullptr;
    int32_t * offsets_d = nullptr;
    int32_t * cursor_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * amax_d = nullptr;

    CUDA_CHECK(cudaMalloc(&src_d, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&idx_d, sizeof(idx_h)));
    CUDA_CHECK(cudaMalloc(&counts_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&offsets_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&cursor_d, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, (size_t) compact_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&amax_d, (size_t) rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_d, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(idx_d, idx_h, sizeof(idx_h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(counts_d, 0, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(offsets_d, 0xff, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(cursor_d, 0, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(indices_d, 0, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(values_d, 0, (size_t) compact_capacity * sizeof(float)));

    ggml_cuda_nvfp4_kcache_outlier_extract(
            "test_compact", src_d, idx_d, counts_d, offsets_d, cursor_d, indices_d, values_d, amax_d,
            k, rows, k, 1, sidecar_rows, compact_capacity, threshold, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> counts((size_t) sidecar_rows);
    std::vector<int32_t> offsets((size_t) sidecar_rows);
    std::vector<int32_t> indices((size_t) compact_capacity);
    std::vector<float> values((size_t) compact_capacity);
    int32_t cursor = 0;
    CUDA_CHECK(cudaMemcpy(counts.data(), counts_d, counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), offsets_d, offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices.data(), indices_d, indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values.data(), values_d, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&cursor, cursor_d, sizeof(cursor), cudaMemcpyDeviceToHost));

    require(counts[5] == 2, "compact row 5 count");
    require(counts[2] == 2, "compact row 2 count");
    require(cursor == 4, "compact cursor should reserve exactly stored entries");

    const int32_t off5 = offsets[5];
    const int32_t off2 = offsets[2];
    require(off5 >= 0 && off2 >= 0 && off5 != off2, "compact rows should have distinct offsets");
    require(off5 == 5 * row_capacity, "compact row 5 fixed offset");
    require(off2 == 2 * row_capacity, "compact row 2 fixed offset");
    require(indices[(size_t) off5 + 0] == 3, "compact row 5 first index");
    require(indices[(size_t) off5 + 1] == 7, "compact row 5 second index");
    require(indices[(size_t) off2 + 0] == 4, "compact row 2 first index");
    require(indices[(size_t) off2 + 1] == 6, "compact row 2 second index");
    require(nearly_equal(values[(size_t) off5 + 0], 17.5f), "compact row 5 first value");
    require(nearly_equal(values[(size_t) off5 + 1], -19.0f), "compact row 5 second value");
    require(nearly_equal(values[(size_t) off2 + 0], -16.25f), "compact row 2 first value");
    require(nearly_equal(values[(size_t) off2 + 1], 23.0f), "compact row 2 second value");

    CUDA_CHECK(cudaFree(src_d));
    CUDA_CHECK(cudaFree(idx_d));
    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    CUDA_CHECK(cudaFree(cursor_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(amax_d));
}

static void test_extract_compact_requires_full_row_capacity() {
    constexpr int64_t k = 16;
    constexpr int64_t rows = 2;
    constexpr int64_t sidecar_rows = 8;
    constexpr int64_t row_capacity = 1;
    constexpr int64_t compact_capacity = sidecar_rows * row_capacity;
    constexpr float threshold = 16.0f;

    std::vector<float> src((size_t) rows * (size_t) k, 0.0f);
    src[1] = 17.0f;
    src[2] = 18.0f;
    src[(size_t) k + 3] = 19.0f;
    src[(size_t) k + 4] = 20.0f;
    const int64_t idx_h[2] = { 0, 1 };

    float * src_d = nullptr;
    int64_t * idx_d = nullptr;
    int32_t * counts_d = nullptr;
    int32_t * offsets_d = nullptr;
    int32_t * cursor_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * amax_d = nullptr;

    CUDA_CHECK(cudaMalloc(&src_d, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&idx_d, sizeof(idx_h)));
    CUDA_CHECK(cudaMalloc(&counts_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&offsets_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&cursor_d, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, (size_t) compact_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&amax_d, (size_t) rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_d, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(idx_d, idx_h, sizeof(idx_h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(counts_d, 0, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(offsets_d, 0xff, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(cursor_d, 0x7f, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(indices_d, 0, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(values_d, 0, (size_t) compact_capacity * sizeof(float)));

    ggml_cuda_nvfp4_kcache_outlier_extract(
            "test_compact_overflow", src_d, idx_d, counts_d, offsets_d, cursor_d, indices_d, values_d, amax_d,
            k, rows, k, 1, sidecar_rows, compact_capacity, threshold, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> counts((size_t) sidecar_rows);
    std::vector<int32_t> offsets((size_t) sidecar_rows);
    int32_t cursor = 0;
    CUDA_CHECK(cudaMemcpy(counts.data(), counts_d, counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), offsets_d, offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&cursor, cursor_d, sizeof(cursor), cudaMemcpyDeviceToHost));

    require(cursor == 4, "compact cursor should count all requested entries after reset");
    require(counts[0] == 0 && offsets[0] == -1, "compact row 0 should be dropped when it exceeds row capacity");
    require(counts[1] == 0 && offsets[1] == -1, "compact row 1 should be dropped when it exceeds row capacity");

    CUDA_CHECK(cudaFree(src_d));
    CUDA_CHECK(cudaFree(idx_d));
    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    CUDA_CHECK(cudaFree(cursor_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(amax_d));
}

static void test_extract_compact_resets_counts_after_assigning_offsets() {
    constexpr int64_t k = 16;
    constexpr int64_t rows = 2;
    constexpr int64_t sidecar_rows = 4;
    constexpr int64_t row_capacity = 2;
    constexpr int64_t compact_capacity = sidecar_rows * row_capacity;
    constexpr float threshold = 16.0f;

    std::vector<float> src((size_t) rows * (size_t) k, 0.0f);
    src[1] = 17.0f;
    src[4] = -18.0f;
    src[(size_t) k + 2] = 19.0f;
    src[(size_t) k + 5] = -20.0f;
    const int64_t idx_h[2] = { 0, 1 };

    float * src_d = nullptr;
    int64_t * idx_d = nullptr;
    int32_t * counts_d = nullptr;
    int32_t * offsets_d = nullptr;
    int32_t * cursor_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * amax_d = nullptr;

    CUDA_CHECK(cudaMalloc(&src_d, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&idx_d, sizeof(idx_h)));
    CUDA_CHECK(cudaMalloc(&counts_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&offsets_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&cursor_d, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, (size_t) compact_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&amax_d, (size_t) rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_d, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(idx_d, idx_h, sizeof(idx_h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(counts_d, 0, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(offsets_d, 0xff, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(cursor_d, 0, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(indices_d, 0, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(values_d, 0, (size_t) compact_capacity * sizeof(float)));

    ggml_cuda_nvfp4_kcache_outlier_extract(
            "test_compact_reset_after_offsets", src_d, idx_d, counts_d, offsets_d, cursor_d, indices_d, values_d, amax_d,
            k, rows, k, 1, sidecar_rows, compact_capacity, threshold, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> counts((size_t) sidecar_rows);
    std::vector<int32_t> offsets((size_t) sidecar_rows);
    std::vector<int32_t> indices((size_t) compact_capacity);
    std::vector<float> values((size_t) compact_capacity);
    int32_t cursor = 0;
    CUDA_CHECK(cudaMemcpy(counts.data(), counts_d, counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), offsets_d, offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices.data(), indices_d, indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values.data(), values_d, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&cursor, cursor_d, sizeof(cursor), cudaMemcpyDeviceToHost));

    require(cursor == 4, "fused offset assignment should preserve compact cursor accounting");
    require(counts[0] == 2 && counts[1] == 2, "fused offset assignment should leave final fill counts");
    require(offsets[0] == 0 && offsets[1] == row_capacity, "fused offset assignment should preserve fixed row offsets");
    require(indices[0] == 1 && nearly_equal(values[0], 17.0f), "row 0 first outlier after fused reset");
    require(indices[1] == 4 && nearly_equal(values[1], -18.0f), "row 0 second outlier after fused reset");
    require(indices[2] == 2 && nearly_equal(values[2], 19.0f), "row 1 first outlier after fused reset");
    require(indices[3] == 5 && nearly_equal(values[3], -20.0f), "row 1 second outlier after fused reset");

    CUDA_CHECK(cudaFree(src_d));
    CUDA_CHECK(cudaFree(idx_d));
    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    CUDA_CHECK(cudaFree(cursor_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(amax_d));
}

static void test_extract_preserves_rows_across_multiple_set_rows_calls() {
    constexpr int64_t k = 16;
    constexpr int64_t rows = 2;
    constexpr int64_t sidecar_rows = 4;
    constexpr int64_t row_capacity = 2;
    constexpr int64_t compact_capacity = sidecar_rows * row_capacity;
    constexpr float threshold = 16.0f;

    const int64_t first_idx_h[2] = { 0, 1 };
    const int64_t second_idx_h[2] = { 2, 3 };
    std::vector<float> first_src((size_t) rows * (size_t) k, 0.0f);
    std::vector<float> second_src((size_t) rows * (size_t) k, 0.0f);
    first_src[1] = 17.0f;
    first_src[5] = -18.0f;
    first_src[(size_t) k + 2] = 19.0f;
    first_src[(size_t) k + 6] = -20.0f;
    second_src[3] = 21.0f;
    second_src[7] = -22.0f;
    second_src[(size_t) k + 4] = 23.0f;
    second_src[(size_t) k + 8] = -24.0f;

    float * first_src_d = nullptr;
    float * second_src_d = nullptr;
    int64_t * first_idx_d = nullptr;
    int64_t * second_idx_d = nullptr;
    int32_t * counts_d = nullptr;
    int32_t * offsets_d = nullptr;
    int32_t * cursor_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * amax_d = nullptr;

    CUDA_CHECK(cudaMalloc(&first_src_d, first_src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&second_src_d, second_src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&first_idx_d, sizeof(first_idx_h)));
    CUDA_CHECK(cudaMalloc(&second_idx_d, sizeof(second_idx_h)));
    CUDA_CHECK(cudaMalloc(&counts_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&offsets_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&cursor_d, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, (size_t) compact_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&amax_d, (size_t) rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(first_src_d, first_src.data(), first_src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(second_src_d, second_src.data(), second_src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(first_idx_d, first_idx_h, sizeof(first_idx_h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(second_idx_d, second_idx_h, sizeof(second_idx_h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(counts_d, 0, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(offsets_d, 0xff, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(cursor_d, 0, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(indices_d, 0, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(values_d, 0, (size_t) compact_capacity * sizeof(float)));

    ggml_cuda_nvfp4_kcache_outlier_extract(
            "test_preserve_first", first_src_d, first_idx_d, counts_d, offsets_d, cursor_d, indices_d, values_d, amax_d,
            k, rows, k, 1, sidecar_rows, compact_capacity, threshold, nullptr);
    ggml_cuda_nvfp4_kcache_outlier_extract(
            "test_preserve_second", second_src_d, second_idx_d, counts_d, offsets_d, cursor_d, indices_d, values_d, amax_d,
            k, rows, k, 1, sidecar_rows, compact_capacity, threshold, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> counts((size_t) sidecar_rows);
    std::vector<int32_t> offsets((size_t) sidecar_rows);
    std::vector<int32_t> indices((size_t) compact_capacity);
    std::vector<float> values((size_t) compact_capacity);
    CUDA_CHECK(cudaMemcpy(counts.data(), counts_d, counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), offsets_d, offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices.data(), indices_d, indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values.data(), values_d, values.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int row = 0; row < (int) sidecar_rows; ++row) {
        require(counts[(size_t) row] == 2, "each row should retain its two outliers");
        require(offsets[(size_t) row] == row * row_capacity, "row offset should be stable across set_rows calls");
    }
    require(indices[0] == 1 && nearly_equal(values[0], 17.0f), "row 0 first outlier should survive second extract");
    require(indices[1] == 5 && nearly_equal(values[1], -18.0f), "row 0 second outlier should survive second extract");
    require(indices[2] == 2 && nearly_equal(values[2], 19.0f), "row 1 first outlier should survive second extract");
    require(indices[3] == 6 && nearly_equal(values[3], -20.0f), "row 1 second outlier should survive second extract");
    require(indices[4] == 3 && nearly_equal(values[4], 21.0f), "row 2 first outlier should be stored");
    require(indices[5] == 7 && nearly_equal(values[5], -22.0f), "row 2 second outlier should be stored");
    require(indices[6] == 4 && nearly_equal(values[6], 23.0f), "row 3 first outlier should be stored");
    require(indices[7] == 8 && nearly_equal(values[7], -24.0f), "row 3 second outlier should be stored");

    CUDA_CHECK(cudaFree(first_src_d));
    CUDA_CHECK(cudaFree(second_src_d));
    CUDA_CHECK(cudaFree(first_idx_d));
    CUDA_CHECK(cudaFree(second_idx_d));
    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    CUDA_CHECK(cudaFree(cursor_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(amax_d));
}

static void test_extract_split_matches_full_for_same_rows() {
    constexpr int64_t k = 16;
    constexpr int64_t rows = 512;
    constexpr int64_t split_rows = 128;
    constexpr int64_t sidecar_rows = rows;
    constexpr int64_t row_capacity = 2;
    constexpr int64_t compact_capacity = sidecar_rows * row_capacity;
    constexpr float threshold = 16.0f;

    std::vector<float> src((size_t) rows * (size_t) k, 0.0f);
    std::vector<int64_t> dst_rows((size_t) rows);
    for (int64_t r = 0; r < rows; ++r) {
        dst_rows[(size_t) r] = r;
        src[(size_t) r * k + (size_t) (r % k)] = 1.0f + (float) (r % 7);
        if (r % 17 == 0) {
            src[(size_t) r * k + (size_t) ((r + 3) % k)] = 17.0f + 0.125f * (float) (r % 11);
        }
        if (r % 97 == 0) {
            src[(size_t) r * k + (size_t) ((r + 5) % k)] = -18.0f - 0.25f * (float) (r % 5);
        }
    }

    float * src_d = nullptr;
    int64_t * dst_rows_d = nullptr;
    int32_t * split_counts_d = nullptr;
    int32_t * split_offsets_d = nullptr;
    int32_t * split_cursor_d = nullptr;
    int32_t * split_indices_d = nullptr;
    float * split_values_d = nullptr;
    int32_t * full_counts_d = nullptr;
    int32_t * full_offsets_d = nullptr;
    int32_t * full_cursor_d = nullptr;
    int32_t * full_indices_d = nullptr;
    float * full_values_d = nullptr;
    float * amax_d = nullptr;

    CUDA_CHECK(cudaMalloc(&src_d, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst_rows_d, dst_rows.size() * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&split_counts_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&split_offsets_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&split_cursor_d, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&split_indices_d, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&split_values_d, (size_t) compact_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&full_counts_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&full_offsets_d, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&full_cursor_d, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&full_indices_d, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&full_values_d, (size_t) compact_capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&amax_d, (size_t) rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_d, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst_rows_d, dst_rows.data(), dst_rows.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(split_counts_d, 0, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(split_offsets_d, 0xff, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(split_cursor_d, 0, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(split_indices_d, 0, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(split_values_d, 0, (size_t) compact_capacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(full_counts_d, 0, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(full_offsets_d, 0xff, (size_t) sidecar_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(full_cursor_d, 0, sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(full_indices_d, 0, (size_t) compact_capacity * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(full_values_d, 0, (size_t) compact_capacity * sizeof(float)));

    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_DETERMINISTIC_FILL", "1");
    for (int64_t off = 0; off < rows; off += split_rows) {
        ggml_cuda_nvfp4_kcache_outlier_extract(
                "test_split", src_d + off * k, dst_rows_d + off,
                split_counts_d, split_offsets_d, split_cursor_d, split_indices_d, split_values_d, amax_d,
                k, split_rows, k, 1, sidecar_rows, compact_capacity, threshold, nullptr);
    }
    ggml_cuda_nvfp4_kcache_outlier_extract(
            "test_full", src_d, dst_rows_d,
            full_counts_d, full_offsets_d, full_cursor_d, full_indices_d, full_values_d, amax_d,
            k, rows, k, 1, sidecar_rows, compact_capacity, threshold, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_DETERMINISTIC_FILL", nullptr);

    std::vector<int32_t> split_counts((size_t) sidecar_rows);
    std::vector<int32_t> split_offsets((size_t) sidecar_rows);
    std::vector<int32_t> split_indices((size_t) compact_capacity);
    std::vector<float> split_values((size_t) compact_capacity);
    std::vector<int32_t> full_counts((size_t) sidecar_rows);
    std::vector<int32_t> full_offsets((size_t) sidecar_rows);
    std::vector<int32_t> full_indices((size_t) compact_capacity);
    std::vector<float> full_values((size_t) compact_capacity);
    CUDA_CHECK(cudaMemcpy(split_counts.data(), split_counts_d, split_counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(split_offsets.data(), split_offsets_d, split_offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(split_indices.data(), split_indices_d, split_indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(split_values.data(), split_values_d, split_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(full_counts.data(), full_counts_d, full_counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(full_offsets.data(), full_offsets_d, full_offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(full_indices.data(), full_indices_d, full_indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(full_values.data(), full_values_d, full_values.size() * sizeof(float), cudaMemcpyDeviceToHost));

    require(split_counts == full_counts, "split extract counts should match full extract counts");
    require(split_offsets == full_offsets, "split extract offsets should match full extract offsets");
    require(split_indices == full_indices, "split extract indices should match full extract indices");
    require(split_values == full_values, "split extract values should match full extract values");

    CUDA_CHECK(cudaFree(src_d));
    CUDA_CHECK(cudaFree(dst_rows_d));
    CUDA_CHECK(cudaFree(split_counts_d));
    CUDA_CHECK(cudaFree(split_offsets_d));
    CUDA_CHECK(cudaFree(split_cursor_d));
    CUDA_CHECK(cudaFree(split_indices_d));
    CUDA_CHECK(cudaFree(split_values_d));
    CUDA_CHECK(cudaFree(full_counts_d));
    CUDA_CHECK(cudaFree(full_offsets_d));
    CUDA_CHECK(cudaFree(full_cursor_d));
    CUDA_CHECK(cudaFree(full_indices_d));
    CUDA_CHECK(cudaFree(full_values_d));
    CUDA_CHECK(cudaFree(amax_d));
}

static void test_nvfp4_outlier_k_scale_mode_selects_row_or_threshold_amax() {
    constexpr float row_amax = 4.0f;
    constexpr float threshold = 16.0f;
    constexpr float expected_row_global_scale = 1344.0f / row_amax;
    constexpr float expected_row_input_scale = row_amax / 1344.0f;
    constexpr float expected_threshold_global_scale = 1344.0f / threshold;
    constexpr float expected_threshold_input_scale = threshold / 1344.0f;

    require(nearly_equal(
                ggml_cuda_nvfp4_kcache_outlier_k_global_scale(row_amax, threshold, false),
                expected_row_global_scale),
            "NVFP4 outlier K global scale should default to residual row amax");
    require(nearly_equal(
                ggml_cuda_nvfp4_kcache_outlier_k_input_scale(row_amax, threshold, false),
                expected_row_input_scale),
            "NVFP4 outlier K input scale should default to residual row amax reciprocal");

    require(nearly_equal(
                ggml_cuda_nvfp4_kcache_outlier_k_global_scale(row_amax, threshold, true),
                expected_threshold_global_scale),
            "NVFP4 outlier K tensor-scale mode should use threshold as per-tensor amax");
    require(nearly_equal(
                ggml_cuda_nvfp4_kcache_outlier_k_input_scale(row_amax, threshold, true),
                expected_threshold_input_scale),
            "NVFP4 outlier K tensor-scale mode should store reciprocal threshold global scale");
}

static void test_nvfp4_outlier_q_scale_uses_dynamic_tensor_amax() {
    constexpr float amax = 21.0f;
    constexpr float out_scale = 0.5f;
    constexpr float expected_global_scale = 1344.0f / amax;
    constexpr float expected_input_scale = out_scale / expected_global_scale;

    require(nearly_equal(ggml_cuda_nvfp4_kcache_outlier_q_global_scale(amax), expected_global_scale),
            "NVFP4 outlier Q global scale should use dynamic per-tensor amax");
    require(nearly_equal(ggml_cuda_nvfp4_kcache_outlier_q_input_scale(amax, out_scale), expected_input_scale),
            "NVFP4 outlier Q input scale should use one dynamic per-tensor scale");
}

static void test_kcache_outlier_diagnostic_switches_default_off() {
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_DETERMINISTIC_FILL", nullptr);
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_NO_CORRECTION", nullptr);
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_FINGERPRINT", nullptr);
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_OVERFLOW_LOG", nullptr);

    require(!ggml_cuda_nvfp4_kcache_outlier_deterministic_fill_enabled(),
            "deterministic fill diagnostic switch should default off");
    require(!ggml_cuda_nvfp4_kcache_outlier_no_correction_enabled(),
            "no-correction diagnostic switch should default off");
    require(!ggml_cuda_nvfp4_kcache_outlier_fingerprint_enabled(),
            "fingerprint diagnostic switch should default off");
    require(!ggml_cuda_nvfp4_kcache_outlier_overflow_log_enabled(),
            "overflow-log diagnostic switch should default off");
}

static void test_kcache_outlier_diagnostic_switches_parse_env() {
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_DETERMINISTIC_FILL", "1");
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_NO_CORRECTION", "1");
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_FINGERPRINT", "1");
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_OVERFLOW_LOG", "1");

    require(ggml_cuda_nvfp4_kcache_outlier_deterministic_fill_enabled(),
            "deterministic fill diagnostic switch should enable on 1");
    require(ggml_cuda_nvfp4_kcache_outlier_no_correction_enabled(),
            "no-correction diagnostic switch should enable on 1");
    require(ggml_cuda_nvfp4_kcache_outlier_fingerprint_enabled(),
            "fingerprint diagnostic switch should enable on 1");
    require(ggml_cuda_nvfp4_kcache_outlier_overflow_log_enabled(),
            "overflow-log diagnostic switch should enable on 1");

    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_DETERMINISTIC_FILL", "0");
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_NO_CORRECTION", "0");
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_FINGERPRINT", "0");
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_OVERFLOW_LOG", "0");

    require(!ggml_cuda_nvfp4_kcache_outlier_deterministic_fill_enabled(),
            "deterministic fill diagnostic switch should disable on 0");
    require(!ggml_cuda_nvfp4_kcache_outlier_no_correction_enabled(),
            "no-correction diagnostic switch should disable on 0");
    require(!ggml_cuda_nvfp4_kcache_outlier_fingerprint_enabled(),
            "fingerprint diagnostic switch should disable on 0");
    require(!ggml_cuda_nvfp4_kcache_outlier_overflow_log_enabled(),
            "overflow-log diagnostic switch should disable on 0");
}

static void test_apply_correction_filters_head() {
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_len = 2;
    constexpr int64_t q_len = 3;
    constexpr int64_t q_heads = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t compact_capacity = 4;

    std::vector<int32_t> counts = { 3, 1 };
    std::vector<int32_t> offsets = { 0, 3 };
    std::vector<int32_t> indices = { 1, 5, 3, 2 };
    std::vector<float> values = { 2.0f, 100.0f, -1.5f, 4.0f };

    std::vector<float> q((size_t) head_dim * q_len * q_heads, 0.0f);
    for (int64_t qh = 0; qh < q_heads; ++qh) {
        for (int64_t qt = 0; qt < q_len; ++qt) {
            for (int64_t d = 0; d < head_dim; ++d) {
                q[(size_t) qh * q_len * head_dim + (size_t) qt * head_dim + (size_t) d] =
                        0.1f * (float) qh + 1.0f * (float) qt + 0.25f * (float) d;
            }
        }
    }

    std::vector<float> kq((size_t) kv_len * q_len, 1.0f);

    int32_t * counts_d = nullptr;
    int32_t * offsets_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * q_d = nullptr;
    float * kq_d = nullptr;
    CUDA_CHECK(cudaMalloc(&counts_d, counts.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&offsets_d, offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_d, q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&kq_d, kq.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(counts_d, counts.data(), counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(offsets_d, offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indices_d, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(values_d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(q_d, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kq_d, kq.data(), kq.size() * sizeof(float), cudaMemcpyHostToDevice));

    ggml_cuda_nvfp4_kcache_outlier_apply_correction(
            counts_d, offsets_d, indices_d, values_d, q_d + q_len * head_dim, kq_d, nullptr,
            head_dim, kv_len, q_len, q_heads, kv_heads,
            1, compact_capacity,
            1, head_dim, 1, kv_len,
            nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(kq.data(), kq_d, kq.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int64_t qt = 0; qt < q_len; ++qt) {
        const float q1 = q[(size_t) 1 * q_len * head_dim + (size_t) qt * head_dim + 1];
        const float q3 = q[(size_t) 1 * q_len * head_dim + (size_t) qt * head_dim + 3];
        const float expected0 = 1.0f + 2.0f * q1 - 1.5f * q3;
        require(nearly_equal(kq[(size_t) qt * kv_len + 0], expected0), "corrected token 0 logit");

        const float q2 = q[(size_t) 1 * q_len * head_dim + (size_t) qt * head_dim + 2];
        const float expected1 = 1.0f + 4.0f * q2;
        require(nearly_equal(kq[(size_t) qt * kv_len + 1], expected1), "corrected token 1 logit");
    }

    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(q_d));
    CUDA_CHECK(cudaFree(kq_d));
}

static void test_apply_correction_compact_filters_gqa_head() {
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_len = 1;
    constexpr int64_t q_len = 2;
    constexpr int64_t q_heads = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t compact_capacity = 2;

    std::vector<int32_t> counts = { 2 };
    std::vector<int32_t> offsets = { 0 };
    std::vector<int32_t> indices = { 1, (int32_t) head_dim + 2 };
    std::vector<float> values = { 3.0f, 5.0f };
    std::vector<float> q((size_t) head_dim * q_len * q_heads, 0.0f);
    for (int64_t qh = 0; qh < q_heads; ++qh) {
        for (int64_t qt = 0; qt < q_len; ++qt) {
            for (int64_t d = 0; d < head_dim; ++d) {
                q[(size_t) qh * q_len * head_dim + (size_t) qt * head_dim + (size_t) d] =
                        10.0f * (float) qh + 1.0f * (float) qt + 0.25f * (float) d;
            }
        }
    }
    std::vector<float> kq((size_t) kv_len * q_len, 1.0f);

    int32_t * counts_d = nullptr;
    int32_t * offsets_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * q_d = nullptr;
    float * kq_d = nullptr;
    CUDA_CHECK(cudaMalloc(&counts_d, counts.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&offsets_d, offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_d, q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&kq_d, kq.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(counts_d, counts.data(), counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(offsets_d, offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indices_d, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(values_d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(q_d, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kq_d, kq.data(), kq.size() * sizeof(float), cudaMemcpyHostToDevice));

    ggml_cuda_nvfp4_kcache_outlier_apply_correction(
            counts_d, offsets_d, indices_d, values_d, q_d + 2 * q_len * head_dim, kq_d, nullptr,
            head_dim, kv_len, q_len, q_heads, kv_heads,
            2, compact_capacity,
            1, head_dim, 1, kv_len,
            nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(kq.data(), kq_d, kq.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (int64_t qt = 0; qt < q_len; ++qt) {
        const float qv = q[(size_t) 2 * q_len * head_dim + (size_t) qt * head_dim + 2];
        require(nearly_equal(kq[(size_t) qt], 1.0f + 5.0f * qv), "compact correction should filter GQA KV head");
    }

    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(q_d));
    CUDA_CHECK(cudaFree(kq_d));
}

static void test_apply_correction_compensates_for_downstream_k_scale() {
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_len = 1;
    constexpr int64_t q_len = 1;
    constexpr int64_t q_heads = 1;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t compact_capacity = 1;
    constexpr float downstream_k_scale = 0.25f;

    std::vector<int32_t> counts = { 1 };
    std::vector<int32_t> offsets = { 0 };
    std::vector<int32_t> indices = { 2 };
    std::vector<float> values = { 8.0f };
    std::vector<float> q = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> kq = { 2.0f };
    std::vector<float> k_scale = { downstream_k_scale };

    int32_t * counts_d = nullptr;
    int32_t * offsets_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * q_d = nullptr;
    float * kq_d = nullptr;
    float * k_scale_d = nullptr;
    CUDA_CHECK(cudaMalloc(&counts_d, counts.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&offsets_d, offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_d, q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&kq_d, kq.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&k_scale_d, k_scale.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(counts_d, counts.data(), counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(offsets_d, offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indices_d, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(values_d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(q_d, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kq_d, kq.data(), kq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(k_scale_d, k_scale.data(), k_scale.size() * sizeof(float), cudaMemcpyHostToDevice));

    ggml_cuda_nvfp4_kcache_outlier_apply_correction(
            counts_d, offsets_d, indices_d, values_d, q_d, kq_d, k_scale_d,
            head_dim, kv_len, q_len, q_heads, kv_heads,
            0, compact_capacity,
            1, head_dim, 1, kv_len,
            nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(kq.data(), kq_d, kq.size() * sizeof(float), cudaMemcpyDeviceToHost));

    const float final_after_graph_scale = kq[0] * downstream_k_scale;
    const float residual_after_graph_scale = 2.0f * downstream_k_scale;
    const float outlier_dot = 8.0f * 3.0f;
    require(nearly_equal(final_after_graph_scale, residual_after_graph_scale + outlier_dot),
            "correction should survive downstream K scale multiply exactly once");

    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(q_d));
    CUDA_CHECK(cudaFree(kq_d));
    CUDA_CHECK(cudaFree(k_scale_d));
}

static void test_apply_correction_no_correction_switch_leaves_kq_unchanged() {
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_len = 1;
    constexpr int64_t q_len = 1;
    constexpr int64_t q_heads = 1;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t compact_capacity = 1;

    std::vector<int32_t> counts = { 1 };
    std::vector<int32_t> offsets = { 0 };
    std::vector<int32_t> indices = { 2 };
    std::vector<float> values = { 8.0f };
    std::vector<float> q = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> kq = { 2.0f };

    int32_t * counts_d = nullptr;
    int32_t * offsets_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * q_d = nullptr;
    float * kq_d = nullptr;
    CUDA_CHECK(cudaMalloc(&counts_d, counts.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&offsets_d, offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_d, q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&kq_d, kq.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(counts_d, counts.data(), counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(offsets_d, offsets.data(), offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indices_d, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(values_d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(q_d, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kq_d, kq.data(), kq.size() * sizeof(float), cudaMemcpyHostToDevice));

    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_NO_CORRECTION", "1");
    ggml_cuda_nvfp4_kcache_outlier_apply_correction(
            counts_d, offsets_d, indices_d, values_d, q_d, kq_d, nullptr,
            head_dim, kv_len, q_len, q_heads, kv_heads,
            0, compact_capacity,
            1, head_dim, 1, kv_len,
            nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_NO_CORRECTION", nullptr);

    CUDA_CHECK(cudaMemcpy(kq.data(), kq_d, kq.size() * sizeof(float), cudaMemcpyDeviceToHost));
    require(nearly_equal(kq[0], 2.0f), "no-correction switch should leave KQ unchanged");

    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(q_d));
    CUDA_CHECK(cudaFree(kq_d));
}

int main() {
    if (!cuda_available()) {
        std::printf("SKIP (no CUDA device)\n");
        return 0;
    }

    test_extract_compact_offsets_and_pool();
    test_extract_compact_requires_full_row_capacity();
    test_extract_compact_resets_counts_after_assigning_offsets();
    test_extract_preserves_rows_across_multiple_set_rows_calls();
    test_extract_split_matches_full_for_same_rows();
    test_nvfp4_outlier_k_scale_mode_selects_row_or_threshold_amax();
    test_nvfp4_outlier_q_scale_uses_dynamic_tensor_amax();
    test_kcache_outlier_diagnostic_switches_default_off();
    test_kcache_outlier_diagnostic_switches_parse_env();
    test_apply_correction_filters_head();
    test_apply_correction_compact_filters_gqa_head();
    test_apply_correction_compensates_for_downstream_k_scale();
    test_apply_correction_no_correction_switch_leaves_kq_unchanged();

    std::printf("OK\n");
    return 0;
}
