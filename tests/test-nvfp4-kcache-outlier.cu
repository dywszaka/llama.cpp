#include <ggml.h>
#include <ggml-cuda.h>

#include "../ggml/src/ggml-cuda/expt/nvfp4/kcache-outlier.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        cudaGetLastError();
        return false;
    }
    return true;
}

static void test_extract_counts_positions_and_residual_amax() {
    constexpr int64_t k = 16;
    constexpr int64_t rows = 2;
    constexpr int64_t max_outliers = 2;
    constexpr float threshold = 16.0f;

    std::vector<float> src((size_t) rows * (size_t) k, 0.0f);
    for (int64_t i = 0; i < rows * k; ++i) {
        src[(size_t) i] = 0.25f * (float) (i % k);
    }
    src[3] = 17.5f;
    src[7] = -19.0f;
    src[11] = 25.0f; // counted but not stored because max_outliers=2
    src[(size_t) k + 4] = -16.25f;
    src[(size_t) k + 9] = 15.99f;

    const int64_t idx_h[2] = { 5, 2 };

    float * src_d = nullptr;
    int64_t * idx_d = nullptr;
    int32_t * counts_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * amax_d = nullptr;

    CUDA_CHECK(cudaMalloc(&src_d, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&idx_d, sizeof(idx_h)));
    CUDA_CHECK(cudaMalloc(&counts_d, 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, (size_t) max_outliers * 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, (size_t) max_outliers * 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&amax_d, (size_t) rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_d, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(idx_d, idx_h, sizeof(idx_h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(counts_d, 0x7f, 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(indices_d, 0, (size_t) max_outliers * 8 * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(values_d, 0, (size_t) max_outliers * 8 * sizeof(float)));

    ggml_cuda_nvfp4_kcache_outlier_extract(
            src_d, idx_d, counts_d, nullptr, nullptr, indices_d, values_d, amax_d,
            k, rows, k, 1, 8, max_outliers, max_outliers, threshold, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> counts(8);
    std::vector<int32_t> indices((size_t) max_outliers * 8);
    std::vector<float> values((size_t) max_outliers * 8);
    std::vector<float> amax((size_t) rows);
    CUDA_CHECK(cudaMemcpy(counts.data(), counts_d, counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices.data(), indices_d, indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values.data(), values_d, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(amax.data(), amax_d, amax.size() * sizeof(float), cudaMemcpyDeviceToHost));

    require(counts[5] == 3, "row 0 true outlier count should include overflow");
    require(indices[5 * max_outliers + 0] == 3, "row 0 first outlier index");
    require(indices[5 * max_outliers + 1] == 7, "row 0 second outlier index");
    require(nearly_equal(values[5 * max_outliers + 0], 17.5f), "row 0 first outlier value");
    require(nearly_equal(values[5 * max_outliers + 1], -19.0f), "row 0 second outlier value");
    require(nearly_equal(amax[0], 3.75f), "row 0 residual amax should ignore outliers");

    require(counts[2] == 1, "row 1 outlier count");
    require(indices[2 * max_outliers + 0] == 4, "row 1 first outlier index");
    require(nearly_equal(values[2 * max_outliers + 0], -16.25f), "row 1 first outlier value");
    require(nearly_equal(amax[1], 15.99f), "row 1 residual amax includes non-outlier threshold-near value");

    CUDA_CHECK(cudaFree(src_d));
    CUDA_CHECK(cudaFree(idx_d));
    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(amax_d));
}

static void test_extract_compact_offsets_and_pool() {
    constexpr int64_t k = 16;
    constexpr int64_t rows = 2;
    constexpr int64_t sidecar_rows = 8;
    constexpr int64_t compact_capacity = 8;
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
            src_d, idx_d, counts_d, offsets_d, cursor_d, indices_d, values_d, amax_d,
            k, rows, k, 1, sidecar_rows, compact_capacity, compact_capacity, threshold, nullptr);
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

static void test_apply_correction_filters_head() {
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_len = 2;
    constexpr int64_t q_len = 3;
    constexpr int64_t q_heads = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t max_outliers = 3;

    std::vector<int32_t> counts = { 3, 1 };
    std::vector<int32_t> indices((size_t) max_outliers * kv_len, 0);
    std::vector<float> values((size_t) max_outliers * kv_len, 0.0f);
    indices[0 * max_outliers + 0] = 1;
    values [0 * max_outliers + 0] = 2.0f;
    indices[0 * max_outliers + 1] = 5;
    values [0 * max_outliers + 1] = 100.0f; // other KV head for qh=1, ignored
    indices[0 * max_outliers + 2] = 3;
    values [0 * max_outliers + 2] = -1.5f;
    indices[1 * max_outliers + 0] = 2;
    values [1 * max_outliers + 0] = 4.0f;

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
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * q_d = nullptr;
    float * kq_d = nullptr;
    CUDA_CHECK(cudaMalloc(&counts_d, counts.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_d, q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&kq_d, kq.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(counts_d, counts.data(), counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indices_d, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(values_d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(q_d, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kq_d, kq.data(), kq.size() * sizeof(float), cudaMemcpyHostToDevice));

    ggml_cuda_nvfp4_kcache_outlier_apply_correction(
            counts_d, nullptr, indices_d, values_d, q_d + q_len * head_dim, kq_d, nullptr,
            head_dim, kv_len, q_len, q_heads, kv_heads,
            1, max_outliers, max_outliers,
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
    constexpr int64_t max_outliers = 1;
    constexpr float downstream_k_scale = 0.25f;

    std::vector<int32_t> counts = { 1 };
    std::vector<int32_t> indices = { 2 };
    std::vector<float> values = { 8.0f };
    std::vector<float> q = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> kq = { 2.0f };
    std::vector<float> k_scale = { downstream_k_scale };

    int32_t * counts_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;
    float * q_d = nullptr;
    float * kq_d = nullptr;
    float * k_scale_d = nullptr;
    CUDA_CHECK(cudaMalloc(&counts_d, counts.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, indices.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, values.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q_d, q.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&kq_d, kq.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&k_scale_d, k_scale.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(counts_d, counts.data(), counts.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indices_d, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(values_d, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(q_d, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kq_d, kq.data(), kq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(k_scale_d, k_scale.data(), k_scale.size() * sizeof(float), cudaMemcpyHostToDevice));

    ggml_cuda_nvfp4_kcache_outlier_apply_correction(
            counts_d, nullptr, indices_d, values_d, q_d, kq_d, k_scale_d,
            head_dim, kv_len, q_len, q_heads, kv_heads,
            0, max_outliers, max_outliers,
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
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
    CUDA_CHECK(cudaFree(q_d));
    CUDA_CHECK(cudaFree(kq_d));
    CUDA_CHECK(cudaFree(k_scale_d));
}

static void test_f16_set_rows_extracts_outliers_and_writes_residual() {
    constexpr int64_t k = 8;
    constexpr int64_t rows = 1;
    constexpr int64_t dst_rows = 4;
    constexpr int64_t max_outliers = 3;
    constexpr float threshold = 16.0f;

    std::vector<float> src = {
        1.0f, 17.0f, -2.5f, -18.0f, 3.0f, 15.0f, 22.0f, -4.0f,
    };
    const int64_t idx_h[1] = { 2 };

    float * src_d = nullptr;
    int64_t * idx_d = nullptr;
    __half * dst_d = nullptr;
    int32_t * counts_d = nullptr;
    int32_t * indices_d = nullptr;
    float * values_d = nullptr;

    CUDA_CHECK(cudaMalloc(&src_d, src.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&idx_d, sizeof(idx_h)));
    CUDA_CHECK(cudaMalloc(&dst_d, (size_t) dst_rows * k * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&counts_d, (size_t) dst_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&indices_d, (size_t) max_outliers * dst_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&values_d, (size_t) max_outliers * dst_rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(src_d, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(idx_d, idx_h, sizeof(idx_h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dst_d, 0, (size_t) dst_rows * k * sizeof(__half)));
    CUDA_CHECK(cudaMemset(counts_d, 0, (size_t) dst_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(indices_d, 0, (size_t) max_outliers * dst_rows * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(values_d, 0, (size_t) max_outliers * dst_rows * sizeof(float)));

    ggml_cuda_f16_kcache_outlier_set_rows(
            src_d, idx_d, dst_d, counts_d, nullptr, nullptr, indices_d, values_d,
            k, rows, k, 1, k, dst_rows, max_outliers, max_outliers, threshold, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> dst((size_t) dst_rows * k);
    std::vector<int32_t> counts((size_t) dst_rows);
    std::vector<int32_t> indices((size_t) max_outliers * dst_rows);
    std::vector<float> values((size_t) max_outliers * dst_rows);
    CUDA_CHECK(cudaMemcpy(dst.data(), dst_d, dst.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts.data(), counts_d, counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices.data(), indices_d, indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values.data(), values_d, values.size() * sizeof(float), cudaMemcpyDeviceToHost));

    require(counts[2] == 3, "F16 row should record all outliers");
    require(indices[2 * max_outliers + 0] == 1, "F16 first outlier index");
    require(indices[2 * max_outliers + 1] == 3, "F16 second outlier index");
    require(indices[2 * max_outliers + 2] == 6, "F16 third outlier index");
    require(nearly_equal(values[2 * max_outliers + 0], 17.0f), "F16 first outlier value");
    require(nearly_equal(values[2 * max_outliers + 1], -18.0f), "F16 second outlier value");
    require(nearly_equal(values[2 * max_outliers + 2], 22.0f), "F16 third outlier value");

    const size_t base = 2 * k;
    require(nearly_equal(__half2float(dst[base + 0]), 1.0f), "F16 residual keeps non-outlier 0");
    require(nearly_equal(__half2float(dst[base + 1]), 0.0f), "F16 residual zeros outlier 1");
    require(nearly_equal(__half2float(dst[base + 2]), -2.5f), "F16 residual keeps non-outlier 2");
    require(nearly_equal(__half2float(dst[base + 3]), 0.0f), "F16 residual zeros outlier 3");
    require(nearly_equal(__half2float(dst[base + 6]), 0.0f), "F16 residual zeros outlier 6");

    CUDA_CHECK(cudaFree(src_d));
    CUDA_CHECK(cudaFree(idx_d));
    CUDA_CHECK(cudaFree(dst_d));
    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(indices_d));
    CUDA_CHECK(cudaFree(values_d));
}

int main() {
    if (!cuda_available()) {
        std::printf("SKIP (no CUDA device)\n");
        return 0;
    }

    test_extract_counts_positions_and_residual_amax();
    test_extract_compact_offsets_and_pool();
    test_nvfp4_outlier_k_scale_mode_selects_row_or_threshold_amax();
    test_nvfp4_outlier_q_scale_uses_dynamic_tensor_amax();
    test_apply_correction_filters_head();
    test_apply_correction_compensates_for_downstream_k_scale();
    test_f16_set_rows_extracts_outliers_and_writes_residual();

    std::printf("OK\n");
    return 0;
}
