#include <ggml.h>

#include "../ggml/src/ggml-quants.h"

#include <cuda_runtime.h>
#include <cublas_api.h>
#include <cublasLt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do {                                                                  \
    cudaError_t err__ = (call);                                                                \
    if (err__ != cudaSuccess) {                                                                \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1);                                                                          \
    }                                                                                          \
} while (0)

#define CUBLASLT_CHECK(call) do {                                                              \
    cublasStatus_t st__ = (call);                                                              \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                                       \
        std::fprintf(stderr, "cuBLASLt error %s:%d: status=%d\n", __FILE__, __LINE__, (int) st__); \
        std::exit(1);                                                                          \
    }                                                                                          \
} while (0)

static inline int64_t pad_i64(int64_t x, int64_t a) {
    return ((x + a - 1) / a) * a;
}

static inline int64_t scale_tiled_index(int64_t outer, int64_t inner, int64_t n_inner_padded) {
    const int64_t outer_tile = outer / 128;
    const int64_t outer_in_tile = outer % 128;
    const int64_t inner_tile = inner / 4;
    const int64_t inner_in_tile = inner % 4;

    const int64_t tiles_per_outer_block = n_inner_padded / 4;
    const int64_t tile_base = (outer_tile * tiles_per_outer_block + inner_tile) * 512;
    const int64_t tile_offset = (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile;
    return tile_base + tile_offset;
}

static void quantize_matrix_nvfp4(
        const std::vector<float> & src,
        std::vector<block_nvfp4> & dst,
        int rows,
        int k,
        float global_scale) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nblk_k = k / QK_NVFP4;
    dst.resize((size_t) rows * (size_t) nblk_k);

    for (int r = 0; r < rows; ++r) {
        quantize_row_nvfp4_ref(
                src.data() + (size_t) r * (size_t) k,
                dst.data() + (size_t) r * (size_t) nblk_k,
                k,
                global_scale);
    }
}

static void dequantize_matrix_nvfp4(
        const std::vector<block_nvfp4> & src,
        std::vector<float> & dst,
        int rows,
        int k,
        float global_scale) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nblk_k = k / QK_NVFP4;
    dst.resize((size_t) rows * (size_t) k);

    for (int r = 0; r < rows; ++r) {
        dequantize_row_nvfp4(
                src.data() + (size_t) r * (size_t) nblk_k,
                dst.data() + (size_t) r * (size_t) k,
                k,
                global_scale);
    }
}

static void fp32_reference_matmul(
        const std::vector<float> & a_deq,
        const std::vector<float> & b_deq,
        std::vector<float> & c_ref,
        int m,
        int n,
        int k) {
    c_ref.assign((size_t) m * (size_t) n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int x = 0; x < k; ++x) {
                acc += a_deq[(size_t) i * (size_t) k + (size_t) x] *
                       b_deq[(size_t) j * (size_t) k + (size_t) x];
            }
            c_ref[(size_t) i * (size_t) n + (size_t) j] = acc;
        }
    }
}

static void split_nvfp4_blocks(
        const std::vector<block_nvfp4> & src,
        int64_t k,
        int64_t n_outer_valid,
        int64_t n_outer_alloc,
        bool linear_scale_layout,
        std::vector<uint8_t> & out_data,
        std::vector<uint8_t> & out_scale,
        int64_t & out_scale_inner_padded,
        int64_t & out_scale_outer_padded) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int64_t nblk_k = k / QK_NVFP4;
    const int64_t row_data_bytes = k / 2;
    const int64_t inner_padded = pad_i64(nblk_k, 4);
    const int64_t outer_padded = pad_i64(n_outer_alloc, 128);

    out_data.assign((size_t) n_outer_alloc * (size_t) row_data_bytes, 0);
    out_scale.assign((size_t) inner_padded * (size_t) outer_padded, 0);

    for (int64_t outer = 0; outer < n_outer_valid; ++outer) {
        for (int64_t inner = 0; inner < nblk_k; ++inner) {
            const block_nvfp4 & b = src[(size_t) outer * (size_t) nblk_k + (size_t) inner];

            uint8_t * data_dst = out_data.data() + (size_t) outer * (size_t) row_data_bytes + (size_t) inner * (QK_NVFP4 / 2);
            std::memcpy(data_dst, b.qs, QK_NVFP4 / 2);

            const int64_t sidx = linear_scale_layout
                    ? (outer * inner_padded + inner)
                    : scale_tiled_index(outer, inner, inner_padded);
            out_scale[(size_t) sidx] = b.e;
        }
    }

    out_scale_inner_padded = inner_padded;
    out_scale_outer_padded = outer_padded;
}

static bool run_case(int m, int n, int k, float global_scale_a, float global_scale_b, uint32_t seed) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((n % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a_fp32((size_t) m * (size_t) k);
    std::vector<float> b_fp32((size_t) n * (size_t) k);
    for (float & v : a_fp32) {
        v = dist(rng);
    }
    for (float & v : b_fp32) {
        v = dist(rng);
    }

    std::vector<block_nvfp4> a_nvfp4;
    std::vector<block_nvfp4> b_nvfp4;
    quantize_matrix_nvfp4(a_fp32, a_nvfp4, m, k, global_scale_a);
    quantize_matrix_nvfp4(b_fp32, b_nvfp4, n, k, global_scale_b);

    std::vector<float> a_deq;
    std::vector<float> b_deq;
    dequantize_matrix_nvfp4(a_nvfp4, a_deq, m, k, global_scale_a);
    dequantize_matrix_nvfp4(b_nvfp4, b_deq, n, k, global_scale_b);

    std::vector<float> c_ref;
    fp32_reference_matmul(a_deq, b_deq, c_ref, m, n, k);

    std::vector<uint8_t> a_data;
    std::vector<uint8_t> a_scale;
    std::vector<uint8_t> b_data;
    std::vector<uint8_t> b_scale;
    int64_t a_scale_inner = 0;
    int64_t a_scale_outer = 0;
    int64_t b_scale_inner = 0;
    int64_t b_scale_outer = 0;

    const bool linear_scale_layout = false;
    split_nvfp4_blocks(a_nvfp4, k, m, m, linear_scale_layout, a_data, a_scale, a_scale_inner, a_scale_outer);
    split_nvfp4_blocks(b_nvfp4, k, n, n, linear_scale_layout, b_data, b_scale, b_scale_inner, b_scale_outer);

    uint8_t * d_a_data = nullptr;
    uint8_t * d_b_data = nullptr;
    uint8_t * d_a_scale = nullptr;
    uint8_t * d_b_scale = nullptr;
    float * d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_data, a_data.size()));
    CUDA_CHECK(cudaMalloc(&d_b_data, b_data.size()));
    CUDA_CHECK(cudaMalloc(&d_a_scale, a_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_b_scale, b_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_c, (size_t) m * (size_t) n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a_data, a_data.data(), a_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_data, b_data.data(), b_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_scale, a_scale.data(), a_scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_scale, b_scale.data(), b_scale.size(), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt = nullptr;
    CUBLASLT_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_n = CUBLAS_OP_N;
    const cublasOperation_t op_t = CUBLAS_OP_T;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_t, sizeof(op_t)));

#if defined(CUBLAS_VER_MAJOR) && (CUBLAS_VER_MAJOR >= 13)
    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    const void * a_scale_ptr = d_a_scale;
    const void * b_scale_ptr = d_b_scale;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));
#else
    std::fprintf(stderr, "Skip: cuBLASLt FP4 scale-channel attributes are unavailable in this toolkit.\n");
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));
    return true;
#endif

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) m, (uint64_t) k, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) n, (uint64_t) k, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F,     (uint64_t) m, (uint64_t) n, (int64_t) n));

    const cublasLtOrder_t order_row = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));

    const float alpha = 1.0f / (global_scale_a * global_scale_b);
    const float beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(
            lt,
            op_desc,
            &alpha,
            d_a_data, a_desc,
            d_b_data, b_desc,
            &beta,
            d_c, c_desc,
            d_c, c_desc,
            nullptr,
            nullptr, 0,
            nullptr));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> c_gpu((size_t) m * (size_t) n, 0.0f);
    CUDA_CHECK(cudaMemcpy(c_gpu.data(), d_c, (size_t) m * (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));

    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_i = 0;
    int worst_j = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const float ref = c_ref[(size_t) i * (size_t) n + (size_t) j];
            const float got = c_gpu[(size_t) i * (size_t) n + (size_t) j];
            const float abs_err = std::fabs(got - ref);
            const float rel_err = abs_err / (std::fabs(ref) + 1e-6f);
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                worst_i = i;
                worst_j = j;
            }
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
            }
        }
    }

    const float tol_abs = 1e-2f;
    const float tol_rel = 1e-2f;
    const bool ok = max_abs_err <= tol_abs || max_rel_err <= tol_rel;

    std::printf("case m=%d n=%d k=%d gs_a=%.3f gs_b=%.3f | max_abs=%.6g max_rel=%.6g | %s\n",
            m, n, k, global_scale_a, global_scale_b, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  worst element: (%d, %d), ref=%.8f, gpu=%.8f\n",
                worst_i,
                worst_j,
                c_ref[(size_t) worst_i * (size_t) n + (size_t) worst_j],
                c_gpu[(size_t) worst_i * (size_t) n + (size_t) worst_j]);
    }

    return ok;
}

// Reproduces the descriptor/layout strategy currently used by ggml_cuda_mul_mat_nvfp4_native:
// - column-major default layouts (no ORDER_ROW)
// - TRANSA=T, TRANSB=N
// - optional N padding to multiple-of-16
static bool run_case_integration_style(int m, int n, int k, float global_scale_a, float global_scale_b, uint32_t seed) {
    GGML_ASSERT((m % 16) == 0);
    GGML_ASSERT((k % 16) == 0);
    GGML_ASSERT((k % QK_NVFP4) == 0);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> a_fp32((size_t) m * (size_t) k);
    std::vector<float> b_fp32((size_t) n * (size_t) k);
    for (float & v : a_fp32) {
        v = dist(rng);
    }
    for (float & v : b_fp32) {
        v = dist(rng);
    }

    std::vector<block_nvfp4> a_nvfp4;
    std::vector<block_nvfp4> b_nvfp4;
    quantize_matrix_nvfp4(a_fp32, a_nvfp4, m, k, global_scale_a);
    quantize_matrix_nvfp4(b_fp32, b_nvfp4, n, k, global_scale_b);

    std::vector<float> a_deq;
    std::vector<float> b_deq;
    dequantize_matrix_nvfp4(a_nvfp4, a_deq, m, k, global_scale_a);
    dequantize_matrix_nvfp4(b_nvfp4, b_deq, n, k, global_scale_b);

    std::vector<float> c_ref;
    fp32_reference_matmul(a_deq, b_deq, c_ref, m, n, k);

    const int n_padded = (int) pad_i64(n, 16);

    std::vector<uint8_t> a_data;
    std::vector<uint8_t> a_scale;
    std::vector<uint8_t> b_data;
    std::vector<uint8_t> b_scale;
    int64_t a_scale_inner = 0;
    int64_t a_scale_outer = 0;
    int64_t b_scale_inner = 0;
    int64_t b_scale_outer = 0;

    const bool linear_scale_layout = false;
    split_nvfp4_blocks(a_nvfp4, k, m, m, linear_scale_layout, a_data, a_scale, a_scale_inner, a_scale_outer);
    split_nvfp4_blocks(b_nvfp4, k, n, n_padded, linear_scale_layout, b_data, b_scale, b_scale_inner, b_scale_outer);

    uint8_t * d_a_data = nullptr;
    uint8_t * d_b_data = nullptr;
    uint8_t * d_a_scale = nullptr;
    uint8_t * d_b_scale = nullptr;
    float * d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a_data, a_data.size()));
    CUDA_CHECK(cudaMalloc(&d_b_data, b_data.size()));
    CUDA_CHECK(cudaMalloc(&d_a_scale, a_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_b_scale, b_scale.size()));
    CUDA_CHECK(cudaMalloc(&d_c, (size_t) m * (size_t) n_padded * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a_data, a_data.data(), a_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_data, b_data.data(), b_data.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_a_scale, a_scale.data(), a_scale.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_scale, b_scale.data(), b_scale.size(), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt = nullptr;
    CUBLASLT_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

#if defined(CUBLAS_VER_MAJOR) && (CUBLAS_VER_MAJOR >= 13)
    const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    const void * a_scale_ptr = d_a_scale;
    const void * b_scale_ptr = d_b_scale;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr)));
#else
    std::fprintf(stderr, "Skip: cuBLASLt FP4 scale-channel attributes are unavailable in this toolkit.\n");
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));
    return true;
#endif

    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) k, (uint64_t) m, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) k, (uint64_t) n_padded, (int64_t) k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F,     (uint64_t) m, (uint64_t) n_padded, (int64_t) m));

    const float alpha = 1.0f / (global_scale_a * global_scale_b);
    const float beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(
            lt,
            op_desc,
            &alpha,
            d_a_data, a_desc,
            d_b_data, b_desc,
            &beta,
            d_c, c_desc,
            d_c, c_desc,
            nullptr,
            nullptr, 0,
            nullptr));

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> c_gpu_padded((size_t) m * (size_t) n_padded, 0.0f);
    CUDA_CHECK(cudaMemcpy(
            c_gpu_padded.data(),
            d_c,
            (size_t) m * (size_t) n_padded * sizeof(float),
            cudaMemcpyDeviceToHost));

    // c_desc uses default column-major layout with ld=m. Convert to row-major for comparison.
    std::vector<float> c_gpu((size_t) m * (size_t) n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            c_gpu[(size_t) i * (size_t) n + (size_t) j] =
                    c_gpu_padded[(size_t) j * (size_t) m + (size_t) i];
        }
    }
    std::printf("  integration-style probe e00: ref=%.8f raw0=%.8f\n", c_ref[0], c_gpu_padded[0]);

    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLASLT_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b_scale));
    CUDA_CHECK(cudaFree(d_a_scale));
    CUDA_CHECK(cudaFree(d_b_data));
    CUDA_CHECK(cudaFree(d_a_data));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int worst_i = 0;
    int worst_j = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const float ref = c_ref[(size_t) i * (size_t) n + (size_t) j];
            const float got = c_gpu[(size_t) i * (size_t) n + (size_t) j];
            const float abs_err = std::fabs(got - ref);
            const float rel_err = abs_err / (std::fabs(ref) + 1e-6f);
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                worst_i = i;
                worst_j = j;
            }
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
            }
        }
    }

    const float tol_abs = 1e-2f;
    const float tol_rel = 1e-2f;
    const bool ok = max_abs_err <= tol_abs || max_rel_err <= tol_rel;

    std::printf("integration-style case m=%d n=%d(kpad=%d) k=%d gs_a=%.3f gs_b=%.3f | max_abs=%.6g max_rel=%.6g | %s\n",
            m, n, n_padded, k, global_scale_a, global_scale_b, max_abs_err, max_rel_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        std::printf("  worst element: (%d, %d), ref=%.8f, gpu=%.8f\n",
                worst_i,
                worst_j,
                c_ref[(size_t) worst_i * (size_t) n + (size_t) worst_j],
                c_gpu[(size_t) worst_i * (size_t) n + (size_t) worst_j]);
    }

    return ok;
}

int main() {
    int dev_count = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count <= 0) {
        std::printf("test-nvfp4-matmul: SKIP (no CUDA device)\n");
        return 0;
    }

    CUDA_CHECK(cudaSetDevice(0));

    bool ok = true;
    ok = run_case(64, 64, 128, 1.00f, 1.00f, 1u) && ok;
    ok = run_case(48, 80, 256, 0.75f, 1.25f, 2u) && ok;
    ok = run_case(96, 96, 192, 1.50f, 0.90f, 3u) && ok;
    ok = run_case(256, 256, 128, 1.00f, 1.00f, 4u) && ok;
    ok = run_case(256, 64, 256, 1.00f, 1.00f, 5u) && ok;

    // Mirror current ggml native descriptor path to detect integration mismatch.
    ok = run_case_integration_style(64, 64, 128, 1.00f, 1.00f, 11u) && ok;
    ok = run_case_integration_style(64, 9, 128, 1.00f, 1.00f, 12u) && ok;
    ok = run_case_integration_style(96, 5, 192, 1.50f, 0.90f, 13u) && ok;

    if (!ok) {
        std::fprintf(stderr, "test-nvfp4-matmul: FAILED\n");
        return 1;
    }

    std::printf("test-nvfp4-matmul: all cases passed\n");
    return 0;
}
