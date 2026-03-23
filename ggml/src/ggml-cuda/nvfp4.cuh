#pragma once

#include "common.cuh"

#include <cstddef>
#include <cstdint>
#include <unordered_map>

struct ggml_cuda_nvfp4_weight_cache {
    int device = -1;

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t rows_pad = 0;
    int64_t cols_pad = 0;
    int64_t blocks_per_row = 0;
    int64_t blocks_per_row_pad = 0;

    void * qdata = nullptr;
    void * scales = nullptr;
    const void * src0_data = nullptr;

    size_t qdata_size = 0;
    size_t scales_size = 0;
};

struct ggml_cuda_nvfp4_plan {
    int64_t rows_pad = 0;
    int64_t cols_pad = 0;
    int64_t k_pad = 0;

    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulAlgo_t   algo {};
    size_t                 workspace_size = 0;
};

static inline int64_t ggml_cuda_nvfp4_pad_rows(int64_t rows) {
    return GGML_PAD(rows, 128);
}

static inline int64_t ggml_cuda_nvfp4_pad_blocks(int64_t blocks) {
    return GGML_PAD(blocks, 4);
}

static inline __host__ __device__ size_t ggml_cuda_nvfp4_scale_offset(int64_t row, int64_t block, int64_t blocks_per_row_pad) {
    return (size_t) row * (size_t) blocks_per_row_pad + (size_t) block;
}

static inline uint64_t ggml_cuda_nvfp4_plan_key(int64_t rows_pad, int64_t cols_pad, int64_t k_pad) {
    return ((uint64_t)rows_pad << 42) ^ ((uint64_t)cols_pad << 21) ^ (uint64_t)k_pad;
}

bool ggml_cuda_should_use_nvfp4(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_nvfp4_destroy_weight_cache(ggml_cuda_nvfp4_weight_cache * cache);
void ggml_cuda_nvfp4_destroy_plan(ggml_cuda_nvfp4_plan * plan);

void ggml_cuda_op_mul_mat_cublaslt_nvfp4(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
