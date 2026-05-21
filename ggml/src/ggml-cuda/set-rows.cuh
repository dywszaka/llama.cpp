#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

struct ggml_cuda_set_rows_params {
    const float * src0_d;
    const int64_t * src1_d;

    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t ne10;
    int64_t ne11;
    int64_t ne12;
    int64_t ne13;

    size_t nb01;
    size_t nb02;
    size_t nb03;
    size_t nb10;
    size_t nb11;
    size_t nb12;
    size_t nb1;
    size_t nb2;
    size_t nb3;

    cudaStream_t stream;
};

bool ggml_cuda_is_experimental_nvfp4_vcache_set_rows_node(const ggml_tensor * dst);
void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
