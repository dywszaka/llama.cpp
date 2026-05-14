#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

bool ggml_cuda_is_experimental_nvfp4_vcache_set_rows_node(const ggml_tensor * dst);
void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
