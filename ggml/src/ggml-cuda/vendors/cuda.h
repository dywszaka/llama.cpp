#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cublasLt.h>
#define GGML_CUDA_HAS_CUBLASLT 1
#else
#define GGML_CUDA_HAS_CUBLASLT 0
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if CUDART_VERSION >= 12050
#include <cuda_fp8.h>
#endif // CUDART_VERSION >= 12050

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
#if defined(__has_include)
#if __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
#define GGML_CUDA_HAS_FP4 1
#endif // __has_include(<cuda_fp4.h>)
#endif // defined(__has_include)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080

#ifndef GGML_CUDA_HAS_FP4
#define GGML_CUDA_HAS_FP4 0
#endif

#if CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
