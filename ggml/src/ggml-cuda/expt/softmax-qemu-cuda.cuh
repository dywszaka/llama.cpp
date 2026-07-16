#pragma once

#include "softmax-qemu.cuh"

void ggml_cuda_soft_max_qemu_cuda_preprocess(
        const ggml_cuda_soft_max_qemu_params & params,
        uint16_t * input_bf16,
        uint16_t * sinks_bf16,
        cudaStream_t stream);

void ggml_cuda_soft_max_qemu_cuda_run_preprocessed(
        const ggml_cuda_soft_max_qemu_params & params,
        const uint16_t * input_bf16,
        const uint16_t * sinks_bf16,
        uint16_t * output_bf16,
        uint32_t * exponent_values,
        float * output_f32,
        cudaStream_t stream);

void ggml_cuda_soft_max_qemu_cuda_output_to_f32(
        const uint16_t * input_bf16,
        float * output_f32,
        size_t elements,
        cudaStream_t stream);

