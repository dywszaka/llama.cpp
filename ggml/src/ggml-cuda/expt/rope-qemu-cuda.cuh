#pragma once

#include "rope-qemu.cuh"

void ggml_cuda_rope_qemu_cuda_preprocess(
        const ggml_cuda_rope_qemu_params & params,
        uint16_t * src0_bf16,
        cudaStream_t stream);

void ggml_cuda_rope_qemu_cuda_run_bf16(
        const ggml_cuda_rope_qemu_params & params,
        const uint16_t * src0_bf16,
        uint16_t * dst_bf16,
        cudaStream_t stream);

void ggml_cuda_rope_qemu_cuda_output(
        const ggml_cuda_rope_qemu_params & params,
        const uint16_t * output_bf16,
        cudaStream_t stream);
