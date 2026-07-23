#include "rms-norm-qemu-cuda.cuh"
#include "rms-norm-bf16-core.cuh"

#include <algorithm>

static __global__ void rms_norm_qemu_preprocess_kernel(
        const float * input,
        uint16_t * output,
        ggml_cuda_rms_norm_qemu_params params,
        size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        const int64_t col = (int64_t) (index % (size_t) params.ncols);
        const int64_t packed_row = (int64_t) (index / (size_t) params.ncols);
        const int64_t row = packed_row % params.nrows;
        const int64_t packed_channel = packed_row / params.nrows;
        const int64_t channel = packed_channel % params.nchannels;
        const int64_t sample = packed_channel / params.nchannels;
        const int64_t source_index = sample * params.stride_sample +
                channel * params.stride_channel + row * params.stride_row + col;
        output[index] = rms_norm_bf16_from_f32_rz(input[source_index]);
    }
}

static __global__ void rms_norm_qemu_bf16_to_f32_kernel(
        const uint16_t * input,
        float * output,
        size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        output[index] = rms_norm_bf16_to_f32(input[index]);
    }
}

static int rms_norm_qemu_cuda_blocks(size_t elements) {
    constexpr size_t threads = 256;
    return (int) std::min<size_t>((elements + threads - 1) / threads, 65535);
}

void ggml_cuda_rms_norm_qemu_cuda_preprocess(
        const ggml_cuda_rms_norm_qemu_params & params,
        uint16_t * input_bf16,
        cudaStream_t stream) {
    const size_t elements = (size_t) params.ncols * (size_t) params.nrows *
            (size_t) params.nchannels * (size_t) params.nsamples;
    if (elements == 0) {
        return;
    }
    constexpr int threads = 256;
    rms_norm_qemu_preprocess_kernel<<<rms_norm_qemu_cuda_blocks(elements), threads, 0, stream>>>(
            params.src0, input_bf16, params, elements);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_rms_norm_qemu_cuda_run_bf16(
        const ggml_cuda_rms_norm_qemu_params & params,
        const uint16_t * input_bf16,
        uint16_t * output_bf16,
        cudaStream_t stream) {
    const size_t rows = (size_t) params.nrows * (size_t) params.nchannels *
            (size_t) params.nsamples;
    if (rows == 0) {
        return;
    }
    GGML_ASSERT(rows <= UINT32_MAX);
    rms_norm_bf16_bitexact_kernel<<<(unsigned int) rows, 32, 0, stream>>>(
            input_bf16, output_bf16, params.ncols, params.eps);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_rms_norm_qemu_cuda_output_to_f32(
        const uint16_t * input_bf16,
        float * output_f32,
        size_t elements,
        cudaStream_t stream) {
    if (elements == 0) {
        return;
    }
    constexpr int threads = 256;
    rms_norm_qemu_bf16_to_f32_kernel<<<rms_norm_qemu_cuda_blocks(elements), threads, 0, stream>>>(
            input_bf16, output_f32, elements);
    CUDA_CHECK(cudaGetLastError());
}
