#include "rms-norm-qemu-cuda.cuh"

#include <algorithm>

static __device__ __forceinline__ uint16_t rms_norm_qemu_f32_to_bf16_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t exponent = bits & UINT32_C(0x7f800000);
    const uint32_t mantissa = bits & UINT32_C(0x007fffff);
    if (exponent == UINT32_C(0x7f800000) && mantissa != 0) {
        uint16_t result = (uint16_t) (bits >> 16);
        result |= UINT16_C(0x0040);
        return result;
    }

    const uint32_t upper = bits >> 16;
    const uint32_t lower = bits & UINT32_C(0xffff);
    return (uint16_t) (upper +
            (lower > UINT32_C(0x8000) ||
             (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0)));
}

static __device__ __forceinline__ float rms_norm_qemu_bf16_bits_to_f32(uint16_t value) {
    return __uint_as_float((uint32_t) value << 16);
}

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
        output[index] = rms_norm_qemu_f32_to_bf16_bits(input[source_index]);
    }
}

static __global__ void rms_norm_qemu_bf16_kernel(
        const uint16_t * input,
        uint16_t * output,
        int ncols,
        float eps) {
    const int row = (int) blockIdx.x;
    const int lane = (int) threadIdx.x;
    const uint16_t * row_input = input + (size_t) row * (size_t) ncols;
    uint16_t * row_output = output + (size_t) row * (size_t) ncols;

    float sum_squares = 0.0f;
    for (int col = lane; col < ncols; col += 16) {
        const float value = rms_norm_qemu_bf16_bits_to_f32(row_input[col]);
        sum_squares = fmaf(value, value, sum_squares);
    }

    const unsigned int mask = __activemask();
    for (int offset = 8; offset > 0; offset >>= 1) {
        sum_squares += __shfl_down_sync(mask, sum_squares, offset, 16);
    }

    __shared__ float scale;
    if (lane == 0) {
        scale = 1.0f / sqrtf(sum_squares / (float) ncols + eps);
    }
    __syncthreads();

    for (int col = lane; col < ncols; col += 16) {
        const float value = rms_norm_qemu_bf16_bits_to_f32(row_input[col]);
        row_output[col] = rms_norm_qemu_f32_to_bf16_bits(value * scale);
    }
}

static __global__ void rms_norm_qemu_bf16_to_f32_kernel(
        const uint16_t * input,
        float * output,
        size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        output[index] = rms_norm_qemu_bf16_bits_to_f32(input[index]);
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
    rms_norm_qemu_bf16_kernel<<<(unsigned int) rows, 16, 0, stream>>>(
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
