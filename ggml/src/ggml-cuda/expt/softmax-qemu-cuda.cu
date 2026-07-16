#include "softmax-qemu-cuda.cuh"
#include "softmax-bf16-core.cuh"

#include <algorithm>

template <typename T>
static __device__ __forceinline__ float softmax_qemu_mask_to_f32(T value) {
    return (float) value;
}

template <>
__device__ __forceinline__ float softmax_qemu_mask_to_f32<half>(half value) {
    return __half2float(value);
}

static __device__ __forceinline__ uint16_t softmax_qemu_f32_to_bf16_bits(float value) {
    uint32_t bits = __float_as_uint(value);
    const uint32_t absolute = bits & UINT32_C(0x7fffffff);
    if (absolute > UINT32_C(0x7f800000)) {
        uint16_t result = (uint16_t) (bits >> 16);
        return (result & UINT16_C(0x007f)) != 0 ? result : (uint16_t) (result | UINT16_C(1));
    }
    bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1u);
    return (uint16_t) (bits >> 16);
}

static __device__ __forceinline__ float softmax_qemu_bf16_bits_to_f32(uint16_t value) {
    return __uint_as_float((uint32_t) value << 16);
}

template <typename T>
static __global__ void softmax_qemu_preprocess_kernel(
        const float * input,
        const T * mask,
        uint16_t * output,
        ggml_cuda_soft_max_qemu_params params,
        size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        const int64_t row = (int64_t) (index / (size_t) params.ncols);
        const int64_t col = (int64_t) (index % (size_t) params.ncols);
        const int64_t i01 = row % params.ne01;
        const int64_t row_head_outer = row / params.ne01;
        const int64_t i02 = row_head_outer % params.ne02;
        const int64_t i03 = row_head_outer / params.ne02;
        const float slope = get_alibi_slope(
                params.max_bias, (uint32_t) i02, params.n_head_log2, params.m0, params.m1);

        float value = input[index] * params.scale;
        if (mask != nullptr) {
            const int64_t i12 = i02 % params.ne12;
            const int64_t i13 = i03 % params.ne13;
            const size_t mask_offset = (size_t) (
                    i01 * params.nb11 + i12 * params.nb12 + i13 * params.nb13) / sizeof(T);
            value += slope * softmax_qemu_mask_to_f32(mask[mask_offset + col]);
        }
        output[index] = softmax_qemu_f32_to_bf16_bits(value);
    }
}

static __global__ void softmax_qemu_preprocess_sinks_kernel(
        const float * input, uint16_t * output, size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        output[index] = softmax_qemu_f32_to_bf16_bits(input[index]);
    }
}

static __global__ void softmax_qemu_bf16_kernel(
        const uint16_t * input,
        const uint16_t * sinks,
        uint16_t * output,
        uint32_t * exponent_values,
        int64_t ncols,
        int64_t ne01,
        int64_t ne02) {
    __shared__ int maximum_q16;
    __shared__ unsigned long long sum_q31;

    const int64_t row = (int64_t) blockIdx.x;
    const uint16_t * row_input = input + row * ncols;
    uint16_t * row_output = output + row * ncols;
    uint32_t * row_exponents = exponent_values + row * ncols;
    const int64_t head = (row / ne01) % ne02;

    if (threadIdx.x == 0) {
        maximum_q16 = sinks != nullptr ? softmax_bf16_to_q16(sinks[head]) : INT32_MIN;
        sum_q31 = 0;
    }
    __syncthreads();

    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        atomicMax(&maximum_q16, softmax_bf16_to_q16(row_input[col]));
    }
    __syncthreads();

    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const int32_t value = softmax_bf16_to_q16(row_input[col]);
        const uint64_t delta = (uint64_t) ((int64_t) maximum_q16 - (int64_t) value);
        const uint32_t exponent = softmax_bf16_exp_neg_q31(delta);
        row_exponents[col] = exponent;
        atomicAdd(&sum_q31, (unsigned long long) exponent);
    }
    __syncthreads();

    if (threadIdx.x == 0 && sinks != nullptr) {
        const int32_t sink = softmax_bf16_to_q16(sinks[head]);
        const uint64_t delta = (uint64_t) ((int64_t) maximum_q16 - (int64_t) sink);
        sum_q31 += softmax_bf16_exp_neg_q31(delta);
    }
    __syncthreads();

    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        row_output[col] = softmax_bf16_probability_bits(row_exponents[col], sum_q31);
    }
}

static __global__ void softmax_qemu_bf16_to_f32_kernel(
        const uint16_t * input, float * output, size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        output[index] = softmax_qemu_bf16_bits_to_f32(input[index]);
    }
}

static int softmax_qemu_cuda_blocks(size_t elements) {
    constexpr size_t threads = 256;
    return (int) std::min<size_t>((elements + threads - 1) / threads, 65535);
}

void ggml_cuda_soft_max_qemu_cuda_preprocess(
        const ggml_cuda_soft_max_qemu_params & params,
        uint16_t * input_bf16,
        uint16_t * sinks_bf16,
        cudaStream_t stream) {
    const size_t elements = (size_t) params.ne01 * (size_t) params.ne02 *
            (size_t) params.ne03 * (size_t) params.ncols;
    constexpr int threads = 256;
    const int blocks = softmax_qemu_cuda_blocks(elements);
    if (elements != 0) {
        switch (params.mask_type) {
            case GGML_CUDA_SOFT_MAX_MASK_F16:
                softmax_qemu_preprocess_kernel<<<blocks, threads, 0, stream>>>(
                        params.src0, (const half *) params.src1, input_bf16, params, elements);
                break;
            case GGML_CUDA_SOFT_MAX_MASK_F32:
                softmax_qemu_preprocess_kernel<<<blocks, threads, 0, stream>>>(
                        params.src0, (const float *) params.src1, input_bf16, params, elements);
                break;
            case GGML_CUDA_SOFT_MAX_MASK_NONE:
                softmax_qemu_preprocess_kernel<<<blocks, threads, 0, stream>>>(
                        params.src0, (const float *) nullptr, input_bf16, params, elements);
                break;
        }
    }
    if (params.src2 != nullptr) {
        const int sink_blocks = softmax_qemu_cuda_blocks((size_t) params.ne02);
        softmax_qemu_preprocess_sinks_kernel<<<sink_blocks, threads, 0, stream>>>(
                params.src2, sinks_bf16, (size_t) params.ne02);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_soft_max_qemu_cuda_run_preprocessed(
        const ggml_cuda_soft_max_qemu_params & params,
        const uint16_t * input_bf16,
        const uint16_t * sinks_bf16,
        uint16_t * output_bf16,
        uint32_t * exponent_values,
        float * output_f32,
        cudaStream_t stream) {
    const size_t rows = (size_t) params.ne01 * (size_t) params.ne02 * (size_t) params.ne03;
    const size_t elements = rows * (size_t) params.ncols;
    if (rows != 0) {
        softmax_qemu_bf16_kernel<<<(unsigned int) rows, 256, 0, stream>>>(
                input_bf16, sinks_bf16, output_bf16, exponent_values,
                params.ncols, params.ne01, params.ne02);
    }
    ggml_cuda_soft_max_qemu_cuda_output_to_f32(output_bf16, output_f32, elements, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_soft_max_qemu_cuda_output_to_f32(
        const uint16_t * input_bf16,
        float * output_f32,
        size_t elements,
        cudaStream_t stream) {
    if (elements == 0) {
        return;
    }
    constexpr int threads = 256;
    const int blocks = softmax_qemu_cuda_blocks(elements);
    softmax_qemu_bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
            input_bf16, output_f32, elements);
}
