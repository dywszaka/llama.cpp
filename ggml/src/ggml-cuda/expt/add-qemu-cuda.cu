#include "add-qemu-cuda.cuh"

#include <algorithm>

static __device__ __forceinline__ uint16_t add_bf16_from_f32_rz(float value) {
    return (uint16_t) (__float_as_uint(value) >> 16);
}

static __device__ __forceinline__ uint16_t add_bf16_from_f32_rne(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t exponent = bits & UINT32_C(0x7f800000);
    const uint32_t mantissa = bits & UINT32_C(0x007fffff);
    if (exponent == UINT32_C(0x7f800000) && mantissa != 0) {
        return UINT16_C(0x7fc0);
    }

    const uint32_t upper = bits >> 16;
    const uint32_t lower = bits & UINT32_C(0xffff);
    return (uint16_t) (upper +
            (lower > UINT32_C(0x8000) ||
             (lower == UINT32_C(0x8000) && (upper & UINT32_C(1)) != 0)));
}

static __device__ __forceinline__ float add_bf16_to_f32(uint16_t value) {
    return __uint_as_float((uint32_t) value << 16);
}

static __device__ __forceinline__ float load_native_value(
        const void * data,
        ggml_type type,
        int64_t index) {
    return type == GGML_TYPE_F32 ?
            ((const float *) data)[index] :
            __half2float(((const half *) data)[index]);
}

static __global__ void add_qemu_preprocess_kernel(
        ggml_cuda_add_qemu_params params,
        uint16_t * src0_bf16,
        uint16_t * src1_bf16,
        size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        const int64_t i0 = (int64_t) (index % (size_t) params.ne[0]);
        size_t remaining = index / (size_t) params.ne[0];
        const int64_t i1 = (int64_t) (remaining % (size_t) params.ne[1]);
        remaining /= (size_t) params.ne[1];
        const int64_t i2 = (int64_t) (remaining % (size_t) params.ne[2]);
        const int64_t i3 = (int64_t) (remaining / (size_t) params.ne[2]);

        const int64_t src0_index = i0 * params.s0[0] +
                i1 * params.s0[1] + i2 * params.s0[2] + i3 * params.s0[3];
        const int64_t src1_index = (i0 % params.ne1[0]) * params.s1[0] +
                (i1 % params.ne1[1]) * params.s1[1] +
                (i2 % params.ne1[2]) * params.s1[2] +
                (i3 % params.ne1[3]) * params.s1[3];

        src0_bf16[index] = add_bf16_from_f32_rz(
                load_native_value(params.src0, params.src0_type, src0_index));
        src1_bf16[index] = add_bf16_from_f32_rz(
                load_native_value(params.src1, params.src1_type, src1_index));
    }
}

static __global__ void add_qemu_bf16_kernel(
        const uint16_t * src0,
        const uint16_t * src1,
        uint16_t * dst,
        size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        const float sum = __fadd_rn(
                add_bf16_to_f32(src0[index]),
                add_bf16_to_f32(src1[index]));
        dst[index] = add_bf16_from_f32_rne(sum);
    }
}

static __global__ void add_qemu_output_kernel(
        ggml_cuda_add_qemu_params params,
        const uint16_t * input,
        size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        const int64_t i0 = (int64_t) (index % (size_t) params.ne[0]);
        size_t remaining = index / (size_t) params.ne[0];
        const int64_t i1 = (int64_t) (remaining % (size_t) params.ne[1]);
        remaining /= (size_t) params.ne[1];
        const int64_t i2 = (int64_t) (remaining % (size_t) params.ne[2]);
        const int64_t i3 = (int64_t) (remaining / (size_t) params.ne[2]);
        const int64_t dst_index = i0 * params.sd[0] +
                i1 * params.sd[1] + i2 * params.sd[2] + i3 * params.sd[3];
        const float value = add_bf16_to_f32(input[index]);

        if (params.dst_type == GGML_TYPE_F32) {
            ((float *) params.dst)[dst_index] = value;
        } else {
            ((half *) params.dst)[dst_index] = __float2half_rn(value);
        }
    }
}

static int add_qemu_cuda_blocks(size_t elements) {
    constexpr size_t threads = 256;
    return (int) std::min<size_t>((elements + threads - 1) / threads, 65535);
}

static size_t add_qemu_elements(const ggml_cuda_add_qemu_params & params) {
    return (size_t) params.ne[0] * (size_t) params.ne[1] *
            (size_t) params.ne[2] * (size_t) params.ne[3];
}

void ggml_cuda_add_qemu_cuda_preprocess(
        const ggml_cuda_add_qemu_params & params,
        uint16_t * src0_bf16,
        uint16_t * src1_bf16,
        cudaStream_t stream) {
    const size_t elements = add_qemu_elements(params);
    if (elements == 0) {
        return;
    }
    constexpr int threads = 256;
    add_qemu_preprocess_kernel<<<add_qemu_cuda_blocks(elements), threads, 0, stream>>>(
            params, src0_bf16, src1_bf16, elements);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_add_qemu_cuda_run_bf16(
        const ggml_cuda_add_qemu_params & params,
        const uint16_t * src0_bf16,
        const uint16_t * src1_bf16,
        uint16_t * dst_bf16,
        cudaStream_t stream) {
    const size_t elements = add_qemu_elements(params);
    if (elements == 0) {
        return;
    }
    constexpr int threads = 256;
    add_qemu_bf16_kernel<<<add_qemu_cuda_blocks(elements), threads, 0, stream>>>(
            src0_bf16, src1_bf16, dst_bf16, elements);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_add_qemu_cuda_output(
        const ggml_cuda_add_qemu_params & params,
        const uint16_t * output_bf16,
        cudaStream_t stream) {
    const size_t elements = add_qemu_elements(params);
    if (elements == 0) {
        return;
    }
    constexpr int threads = 256;
    add_qemu_output_kernel<<<add_qemu_cuda_blocks(elements), threads, 0, stream>>>(
            params, output_bf16, elements);
    CUDA_CHECK(cudaGetLastError());
}
