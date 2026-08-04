#include "rope-qemu-cuda.cuh"
#include "rope-qemu-protocol.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

static constexpr size_t ROPE_FP32_TABLE_FLOATS =
        (size_t) ROPE_FP32_TABLE_POSITIONS * ROPE_FP32_TABLE_CHANNELS * 2;
static constexpr size_t ROPE_FP32_TABLE_BYTES =
        ROPE_FP32_TABLE_FLOATS * sizeof(float);

static __device__ __forceinline__ uint16_t rope_bf16_from_f32_rz(float value) {
    return (uint16_t) (__float_as_uint(value) >> 16);
}

static __device__ __forceinline__ uint16_t rope_bf16_from_f32_rne(float value) {
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

static __device__ __forceinline__ float rope_bf16_to_f32(uint16_t value) {
    return __uint_as_float((uint32_t) value << 16);
}

static __device__ __forceinline__ float rope_load_native(
        const void * data,
        ggml_type type,
        int64_t index) {
    return type == GGML_TYPE_F32 ?
            ((const float *) data)[index] :
            __half2float(((const half *) data)[index]);
}

static __global__ void rope_qemu_preprocess_kernel(
        ggml_cuda_rope_qemu_params params,
        uint16_t * output,
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
        const int64_t source_index = i0 * params.s0[0] + i1 * params.s0[1] +
                i2 * params.s0[2] + i3 * params.s0[3];
        output[index] = rope_bf16_from_f32_rz(
                rope_load_native(params.src0, params.src0_type, source_index));
    }
}

static __global__ void rope_qemu_bf16_kernel(
        ggml_cuda_rope_qemu_params params,
        const uint16_t * input,
        const float * table,
        uint16_t * output,
        size_t elements) {
    for (size_t index = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
            index < elements;
            index += (size_t) blockDim.x * gridDim.x) {
        const size_t ne0 = (size_t) params.ne[0];
        const size_t i0 = index % ne0;
        const size_t row = index / ne0;
        if (i0 < 64) {
            const size_t token = row / (size_t) params.ne[1];
            const int32_t position = params.positions[token];
            if (position < 0 || position >= (int32_t) ROPE_FP32_TABLE_POSITIONS) {
                output[index] = UINT16_C(0x7fc0);
                output[index + 64] = UINT16_C(0x7fc0);
                continue;
            }
            const size_t table_index =
                    ((size_t) position * ROPE_FP32_TABLE_CHANNELS + i0) * 2;
            const float cosine = table[table_index];
            const float sine = table[table_index + 1];
            const float x0 = rope_bf16_to_f32(input[index]);
            const float x1 = rope_bf16_to_f32(input[index + 64]);
            const float x1_sine = __fmul_rn(x1, sine);
            const float x0_sine = __fmul_rn(x0, sine);
            const float y0 = __fmaf_rn(x0, cosine, -x1_sine);
            const float y1 = __fmaf_rn(x1, cosine, x0_sine);
            output[index] = rope_bf16_from_f32_rne(y0);
            output[index + 64] = rope_bf16_from_f32_rne(y1);
        } else if (i0 >= 128) {
            output[index] = input[index];
        }
    }
}

static __global__ void rope_qemu_output_kernel(
        ggml_cuda_rope_qemu_params params,
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
        const int64_t destination_index = i0 * params.sd[0] + i1 * params.sd[1] +
                i2 * params.sd[2] + i3 * params.sd[3];
        const float value = rope_bf16_to_f32(input[index]);
        if (params.dst_type == GGML_TYPE_F32) {
            ((float *) params.dst)[destination_index] = value;
        } else {
            ((half *) params.dst)[destination_index] = __float2half_rn(value);
        }
    }
}

static size_t rope_elements(const ggml_cuda_rope_qemu_params & params) {
    return (size_t) params.ne[0] * (size_t) params.ne[1] *
            (size_t) params.ne[2] * (size_t) params.ne[3];
}

static int rope_blocks(size_t elements) {
    constexpr size_t threads = 256;
    return (int) std::min<size_t>((elements + threads - 1) / threads, 65535);
}

static const float * rope_device_table() {
    static std::mutex mutex;
    static std::map<int, float *> tables;
    std::lock_guard<std::mutex> lock(mutex);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const auto found = tables.find(device);
    if (found != tables.end()) {
        return found->second;
    }

    const char * configured = std::getenv("GGML_CUDA_ROPE_QEMU_TABLE");
    const std::string path = configured != nullptr && configured[0] != '\0' ?
            configured :
            "/home/lerong.chen/0729-rope-node4/rope-cos-sin-f32.bin";
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != (std::streamoff) ROPE_FP32_TABLE_BYTES) {
        GGML_ABORT("%s: invalid static ROPE table path=%s expected_bytes=%zu\n",
                __func__, path.c_str(), ROPE_FP32_TABLE_BYTES);
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> host(ROPE_FP32_TABLE_FLOATS);
    if (!input.read((char *) host.data(), (std::streamsize) ROPE_FP32_TABLE_BYTES)) {
        GGML_ABORT("%s: failed to read static ROPE table path=%s\n",
                __func__, path.c_str());
    }

    float * device_table = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &device_table, ROPE_FP32_TABLE_BYTES));
    CUDA_CHECK(cudaMemcpy(device_table, host.data(), ROPE_FP32_TABLE_BYTES,
            cudaMemcpyHostToDevice));
    tables.emplace(device, device_table);
    GGML_LOG_INFO("%s: loaded static F32 ROPE table once device=%d bytes=%zu path=%s\n",
            __func__, device, ROPE_FP32_TABLE_BYTES, path.c_str());
    return device_table;
}

void ggml_cuda_rope_qemu_cuda_preprocess(
        const ggml_cuda_rope_qemu_params & params,
        uint16_t * src0_bf16,
        cudaStream_t stream) {
    const size_t elements = rope_elements(params);
    if (elements == 0) {
        return;
    }
    constexpr int threads = 256;
    rope_qemu_preprocess_kernel<<<rope_blocks(elements), threads, 0, stream>>>(
            params, src0_bf16, elements);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_rope_qemu_cuda_run_bf16(
        const ggml_cuda_rope_qemu_params & params,
        const uint16_t * src0_bf16,
        uint16_t * dst_bf16,
        cudaStream_t stream) {
    const size_t elements = rope_elements(params);
    if (elements == 0) {
        return;
    }
    const float * table = rope_device_table();
    constexpr int threads = 256;
    rope_qemu_bf16_kernel<<<rope_blocks(elements), threads, 0, stream>>>(
            params, src0_bf16, table, dst_bf16, elements);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_rope_qemu_cuda_output(
        const ggml_cuda_rope_qemu_params & params,
        const uint16_t * output_bf16,
        cudaStream_t stream) {
    const size_t elements = rope_elements(params);
    if (elements == 0) {
        return;
    }
    constexpr int threads = 256;
    rope_qemu_output_kernel<<<rope_blocks(elements), threads, 0, stream>>>(
            params, output_bf16, elements);
    CUDA_CHECK(cudaGetLastError());
}
