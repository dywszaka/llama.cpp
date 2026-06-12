#include "nvfp4-quantize.cuh"

#include "../../common.cuh"
#include "kcache-outlier.cuh"
#include "nvfp4-common.cuh"
#include "nvfp4-log.cuh"
#include "nvfp4-quantize-core.cuh"

#include <cstdlib>

namespace {

static __device__ __forceinline__ void quantize_row_nvfp4_bf16_block(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float global_scale,
        const bool bf16_internal_arith,
        const bool bf16_block_scale) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const float xi = (lane_active && k0 < ne00) ? x[(int64_t) i1 * s01 + k0] : 0.0f;
    ggml_cuda_nvfp4_core_quantize_block_bf16_trunc_nn(
            xi,
            lane_active,
            global_scale,
            bf16_internal_arith,
            bf16_block_scale,
            y + (int64_t) i1 * (ne00 / QK_NVFP4) + ib);
}

static __global__ void quantize_row_nvfp4_bf16_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ global_scales,
        const bool per_tensor_scale,
        const bool bf16_internal_arith,
        const bool bf16_block_scale) {
    const int i1 = blockIdx.y;
    const float global_scale = per_tensor_scale ? global_scales[0] : global_scales[i1];
    quantize_row_nvfp4_bf16_block(x, y, ne00, s01, global_scale, bf16_internal_arith, bf16_block_scale);
}

static __global__ void quantize_row_nvfp4_dynamic_bf16_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        const int64_t ne00,
        const int64_t s01,
        const float * __restrict__ amax_rows,
        const bool per_tensor_scale,
        const bool bf16_internal_arith,
        const bool bf16_block_scale) {
    const int i1 = blockIdx.y;
    const float amax_f = per_tensor_scale ? amax_rows[0] : amax_rows[i1];
    const float global_scale = ggml_cuda_nvfp4_kcache_outlier_q_global_scale(amax_f);
    quantize_row_nvfp4_bf16_block(x, y, ne00, s01, global_scale, bf16_internal_arith, bf16_block_scale);
}

} // namespace

bool ggml_cuda_nvfp4_bf16_quant_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_BF16_QUANT");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_bf16_quant_once(env, cached != 0);
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_bf16_quant_trunc_nn_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_bf16_quant_trunc_nn_once(env, cached != 0);
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_bf16_quant_bf16_internal_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_BF16_QUANT_BF16_INTERNAL");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_bf16_quant_bf16_internal_once(env, cached != 0);
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_bf16_quant_bf16_block_scale_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_BF16_QUANT_BF16_BLOCK_SCALE");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_bf16_quant_bf16_block_scale_once(env, cached != 0);
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_trunc_bf16_input_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_TRUNC_BF16_INPUT");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_trunc_bf16_input_once(env, cached != 0);
    }
    return cached != 0;
}

void ggml_cuda_nvfp4_quantize_rows_bf16_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * global_scales,
        bool per_tensor_scale,
        bool bf16_internal_arith,
        bool bf16_block_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
            x, y, ne00, s01, global_scales, per_tensor_scale, bf16_internal_arith, bf16_block_scale);
}

void ggml_cuda_nvfp4_quantize_rows_dynamic_bf16_f32(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        const float * amax_rows,
        bool per_tensor_scale,
        bool bf16_internal_arith,
        bool bf16_block_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);
    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_dynamic_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(
            x, y, ne00, s01, amax_rows, per_tensor_scale, bf16_internal_arith, bf16_block_scale);
}
