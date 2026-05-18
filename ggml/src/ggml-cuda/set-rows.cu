#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "../ggml-quants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

static constexpr float GGML_CUDA_NVFP4_FP4_MAX = 6.0f;
static constexpr float GGML_CUDA_NVFP4_E4M3_HALF_MAX = 224.0f;
static constexpr float GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX = GGML_CUDA_NVFP4_FP4_MAX * GGML_CUDA_NVFP4_E4M3_HALF_MAX;

static bool ggml_cuda_nvfp4_vcache_experiment_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("LLAMA_EXPERIMENT_NVFP4_VCACHE");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_cuda_is_nvfp4_vcache_transposed_set_rows(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    if (!ggml_cuda_nvfp4_vcache_experiment_enabled()) {
        return false;
    }

    if (dst->type != GGML_TYPE_NVFP4 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I64) {
        return false;
    }

    if (dst->view_src == nullptr || ggml_tensor_get_nvfp4_scale(dst) == nullptr) {
        return false;
    }

    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }

    if (dst->ne[0] != QK_NVFP4 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (src0->ne[0] <= 0 || src0->ne[1] <= 0) {
        return false;
    }

    if (src0->ne[0] != QK_NVFP4) {
        return false;
    }

    if (src1->ne[0] * QK_NVFP4 != src0->ne[0] * src0->ne[1]) {
        return false;
    }

    return true;
}

bool ggml_cuda_is_experimental_nvfp4_vcache_set_rows_node(const ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_SET_ROWS) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    if (src0 == nullptr || src1 == nullptr || src2 == nullptr) {
        return false;
    }

    return ggml_cuda_is_nvfp4_vcache_transposed_set_rows(src1, src2, dst);
}

static bool ggml_cuda_fp8_e4m3_e8m0_32_e4m2_experiment_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2");
        cached = (env != nullptr && atoi(env) != 0) ? 1 : 0;
    }
    return cached != 0;
}

static void ggml_cuda_log_fp8_e4m3_e8m0_32_e4m2_set_rows_once(const ggml_tensor * dst, bool enabled) {
    static int logged_enabled = 0;
    static int logged_disabled = 0;

    int * logged = enabled ? &logged_enabled : &logged_disabled;
    if (*logged != 0) {
        return;
    }
    *logged = 1;

    const char * env = getenv("GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2");
    GGML_LOG_INFO(
            "%s: GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2=%s -> %s; dst=%s type=%s ne=[%lld,%lld,%lld,%lld]\n",
            __func__,
            env != nullptr ? env : "(unset)",
            enabled ? "enabled, CUDA set_rows will mask FP8 mantissa low bit (E4M2 experiment)"
                    : "disabled, CUDA set_rows keeps FP8 E4M3",
            ggml_get_name(dst),
            ggml_type_name(dst->type),
            (long long) dst->ne[0], (long long) dst->ne[1],
            (long long) dst->ne[2], (long long) dst->ne[3]);
}

static __device__ void quantize_f32_fp8_e4m3_e8m0_32_e4m2_block(const float * __restrict__ x, block_fp8_e4m3_e8m0_32 * __restrict__ y) {
    quantize_f32_fp8_e4m3_e8m0_32_block(x, y, true);
}

static __device__ void quantize_f32_fp8_e4m3_e8m0_32_e4m3_block(const float * __restrict__ x, block_fp8_e4m3_e8m0_32 * __restrict__ y) {
    quantize_f32_fp8_e4m3_e8m0_32_block(x, y, false);
}

static __device__ __forceinline__ void ggml_cuda_atomic_max_f32(float * addr, float value) {
    int * addr_i = (int *) addr;
    int old = *addr_i;

    while (__int_as_float(old) < value) {
        const int assumed = old;
        old = atomicCAS(addr_i, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_nvfp4_set_rows(float x) {
    uint8_t best_index = 0;
    float best_err = fabsf((float) kvalues_nvfp4[0] - x);

#pragma unroll
    for (int i = 1; i < 16; ++i) {
        const float err = fabsf((float) kvalues_nvfp4[i] - x);
        if (err < best_err) {
            best_err = err;
            best_index = (uint8_t) i;
        }
    }

    return best_index;
}

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_e4m3_set_rows(float x) {
    uint8_t best_index = 0;
    float best_err = INFINITY;

    for (int i = 0; i < 256; ++i) {
        const float v = ggml_cuda_e4m3_to_fp32((uint8_t) i);
        if (!isfinite(v)) {
            continue;
        }

        const float err = fabsf(v - x);
        if (err < best_err) {
            best_err = err;
            best_index = (uint8_t) i;
        }
    }

    return best_index;
}

static __device__ __forceinline__ float ggml_cuda_dequantize_nvfp4_value_set_rows(
        const block_nvfp4 & block,
        float input_scale,
        int lane) {
    const float d = ggml_cuda_e4m3_to_fp32_half(block.e) * input_scale;
    const uint8_t packed = block.qs[lane / 2];
    const uint8_t q = (lane & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
    return d * (float) kvalues_nvfp4[q];
}

static __global__ void k_set_rows_nvfp4_vcache(
        const float * __restrict__ src0,
        const int64_t * __restrict__ src1,
        block_nvfp4 * __restrict__ dst,
        float * __restrict__ scale,
        int64_t n_rows_local,
        int64_t n_tokens,
        int64_t kv_size_padded,
        int64_t n_blocks,
        int64_t n_row_groups) {
    const int row_local = blockIdx.x;
    const int lane = threadIdx.x;

    if (row_local >= n_rows_local || lane >= WARP_SIZE) {
        return;
    }

    const int64_t row_group = row_local / QK_NVFP4;
    const int64_t row_in_group = row_local - row_group * QK_NVFP4;

    __shared__ float tile[QK_NVFP4];
    __shared__ float reduction[QK_NVFP4];
    __shared__ uint8_t qvals[QK_NVFP4];
    __shared__ int64_t current_row_global;
    __shared__ int64_t current_block;
    __shared__ int pending_flush;
    __shared__ int active_block;
    __shared__ int pending_lane;
    __shared__ float pending_value;

    auto load_block = [&](int64_t row_global, int64_t block_idx) {
        if (lane < QK_NVFP4) {
            const int64_t flat_block = row_global * n_blocks + block_idx;
            const float input_scale = scale[flat_block];
            const block_nvfp4 block = dst[flat_block];
            tile[lane] = ggml_cuda_dequantize_nvfp4_value_set_rows(block, input_scale, lane);
        }
    };

    auto flush_block = [&](int64_t row_global, int64_t block_idx) {
        if (lane < QK_NVFP4) {
            reduction[lane] = fabsf(tile[lane]);
        }
        __syncthreads();

        for (int stride = QK_NVFP4 / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
            }
            __syncthreads();
        }

        const float amax = reduction[0];
        const float global_scale = (amax > 0.0f && isfinite(amax)) ? (GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / amax) : 0.0f;
        float block_scale_f = 0.0f;

        if (lane == 0) {
            const float input_scale = (global_scale != 0.0f && isfinite(global_scale)) ? (1.0f / global_scale) : 0.0f;
            scale[row_global * n_blocks + block_idx] = input_scale;

            const float scale_f = (global_scale != 0.0f) ? (global_scale * (amax / GGML_CUDA_NVFP4_FP4_MAX)) : 0.0f;
            const uint8_t scale_q = ggml_cuda_best_index_e4m3_set_rows(scale_f);
            dst[row_global * n_blocks + block_idx].e = scale_q;
            block_scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
        }
        __syncthreads();

        if (lane == 0) {
            reduction[0] = block_scale_f;
        }
        __syncthreads();

        const float inv_scale = (global_scale != 0.0f && reduction[0] != 0.0f) ? (global_scale / reduction[0]) : 0.0f;
        if (lane < QK_NVFP4) {
            qvals[lane] = ggml_cuda_best_index_nvfp4_set_rows(tile[lane] * inv_scale);
        }
        __syncthreads();

        if (lane < QK_NVFP4 && (lane & 1) == 0) {
            dst[row_global * n_blocks + block_idx].qs[lane / 2] = qvals[lane] | (qvals[lane + 1] << 4);
        }
        __syncthreads();
    };

    if (lane == 0) {
        active_block = 0;
    }
    __syncthreads();

    for (int64_t token = 0; token < n_tokens; ++token) {
        const int64_t flat_group = token * n_row_groups + row_group;
        const int64_t dst_index = src1[flat_group] + row_in_group * kv_size_padded;
        const int64_t row_global = dst_index / kv_size_padded;
        const int64_t token_slot = dst_index - row_global * kv_size_padded;
        const int64_t block_idx = token_slot / QK_NVFP4;
        const int in_block = (int) (token_slot % QK_NVFP4);

        if (lane == 0) {
            pending_flush = active_block && (row_global != current_row_global || block_idx != current_block);
            pending_lane = in_block;
            pending_value = src0[flat_group * QK_NVFP4 + row_in_group];
        }
        __syncthreads();

        if (pending_flush) {
            flush_block(current_row_global, current_block);
        }

        if (lane == 0) {
            if (!active_block || row_global != current_row_global || block_idx != current_block) {
                current_row_global = row_global;
                current_block = block_idx;
                active_block = 1;
            }
        }
        __syncthreads();

        if (pending_flush || token == 0) {
            load_block(current_row_global, current_block);
        }
        __syncthreads();

        if (lane == pending_lane) {
            tile[lane] = pending_value;
        }
        __syncthreads();
    }

    if (active_block) {
        flush_block(current_row_global, current_block);
    }
}

static __global__ void k_abs_max_f32_rows(
        const float * __restrict__ src0,
        float * __restrict__ amax,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t s01) {
    const int64_t row = blockIdx.x;
    if (row >= ne01) {
        return;
    }

    float local_max = 0.0f;
    const int64_t row_off = row * s01;
    for (int64_t i = threadIdx.x; i < ne00; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(src0[row_off + i]));
    }

    __shared__ float shared_max[CUDA_SET_ROWS_BLOCK_SIZE];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        amax[row] = shared_max[0];
    }
}

static __global__ void k_set_rows_nvfp4(
        const float * __restrict__ src0, const int64_t * __restrict__ src1, block_nvfp4 * __restrict__ dst,
        const int64_t ne00, const int64_t ne01,
        const int64_t s01,
        const int64_t s10,
        const int64_t s1,
        const float * __restrict__ amax_rows) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const int64_t row_off = (int64_t) i1 * s01;
    const float xi = (lane_active && k0 < ne00) ? src0[row_off + k0] : 0.0f;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    const int64_t dst_row = *(src1 + i1*s10);
    block_nvfp4 * dst_row_ptr = dst + dst_row*s1 / sizeof(block_nvfp4);

    float scale_f = 0.0f;
    const float amax_f = amax_rows[i1];
    const float global_scale = (amax_f > 0.0f && isfinite(amax_f)) ? (GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / amax_f) : 0.0f;
    if (lane == 0) {
        const float scale = (global_scale != 0.0f) ? (global_scale * (vmax / GGML_CUDA_NVFP4_FP4_MAX)) : 0.0f;
        const uint8_t scale_q = ggml_cuda_best_index_e4m3_set_rows(scale);
        dst_row_ptr[ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4_set_rows(xi * inv_scale);
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        dst_row_ptr[ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static __global__ void k_set_rows_nvfp4_8(
        const float * __restrict__ src0, const int64_t * __restrict__ src1, block_nvfp4_8 * __restrict__ dst,
        const int64_t ne00, const int64_t ne01,
        const int64_t s01,
        const int64_t s10,
        const int64_t s1,
        const float * __restrict__ amax_rows) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4_8;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4_8 + lane;

    const int64_t row_off = (int64_t) i1 * s01;
    const float xi = (lane_active && k0 < ne00) ? src0[row_off + k0] : 0.0f;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    const int64_t dst_row = *(src1 + i1*s10);
    block_nvfp4_8 * dst_row_ptr = dst + dst_row*s1 / sizeof(block_nvfp4_8);

    float scale_f = 0.0f;
    const float amax_f = amax_rows[i1];
    const float global_scale = (amax_f > 0.0f && isfinite(amax_f)) ? (GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / amax_f) : 0.0f;
    if (lane == 0) {
        const float scale = (global_scale != 0.0f) ? (global_scale * (vmax / GGML_CUDA_NVFP4_FP4_MAX)) : 0.0f;
        const uint8_t scale_q = ggml_cuda_best_index_e4m3_set_rows(scale);
        dst_row_ptr[ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4_set_rows(xi * inv_scale);
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (lane_active && (lane & 1) == 0) {
        dst_row_ptr[ib].qs[lane/2] = q | (q_peer << 4);
    }
}

static void ggml_cuda_op_set_rows_nvfp4_vcache(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * src0,
        const ggml_tensor * src1) {
    cudaStream_t stream = ctx.stream();

    ggml_tensor * v_cache = dst->view_src;
    ggml_tensor * v_scale = (ggml_tensor *) ggml_tensor_get_nvfp4_scale(dst);

    GGML_ASSERT(v_cache != nullptr);
    GGML_ASSERT(v_scale != nullptr);
    GGML_ASSERT(v_cache->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(v_scale->type == GGML_TYPE_F32);

    const int64_t kv_size_padded = v_cache->ne[0];
    const int64_t n_rows_local = v_cache->ne[1];
    const int64_t n_row_groups = n_rows_local / QK_NVFP4;
    const int64_t n_tokens = src1->ne[0] / n_row_groups;
    const int64_t n_blocks = kv_size_padded / QK_NVFP4;

    GGML_ASSERT(kv_size_padded % QK_NVFP4 == 0);
    GGML_ASSERT(n_rows_local % QK_NVFP4 == 0);
    GGML_ASSERT(n_rows_local > 0);
    GGML_ASSERT(n_row_groups > 0);
    GGML_ASSERT(src1->ne[0] % n_row_groups == 0);
    GGML_ASSERT(src0->ne[0] == QK_NVFP4);
    GGML_ASSERT(src0->ne[1] == src1->ne[0]);
    if (n_tokens > 0) {
        k_set_rows_nvfp4_vcache<<<(uint32_t) n_rows_local, WARP_SIZE, 0, stream>>>(
                (const float *) src0->data,
                (const int64_t *) src1->data,
                (block_nvfp4 *) v_cache->data,
                (float *) v_scale->data,
                n_rows_local,
                n_tokens,
                kv_size_padded,
                n_blocks,
                n_row_groups);
    }
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void k_set_rows_scale(
        const int64_t * __restrict__ src1,
        float * __restrict__ scale,
        const int64_t ne10,
        const int64_t s10,
        const float * __restrict__ amax_rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne10) {
        return;
    }

    const float amax_f = amax_rows[i];
    const float global_scale = (amax_f > 0.0f && isfinite(amax_f)) ? (GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / amax_f) : 0.0f;
    const float input_scale = (global_scale != 0.0f && isfinite(global_scale)) ? (1.0f / global_scale) : 0.0f;
    const int64_t dst_row = *(src1 + i*s10);
    scale[dst_row] = input_scale;
}

// Generic quantized set_rows kernel template
template<typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static __global__ void k_set_rows_quant(
        const float * __restrict__ src0, const int64_t * __restrict__ src1, block_type * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1, const int64_t s2, const int64_t s3) {

    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    const int64_t i03 = i_base / (ne00 * ne01 * ne02);
    const int64_t i02 = (i_base - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i_base - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i_base - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const int64_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(int64_t);
    const int64_t s11 = nb11/sizeof(int64_t);
    const int64_t s12 = nb12/sizeof(int64_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0) {
        k_set_rows_quant<block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            s01, s02, s03,
            s10, s11, s12,
            s1, s2, s3);
    }
}

template<typename src_t, typename dst_t>
static __global__ void k_set_rows(
        const src_t * __restrict__ src0, const int64_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1, const int64_t s2, const int64_t s3) {

    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;

    if (i >= ne_total) {
        return;
    }

    const int64_t i03 = i / (ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const int64_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(int64_t);
    const int64_t s11 = nb11/sizeof(int64_t);
    const int64_t s12 = nb12/sizeof(int64_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0) {
        k_set_rows<<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            s01, s02, s03,
            s10, s11, s12,
            s1, s2, s3);
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64);

    GGML_TENSOR_BINARY_OP_LOCALS

    const float * src0_d   = (const float *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    cudaStream_t stream = ctx.stream();

    if (ggml_cuda_is_nvfp4_vcache_transposed_set_rows(src0, src1, dst)) {
        ggml_cuda_op_set_rows_nvfp4_vcache(ctx, dst, src0, src1);
        return;
    }



    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_FP8_E4M3_E8M0_32) {
        const bool fp8_e4m2_experiment = ggml_cuda_fp8_e4m3_e8m0_32_e4m2_experiment_enabled();
        ggml_cuda_log_fp8_e4m3_e8m0_32_e4m2_set_rows_once(dst, fp8_e4m2_experiment);
        if (fp8_e4m2_experiment) {
            set_rows_cuda_quant<block_fp8_e4m3_e8m0_32, QK_FP8_E4M3_E8M0_32, quantize_f32_fp8_e4m3_e8m0_32_e4m2_block>(
                src0_d, src1_d, (block_fp8_e4m3_e8m0_32 *) dst->data,
                ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
        } else {
            set_rows_cuda_quant<block_fp8_e4m3_e8m0_32, QK_FP8_E4M3_E8M0_32, quantize_f32_fp8_e4m3_e8m0_32_e4m3_block>(
                src0_d, src1_d, (block_fp8_e4m3_e8m0_32 *) dst->data,
                ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream
            );
        }
    } else if (dst->type == GGML_TYPE_FP8_E4M3_E8M0_16) {
        set_rows_cuda_quant<block_fp8_e4m3_e8m0_16, QK_FP8_E4M3_E8M0_16, quantize_f32_fp8_e4m3_e8m0_16_block>(
            src0_d, src1_d, (block_fp8_e4m3_e8m0_16 *) dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_NVFP4) {
        GGML_ASSERT(ne02 == 1 && ne03 == 1);
        GGML_ASSERT(ne10 == ne01 && ne11 == 1 && ne12 == 1 && ne13 == 1);
        GGML_ASSERT(ne00 % QK_NVFP4 == 0);

        const ggml_tensor * scale_tensor = ggml_tensor_get_nvfp4_scale(dst);
        GGML_ASSERT(scale_tensor != nullptr);
        GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);
        GGML_ASSERT(scale_tensor->data != nullptr);
        ggml_cuda_pool_alloc<float> amax_d(ctx.pool(), (size_t) ne01);
        if (ne01 > 0) {
            k_abs_max_f32_rows<<<(int) ne01, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                    src0_d, amax_d.get(),
                    ne00, ne01,
                    nb01/sizeof(float));
            CUDA_CHECK(cudaGetLastError());
        }

        if (ne01 > 0) {
            const dim3 block_size(QK_NVFP4);
            const dim3 grid_size((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
            k_set_rows_nvfp4<<<grid_size, block_size, 0, stream>>>(
                    src0_d, src1_d, (block_nvfp4 *) dst->data,
                    ne00, ne01,
                    nb01/sizeof(float),
                    nb10/sizeof(int64_t),
                    nb1,
                    amax_d.get());
            CUDA_CHECK(cudaGetLastError());

            const int scale_blocks = (int) ((ne10 + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE);
            k_set_rows_scale<<<scale_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                    src1_d,
                    (float *) scale_tensor->data,
                    ne10,
                    nb10/sizeof(int64_t),
                    amax_d.get());
            CUDA_CHECK(cudaGetLastError());
        }
    } else if (dst->type == GGML_TYPE_NVFP4_8) {
        GGML_ASSERT(ne02 == 1 && ne03 == 1);
        GGML_ASSERT(ne10 == ne01 && ne11 == 1 && ne12 == 1 && ne13 == 1);
        GGML_ASSERT(ne00 % QK_NVFP4_8 == 0);

        const ggml_tensor * scale_tensor = ggml_tensor_get_nvfp4_scale(dst);
        GGML_ASSERT(scale_tensor != nullptr);
        GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);
        GGML_ASSERT(scale_tensor->data != nullptr);
        ggml_cuda_pool_alloc<float> amax_d(ctx.pool(), (size_t) ne01);
        if (ne01 > 0) {
            k_abs_max_f32_rows<<<(int) ne01, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                    src0_d, amax_d.get(),
                    ne00, ne01,
                    nb01/sizeof(float));
            CUDA_CHECK(cudaGetLastError());
        }

        if (ne01 > 0) {
            const dim3 block_size(WARP_SIZE);
            const dim3 grid_size((uint32_t) (ne00 / QK_NVFP4_8), (uint32_t) ne01, 1);
            k_set_rows_nvfp4_8<<<grid_size, block_size, 0, stream>>>(
                    src0_d, src1_d, (block_nvfp4_8 *) dst->data,
                    ne00, ne01,
                    nb01/sizeof(float),
                    nb10/sizeof(int64_t),
                    nb1,
                    amax_d.get());
            CUDA_CHECK(cudaGetLastError());

            const int scale_blocks = (int) ((ne10 + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE);
            k_set_rows_scale<<<scale_blocks, CUDA_SET_ROWS_BLOCK_SIZE, 0, stream>>>(
                    src1_d,
                    (float *) scale_tensor->data,
                    ne10,
                    nb10/sizeof(int64_t),
                    amax_d.get());
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}
