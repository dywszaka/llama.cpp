#include "vcache-nvfp4-set-rows.cuh"

#include <cstdlib>

static constexpr const char * GGML_CUDA_NVFP4_VCACHE_FAST_UPDATE_ENV = "LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE";

static bool ggml_cuda_nvfp4_vcache_experiment_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("LLAMA_EXPERIMENT_NVFP4_VCACHE");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_vcache_fast_update_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv(GGML_CUDA_NVFP4_VCACHE_FAST_UPDATE_ENV);
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

void ggml_cuda_log_nvfp4_vcache_fast_update_once(bool enabled) {
    static int logged = 0;
    if (logged != 0) {
        return;
    }
    logged = 1;

    const char * env = getenv(GGML_CUDA_NVFP4_VCACHE_FAST_UPDATE_ENV);
    GGML_LOG_INFO(
            "%s: %s=%s -> %s\n",
            __func__,
            GGML_CUDA_NVFP4_VCACHE_FAST_UPDATE_ENV,
            env != nullptr ? env : "(unset)",
            enabled ? "enabled, CUDA NVFP4 V-cache set_rows may patch single-token updates without requantizing the block"
                    : "disabled");
}

bool ggml_cuda_is_nvfp4_vcache_transposed_set_rows(
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

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_nvfp4_vcache_set_rows(float x) {
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

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_e4m3_vcache_set_rows(float x) {
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

static __device__ __forceinline__ float ggml_cuda_dequantize_nvfp4_value_set_rows_global(
        const block_nvfp4 & block,
        float global_scale,
        int lane) {
    const float d = global_scale > 0.0f ? (ggml_cuda_e4m3_to_fp32_half(block.e) / global_scale) : 0.0f;
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
        int64_t n_row_groups,
        int64_t rows_per_scale,
        bool scale_is_global,
        bool fast_update) {
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
    __shared__ int fast_update_done;
    __shared__ int pending_lane;
    __shared__ float pending_value;

    if (fast_update && n_tokens == 1) {
        const int64_t flat_group = row_group;
        const int64_t dst_index = src1[flat_group] + row_in_group * kv_size_padded;
        const int64_t row_global = dst_index / kv_size_padded;
        const int64_t token_slot = dst_index - row_global * kv_size_padded;
        const int64_t block_idx = token_slot / QK_NVFP4;
        const int in_block = (int) (token_slot % QK_NVFP4);
        const int64_t flat_block = row_global * n_blocks + block_idx;

        if (lane == 0) {
            fast_update_done = 0;
            const float value = src0[flat_group * QK_NVFP4 + row_in_group];
            const float input_scale = scale_is_global ? 0.0f : scale[flat_block];
            const float global_scale = scale_is_global ? scale[row_global / rows_per_scale] : (input_scale > 0.0f ? 1.0f / input_scale : 0.0f);
            const block_nvfp4 block = dst[flat_block];
            const float block_scale_f = ggml_cuda_e4m3_to_fp32_half(block.e);
            float current_amax_q = 0.0f;
            for (int byte = 0; byte < QK_NVFP4 / 2; ++byte) {
                const uint8_t packed = block.qs[byte];
                current_amax_q = fmaxf(current_amax_q, fabsf((float) kvalues_nvfp4[packed & 0x0F]));
                current_amax_q = fmaxf(current_amax_q, fabsf((float) kvalues_nvfp4[packed >> 4]));
            }
            const float current_amax = scale_is_global ?
                (global_scale > 0.0f ? current_amax_q * block_scale_f / global_scale : 0.0f) :
                current_amax_q * block_scale_f * input_scale;

            if ((scale_is_global ? global_scale > 0.0f : input_scale > 0.0f) && block_scale_f > 0.0f &&
                    isfinite(scale_is_global ? global_scale : input_scale) && isfinite(block_scale_f) && isfinite(value) &&
                    fabsf(value) <= current_amax) {
                const float inv_scale = global_scale / block_scale_f;
                const uint8_t q = ggml_cuda_best_index_nvfp4_vcache_set_rows(value * inv_scale);
                const int byte = in_block / 2;
                const uint8_t old = block.qs[byte];
                const uint8_t patched = (in_block & 1) == 0
                    ? (uint8_t) ((old & 0xF0) | q)
                    : (uint8_t) ((old & 0x0F) | (q << 4));
                dst[flat_block].qs[byte] = patched;
                fast_update_done = 1;
            }
        }
        __syncthreads();

        if (fast_update_done != 0) {
            return;
        }
    }

    auto load_block = [&](int64_t row_global, int64_t block_idx) {
        if (lane < QK_NVFP4) {
            const int64_t flat_block = row_global * n_blocks + block_idx;
            const block_nvfp4 block = dst[flat_block];
            if (scale_is_global) {
                tile[lane] = ggml_cuda_dequantize_nvfp4_value_set_rows_global(block, scale[row_global / rows_per_scale], lane);
            } else {
                const float input_scale = scale[flat_block];
                tile[lane] = ggml_cuda_dequantize_nvfp4_value_set_rows(block, input_scale, lane);
            }
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
        const float global_scale = scale_is_global ? scale[row_global / rows_per_scale] :
            ((amax > 0.0f && isfinite(amax)) ? (GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / amax) : 0.0f);
        float block_scale_f = 0.0f;

        if (lane == 0) {
            if (!scale_is_global) {
                const float input_scale = (global_scale != 0.0f && isfinite(global_scale)) ? (1.0f / global_scale) : 0.0f;
                scale[row_global * n_blocks + block_idx] = input_scale;
            }

            const float scale_f = (global_scale != 0.0f) ? (global_scale * (amax / GGML_CUDA_NVFP4_FP4_MAX)) : 0.0f;
            const uint8_t scale_q = ggml_cuda_best_index_e4m3_vcache_set_rows(scale_f);
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
            qvals[lane] = ggml_cuda_best_index_nvfp4_vcache_set_rows(tile[lane] * inv_scale);
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

void ggml_cuda_op_set_rows_nvfp4_vcache(
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
    const int64_t n_scales = ggml_nelements(v_scale);
    const bool scale_is_global = v_scale->ne[0] == 1 && n_scales > 0 &&
        (v_cache->ne[2] == n_scales || n_rows_local % n_scales == 0);
    const int64_t rows_per_scale = scale_is_global ?
        (v_cache->ne[2] == n_scales ? n_rows_local : n_rows_local / n_scales) : 0;
    const bool fast_update = ggml_cuda_nvfp4_vcache_fast_update_enabled();

    GGML_ASSERT(kv_size_padded % QK_NVFP4 == 0);
    GGML_ASSERT(n_rows_local % QK_NVFP4 == 0);
    GGML_ASSERT(n_rows_local > 0);
    GGML_ASSERT(n_row_groups > 0);
    GGML_ASSERT(src1->ne[0] % n_row_groups == 0);
    GGML_ASSERT(src0->ne[0] == QK_NVFP4);
    GGML_ASSERT(src0->ne[1] == src1->ne[0]);

    ggml_cuda_log_nvfp4_vcache_fast_update_once(fast_update);

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
                n_row_groups,
                rows_per_scale,
                scale_is_global,
                fast_update);
    }
    CUDA_CHECK(cudaGetLastError());
}
