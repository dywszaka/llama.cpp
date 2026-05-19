#include "vcache-nvfp4-matmul.cuh"

#include <atomic>
#include <cstdlib>

static constexpr float GGML_CUDA_VCACHE_NVFP4_FP4_MAX = 6.0f;
static constexpr float GGML_CUDA_VCACHE_NVFP4_E4M3_HALF_MAX = 224.0f;
static constexpr float GGML_CUDA_VCACHE_NVFP4_GLOBAL_SCALE_MAX =
        GGML_CUDA_VCACHE_NVFP4_FP4_MAX * GGML_CUDA_VCACHE_NVFP4_E4M3_HALF_MAX;
static constexpr int64_t GGML_CUDA_VCACHE_NVFP4_FP4_P_AMAX_PREPASS_MIN_KV = 2048;
static constexpr const char * GGML_CUDA_VCACHE_NVFP4_FP4_PV_ENV = "LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV";

static bool ggml_cuda_vcache_nvfp4_fp4_pv_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv(GGML_CUDA_VCACHE_NVFP4_FP4_PV_ENV);
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static void ggml_cuda_vcache_nvfp4_log_fp4_pv_once(bool enabled) {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    const char * env = getenv(GGML_CUDA_VCACHE_NVFP4_FP4_PV_ENV);
    GGML_LOG_INFO(
            "%s: %s=%s -> %s\n",
            __func__,
            GGML_CUDA_VCACHE_NVFP4_FP4_PV_ENV,
            env != nullptr ? env : "(unset)",
            enabled ? "enabled, CUDA NVFP4 V-cache p*v quantizes P to dynamic NVFP4 before dot"
                    : "disabled, CUDA NVFP4 V-cache p*v uses F32 P");
}

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_nvfp4_vcache(float x) {
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

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_e4m3_vcache(float x) {
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

static bool ggml_cuda_is_experimental_vcache_nvfp4_tensor(const ggml_tensor * src0) {
    if (src0 == nullptr || src0->type != GGML_TYPE_NVFP4) {
        return false;
    }

    const ggml_tensor * scale = ggml_tensor_get_nvfp4_scale(src0);
    if (scale == nullptr || scale->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] % QK_NVFP4 != 0 || src0->ne[1] <= 0) {
        return false;
    }

    return true;
}

static bool ggml_cuda_match_vcache_nvfp4_scale_layout(
        const ggml_tensor * src0,
        const ggml_tensor * scale,
        int64_t & blocks,
        int64_t & rows,
        int64_t & heads,
        int64_t & streams,
        int64_t & scale_row_nb,
        int64_t & scale_head_nb,
        int64_t & scale_stream_nb) {
    blocks = src0->ne[0] / QK_NVFP4;
    rows = src0->ne[1];
    heads = src0->ne[2];
    streams = src0->ne[3];

    if (scale->ne[0] == blocks &&
        scale->ne[1] == rows &&
        scale->ne[2] == heads &&
        scale->ne[3] == streams) {
        scale_row_nb = scale->nb[1];
        scale_head_nb = scale->nb[2];
        scale_stream_nb = scale->nb[3];
        return true;
    }

    if (scale->ne[0] == blocks &&
        scale->ne[1] == heads &&
        scale->ne[2] == rows &&
        scale->ne[3] == streams) {
        scale_row_nb = scale->nb[2];
        scale_head_nb = scale->nb[1];
        scale_stream_nb = scale->nb[3];
        return true;
    }

    return false;
}

static __global__ void k_p_rows_abs_max_f32(
        const float * __restrict__ p_data,
        float * __restrict__ p_amax,
        int64_t kv_size,
        int64_t cols,
        int64_t q_heads,
        int64_t q_streams,
        int64_t p_nb1,
        int64_t p_nb2,
        int64_t p_nb3) {
    const int64_t p_row = blockIdx.x;
    const int64_t p_rows = cols * q_heads * q_streams;
    if (p_row >= p_rows) {
        return;
    }

    const int64_t stream = p_row / (cols * q_heads);
    const int64_t rem = p_row - stream * cols * q_heads;
    const int64_t head = rem / cols;
    const int64_t col = rem - head * cols;
    if (stream >= q_streams) {
        return;
    }

    float local_max = 0.0f;
    for (int64_t k = threadIdx.x; k < kv_size; k += blockDim.x) {
        const char * p_ptr = (const char *) p_data + k * (int64_t) sizeof(float) + col * p_nb1 + head * p_nb2 + stream * p_nb3;
        local_max = fmaxf(local_max, fabsf(*(const float *) p_ptr));
    }

    __shared__ float shared_max[256];
    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        p_amax[p_row] = shared_max[0];
    }
}

static __global__ void k_quantize_p_rows_nvfp4_dynamic(
        const float * __restrict__ p_data,
        const float * __restrict__ p_amax,
        block_nvfp4 * __restrict__ p_q,
        float * __restrict__ p_scale,
        int64_t kv_size,
        int64_t cols,
        int64_t q_heads,
        int64_t q_streams,
        int64_t p_nb1,
        int64_t p_nb2,
        int64_t p_nb3) {
    const int64_t block = blockIdx.x;
    const int64_t col = blockIdx.y;
    const int64_t head = blockIdx.z % q_heads;
    const int64_t stream = blockIdx.z / q_heads;
    const int lane = threadIdx.x;

    if (block >= kv_size / QK_NVFP4 || col >= cols || head >= q_heads || stream >= q_streams || lane >= WARP_SIZE) {
        return;
    }

    const int64_t p_row = (stream * q_heads + head) * cols + col;
    const int64_t k = block * QK_NVFP4 + lane;
    const bool active = lane < QK_NVFP4 && k < kv_size;

    const char * p_ptr = (const char *) p_data + k * (int64_t) sizeof(float) + col * p_nb1 + head * p_nb2 + stream * p_nb3;
    const float x = active ? *(const float *) p_ptr : 0.0f;

    float vmax = fabsf(x);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    float row_amax = 0.0f;
    if (p_amax != nullptr) {
        row_amax = p_amax[p_row];
    } else {
        for (int64_t i = lane; i < kv_size; i += WARP_SIZE) {
            const char * row_p_ptr = (const char *) p_data + i * (int64_t) sizeof(float) + col * p_nb1 + head * p_nb2 + stream * p_nb3;
            row_amax = fmaxf(row_amax, fabsf(*(const float *) row_p_ptr));
        }
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFF, row_amax, 16, WARP_SIZE));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFF, row_amax, 8, WARP_SIZE));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFF, row_amax, 4, WARP_SIZE));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFF, row_amax, 2, WARP_SIZE));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFF, row_amax, 1, WARP_SIZE));
        row_amax = __shfl_sync(0xFFFFFFFF, row_amax, 0, WARP_SIZE);
    }
    const float global_scale = (row_amax > 0.0f && isfinite(row_amax)) ?
            (GGML_CUDA_VCACHE_NVFP4_GLOBAL_SCALE_MAX / row_amax) : 0.0f;
    if (lane == 0) {
        p_scale[p_row] = global_scale != 0.0f ? (1.0f / global_scale) : 0.0f;
    }

    float scale_f = 0.0f;
    if (lane == 0) {
        const float block_scale = (global_scale != 0.0f) ?
            (global_scale * (vmax / GGML_CUDA_VCACHE_NVFP4_FP4_MAX)) : 0.0f;
        const uint8_t scale_q = ggml_cuda_best_index_e4m3_vcache(block_scale);
        p_q[p_row * (kv_size / QK_NVFP4) + block].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4_vcache(x * inv_scale);
    const uint8_t q_peer = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);

    if (active && (lane & 1) == 0) {
        p_q[p_row * (kv_size / QK_NVFP4) + block].qs[lane / 2] = q | (q_peer << 4);
    }
}

static __global__ void k_vcache_nvfp4_matmul_4d(
        const block_nvfp4 * __restrict__ v_data,
        const float * __restrict__ v_scale,
        const float * __restrict__ p_data,
        float * __restrict__ dst_data,
        int64_t kv_size,
        int64_t rows,
        int64_t cols,
        int64_t kv_heads,
        int64_t q_heads,
        int64_t kv_streams,
        int64_t q_streams,
        int64_t v_nb0,
        int64_t v_nb1,
        int64_t v_nb2,
        int64_t v_nb3,
        int64_t scale_nb0,
        int64_t scale_row_nb,
        int64_t scale_head_nb,
        int64_t scale_stream_nb,
        int64_t p_nb1,
        int64_t p_nb2,
        int64_t p_nb3,
        int64_t dst_nb1,
        int64_t dst_nb2,
        int64_t dst_nb3,
        int64_t r2,
        int64_t r3) {
    const int64_t row = blockIdx.x;
    const int64_t col = blockIdx.y;
    const int64_t head = blockIdx.z % q_heads;
    const int64_t stream = blockIdx.z / q_heads;
    const int64_t kv_head = head / r2;
    const int64_t kv_stream = stream / r3;

    if (row >= rows || col >= cols || head >= q_heads || stream >= q_streams || kv_head >= kv_heads || kv_stream >= kv_streams) {
        return;
    }

    const char * v_base = (const char *) v_data + row * v_nb1 + kv_head * v_nb2 + kv_stream * v_nb3;
    const char * scale_base = (const char *) v_scale + row * scale_row_nb + kv_head * scale_head_nb + kv_stream * scale_stream_nb;

    float thread_sum = 0.0f;
    for (int64_t lane = threadIdx.x; lane < kv_size; lane += blockDim.x) {
        const int64_t block = lane / QK_NVFP4;
        const int64_t in_block = lane % QK_NVFP4;

        const block_nvfp4 * v_block_ptr = (const block_nvfp4 *) (v_base + block * v_nb0);
        const float input_scale = *(const float *) (scale_base + block * scale_nb0);

        const block_nvfp4 vb = *v_block_ptr;
        const float d = ggml_cuda_e4m3_to_fp32_half(vb.e) * input_scale;
        const uint8_t packed = vb.qs[in_block / 2];
        const uint8_t q = (in_block & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
        const float v = d * (float) kvalues_nvfp4[q];

        const char * p_ptr = (const char *) p_data + lane * sizeof(float) + col * p_nb1 + head * p_nb2 + stream * p_nb3;
        const float p = *(const float *) p_ptr;
        thread_sum += v * p;
    }

    __shared__ float sum[256];
    sum[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        char * dst_ptr = (char *) dst_data + row * sizeof(float) + col * dst_nb1 + head * dst_nb2 + stream * dst_nb3;
        *(float *) dst_ptr = sum[0];
    }
}

static __global__ void k_vcache_nvfp4_matmul_fp4_p_4d(
        const block_nvfp4 * __restrict__ v_data,
        const float * __restrict__ v_scale,
        const block_nvfp4 * __restrict__ p_q,
        const float * __restrict__ p_scale,
        float * __restrict__ dst_data,
        int64_t kv_size,
        int64_t rows,
        int64_t cols,
        int64_t kv_heads,
        int64_t q_heads,
        int64_t kv_streams,
        int64_t q_streams,
        int64_t v_nb0,
        int64_t v_nb1,
        int64_t v_nb2,
        int64_t v_nb3,
        int64_t scale_nb0,
        int64_t scale_row_nb,
        int64_t scale_head_nb,
        int64_t scale_stream_nb,
        int64_t dst_nb1,
        int64_t dst_nb2,
        int64_t dst_nb3,
        int64_t r2,
        int64_t r3) {
    const int64_t row = blockIdx.x;
    const int64_t col = blockIdx.y;
    const int64_t head = blockIdx.z % q_heads;
    const int64_t stream = blockIdx.z / q_heads;
    const int64_t kv_head = head / r2;
    const int64_t kv_stream = stream / r3;

    if (row >= rows || col >= cols || head >= q_heads || stream >= q_streams || kv_head >= kv_heads || kv_stream >= kv_streams) {
        return;
    }

    const int64_t n_blocks = kv_size / QK_NVFP4;
    const int64_t p_row = (stream * q_heads + head) * cols + col;
    const char * v_base = (const char *) v_data + row * v_nb1 + kv_head * v_nb2 + kv_stream * v_nb3;
    const char * scale_base = (const char *) v_scale + row * scale_row_nb + kv_head * scale_head_nb + kv_stream * scale_stream_nb;
    const block_nvfp4 * p_row_q = p_q + p_row * n_blocks;
    const float p_row_scale = p_scale[p_row];

    float thread_sum = 0.0f;
    for (int64_t block = threadIdx.x; block < n_blocks; block += blockDim.x) {
        const block_nvfp4 * v_block_ptr = (const block_nvfp4 *) (v_base + block * v_nb0);
        const block_nvfp4 vb = *v_block_ptr;
        const block_nvfp4 pb = p_row_q[block];
        const float v_d = ggml_cuda_e4m3_to_fp32_half(vb.e) * (*(const float *) (scale_base + block * scale_nb0));
        const float p_d = ggml_cuda_e4m3_to_fp32_half(pb.e) * p_row_scale;
        const float d = v_d * p_d;

#pragma unroll
        for (int i = 0; i < QK_NVFP4 / 2; ++i) {
            const uint8_t v_packed = vb.qs[i];
            const uint8_t p_packed = pb.qs[i];
            thread_sum += d * (float) kvalues_nvfp4[v_packed & 0x0F] * (float) kvalues_nvfp4[p_packed & 0x0F];
            thread_sum += d * (float) kvalues_nvfp4[v_packed >> 4]    * (float) kvalues_nvfp4[p_packed >> 4];
        }
    }

    __shared__ float sum[256];
    sum[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        char * dst_ptr = (char *) dst_data + row * sizeof(float) + col * dst_nb1 + head * dst_nb2 + stream * dst_nb3;
        *(float *) dst_ptr = sum[0];
    }
}

bool ggml_cuda_mul_mat_vcache_nvfp4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (!ggml_cuda_is_experimental_vcache_nvfp4_tensor(src0)) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * scale = ggml_tensor_get_nvfp4_scale(src0);
    int64_t blocks = 0;
    int64_t rows = 0;
    int64_t kv_heads = 0;
    int64_t kv_streams = 0;
    int64_t scale_row_nb = 0;
    int64_t scale_head_nb = 0;
    int64_t scale_stream_nb = 0;
    if (!ggml_cuda_match_vcache_nvfp4_scale_layout(src0, scale, blocks, rows, kv_heads, kv_streams, scale_row_nb, scale_head_nb, scale_stream_nb)) {
        return false;
    }

    const int64_t kv_size = src0->ne[0];
    const int64_t cols = src1->ne[1];
    const int64_t q_heads = src1->ne[2];
    const int64_t q_streams = src1->ne[3];

    if (src1->ne[0] != kv_size) {
        return false;
    }

    if (kv_heads <= 0 || kv_streams <= 0 || q_heads % kv_heads != 0 || q_streams % kv_streams != 0) {
        return false;
    }

    if (dst->ne[0] != rows || dst->ne[1] != cols || dst->ne[2] != q_heads || dst->ne[3] != q_streams) {
        return false;
    }

    if (kv_size <= 0) {
        return false;
    }

    if (src0->nb[0] != (int64_t) sizeof(block_nvfp4) || scale->nb[0] != (int64_t) sizeof(float) || src1->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }

    const int64_t r2 = q_heads / kv_heads;
    const int64_t r3 = q_streams / kv_streams;
    int block_threads = 16;
    while (block_threads < kv_size && block_threads < 256) {
        block_threads *= 2;
    }

    const bool fp4_p_enabled = ggml_cuda_vcache_nvfp4_fp4_pv_enabled();
    ggml_cuda_vcache_nvfp4_log_fp4_pv_once(fp4_p_enabled);

    const dim3 grid((uint32_t) rows, (uint32_t) cols, (uint32_t) (q_heads * q_streams));
    if (fp4_p_enabled && kv_size % QK_NVFP4 == 0) {
        const int64_t n_blocks = kv_size / QK_NVFP4;
        const int64_t p_rows = cols * q_heads * q_streams;
        ggml_cuda_pool_alloc<block_nvfp4> p_q(ctx.pool(), (size_t) p_rows * (size_t) n_blocks);
        const bool use_amax_prepass = kv_size >= GGML_CUDA_VCACHE_NVFP4_FP4_P_AMAX_PREPASS_MIN_KV;
        ggml_cuda_pool_alloc<float> p_amax(ctx.pool(), use_amax_prepass ? (size_t) p_rows : 0);
        ggml_cuda_pool_alloc<float> p_scale(ctx.pool(), (size_t) p_rows);
        if (use_amax_prepass) {
            const int p_amax_threads = 256;
            k_p_rows_abs_max_f32<<<(uint32_t) p_rows, p_amax_threads, 0, ctx.stream()>>>(
                    (const float *) src1->data,
                    p_amax.get(),
                    kv_size,
                    cols,
                    q_heads,
                    q_streams,
                    src1->nb[1],
                    src1->nb[2],
                    src1->nb[3]);
            CUDA_CHECK(cudaGetLastError());
        }

        const dim3 q_grid((uint32_t) n_blocks, (uint32_t) cols, (uint32_t) (q_heads * q_streams));
        const dim3 q_block(WARP_SIZE, 1, 1);
        k_quantize_p_rows_nvfp4_dynamic<<<q_grid, q_block, 0, ctx.stream()>>>(
                (const float *) src1->data,
                use_amax_prepass ? p_amax.get() : nullptr,
                p_q.get(),
                p_scale.get(),
                kv_size,
                cols,
                q_heads,
                q_streams,
                src1->nb[1],
                src1->nb[2],
                src1->nb[3]);
        CUDA_CHECK(cudaGetLastError());

        int fp4_block_threads = 16;
        while (fp4_block_threads < n_blocks && fp4_block_threads < 256) {
            fp4_block_threads *= 2;
        }
        k_vcache_nvfp4_matmul_fp4_p_4d<<<grid, dim3((uint32_t) fp4_block_threads, 1, 1), 0, ctx.stream()>>>(
                (const block_nvfp4 *) src0->data,
                (const float *) scale->data,
                p_q.get(),
                p_scale.get(),
                (float *) dst->data,
                kv_size,
                rows,
                cols,
                kv_heads,
                q_heads,
                kv_streams,
                q_streams,
                src0->nb[0],
                src0->nb[1],
                src0->nb[2],
                src0->nb[3],
                scale->nb[0],
                scale_row_nb,
                scale_head_nb,
                scale_stream_nb,
                dst->nb[1],
                dst->nb[2],
                dst->nb[3],
                r2,
                r3);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    const dim3 block((uint32_t) block_threads, 1, 1);
    k_vcache_nvfp4_matmul_4d<<<grid, block, 0, ctx.stream()>>>(
            (const block_nvfp4 *) src0->data,
            (const float *) scale->data,
            (const float *) src1->data,
            (float *) dst->data,
            kv_size,
            rows,
            cols,
            kv_heads,
            q_heads,
            kv_streams,
            q_streams,
            src0->nb[0],
            src0->nb[1],
            src0->nb[2],
            src0->nb[3],
            scale->nb[0],
            scale_row_nb,
            scale_head_nb,
            scale_stream_nb,
            src1->nb[1],
            src1->nb[2],
            src1->nb[3],
            dst->nb[1],
            dst->nb[2],
            dst->nb[3],
            r2,
            r3);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
