#include "fattn-nvfp4.cuh"

#include "nvfp4-matmul.cuh"
#include "ggml-quants.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static constexpr float FP8_E4M3FN_MAX = 448.0f;
static constexpr float FP4_E2M1_MAX   = 6.0f;

static bool env_enabled(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 && std::strcmp(v, "FALSE") != 0;
}

static float max_abs(const std::vector<float> & x) {
    float amax = 0.0f;
    for (float v : x) {
        amax = std::max(amax, std::fabs(v));
    }
    return amax;
}

static __host__ __device__ float alibi_slope(float max_bias, int64_t head, int64_t n_head) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    uint32_t n_head_log2 = 1;
    while ((int64_t) (n_head_log2 << 1) <= n_head) {
        n_head_log2 <<= 1;
    }

    const float m0 = powf(2.0f, -max_bias / (float) n_head_log2);
    const float m1 = powf(2.0f, -max_bias / 2.0f / (float) n_head_log2);
    const float base = head < (int64_t) n_head_log2 ? m0 : m1;
    const int exph = head < (int64_t) n_head_log2 ? (int) head + 1 : 2 * ((int) head - (int) n_head_log2) + 1;

    return powf(base, exph);
}

static void nvfp4_quant_dequant_rows(
        const std::vector<float> & src,
        std::vector<float> & dst,
        int64_t rows,
        int64_t cols,
        float global_scale_inv) {
    GGML_ASSERT(cols % QK_NVFP4 == 0);
    const int64_t nblk = cols / QK_NVFP4;
    std::vector<block_nvfp4> q((size_t) rows * (size_t) nblk);
    dst.assign(src.size(), 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
        quantize_row_nvfp4_ref(src.data() + (size_t) r * (size_t) cols, q.data() + (size_t) r * (size_t) nblk, cols, global_scale_inv);
        dequantize_row_nvfp4(q.data() + (size_t) r * (size_t) nblk, dst.data() + (size_t) r * (size_t) cols, cols, global_scale_inv);
    }
}

static float read_f32_or_f16(const std::vector<uint8_t> & data, const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const size_t off = (size_t) (i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
    if (t->type == GGML_TYPE_F32) {
        return *(const float *) (data.data() + off);
    }
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (data.data() + off));
    }
    GGML_ABORT("%s: unsupported tensor type %s", __func__, ggml_type_name(t->type));
}

static size_t tensor_data_span_bytes(const ggml_tensor * t) {
    if (ggml_is_empty(t)) {
        return 0;
    }
    size_t span = ggml_element_size(t);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] > 0) {
            span += (size_t) (t->ne[i] - 1) * (size_t) t->nb[i];
        }
    }
    return span;
}

static float read_f16(const std::vector<uint8_t> & data, const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    GGML_ASSERT(t->type == GGML_TYPE_F16);
    const size_t off = (size_t) (i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
    return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (data.data() + off));
}

static void write_f32(std::vector<uint8_t> & data, const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, float value) {
    const size_t off = (size_t) (i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
    *(float *) (data.data() + off) = value;
}

static std::vector<int64_t> infer_visible_kv_lens(
        const std::vector<uint8_t> & mask_raw,
        const ggml_tensor * mask,
        int64_t batch,
        int64_t q_len,
        int64_t kv_len) {
    std::vector<int64_t> visible((size_t) batch, kv_len);
    if (mask == nullptr) {
        return visible;
    }

    std::fill(visible.begin(), visible.end(), 0);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t mh = 0; mh < mask->ne[2]; ++mh) {
            for (int64_t qt = 0; qt < q_len; ++qt) {
                for (int64_t kt = 0; kt < kv_len; ++kt) {
                    const float mv = read_f16(mask_raw, mask, kt, qt, mh, b % mask->ne[3]);
                    if (mv != -INFINITY) {
                        visible[(size_t) b] = std::max<int64_t>(visible[(size_t) b], kt + 1);
                    }
                }
            }
        }
        if (visible[(size_t) b] == 0) {
            visible[(size_t) b] = kv_len;
        }
    }
    return visible;
}

static __device__ __forceinline__ float read_tensor_f32_or_f16_device(
        const void * data,
        int type,
        int64_t nb0,
        int64_t nb1,
        int64_t nb2,
        int64_t nb3,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3) {
    const char * ptr = (const char *) data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3;
    if (type == GGML_TYPE_F32) {
        return *(const float *) ptr;
    }
    return __half2float(*(const half *) ptr);
}

static __device__ __forceinline__ void atomic_max_f32_positive(float * addr, float value) {
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

static __global__ void absmax_kernel(const float * x, float * out, int64_t n) {
    float local = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        local = fmaxf(local, fabsf(x[i]));
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomic_max_f32_positive(out, smem[0]);
    }
}

static __global__ void q_smooth_kernel(
        const void * q,
        int q_type,
        int64_t q_nb0,
        int64_t q_nb1,
        int64_t q_nb2,
        int64_t q_nb3,
        float * q_centered,
        float * q_mean,
        int64_t d,
        int64_t q_len,
        int64_t q_heads) {
    const int64_t i = blockIdx.x;
    const int64_t block = blockIdx.y;
    const int64_t bh = blockIdx.z;
    const int64_t b = bh / q_heads;
    const int64_t h = bh - b*q_heads;
    const int64_t block_start = block * 128;
    const int64_t block_end = min(block_start + 128, q_len);

    float v = 0.0f;
    const int64_t t = block_start + threadIdx.x;
    if (threadIdx.x < 128 && t < block_end) {
        v = read_tensor_f32_or_f16_device(q, q_type, q_nb0, q_nb1, q_nb2, q_nb3, i, t, h, b);
    }

    __shared__ float smem[128];
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float mean = smem[0] / 128.0f;
    if (threadIdx.x < 128 && t < block_end) {
        const int64_t row = (b*q_heads + h)*q_len + t;
        q_mean[row*d + i] = mean;
        q_centered[row*d + i] = v - mean;
    }
}

static __global__ void k_smooth_kernel(
        const void * k,
        int k_type,
        int64_t k_nb0,
        int64_t k_nb1,
        int64_t k_nb2,
        int64_t k_nb3,
        const int64_t * visible_lens,
        float * k_centered,
        int64_t d,
        int64_t kv_len,
        int64_t kv_heads) {
    const int64_t i = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t b = bh / kv_heads;
    const int64_t h = bh - b*kv_heads;
    const int64_t visible = visible_lens[b];

    float local = 0.0f;
    for (int64_t t = threadIdx.x; t < visible; t += blockDim.x) {
        local += read_tensor_f32_or_f16_device(k, k_type, k_nb0, k_nb1, k_nb2, k_nb3, i, t, h, b);
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float mean = visible > 0 ? smem[0] / (float) visible : 0.0f;
    for (int64_t t = threadIdx.x; t < kv_len; t += blockDim.x) {
        const int64_t row = (b*kv_heads + h)*kv_len + t;
        const float kv = t < visible ? read_tensor_f32_or_f16_device(k, k_type, k_nb0, k_nb1, k_nb2, k_nb3, i, t, h, b) - mean : 0.0f;
        k_centered[row*d + i] = kv;
    }
}

static __global__ void v_by_dim_kernel(
        const void * v,
        int v_type,
        int64_t v_nb0,
        int64_t v_nb1,
        int64_t v_nb2,
        int64_t v_nb3,
        const int64_t * visible_lens,
        float * v_by_dim,
        int64_t d,
        int64_t kv_len,
        int64_t kv_heads,
        int64_t total) {
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t) blockDim.x * gridDim.x) {
        int64_t rem = idx;
        const int64_t t = rem % kv_len;
        rem /= kv_len;
        const int64_t i = rem % d;
        rem /= d;
        const int64_t h = rem % kv_heads;
        const int64_t b = rem / kv_heads;
        const int64_t visible = visible_lens[b];
        v_by_dim[idx] = t < visible ? read_tensor_f32_or_f16_device(v, v_type, v_nb0, v_nb1, v_nb2, v_nb3, i, t, h, b) : 0.0f;
    }
}

static __global__ void qmean_kcorr_kernel(
        const float * q_mean,
        const float * k_centered,
        float * corr,
        int64_t d,
        int64_t q_len,
        int64_t q_heads,
        int64_t kv_len,
        int64_t kv_heads,
        int64_t gqa_ratio) {
    const int64_t kt = blockIdx.x;
    const int64_t qt = blockIdx.y;
    const int64_t bh = blockIdx.z;
    const int64_t b = bh / q_heads;
    const int64_t qh = bh - b*q_heads;
    const int64_t kh = qh / gqa_ratio;

    float local = 0.0f;
    const int64_t qrow = (b*q_heads + qh)*q_len + qt;
    const int64_t krow = (b*kv_heads + kh)*kv_len + kt;
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        local += q_mean[qrow*d + i] * k_centered[krow*d + i];
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        corr[bh*kv_len*q_len + kt + qt*kv_len] = smem[0];
    }
}

static __global__ void softmax_kernel(
        const float * qk,
        const float * corr,
        const half * mask,
        float * probs,
        int64_t mask_nb1_el,
        int64_t mask_nb2_el,
        int64_t mask_nb3_el,
        int64_t mask_heads,
        int64_t mask_batch,
        int64_t q_len,
        int64_t q_heads,
        int64_t kv_len,
        float scale,
        float max_bias,
        float logit_softcap) {
    const int64_t qt = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t b = bh / q_heads;
    const int64_t qh = bh - b*q_heads;
    const float slope = alibi_slope(max_bias, qh, q_heads);

    float local_max = -INFINITY;
    for (int64_t kt = threadIdx.x; kt < kv_len; kt += blockDim.x) {
        float score = (qk[bh*kv_len*q_len + kt + qt*kv_len] + corr[bh*kv_len*q_len + kt + qt*kv_len]) * scale;
        if (logit_softcap != 0.0f) {
            score = logit_softcap * tanhf(score);
        }
        if (mask != nullptr) {
            const int64_t mh = qh % mask_heads;
            const int64_t mb = b % mask_batch;
            score += slope * __half2float(mask[kt + qt*mask_nb1_el + mh*mask_nb2_el + mb*mask_nb3_el]);
        }
        local_max = fmaxf(local_max, score);
        probs[bh*q_len*kv_len + qt*kv_len + kt] = score;
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = smem[0];

    float local_sum = 0.0f;
    for (int64_t kt = threadIdx.x; kt < kv_len; kt += blockDim.x) {
        const int64_t idx = bh*q_len*kv_len + qt*kv_len + kt;
        const float p = expf(probs[idx] - row_max);
        probs[idx] = p;
        local_sum += p;
    }

    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_sum = smem[0] > 0.0f ? 1.0f / smem[0] : 0.0f;

    for (int64_t kt = threadIdx.x; kt < kv_len; kt += blockDim.x) {
        const int64_t idx = bh*q_len*kv_len + qt*kv_len + kt;
        probs[idx] *= inv_sum;
    }
}

static __global__ void write_vp_output_kernel(
        const float * vp,
        const float * p_first_scales,
        float * dst,
        int64_t dst_nb1_el,
        int64_t dst_nb2_el,
        int64_t dst_nb3_el,
        int64_t d,
        int64_t q_len,
        int64_t q_heads,
        int64_t total) {
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t) blockDim.x * gridDim.x) {
        int64_t rem = idx;
        const int64_t i = rem % d;
        rem /= d;
        const int64_t qt = rem % q_len;
        rem /= q_len;
        const int64_t qh = rem % q_heads;
        const int64_t b = rem / q_heads;
        const int64_t bh = b*q_heads + qh;
        dst[i + qh*dst_nb1_el + qt*dst_nb2_el + b*dst_nb3_el] =
                vp[bh*d*q_len + i + qt*d] * p_first_scales[bh*q_len + qt];
    }
}

static __global__ void probs_twolevel_scale_kernel(
        const float * probs,
        float * probs_scaled,
        float * first_scales,
        int64_t q_len,
        int64_t kv_len) {
    const int64_t qt = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t row = bh*q_len + qt;

    float local_max = 0.0f;
    for (int64_t kt = threadIdx.x; kt < kv_len; kt += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(probs[row*kv_len + kt]));
    }

    __shared__ float smem[256];
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float first_scale = smem[0] > 0.0f ? smem[0] / (FP8_E4M3FN_MAX * FP4_E2M1_MAX) : 0.0f;
    if (threadIdx.x == 0) {
        first_scales[row] = first_scale;
    }
    const float inv_scale = first_scale > 0.0f ? 1.0f / first_scale : 0.0f;
    for (int64_t kt = threadIdx.x; kt < kv_len; kt += blockDim.x) {
        probs_scaled[row*kv_len + kt] = probs[row*kv_len + kt] * inv_scale;
    }
}

static ggml_tensor make_cuda_temp_tensor_2d(ggml_type type, void * data, int64_t ne0, int64_t ne1) {
    ggml_tensor t = {};
    t.type = type;
    t.op = GGML_OP_NONE;
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = ggml_row_size(type, ne0);
    t.nb[2] = t.nb[1] * ne1;
    t.nb[3] = t.nb[2];
    t.data = data;
    return t;
}

static ggml_tensor make_cuda_mul_mat_dst(
        float * data,
        int64_t ne0,
        int64_t ne1,
        const ggml_tensor * weight_scale,
        const ggml_tensor * input_scale = nullptr) {
    ggml_tensor t = make_cuda_temp_tensor_2d(GGML_TYPE_F32, data, ne0, ne1);
    t.op = GGML_OP_MUL_MAT;
    if (weight_scale != nullptr) {
        ggml_mul_mat_set_nvfp4_weight_scale(&t, weight_scale);
    }
    if (input_scale != nullptr) {
        ggml_mul_mat_set_nvfp4_input_scale(&t, input_scale);
    }
    return t;
}

static ggml_tensor make_host_scalar_tensor_f32(float * value) {
    ggml_tensor t = {};
    t.type = GGML_TYPE_F32;
    t.ne[0] = 1;
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = ggml_type_size(t.type);
    t.nb[1] = sizeof(float);
    t.nb[2] = sizeof(float);
    t.nb[3] = sizeof(float);
    t.data = value;
    return t;
}

static float device_absmax(
        ggml_backend_cuda_context & ctx,
        const float * x,
        int64_t n,
        cudaStream_t stream) {
    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(tmp.get(), 0, sizeof(float), stream));
    const int block_size = 256;
    const int grid_size = (int) std::min<int64_t>(1024, std::max<int64_t>(1, (n + block_size - 1) / block_size));
    absmax_kernel<<<grid_size, block_size, 0, stream>>>(x, tmp.get(), n);
    CUDA_CHECK(cudaGetLastError());

    float out = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&out, tmp.get(), sizeof(out), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
}

static bool ggml_cuda_flash_attn_ext_nvfp4_gpu_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if GGML_CUDA_HAS_CUBLASLT && GGML_CUDA_HAS_FP4 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (!env_enabled("GGML_CUDA_NVFP4_FATTN")) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (q == nullptr || k == nullptr || v == nullptr || q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!((k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32) &&
          (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32))) {
        return false;
    }
    if (q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0] || q->ne[0] % QK_NVFP4 != 0 || k->ne[1] % QK_NVFP4 != 0) {
        return false;
    }
    if (q->ne[2] % k->ne[2] != 0 || q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    if (!ggml_is_contiguous(dst)) {
        return false;
    }
    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F16 || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1]) {
            return false;
        }
        if (q->ne[2] % mask->ne[2] != 0 || q->ne[3] % mask->ne[3] != 0) {
            return false;
        }
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int64_t d        = q->ne[0];
    const int64_t q_len    = q->ne[1];
    const int64_t q_heads  = q->ne[2];
    const int64_t batch    = q->ne[3];
    const int64_t kv_len   = k->ne[1];
    const int64_t kv_heads = k->ne[2];
    const int64_t gqa_ratio = q_heads / kv_heads;

    cudaStream_t stream = ctx.stream();
    const bool debug_log = env_enabled("GGML_CUDA_NVFP4_FATTN_DEBUG");

    std::vector<uint8_t> mask_raw(mask != nullptr ? tensor_data_span_bytes(mask) : 0);
    if (mask != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(mask_raw.data(), mask->data, mask_raw.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const std::vector<int64_t> visible_kv_lens = infer_visible_kv_lens(mask_raw, mask, batch, q_len, kv_len);

    ggml_cuda_pool_alloc<int64_t> d_visible(ctx.pool(), (size_t) batch);
    CUDA_CHECK(cudaMemcpyAsync(d_visible.get(), visible_kv_lens.data(), (size_t) batch * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    const int64_t q_rows = batch*q_heads*q_len;
    const int64_t k_rows = batch*kv_heads*kv_len;
    const int64_t v_rows = batch*kv_heads*d;
    const int64_t bh_rows = batch*q_heads;

    ggml_cuda_pool_alloc<float> q_centered(ctx.pool(), (size_t) q_rows * (size_t) d);
    ggml_cuda_pool_alloc<float> q_mean(ctx.pool(), (size_t) q_rows * (size_t) d);
    ggml_cuda_pool_alloc<float> k_centered(ctx.pool(), (size_t) k_rows * (size_t) d);
    ggml_cuda_pool_alloc<float> v_by_dim(ctx.pool(), (size_t) v_rows * (size_t) kv_len);

    q_smooth_kernel<<<dim3((uint32_t) d, (uint32_t) ((q_len + 127) / 128), (uint32_t) (batch*q_heads)), 128, 0, stream>>>(
            q->data, q->type,
            (int64_t) q->nb[0], (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
            q_centered.get(), q_mean.get(), d, q_len, q_heads);
    CUDA_CHECK(cudaGetLastError());

    k_smooth_kernel<<<dim3((uint32_t) d, (uint32_t) (batch*kv_heads), 1), 256, 0, stream>>>(
            k->data, k->type,
            (int64_t) k->nb[0], (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
            d_visible.get(), k_centered.get(), d, kv_len, kv_heads);
    CUDA_CHECK(cudaGetLastError());

    const int block_size = 256;
    const int64_t v_total = v_rows * kv_len;
    v_by_dim_kernel<<<(int) std::min<int64_t>(1024, (v_total + block_size - 1) / block_size), block_size, 0, stream>>>(
            v->data, v->type,
            (int64_t) v->nb[0], (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],
            d_visible.get(), v_by_dim.get(), d, kv_len, kv_heads, v_total);
    CUDA_CHECK(cudaGetLastError());

    const float q_amax = device_absmax(ctx, q_centered.get(), q_rows*d, stream);
    const float k_amax = device_absmax(ctx, k_centered.get(), k_rows*d, stream);
    const float v_amax = device_absmax(ctx, v_by_dim.get(), v_rows*kv_len, stream);
    const float q_global_scale = q_amax > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / q_amax : 0.0f;
    const float k_global_scale = k_amax > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / k_amax : 0.0f;
    const float v_global_scale = v_amax > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / v_amax : 0.0f;
    if (q_global_scale == 0.0f || k_global_scale == 0.0f || v_global_scale == 0.0f) {
        return false;
    }

    if (debug_log) {
        GGML_LOG_INFO(
                "%s: NVFP4 FATTN prefill quantization: "
                "Q/K group_dim=head_dim group_size=%d tensor_global_scale_inv=[q=%g k=%g] "
                "V group_dim=kv_len group_size=%d tensor_global_scale_inv=%g "
                "P format=nvfp4_twolevel group_dim=kv_len first_level=row_max/(448*6) second_level=NVFP4(global_scale_inv=1) "
                "shape=[batch=%lld q_heads=%lld kv_heads=%lld q_len=%lld kv_len=%lld head_dim=%lld]\n",
                __func__,
                QK_NVFP4,
                (double) q_global_scale,
                (double) k_global_scale,
                QK_NVFP4,
                (double) v_global_scale,
                (long long) batch,
                (long long) q_heads,
                (long long) kv_heads,
                (long long) q_len,
                (long long) kv_len,
                (long long) d);
    }

    ggml_cuda_pool_alloc<block_nvfp4> k_q(ctx.pool(), (size_t) k_rows * (size_t) (d / QK_NVFP4));
    ggml_cuda_pool_alloc<block_nvfp4> v_q(ctx.pool(), (size_t) v_rows * (size_t) (kv_len / QK_NVFP4));
    ggml_cuda_nvfp4_quantize_rows_f32(k_centered.get(), k_q.get(), d, d, k_rows, k_global_scale, stream);
    ggml_cuda_nvfp4_quantize_rows_f32(v_by_dim.get(), v_q.get(), kv_len, kv_len, v_rows, v_global_scale, stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<float> qk(ctx.pool(), (size_t) bh_rows * (size_t) kv_len * (size_t) q_len);
    ggml_cuda_pool_alloc<float> corr(ctx.pool(), (size_t) bh_rows * (size_t) kv_len * (size_t) q_len);
    ggml_cuda_pool_alloc<float> probs(ctx.pool(), (size_t) bh_rows * (size_t) q_len * (size_t) kv_len);
    ggml_cuda_pool_alloc<float> probs_scaled(ctx.pool(), (size_t) bh_rows * (size_t) q_len * (size_t) kv_len);
    ggml_cuda_pool_alloc<float> p_first_scales(ctx.pool(), (size_t) bh_rows * (size_t) q_len);
    ggml_cuda_pool_alloc<float> vp(ctx.pool(), (size_t) bh_rows * (size_t) d * (size_t) q_len);

    qmean_kcorr_kernel<<<dim3((uint32_t) kv_len, (uint32_t) q_len, (uint32_t) bh_rows), 256, 0, stream>>>(
            q_mean.get(), k_centered.get(), corr.get(), d, q_len, q_heads, kv_len, kv_heads, gqa_ratio);
    CUDA_CHECK(cudaGetLastError());

    float q_input_scale_value = 1.0f / q_global_scale;
    float k_weight_scale_value = 1.0f / k_global_scale;
    ggml_tensor q_input_scale = make_host_scalar_tensor_f32(&q_input_scale_value);
    ggml_tensor k_weight_scale = make_host_scalar_tensor_f32(&k_weight_scale_value);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t qh = 0; qh < q_heads; ++qh) {
            const int64_t kh = qh / gqa_ratio;
            block_nvfp4 * k_ptr = k_q.get() + ((b*kv_heads + kh)*kv_len) * (d / QK_NVFP4);
            float * q_ptr = q_centered.get() + ((b*q_heads + qh)*q_len) * d;
            float * qk_ptr = qk.get() + (b*q_heads + qh)*kv_len*q_len;

            ggml_tensor k_t = make_cuda_temp_tensor_2d(GGML_TYPE_NVFP4, k_ptr, d, kv_len);
            ggml_tensor q_t = make_cuda_temp_tensor_2d(GGML_TYPE_F32, q_ptr, d, q_len);
            ggml_tensor qk_t = make_cuda_mul_mat_dst(qk_ptr, kv_len, q_len, &k_weight_scale, &q_input_scale);
            ggml_set_name(&k_t, "nvfp4-fattn-k");
            ggml_set_name(&q_t, "nvfp4-fattn-q");
            ggml_set_name(&qk_t, "nvfp4-fattn-qk");
            if (debug_log && b == 0 && qh == 0) {
                GGML_LOG_INFO(
                        "%s: QK matmul requested: backend=cublasLt tensor_core=FP4 lt_type=CUDA_R_4F_E2M1 "
                        "A=K_centered[NVFP4,k=%lld,m=%lld,weight_scale=1/k_global_scale_inv=%g] "
                        "B=Q_centered[F32->NVFP4,k=%lld,n=%lld,input_scale=1/q_global_scale_inv=%g] "
                        "C=F32[%lld,%lld]\n",
                        __func__,
                        (long long) d,
                        (long long) kv_len,
                        (double) k_weight_scale_value,
                        (long long) d,
                        (long long) q_len,
                        (double) q_input_scale_value,
                        (long long) kv_len,
                        (long long) q_len);
            }
            if (!ggml_cuda_mul_mat_nvfp4_native(ctx, &k_t, &q_t, &qk_t)) {
                if (debug_log) {
                    GGML_LOG_WARN("%s: QK matmul did not use native Tensor Core FP4 path; GPU prefill will fallback\n", __func__);
                }
                return false;
            }
            if (debug_log && b == 0 && qh == 0) {
                GGML_LOG_INFO("%s: QK matmul active: native Tensor Core FP4 path confirmed (cublasLt CUDA_R_4F_E2M1)\n", __func__);
            }
        }
    }

    softmax_kernel<<<dim3((uint32_t) q_len, (uint32_t) bh_rows, 1), 256, 0, stream>>>(
            qk.get(), corr.get(), mask != nullptr ? (const half *) mask->data : nullptr, probs.get(),
            mask != nullptr ? (int64_t) (mask->nb[1] / sizeof(half)) : 0,
            mask != nullptr ? (int64_t) (mask->nb[2] / sizeof(half)) : 0,
            mask != nullptr ? (int64_t) (mask->nb[3] / sizeof(half)) : 0,
            mask != nullptr ? mask->ne[2] : 1,
            mask != nullptr ? mask->ne[3] : 1,
            q_len, q_heads, kv_len, scale, max_bias, logit_softcap);
    CUDA_CHECK(cudaGetLastError());

    probs_twolevel_scale_kernel<<<dim3((uint32_t) q_len, (uint32_t) bh_rows, 1), 256, 0, stream>>>(
            probs.get(), probs_scaled.get(), p_first_scales.get(), q_len, kv_len);
    CUDA_CHECK(cudaGetLastError());

    float v_weight_scale_value = 1.0f / v_global_scale;
    float p_input_scale_value = 1.0f;
    ggml_tensor v_weight_scale = make_host_scalar_tensor_f32(&v_weight_scale_value);
    ggml_tensor p_input_scale = make_host_scalar_tensor_f32(&p_input_scale_value);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t qh = 0; qh < q_heads; ++qh) {
            const int64_t kh = qh / gqa_ratio;
            block_nvfp4 * v_ptr = v_q.get() + ((b*kv_heads + kh)*d) * (kv_len / QK_NVFP4);
            float * p_ptr = probs_scaled.get() + (b*q_heads + qh)*q_len*kv_len;
            float * vp_ptr = vp.get() + (b*q_heads + qh)*d*q_len;

            ggml_tensor v_t = make_cuda_temp_tensor_2d(GGML_TYPE_NVFP4, v_ptr, kv_len, d);
            ggml_tensor p_t = make_cuda_temp_tensor_2d(GGML_TYPE_F32, p_ptr, kv_len, q_len);
            ggml_tensor vp_t = make_cuda_mul_mat_dst(vp_ptr, d, q_len, &v_weight_scale, &p_input_scale);
            ggml_set_name(&v_t, "nvfp4-fattn-v");
            ggml_set_name(&p_t, "nvfp4-fattn-p");
            ggml_set_name(&vp_t, "nvfp4-fattn-vp");
            if (debug_log && b == 0 && qh == 0) {
                GGML_LOG_INFO(
                        "%s: VP matmul requested: backend=cublasLt tensor_core=FP4 lt_type=CUDA_R_4F_E2M1 "
                        "A=V[NVFP4,k=%lld,m=%lld,weight_scale=1/v_global_scale_inv=%g] "
                        "B=P_scaled[F32->NVFP4,k=%lld,n=%lld,input_scale=1,twolevel_first_scale_applied_after_matmul] "
                        "C=F32[%lld,%lld]\n",
                        __func__,
                        (long long) kv_len,
                        (long long) d,
                        (double) v_weight_scale_value,
                        (long long) kv_len,
                        (long long) q_len,
                        (long long) d,
                        (long long) q_len);
            }
            if (!ggml_cuda_mul_mat_nvfp4_native(ctx, &v_t, &p_t, &vp_t)) {
                if (debug_log) {
                    GGML_LOG_WARN("%s: VP matmul did not use native Tensor Core FP4 path; GPU prefill will fallback\n", __func__);
                }
                return false;
            }
            if (debug_log && b == 0 && qh == 0) {
                GGML_LOG_INFO("%s: VP matmul active: native Tensor Core FP4 path confirmed (cublasLt CUDA_R_4F_E2M1)\n", __func__);
            }
        }
    }

    const int64_t out_total = batch*q_heads*q_len*d;
    write_vp_output_kernel<<<(int) std::min<int64_t>(1024, (out_total + block_size - 1) / block_size), block_size, 0, stream>>>(
            vp.get(), p_first_scales.get(), (float *) dst->data,
            (int64_t) (dst->nb[1] / sizeof(float)),
            (int64_t) (dst->nb[2] / sizeof(float)),
            (int64_t) (dst->nb[3] / sizeof(float)),
            d, q_len, q_heads, out_total);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#endif
}

static bool ggml_cuda_flash_attn_ext_nvfp4_ref(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!env_enabled("GGML_CUDA_NVFP4_FATTN")) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    if (q == nullptr || k == nullptr || v == nullptr) {
        return false;
    }
    if (q->type != GGML_TYPE_F32) {
        return false;
    }
    if (!((k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32) &&
          (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32))) {
        return false;
    }
    if (q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0] || q->ne[0] % QK_NVFP4 != 0 || k->ne[1] % QK_NVFP4 != 0) {
        return false;
    }
    if (q->ne[2] % k->ne[2] != 0 || q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    if (mask != nullptr) {
        if (mask->type != GGML_TYPE_F16 || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1]) {
            return false;
        }
        if (q->ne[2] % mask->ne[2] != 0 || q->ne[3] % mask->ne[3] != 0) {
            return false;
        }
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    GGML_UNUSED(ctx);
    std::vector<uint8_t> q_raw(tensor_data_span_bytes(q));
    std::vector<uint8_t> k_raw(tensor_data_span_bytes(k));
    std::vector<uint8_t> v_raw(tensor_data_span_bytes(v));
    std::vector<uint8_t> mask_raw(mask != nullptr ? tensor_data_span_bytes(mask) : 0);
    std::vector<uint8_t> out_raw(ggml_nbytes(dst), 0);

    CUDA_CHECK(cudaMemcpy(q_raw.data(), q->data, q_raw.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k_raw.data(), k->data, k_raw.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_raw.data(), v->data, v_raw.size(), cudaMemcpyDeviceToHost));
    if (mask != nullptr) {
        CUDA_CHECK(cudaMemcpy(mask_raw.data(), mask->data, mask_raw.size(), cudaMemcpyDeviceToHost));
    }

    const int64_t d       = q->ne[0];
    const int64_t q_len   = q->ne[1];
    const int64_t q_heads = q->ne[2];
    const int64_t batch   = q->ne[3];
    const int64_t kv_len  = k->ne[1];
    const int64_t kv_heads = k->ne[2];
    const int64_t gqa_ratio = q_heads / kv_heads;
    const std::vector<int64_t> visible_kv_lens = infer_visible_kv_lens(mask_raw, mask, batch, q_len, kv_len);

    std::vector<float> q_centered((size_t) batch * (size_t) q_heads * (size_t) q_len * (size_t) d);
    std::vector<float> q_mean(q_centered.size());
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < q_heads; ++h) {
            for (int64_t block = 0; block < (q_len + 127) / 128; ++block) {
                const int64_t block_start = block * 128;
                const int64_t block_end = std::min<int64_t>(block_start + 128, q_len);
                for (int64_t i = 0; i < d; ++i) {
                    float mean = 0.0f;
                    for (int64_t t = block_start; t < block_end; ++t) {
                        mean += read_f32_or_f16(q_raw, q, i, t, h, b);
                    }
                    mean /= 128.0f;
                    for (int64_t t = block_start; t < block_end; ++t) {
                        const size_t idx = (((size_t) b * (size_t) q_heads + (size_t) h) * (size_t) q_len + (size_t) t) * (size_t) d + (size_t) i;
                        q_mean[idx] = mean;
                        q_centered[idx] = read_f32_or_f16(q_raw, q, i, t, h, b) - mean;
                    }
                }
            }
        }
    }

    std::vector<float> k_centered((size_t) batch * (size_t) kv_heads * (size_t) kv_len * (size_t) d);
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t visible_kv_len = visible_kv_lens[(size_t) b];
        for (int64_t h = 0; h < kv_heads; ++h) {
            for (int64_t i = 0; i < d; ++i) {
                float mean = 0.0f;
                for (int64_t t = 0; t < visible_kv_len; ++t) {
                    mean += read_f32_or_f16(k_raw, k, i, t, h, b);
                }
                mean /= (float) visible_kv_len;
                for (int64_t t = 0; t < visible_kv_len; ++t) {
                    const size_t idx = (((size_t) b * (size_t) kv_heads + (size_t) h) * (size_t) kv_len + (size_t) t) * (size_t) d + (size_t) i;
                    k_centered[idx] = read_f32_or_f16(k_raw, k, i, t, h, b) - mean;
                }
                for (int64_t t = visible_kv_len; t < kv_len; ++t) {
                    const size_t idx = (((size_t) b * (size_t) kv_heads + (size_t) h) * (size_t) kv_len + (size_t) t) * (size_t) d + (size_t) i;
                    k_centered[idx] = 0.0f;
                }
            }
        }
    }

    const float q_gscale_inv = max_abs(q_centered) > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(q_centered) : 0.0f;
    const float k_gscale_inv = max_abs(k_centered) > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(k_centered) : 0.0f;

    std::vector<float> q_deq;
    nvfp4_quant_dequant_rows(q_centered, q_deq, batch * q_heads * q_len, d, q_gscale_inv);

    std::vector<float> k_deq;
    nvfp4_quant_dequant_rows(k_centered, k_deq, batch * kv_heads * kv_len, d, k_gscale_inv);

    std::vector<float> v_by_dim((size_t) batch * (size_t) kv_heads * (size_t) d * (size_t) kv_len);
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t visible_kv_len = visible_kv_lens[(size_t) b];
        for (int64_t h = 0; h < kv_heads; ++h) {
            for (int64_t i = 0; i < d; ++i) {
                for (int64_t t = 0; t < visible_kv_len; ++t) {
                    const size_t idx = (((size_t) b * (size_t) kv_heads + (size_t) h) * (size_t) d + (size_t) i) * (size_t) kv_len + (size_t) t;
                    v_by_dim[idx] = read_f32_or_f16(v_raw, v, i, t, h, b);
                }
                for (int64_t t = visible_kv_len; t < kv_len; ++t) {
                    const size_t idx = (((size_t) b * (size_t) kv_heads + (size_t) h) * (size_t) d + (size_t) i) * (size_t) kv_len + (size_t) t;
                    v_by_dim[idx] = 0.0f;
                }
            }
        }
    }
    const float v_gscale_inv = max_abs(v_by_dim) > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / max_abs(v_by_dim) : 0.0f;
    std::vector<float> v_deq;
    nvfp4_quant_dequant_rows(v_by_dim, v_deq, batch * kv_heads * d, kv_len, v_gscale_inv);

    std::vector<float> scores((size_t) kv_len);
    std::vector<float> probs((size_t) kv_len);
    std::vector<float> probs_deq;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t qh = 0; qh < q_heads; ++qh) {
            const int64_t kh = qh / gqa_ratio;
            const float slope = alibi_slope(max_bias, qh, q_heads);
            for (int64_t qt = 0; qt < q_len; ++qt) {
                float score_max = -INFINITY;
                for (int64_t kt = 0; kt < kv_len; ++kt) {
                    float main = 0.0f;
                    float corr = 0.0f;
                    const size_t qrow = (((size_t) b * (size_t) q_heads + (size_t) qh) * (size_t) q_len + (size_t) qt) * (size_t) d;
                    const size_t krow = (((size_t) b * (size_t) kv_heads + (size_t) kh) * (size_t) kv_len + (size_t) kt) * (size_t) d;
                    for (int64_t i = 0; i < d; ++i) {
                        main += q_deq[qrow + (size_t) i] * k_deq[krow + (size_t) i];
                        corr += q_mean[qrow + (size_t) i] * k_centered[krow + (size_t) i];
                    }
                    float score = (main + corr) * scale;
                    if (logit_softcap != 0.0f) {
                        score = logit_softcap * std::tanh(score);
                    }
                    if (mask != nullptr) {
                        score += slope * read_f16(mask_raw, mask, kt, qt, qh % mask->ne[2], b % mask->ne[3]);
                    }
                    scores[(size_t) kt] = score;
                    score_max = std::max(score_max, scores[(size_t) kt]);
                }

                float prob_sum = 0.0f;
                for (int64_t kt = 0; kt < kv_len; ++kt) {
                    probs[(size_t) kt] = std::exp(scores[(size_t) kt] - score_max);
                    prob_sum += probs[(size_t) kt];
                }
                for (float & p : probs) {
                    p /= prob_sum;
                }

                const float row_max = max_abs(probs);
                const float first_scale = row_max > 0.0f ? row_max / (FP8_E4M3FN_MAX * FP4_E2M1_MAX) : 0.0f;
                std::vector<float> probs_scaled((size_t) kv_len);
                for (int64_t kt = 0; kt < kv_len; ++kt) {
                    probs_scaled[(size_t) kt] = first_scale > 0.0f ? probs[(size_t) kt] / first_scale : 0.0f;
                }
                nvfp4_quant_dequant_rows(probs_scaled, probs_deq, 1, kv_len, 1.0f);
                for (int64_t kt = 0; kt < kv_len; ++kt) {
                    probs_deq[(size_t) kt] *= first_scale;
                }

                for (int64_t i = 0; i < d; ++i) {
                    float out = 0.0f;
                    const size_t vrow = (((size_t) b * (size_t) kv_heads + (size_t) kh) * (size_t) d + (size_t) i) * (size_t) kv_len;
                    for (int64_t kt = 0; kt < kv_len; ++kt) {
                        out += probs_deq[(size_t) kt] * v_deq[vrow + (size_t) kt];
                    }
                    write_f32(out_raw, dst, i, qh, qt, b, out);
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(dst->data, out_raw.data(), out_raw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace

bool ggml_cuda_nvfp4_fattn_no_fallback() {
    return env_enabled("GGML_CUDA_NVFP4_FATTN_NO_FALLBACK");
}

bool ggml_cuda_flash_attn_ext_nvfp4_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_cuda_flash_attn_ext_nvfp4_gpu_prefill(ctx, dst)) {
        return true;
    }
    return ggml_cuda_flash_attn_ext_nvfp4_ref(ctx, dst);
}

bool ggml_cuda_flash_attn_ext_nvfp4_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!env_enabled("GGML_CUDA_NVFP4_FATTN_DECODE")) {
        return false;
    }

    return ggml_cuda_flash_attn_ext_nvfp4_ref(ctx, dst);
}
