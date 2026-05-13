#include "fattn-nvfp4.cuh"

#include "nvfp4-matmul.cuh"
#include "ggml-quants.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
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

static __device__ __forceinline__ float read_tensor_nvfp4_device(
        const void * data,
        const float * scale,
        int64_t nb0,
        int64_t nb1,
        int64_t nb2,
        int64_t nb3,
        int64_t scale_nb0,
        int64_t scale_nb1,
        int64_t scale_nb2,
        int64_t scale_nb3,
        int scale_axis,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3) {
    const char * ptr = (const char *) data + (i0 / QK_NVFP4)*nb0 + i1*nb1 + i2*nb2 + i3*nb3;
    const block_nvfp4 * block = (const block_nvfp4 *) ptr;
    const uint8_t packed = block->qs[(i0 % QK_NVFP4) / 2];
    const uint8_t q = (i0 & 1) == 0 ? (packed & 0x0F) : (packed >> 4);

    float input_scale = 1.0f;
    if (scale != nullptr && scale_axis >= 0) {
        const int64_t scale_i0 = scale_axis == 0 ? i0 : scale_axis == 1 ? i1 : scale_axis == 2 ? i2 : i3;
        const int64_t scale_i1 = scale_axis == 1 ? 0  : i1;
        const int64_t scale_i2 = scale_axis == 2 ? 0  : i2;
        const char * scale_ptr = (const char *) scale + scale_i0*scale_nb0 + scale_i1*scale_nb1 + scale_i2*scale_nb2 + i3*scale_nb3;
        input_scale = *(const float *) scale_ptr;
    }

    return ggml_cuda_e4m3_to_fp32_half(block->e) * input_scale * (float) kvalues_nvfp4[q];
}

static __device__ __forceinline__ float read_nvfp4_scale_device(
        const float * scale,
        int64_t scale_nb0,
        int64_t scale_nb1,
        int64_t scale_nb2,
        int64_t scale_nb3,
        int scale_axis,
        int64_t i0,
        int64_t i1,
        int64_t i2,
        int64_t i3) {
    if (scale == nullptr || scale_axis < 0) {
        return 1.0f;
    }

    const int64_t scale_i0 = scale_axis == 0 ? i0 : scale_axis == 1 ? i1 : scale_axis == 2 ? i2 : i3;
    const int64_t scale_i1 = scale_axis == 1 ? 0  : i1;
    const int64_t scale_i2 = scale_axis == 2 ? 0  : i2;
    const char * scale_ptr = (const char *) scale + scale_i0*scale_nb0 + scale_i1*scale_nb1 + scale_i2*scale_nb2 + i3*scale_nb3;
    return *(const float *) scale_ptr;
}

static bool get_nvfp4_scale_layout(
        const ggml_tensor * src,
        int & scale_axis,
        int64_t & scale_nb0,
        int64_t & scale_nb1,
        int64_t & scale_nb2,
        int64_t & scale_nb3) {
    const ggml_tensor * scale = ggml_tensor_get_nvfp4_scale(src);
    if (scale == nullptr || scale->type != GGML_TYPE_F32 || scale->data == nullptr) {
        return false;
    }

    scale_nb0 = (int64_t) scale->nb[0];
    scale_nb1 = (int64_t) scale->nb[1];
    scale_nb2 = (int64_t) scale->nb[2];
    scale_nb3 = (int64_t) scale->nb[3];

    if (scale->ne[0] == src->ne[1] && scale->ne[3] == src->ne[3]) {
        scale_axis = 1;
        return true;
    }
    if (scale->ne[0] == src->ne[2] && scale->ne[3] == src->ne[3]) {
        scale_axis = 2;
        return true;
    }
    if (scale->ne[0] == src->ne[0] && scale->ne[3] == src->ne[3]) {
        scale_axis = 0;
        return true;
    }
    return false;
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

static __global__ void q_smooth_decode_kernel(
        const void * q,
        int q_type,
        int64_t q_nb0,
        int64_t q_nb1,
        int64_t q_nb2,
        int64_t q_nb3,
        float * q_centered,
        float * q_mean,
        int64_t d,
        int64_t q_heads,
        int64_t total) {
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
            idx < total;
            idx += (int64_t) blockDim.x * gridDim.x) {
        const int64_t i = idx % d;
        const int64_t bh = idx / d;
        const int64_t b = bh / q_heads;
        const int64_t h = bh - b*q_heads;

        const float v = read_tensor_f32_or_f16_device(q, q_type, q_nb0, q_nb1, q_nb2, q_nb3, i, 0, h, b);
        const float mean = v / 128.0f;
        q_mean[idx] = mean;
        q_centered[idx] = v - mean;
    }
}

static __global__ void q_no_smooth_kernel(
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
        int64_t q_heads,
        int64_t total) {
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
            idx < total;
            idx += (int64_t) blockDim.x * gridDim.x) {
        int64_t rem = idx;
        const int64_t i = rem % d;
        rem /= d;
        const int64_t t = rem % q_len;
        rem /= q_len;
        const int64_t h = rem % q_heads;
        const int64_t b = rem / q_heads;

        q_centered[idx] = read_tensor_f32_or_f16_device(q, q_type, q_nb0, q_nb1, q_nb2, q_nb3, i, t, h, b);
        q_mean[idx] = 0.0f;
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

static __global__ void k_no_smooth_kernel(
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
        int64_t kv_heads,
        int64_t total) {
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
            idx < total;
            idx += (int64_t) blockDim.x * gridDim.x) {
        int64_t rem = idx;
        const int64_t i = rem % d;
        rem /= d;
        const int64_t t = rem % kv_len;
        rem /= kv_len;
        const int64_t h = rem % kv_heads;
        const int64_t b = rem / kv_heads;
        const int64_t visible = visible_lens[b];

        k_centered[idx] = t < visible ? read_tensor_f32_or_f16_device(k, k_type, k_nb0, k_nb1, k_nb2, k_nb3, i, t, h, b) : 0.0f;
    }
}

static __global__ void copy_k_nvfp4_head_kernel(
        const void * k,
        block_nvfp4 * k_head,
        int64_t k_nb0,
        int64_t k_nb1,
        int64_t k_nb2,
        int64_t k_nb3,
        int64_t d,
        int64_t kv_len,
        int64_t kh,
        int64_t b,
        int64_t total_blocks) {
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
            idx < total_blocks;
            idx += (int64_t) blockDim.x * gridDim.x) {
        const int64_t ib = idx % (d / QK_NVFP4);
        const int64_t t  = idx / (d / QK_NVFP4);
        const char * src = (const char *) k + ib*k_nb0 + t*k_nb1 + kh*k_nb2 + b*k_nb3;
        k_head[idx] = *(const block_nvfp4 *) src;
    }
}

static __global__ void qk_apply_k_scale_kernel(
        float * qk,
        const float * k_scale,
        int64_t k_scale_nb0,
        int64_t k_scale_nb1,
        int64_t k_scale_nb2,
        int64_t k_scale_nb3,
        int k_scale_axis,
        int64_t q_len,
        int64_t kv_len,
        int64_t kh,
        int64_t b,
        int64_t total) {
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
            idx < total;
            idx += (int64_t) blockDim.x * gridDim.x) {
        const int64_t kt = idx % kv_len;
        const int64_t qt = idx / kv_len;
        const float scale = read_nvfp4_scale_device(
                k_scale,
                k_scale_nb0, k_scale_nb1, k_scale_nb2, k_scale_nb3,
                k_scale_axis,
                0, kt, kh, b);
        qk[qt*kv_len + kt] *= scale;
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

static __global__ void qmean_kcorr_nvfp4_kernel(
        const float * q_mean,
        const void * k,
        const float * k_scale,
        float * corr,
        int64_t k_nb0,
        int64_t k_nb1,
        int64_t k_nb2,
        int64_t k_nb3,
        int64_t k_scale_nb0,
        int64_t k_scale_nb1,
        int64_t k_scale_nb2,
        int64_t k_scale_nb3,
        int k_scale_axis,
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
    for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float kv = read_tensor_nvfp4_device(
                k, k_scale,
                k_nb0, k_nb1, k_nb2, k_nb3,
                k_scale_nb0, k_scale_nb1, k_scale_nb2, k_scale_nb3,
                k_scale_axis,
                i, kt, kh, b);
        local += q_mean[qrow*d + i] * kv;
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

static __global__ void probs_direct_scale_kernel(
        const float * probs,
        float * probs_scaled,
        float * first_scales,
        int64_t q_len,
        int64_t kv_len) {
    const int64_t qt = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t row = bh*q_len + qt;

    if (threadIdx.x == 0) {
        first_scales[row] = 1.0f;
    }
    for (int64_t kt = threadIdx.x; kt < kv_len; kt += blockDim.x) {
        probs_scaled[row*kv_len + kt] = probs[row*kv_len + kt];
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

static void log_nvfp4_fattn_tensor_brief_once(
        const char * label,
        const ggml_tensor * a,
        const ggml_tensor * b,
        const ggml_tensor * dst,
        bool qk) {
    static std::atomic<bool> qk_logged(false);
    static std::atomic<bool> pv_logged(false);
    std::atomic<bool> & logged = qk ? qk_logged : pv_logged;
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_INFO(
            "%s: %s A{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "B{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "dst{name=%s type=%s ne=[%lld,%lld,%lld,%lld]}\n",
            __func__,
            label,
            a != nullptr ? ggml_get_name(a) : "(null)",
            a != nullptr ? ggml_type_name(a->type) : "(null)",
            a != nullptr ? (long long) a->ne[0] : 0, a != nullptr ? (long long) a->ne[1] : 0,
            a != nullptr ? (long long) a->ne[2] : 0, a != nullptr ? (long long) a->ne[3] : 0,
            b != nullptr ? ggml_get_name(b) : "(null)",
            b != nullptr ? ggml_type_name(b->type) : "(null)",
            b != nullptr ? (long long) b->ne[0] : 0, b != nullptr ? (long long) b->ne[1] : 0,
            b != nullptr ? (long long) b->ne[2] : 0, b != nullptr ? (long long) b->ne[3] : 0,
            dst != nullptr ? ggml_get_name(dst) : "(null)",
            dst != nullptr ? ggml_type_name(dst->type) : "(null)",
            dst != nullptr ? (long long) dst->ne[0] : 0, dst != nullptr ? (long long) dst->ne[1] : 0,
            dst != nullptr ? (long long) dst->ne[2] : 0, dst != nullptr ? (long long) dst->ne[3] : 0);
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

static bool ggml_cuda_flash_attn_ext_nvfp4_gpu_native(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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
    const bool no_q_smooth = env_enabled("GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH");
    const bool no_k_smooth = env_enabled("GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH");
    const bool k_nvfp4_cache = k->type == GGML_TYPE_NVFP4;

    if (!((k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32 || k_nvfp4_cache) &&
          (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32))) {
        return false;
    }
    if (k_nvfp4_cache && !no_k_smooth) {
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
    const bool p_direct = env_enabled("GGML_CUDA_NVFP4_FATTN_P_DIRECT");
    const bool q_dynamic = env_enabled("GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC");

    int k_scale_axis = -1;
    int64_t k_scale_nb0 = 0;
    int64_t k_scale_nb1 = 0;
    int64_t k_scale_nb2 = 0;
    int64_t k_scale_nb3 = 0;
    const ggml_tensor * k_scale_tensor = nullptr;
    if (k_nvfp4_cache) {
        if (!get_nvfp4_scale_layout(k, k_scale_axis, k_scale_nb0, k_scale_nb1, k_scale_nb2, k_scale_nb3)) {
            return false;
        }
        k_scale_tensor = ggml_tensor_get_nvfp4_scale(k);
    }

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
    ggml_cuda_pool_alloc<float> k_centered(ctx.pool(), k_nvfp4_cache ? 0 : (size_t) k_rows * (size_t) d);
    ggml_cuda_pool_alloc<float> v_by_dim(ctx.pool(), (size_t) v_rows * (size_t) kv_len);

    const int q_block_size = 256;
    if (no_q_smooth) {
        const int64_t q_total = q_rows*d;
        q_no_smooth_kernel<<<(int) std::min<int64_t>(1024, (q_total + q_block_size - 1) / q_block_size),
                q_block_size, 0, stream>>>(
                q->data, q->type,
                (int64_t) q->nb[0], (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
                q_centered.get(), q_mean.get(), d, q_len, q_heads, q_total);
    } else if (q_len == 1) {
        const int64_t q_total = batch*q_heads*d;
        q_smooth_decode_kernel<<<(int) std::min<int64_t>(1024, (q_total + q_block_size - 1) / q_block_size),
                q_block_size, 0, stream>>>(
                q->data, q->type,
                (int64_t) q->nb[0], (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
                q_centered.get(), q_mean.get(), d, q_heads, q_total);
    } else {
        q_smooth_kernel<<<dim3((uint32_t) d, (uint32_t) ((q_len + 127) / 128), (uint32_t) (batch*q_heads)), 128, 0, stream>>>(
                q->data, q->type,
                (int64_t) q->nb[0], (int64_t) q->nb[1], (int64_t) q->nb[2], (int64_t) q->nb[3],
                q_centered.get(), q_mean.get(), d, q_len, q_heads);
    }
    CUDA_CHECK(cudaGetLastError());

    if (no_k_smooth && !k_nvfp4_cache) {
        k_no_smooth_kernel<<<(int) std::min<int64_t>(1024, (k_rows*d + q_block_size - 1) / q_block_size),
                q_block_size, 0, stream>>>(
                k->data, k->type,
                (int64_t) k->nb[0], (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                d_visible.get(), k_centered.get(), d, kv_len, kv_heads, k_rows*d);
        CUDA_CHECK(cudaGetLastError());
    } else if (!k_nvfp4_cache) {
        k_smooth_kernel<<<dim3((uint32_t) d, (uint32_t) (batch*kv_heads), 1), 256, 0, stream>>>(
                k->data, k->type,
                (int64_t) k->nb[0], (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                d_visible.get(), k_centered.get(), d, kv_len, kv_heads);
        CUDA_CHECK(cudaGetLastError());
    }

    const int block_size = 256;
    const int64_t v_total = v_rows * kv_len;
    v_by_dim_kernel<<<(int) std::min<int64_t>(1024, (v_total + block_size - 1) / block_size), block_size, 0, stream>>>(
            v->data, v->type,
            (int64_t) v->nb[0], (int64_t) v->nb[1], (int64_t) v->nb[2], (int64_t) v->nb[3],
            d_visible.get(), v_by_dim.get(), d, kv_len, kv_heads, v_total);
    CUDA_CHECK(cudaGetLastError());

    const float q_amax = device_absmax(ctx, q_centered.get(), q_rows*d, stream);
    const float k_amax = k_nvfp4_cache ? 1.0f : device_absmax(ctx, k_centered.get(), k_rows*d, stream);
    const float v_amax = device_absmax(ctx, v_by_dim.get(), v_rows*kv_len, stream);
    const float q_global_scale = q_amax > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / q_amax : 0.0f;
    const float k_global_scale = k_nvfp4_cache ? 1.0f : (k_amax > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / k_amax : 0.0f);
    const float v_global_scale = v_amax > 0.0f ? (FP8_E4M3FN_MAX * FP4_E2M1_MAX) / v_amax : 0.0f;
    if (q_global_scale == 0.0f || k_global_scale == 0.0f || v_global_scale == 0.0f) {
        return false;
    }

    if (debug_log) {
        GGML_LOG_INFO(
                "%s: NVFP4 FATTN native quantization: "
                "Q/K group_dim=head_dim group_size=%d tensor_global_scale_inv=[q=%g k=%g] "
                "V group_dim=kv_len group_size=%d tensor_global_scale_inv=%g "
                "P format=%s "
                "Q quant=%s "
                "smooth=[q=%s k=%s] "
                "shape=[batch=%lld q_heads=%lld kv_heads=%lld q_len=%lld kv_len=%lld head_dim=%lld]\n",
                __func__,
                QK_NVFP4,
                (double) q_global_scale,
                (double) k_global_scale,
                QK_NVFP4,
                (double) v_global_scale,
                p_direct ?
                    "nvfp4_direct group_dim=kv_len first_level=none second_level=NVFP4(global_scale_inv=1)" :
                    "nvfp4_twolevel group_dim=kv_len first_level=row_max/(448*6) second_level=NVFP4(global_scale_inv=1)",
                q_dynamic ? "dynamic_per_row" : "static_global",
                no_q_smooth ? "off" : "on",
                no_k_smooth ? "off" : "on",
                (long long) batch,
                (long long) q_heads,
                (long long) kv_heads,
                (long long) q_len,
                (long long) kv_len,
                (long long) d);
    }

    ggml_cuda_pool_alloc<block_nvfp4> k_q(ctx.pool(), k_nvfp4_cache ? 0 : (size_t) k_rows * (size_t) (d / QK_NVFP4));
    ggml_cuda_pool_alloc<block_nvfp4> v_q(ctx.pool(), (size_t) v_rows * (size_t) (kv_len / QK_NVFP4));
    if (!k_nvfp4_cache) {
        ggml_cuda_nvfp4_quantize_rows_f32(k_centered.get(), k_q.get(), d, d, k_rows, k_global_scale, stream);
    }
    ggml_cuda_nvfp4_quantize_rows_f32(v_by_dim.get(), v_q.get(), kv_len, kv_len, v_rows, v_global_scale, stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<float> qk(ctx.pool(), (size_t) bh_rows * (size_t) kv_len * (size_t) q_len);
    ggml_cuda_pool_alloc<float> corr(ctx.pool(), (size_t) bh_rows * (size_t) kv_len * (size_t) q_len);
    ggml_cuda_pool_alloc<float> probs(ctx.pool(), (size_t) bh_rows * (size_t) q_len * (size_t) kv_len);
    ggml_cuda_pool_alloc<float> probs_scaled(ctx.pool(), (size_t) bh_rows * (size_t) q_len * (size_t) kv_len);
    ggml_cuda_pool_alloc<float> p_first_scales(ctx.pool(), (size_t) bh_rows * (size_t) q_len);
    ggml_cuda_pool_alloc<float> vp(ctx.pool(), (size_t) bh_rows * (size_t) d * (size_t) q_len);

    if (k_nvfp4_cache) {
        qmean_kcorr_nvfp4_kernel<<<dim3((uint32_t) kv_len, (uint32_t) q_len, (uint32_t) bh_rows), 256, 0, stream>>>(
                q_mean.get(), k->data, (const float *) k_scale_tensor->data, corr.get(),
                (int64_t) k->nb[0], (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                k_scale_nb0, k_scale_nb1, k_scale_nb2, k_scale_nb3, k_scale_axis,
                d, q_len, q_heads, kv_len, kv_heads, gqa_ratio);
    } else {
        qmean_kcorr_kernel<<<dim3((uint32_t) kv_len, (uint32_t) q_len, (uint32_t) bh_rows), 256, 0, stream>>>(
                q_mean.get(), k_centered.get(), corr.get(), d, q_len, q_heads, kv_len, kv_heads, gqa_ratio);
    }
    CUDA_CHECK(cudaGetLastError());

    float q_input_scale_value = 1.0f / q_global_scale;
    float k_weight_scale_value = 1.0f / k_global_scale;
    ggml_tensor q_input_scale = make_host_scalar_tensor_f32(&q_input_scale_value);
    ggml_tensor k_weight_scale = make_host_scalar_tensor_f32(&k_weight_scale_value);
    ggml_cuda_pool_alloc<block_nvfp4> k_head_q(ctx.pool(), k_nvfp4_cache ? (size_t) kv_len * (size_t) (d / QK_NVFP4) : 0);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t qh = 0; qh < q_heads; ++qh) {
            const int64_t kh = qh / gqa_ratio;
            block_nvfp4 * k_ptr = nullptr;
            if (k_nvfp4_cache) {
                const int64_t k_blocks = kv_len * (d / QK_NVFP4);
                copy_k_nvfp4_head_kernel<<<(int) std::min<int64_t>(1024, (k_blocks + q_block_size - 1) / q_block_size),
                        q_block_size, 0, stream>>>(
                        k->data, k_head_q.get(),
                        (int64_t) k->nb[0], (int64_t) k->nb[1], (int64_t) k->nb[2], (int64_t) k->nb[3],
                        d, kv_len, kh, b, k_blocks);
                CUDA_CHECK(cudaGetLastError());
                k_ptr = k_head_q.get();
            } else {
                k_ptr = k_q.get() + ((b*kv_heads + kh)*kv_len) * (d / QK_NVFP4);
            }
            float * q_ptr = q_centered.get() + ((b*q_heads + qh)*q_len) * d;
            float * qk_ptr = qk.get() + (b*q_heads + qh)*kv_len*q_len;

            ggml_tensor k_t = make_cuda_temp_tensor_2d(GGML_TYPE_NVFP4, k_ptr, d, kv_len);
            ggml_tensor q_t = make_cuda_temp_tensor_2d(GGML_TYPE_F32, q_ptr, d, q_len);
            ggml_tensor qk_t = make_cuda_mul_mat_dst(qk_ptr, kv_len, q_len, &k_weight_scale, q_dynamic ? nullptr : &q_input_scale);
            ggml_set_name(&k_t, "nvfp4-fattn-k");
            ggml_set_name(&q_t, "nvfp4-fattn-q");
            ggml_set_name(&qk_t, "nvfp4-fattn-qk");
            log_nvfp4_fattn_tensor_brief_once("q*k", &k_t, &q_t, &qk_t, true);
            if (debug_log && b == 0 && qh == 0) {
                GGML_LOG_INFO(
                        "%s: QK matmul requested: backend=cublasLt tensor_core=FP4 lt_type=CUDA_R_4F_E2M1 "
                        "A=%s[NVFP4,k=%lld,m=%lld,weight_scale=%g%s] "
                        "B=Q_centered[F32->NVFP4,k=%lld,n=%lld,%s=%g] "
                        "C=F32[%lld,%lld]\n",
                        __func__,
                        k_nvfp4_cache ? "K_cache_direct" : "K_centered",
                        (long long) d,
                        (long long) kv_len,
                        (double) k_weight_scale_value,
                        k_nvfp4_cache ? ",row_scale_after_matmul" : "",
                        (long long) d,
                        (long long) q_len,
                        q_dynamic ? "dynamic_per_row_scale_placeholder" : "input_scale=1/q_global_scale_inv",
                        (double) q_input_scale_value,
                        (long long) kv_len,
                        (long long) q_len);
            }
            if (!ggml_cuda_mul_mat_nvfp4_native(ctx, &k_t, &q_t, &qk_t)) {
                if (debug_log) {
                    GGML_LOG_WARN("%s: QK matmul did not use native Tensor Core FP4 path; NVFP4 FATTN native path unavailable\n", __func__);
                }
                return false;
            }
            if (debug_log && b == 0 && qh == 0) {
                GGML_LOG_INFO("%s: QK matmul active: native Tensor Core FP4 path confirmed (cublasLt CUDA_R_4F_E2M1)\n", __func__);
            }
            if (k_nvfp4_cache) {
                const int64_t qk_total = q_len * kv_len;
                qk_apply_k_scale_kernel<<<(int) std::min<int64_t>(1024, (qk_total + q_block_size - 1) / q_block_size),
                        q_block_size, 0, stream>>>(
                        qk_ptr, (const float *) k_scale_tensor->data,
                        k_scale_nb0, k_scale_nb1, k_scale_nb2, k_scale_nb3, k_scale_axis,
                        q_len, kv_len, kh, b, qk_total);
                CUDA_CHECK(cudaGetLastError());
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

    if (p_direct) {
        probs_direct_scale_kernel<<<dim3((uint32_t) q_len, (uint32_t) bh_rows, 1), 256, 0, stream>>>(
                probs.get(), probs_scaled.get(), p_first_scales.get(), q_len, kv_len);
    } else {
        probs_twolevel_scale_kernel<<<dim3((uint32_t) q_len, (uint32_t) bh_rows, 1), 256, 0, stream>>>(
                probs.get(), probs_scaled.get(), p_first_scales.get(), q_len, kv_len);
    }
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
            log_nvfp4_fattn_tensor_brief_once("p*v", &v_t, &p_t, &vp_t, false);
            if (debug_log && b == 0 && qh == 0) {
                GGML_LOG_INFO(
                        "%s: VP matmul requested: backend=cublasLt tensor_core=FP4 lt_type=CUDA_R_4F_E2M1 "
                        "A=V[NVFP4,k=%lld,m=%lld,weight_scale=1/v_global_scale_inv=%g] "
                        "B=%s[F32->NVFP4,k=%lld,n=%lld,input_scale=1,%s] "
                        "C=F32[%lld,%lld]\n",
                        __func__,
                        (long long) kv_len,
                        (long long) d,
                        (double) v_weight_scale_value,
                        p_direct ? "P_raw" : "P_scaled",
                        (long long) kv_len,
                        (long long) q_len,
                        p_direct ? "no_first_scale" : "twolevel_first_scale_applied_after_matmul",
                        (long long) d,
                        (long long) q_len);
            }
            if (!ggml_cuda_mul_mat_nvfp4_native(ctx, &v_t, &p_t, &vp_t)) {
                if (debug_log) {
                    GGML_LOG_WARN("%s: VP matmul did not use native Tensor Core FP4 path; NVFP4 FATTN native path unavailable\n", __func__);
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

} // namespace

bool ggml_cuda_nvfp4_fattn_no_fallback() {
    return env_enabled("GGML_CUDA_NVFP4_FATTN_NO_FALLBACK");
}

bool ggml_cuda_flash_attn_ext_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    return ggml_cuda_flash_attn_ext_nvfp4_gpu_native(ctx, dst);
}
