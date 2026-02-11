#include "nvfp4-matmul.cuh"

#include "ggml-backend.h"

#include <atomic>
#include <cmath>
#include <cstring>

namespace {

template<typename T>
static bool ggml_cuda_read_scalar_tensor(const ggml_tensor * t, T & out) {
    if (t == nullptr || !ggml_is_scalar(t) || t->data == nullptr) {
        return false;
    }

    if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
        std::memcpy(&out, t->data, sizeof(T));
        return true;
    }

    ggml_backend_tensor_get(t, &out, 0, sizeof(T));
    return true;
}

static bool ggml_cuda_fetch_input_scale_f32(const ggml_tensor * scale, float & out) {
    if (scale == nullptr || !ggml_is_scalar(scale)) {
        return false;
    }

    switch (scale->type) {
        case GGML_TYPE_F32:
            return ggml_cuda_read_scalar_tensor(scale, out);
        case GGML_TYPE_F16: {
            ggml_fp16_t v = 0;
            if (!ggml_cuda_read_scalar_tensor(scale, v)) {
                return false;
            }
            out = ggml_fp16_to_fp32(v);
            return true;
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t v = { 0 };
            if (!ggml_cuda_read_scalar_tensor(scale, v)) {
                return false;
            }
            out = ggml_bf16_to_fp32(v);
            return true;
        }
        default:
            return false;
    }
}

static float ggml_cuda_nvfp4_global_scale(const ggml_tensor * dst) {
    const ggml_tensor * scale = ggml_mul_mat_get_nvfp4_input_scale(dst);
    if (scale == nullptr) {
        return 1.0f;
    }

    float input_scale = 0.0f;
    if (!ggml_cuda_fetch_input_scale_f32(scale, input_scale) || input_scale == 0.0f || !std::isfinite(input_scale)) {
        static std::atomic<bool> logged(false);
        if (!logged.exchange(true)) {
            GGML_LOG_DEBUG("%s: invalid NVFP4 input scale for %s, fallback global_scale=1.0\n",
                    __func__, ggml_get_name(dst));
        }
        return 1.0f;
    }

    return 1.0f / input_scale;
}

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_nvfp4(float x) {
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

static __device__ __forceinline__ uint8_t ggml_cuda_best_index_e4m3_half(float x) {
    uint8_t best_index = 0;
    float best_err = INFINITY;

    for (int i = 0; i < 256; ++i) {
        const float v = ggml_cuda_e4m3_to_fp32_half((uint8_t) i);
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

static __global__ void quantize_row_nvfp4_kernel(
        const float * __restrict__ x,
        block_nvfp4 * __restrict__ y,
        int64_t ne00,
        int64_t s01,
        float global_scale) {
    const int lane = threadIdx.x;
    const bool lane_active = lane < QK_NVFP4;

    const int ib = blockIdx.x;
    const int i1 = blockIdx.y;
    const int64_t k0 = (int64_t) ib * QK_NVFP4 + lane;

    const int64_t row_off = (int64_t) i1 * s01;
    const float xi = (lane_active && k0 < ne00) ? x[row_off + k0] : 0.0f;

    float vmax = fabsf(xi);
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 8, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 4, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 2, WARP_SIZE));
    vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFF, vmax, 1, WARP_SIZE));
    vmax = __shfl_sync(0xFFFFFFFF, vmax, 0, WARP_SIZE);

    float scale_f = 0.0f;
    if (lane == 0) {
        const float scale = global_scale * (vmax / 6.0f);
        const uint8_t scale_q = ggml_cuda_best_index_e4m3_half(scale);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].e = scale_q;
        scale_f = ggml_cuda_e4m3_to_fp32_half(scale_q);
    }
    scale_f = __shfl_sync(0xFFFFFFFF, scale_f, 0, WARP_SIZE);

    const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
    const uint8_t q = ggml_cuda_best_index_nvfp4(xi * inv_scale);

    if (lane_active && (lane & 1) == 0) {
        const uint8_t q_hi = __shfl_xor_sync(0xFFFFFFFF, q, 1, WARP_SIZE);
        y[(int64_t) i1 * (ne00 / QK_NVFP4) + ib].qs[lane/2] = q | (q_hi << 4);
    }
}

static void quantize_row_nvfp4_cuda(
        const float * x,
        block_nvfp4 * y,
        int64_t ne00,
        int64_t s01,
        int64_t ne01,
        float global_scale,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_NVFP4 == 0);

    const dim3 num_blocks((uint32_t) (ne00 / QK_NVFP4), (uint32_t) ne01, 1);
    const dim3 block_size(WARP_SIZE, 1, 1);
    quantize_row_nvfp4_kernel<<<num_blocks, block_size, 0, stream>>>(x, y, ne00, s01, global_scale);
}

static bool ggml_cuda_nvfp4_cache_key_match(
        const ggml_cuda_nvfp4_cache_entry & entry,
        const ggml_tensor * src0) {
    if (entry.src0_data != src0->data) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (entry.ne[i] != src0->ne[i] || entry.nb[i] != src0->nb[i]) {
            return false;
        }
    }

    return true;
}

#if GGML_CUDA_HAS_CUBLASLT
static const void * ggml_cuda_nvfp4_get_repacked_src0(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        cudaStream_t stream) {
    auto & cache = ctx.nvfp4_repack_cache[ctx.device];
    for (const ggml_cuda_nvfp4_cache_entry & entry : cache) {
        if (ggml_cuda_nvfp4_cache_key_match(entry, src0) && entry.repacked != nullptr) {
            return entry.repacked;
        }
    }

    void * repacked = nullptr;
    const size_t nbytes = ggml_nbytes(src0);
    cudaError_t err = cudaMalloc(&repacked, nbytes);
    if (err != cudaSuccess) {
        return nullptr;
    }

    err = cudaMemcpyAsync(repacked, src0->data, nbytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(repacked);
        return nullptr;
    }

    ggml_cuda_nvfp4_cache_entry entry = {};
    entry.src0_data = src0->data;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        entry.ne[i] = src0->ne[i];
        entry.nb[i] = src0->nb[i];
    }
    entry.repacked = repacked;
    entry.nbytes = nbytes;
    cache.push_back(entry);

    return repacked;
}
#endif

} // namespace

bool ggml_cuda_mul_mat_nvfp4_native(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
#if GGML_CUDA_HAS_CUBLASLT && GGML_CUDA_HAS_FP4 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(dst  != nullptr);

    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // This pass intentionally handles only dense, non-batched MUL_MAT.
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (ggml_is_transposed(src0) || ggml_is_transposed(src1) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne01 = src0->ne[1];
    if (src0->ne[0] != ne10 || dst->ne[0] != ne01 || dst->ne[1] != ne11) {
        return false;
    }

    if (ne10 % QK_NVFP4 != 0) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<block_nvfp4> src1_q_nvfp4(ctx.pool(), (size_t) (ne10 / QK_NVFP4) * (size_t) ne11);
    ggml_cuda_pool_alloc<block_nvfp4> src1_repacked(ctx.pool(), (size_t) (ne10 / QK_NVFP4) * (size_t) ne11);

    const float global_scale = ggml_cuda_nvfp4_global_scale(dst);
    quantize_row_nvfp4_cuda(
            (const float *) src1->data, src1_q_nvfp4.get(),
            ne10, src1->nb[1] / (int64_t) sizeof(float), ne11,
            global_scale, stream);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(
            src1_repacked.get(), src1_q_nvfp4.get(),
            (size_t) (ne10 / QK_NVFP4) * (size_t) ne11 * sizeof(block_nvfp4),
            cudaMemcpyDeviceToDevice, stream));

    const void * src0_repacked = ggml_cuda_nvfp4_get_repacked_src0(ctx, src0, stream);
    if (src0_repacked == nullptr) {
        return false;
    }

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) ne10, (uint64_t) ne01, (int64_t) ne10);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) ne10, (uint64_t) ne11, (int64_t) ne10);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, (uint64_t) ne01, (uint64_t) ne11, (int64_t) ne01);
    }

    float out_scale = 1.0f;
    if (const ggml_tensor * scale = ggml_mul_mat_get_nvfp4_weight_scale(dst)) {
        float scale_val = 0.0f;
        if (ggml_cuda_fetch_input_scale_f32(scale, scale_val) && std::isfinite(scale_val)) {
            out_scale = scale_val;
        }
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        const float alpha = out_scale;
        const float beta  = 0.0f;
        st = cublasLtMatmul(
                ctx.cublaslt_handle(),
                op_desc,
                &alpha,
                src0_repacked, a_desc,
                src1_repacked.get(), b_desc,
                &beta,
                dst->data, c_desc,
                dst->data, c_desc,
                nullptr,
                nullptr, 0,
                stream);
    }

    if (c_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc != nullptr) {
        cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (op_desc != nullptr) {
        cublasLtMatmulDescDestroy(op_desc);
    }

    if (st != CUBLAS_STATUS_SUCCESS) {
        static std::atomic<bool> logged(false);
        if (!logged.exchange(true)) {
            GGML_LOG_DEBUG("%s: cublasLt NVFP4 matmul unavailable for %s (status=%d), fallback to mmq/mmvq\n",
                    __func__, ggml_get_name(dst), (int) st);
        }
        return false;
    }

    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#endif
}
