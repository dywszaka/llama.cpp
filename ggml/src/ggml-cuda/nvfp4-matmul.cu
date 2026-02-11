#include "nvfp4-matmul.cuh"

#include "ggml-backend.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
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

static bool ggml_cuda_nvfp4_native_debug_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_NATIVE_DEBUG");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static float ggml_cuda_nvfp4_input_global_scale(const ggml_tensor * dst) {
    const ggml_tensor * scale = ggml_mul_mat_get_nvfp4_input_scale(dst);
    if (scale == nullptr) {
        return 1.0f;
    }

    float input_scale = 0.0f;
    if (!ggml_cuda_fetch_input_scale_f32(scale, input_scale) || input_scale == 0.0f || !std::isfinite(input_scale)) {
        static std::atomic<bool> logged(false);
        if (ggml_cuda_nvfp4_native_debug_enabled() || !logged.exchange(true)) {
            GGML_LOG_WARN("%s: invalid NVFP4 input scale for %s, fallback global_scale=1.0\n",
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
        static std::atomic<bool> logged(false);
        if (ggml_cuda_nvfp4_native_debug_enabled() || !logged.exchange(true)) {
            GGML_LOG_WARN("%s: cudaMalloc failed for repacked src0 (%zu bytes): %s\n",
                    __func__, nbytes, cudaGetErrorString(err));
        }
        return nullptr;
    }

    err = cudaMemcpyAsync(repacked, src0->data, nbytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        static std::atomic<bool> logged(false);
        if (ggml_cuda_nvfp4_native_debug_enabled() || !logged.exchange(true)) {
            GGML_LOG_WARN("%s: cudaMemcpyAsync failed for repacked src0 (%zu bytes): %s\n",
                    __func__, nbytes, cudaGetErrorString(err));
        }
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

    const bool debug_enabled = ggml_cuda_nvfp4_native_debug_enabled();
    auto log_skip = [&](const char * reason) {
        if (debug_enabled) {
            GGML_LOG_INFO(
                    "%s: skip native NVFP4 path: %s | src0=%s src1=%s dst=%s "
                    "src0_type=%s src1_type=%s dst_type=%s "
                    "src0_ne=[%lld,%lld,%lld,%lld] src1_ne=[%lld,%lld,%lld,%lld] dst_ne=[%lld,%lld,%lld,%lld]\n",
                    __func__, reason,
                    ggml_get_name(src0), ggml_get_name(src1), ggml_get_name(dst),
                    ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type),
                    (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
                    (long long) src1->ne[0], (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
                    (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3]);
        }
    };

    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        log_skip("unsupported tensor types");
        return false;
    }

    // This pass intentionally handles only dense, non-batched MUL_MAT.
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        log_skip("batched tensor shape not supported");
        return false;
    }

    if (ggml_is_transposed(src0) || ggml_is_transposed(src1) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        log_skip("requires contiguous non-transposed tensors");
        return false;
    }

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne01 = src0->ne[1];
    if (src0->ne[0] != ne10 || dst->ne[0] != ne01 || dst->ne[1] != ne11) {
        log_skip("incompatible matrix dimensions");
        return false;
    }

    if (ne10 % QK_NVFP4 != 0) {
        log_skip("K dimension is not divisible by QK_NVFP4");
        return false;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<block_nvfp4> src1_q_nvfp4(ctx.pool(), (size_t) (ne10 / QK_NVFP4) * (size_t) ne11);
    ggml_cuda_pool_alloc<block_nvfp4> src1_repacked(ctx.pool(), (size_t) (ne10 / QK_NVFP4) * (size_t) ne11);

    const float global_scale = ggml_cuda_nvfp4_input_global_scale(dst);
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
        static std::atomic<bool> logged(false);
        if (debug_enabled || !logged.exchange(true)) {
            GGML_LOG_WARN("%s: failed to prepare repacked src0 for %s\n", __func__, ggml_get_name(dst));
        }
        return false;
    }

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;

    const char * stage = "matmul_desc_create";
    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    const cublasOperation_t op_t = CUBLAS_OP_T;
    const cublasOperation_t op_n = CUBLAS_OP_N;
    stage = "matmul_desc_set_transa";
    st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "matmul_desc_set_transb";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_a";
        st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) ne10, (uint64_t) ne01, (int64_t) ne10);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_b";
        st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) ne10, (uint64_t) ne11, (int64_t) ne10);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_c";
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
        stage = "matmul";
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
        const cudaError_t cuda_err = cudaPeekAtLastError();
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        int runtime_version = 0;
        int driver_version = 0;
        (void) cudaRuntimeGetVersion(&runtime_version);
        (void) cudaDriverGetVersion(&driver_version);

        const ggml_tensor * in_scale_tensor = ggml_mul_mat_get_nvfp4_input_scale(dst);
        const ggml_tensor * out_scale_tensor = ggml_mul_mat_get_nvfp4_weight_scale(dst);

        const size_t src0_align16 = ((uintptr_t) src0_repacked) & 0xF;
        const size_t src1_align16 = ((uintptr_t) src1_repacked.get()) & 0xF;
        const size_t dst_align16  = ((uintptr_t) dst->data) & 0xF;

        static std::atomic<bool> logged(false);
        if (debug_enabled || !logged.exchange(true)) {
            GGML_LOG_WARN(
                "%s: cublasLt NVFP4 matmul failed for %s stage=%s status=%d (%s) cuda_err=%d (%s) "
                "device=%d cc=%d runtime=%d driver=%d stream=%p "
                "A=[k=%lld,n=%lld,ld=%lld,type=CUDA_R_4F_E2M1] "
                "B=[k=%lld,m=%lld,ld=%lld,type=CUDA_R_4F_E2M1] "
                "C/D=[n=%lld,m=%lld,ld=%lld,type=CUDA_R_32F] alpha=%g beta=0 "
                "global_scale=%g src0_type=%s src1_type=%s dst_type=%s "
                "in_scale_tensor=%p in_scale_type=%s out_scale_tensor=%p out_scale_type=%s "
                "ptr=[src0=%p src1=%p dst=%p] align16=[src0=%zu src1=%zu dst=%zu] "
                "src0_nb=[%zu,%zu,%zu,%zu] src1_nb=[%zu,%zu,%zu,%zu] dst_nb=[%zu,%zu,%zu,%zu]\n",
                __func__,
                ggml_get_name(dst),
                stage,
                (int) st,
                cublas_get_error_str(st),
                (int) cuda_err,
                cudaGetErrorString(cuda_err),
                ctx.device,
                cc,
                runtime_version,
                driver_version,
                (void *) stream,
                (long long) ne10, (long long) ne01, (long long) ne10,
                (long long) ne10, (long long) ne11, (long long) ne10,
                (long long) ne01, (long long) ne11, (long long) ne01,
                (double) out_scale,
                (double) global_scale,
                ggml_type_name(src0->type),
                ggml_type_name(src1->type),
                ggml_type_name(dst->type),
                (const void *) in_scale_tensor,
                in_scale_tensor ? ggml_type_name(in_scale_tensor->type) : "(null)",
                (const void *) out_scale_tensor,
                out_scale_tensor ? ggml_type_name(out_scale_tensor->type) : "(null)",
                src0_repacked,
                (const void *) src1_repacked.get(),
                (const void *) dst->data,
                src0_align16,
                src1_align16,
                dst_align16,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);

            if (st == CUBLAS_STATUS_NOT_SUPPORTED) {
                GGML_LOG_WARN(
                        "%s: hint: CUBLAS_STATUS_NOT_SUPPORTED usually means this GPU/toolkit/shape does not support "
                        "the requested FP4 Lt matmul path; fallback kernels will be used.\n",
                        __func__);
            }
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
