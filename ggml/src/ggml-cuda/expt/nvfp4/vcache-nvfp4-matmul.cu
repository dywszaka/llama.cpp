#include "vcache-nvfp4-matmul.cuh"

#include "nvfp4-fp4mulmat.cuh"
#include "nvfp4-log.cuh"
#include "nvfp4-matmul.cuh"

#include <cstdlib>

static bool ggml_cuda_nvfp4_vcache_cublaslt_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_VCACHE_CUBLASLT_TRACE");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_vcache_cublaslt_trace_switch_once(env, cached != 0);
    }
    return cached != 0;
}

static bool ggml_cuda_is_vcache_nvfp4_tensor(const ggml_tensor * src0) {
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

static bool ggml_cuda_match_vcache_nvfp4_global_scale_layout(
        const ggml_tensor * src0,
        const ggml_tensor * scale,
        int64_t & rows,
        int64_t & heads,
        int64_t & streams,
        int64_t & scale_stream_nb) {
    rows = src0->ne[1];
    heads = src0->ne[2];
    streams = src0->ne[3];

    if (ggml_nelements(scale) != streams) {
        return false;
    }

    scale_stream_nb = streams > 1 ? scale->nb[0] : 0;
    return true;
}

static ggml_tensor ggml_cuda_vcache_nvfp4_make_temp_tensor_2d(
        ggml_type type,
        void * data,
        int64_t ne0,
        int64_t ne1) {
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
    t.buffer = nullptr;
    return t;
}

static ggml_tensor ggml_cuda_vcache_nvfp4_make_temp_mul_mat_dst(
        float * data,
        int64_t ne0,
        int64_t ne1) {
    ggml_tensor t = ggml_cuda_vcache_nvfp4_make_temp_tensor_2d(GGML_TYPE_F32, data, ne0, ne1);
    t.op = GGML_OP_MUL_MAT;
    return t;
}

static void ggml_cuda_vcache_nvfp4_materialize_v_slice(
        ggml_backend_cuda_context & ctx,
        const void * src,
        int64_t src_row_stride,
        int64_t row_bytes,
        int64_t rows,
        ggml_cuda_pool_alloc<char> & storage,
        cudaStream_t stream,
        void *& out) {
    if (src_row_stride == row_bytes) {
        out = const_cast<void *>(src);
        return;
    }

    storage.alloc(ctx.pool(), (size_t) row_bytes * (size_t) rows);
    CUDA_CHECK(cudaMemcpy2DAsync(
            storage.get(),
            (size_t) row_bytes,
            src,
            (size_t) src_row_stride,
            (size_t) row_bytes,
            (size_t) rows,
            cudaMemcpyDeviceToDevice,
            stream));
    out = storage.get();
}

static bool ggml_cuda_vcache_nvfp4_matmul_global_native_slices(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        const ggml_tensor * scale,
        int64_t kv_size,
        int64_t rows,
        int64_t cols,
        int64_t kv_heads,
        int64_t q_heads,
        int64_t kv_streams,
        int64_t q_streams,
        int64_t scale_stream_nb,
        int64_t r2,
        int64_t r3) {
    if (kv_size <= 0 || kv_size % QK_NVFP4 != 0 || rows <= 0 || cols <= 0 ||
            kv_heads <= 0 || q_heads <= 0 || kv_streams <= 0 || q_streams <= 0) {
        return false;
    }

    const int64_t v_row_bytes = (kv_size / QK_NVFP4) * (int64_t) sizeof(block_nvfp4);
    if ((int64_t) src0->nb[0] != (int64_t) sizeof(block_nvfp4) ||
            (int64_t) src0->nb[1] < v_row_bytes ||
            (int64_t) src1->nb[0] != (int64_t) sizeof(float) ||
            (int64_t) src1->nb[1] != kv_size * (int64_t) sizeof(float) ||
            (int64_t) dst->nb[0] != (int64_t) sizeof(float) ||
            (int64_t) dst->nb[1] != rows * (int64_t) sizeof(float)) {
        return false;
    }

    if (scale->data == nullptr) {
        return false;
    }

    for (int64_t q_stream = 0; q_stream < q_streams; ++q_stream) {
        const int64_t kv_stream = q_stream / r3;
        if (kv_stream >= kv_streams) {
            return false;
        }

        for (int64_t q_head = 0; q_head < q_heads; ++q_head) {
            const int64_t kv_head = q_head / r2;
            if (kv_head >= kv_heads) {
                return false;
            }

            void * v_ptr = (char *) src0->data + kv_head * src0->nb[2] + kv_stream * src0->nb[3];
            void * p_ptr = (char *) src1->data + q_head * src1->nb[2] + q_stream * src1->nb[3];
            float * dst_ptr = (float *) ((char *) dst->data + q_head * dst->nb[2] + q_stream * dst->nb[3]);
            ggml_cuda_pool_alloc<char> v_contig(ctx.pool());
            void * v_slice_ptr = nullptr;
            ggml_cuda_vcache_nvfp4_materialize_v_slice(
                    ctx,
                    v_ptr,
                    (int64_t) src0->nb[1],
                    v_row_bytes,
                    rows,
                    v_contig,
                    ctx.stream(),
                    v_slice_ptr);

            ggml_tensor v_slice = ggml_cuda_vcache_nvfp4_make_temp_tensor_2d(GGML_TYPE_NVFP4, v_slice_ptr, kv_size, rows);
            ggml_tensor p_slice = ggml_cuda_vcache_nvfp4_make_temp_tensor_2d(GGML_TYPE_F32, p_ptr, kv_size, cols);
            ggml_tensor out_slice = ggml_cuda_vcache_nvfp4_make_temp_mul_mat_dst(dst_ptr, rows, cols);
            ggml_set_name(&v_slice, "nvfp4-vcache-native-v");
            ggml_set_name(&p_slice, "nvfp4-vcache-native-p");
            ggml_set_name(&out_slice, "nvfp4-vcache-native-pv");

            const float * scale_ptr = (const float *) ((const char *) scale->data + kv_stream * scale_stream_nb);
            if (!ggml_cuda_mul_mat_nvfp4_native_device_weight_scale(
                        ctx, &v_slice, &p_slice, &out_slice, scale_ptr, true)) {
                return false;
            }
        }
    }

    ggml_cuda_nvfp4_log_vcache_native_slice_active_once(rows, cols, kv_size, q_heads, q_streams);
    if (!ggml_cuda_nvfp4_fp4mulmat_enabled() && ggml_cuda_nvfp4_vcache_cublaslt_trace_enabled()) {
        const int64_t lt_k = ggml_cuda_nvfp4_pad_i64(kv_size, 32);
        ggml_cuda_nvfp4_log_vcache_cublaslt_trace(rows, cols, kv_size, lt_k, q_heads, q_streams);
    }
    return true;
}

bool ggml_cuda_mul_mat_vcache_nvfp4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (!ggml_cuda_is_vcache_nvfp4_tensor(src0)) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * scale = ggml_tensor_get_nvfp4_scale(src0);
    int64_t rows = 0;
    int64_t kv_heads = 0;
    int64_t kv_streams = 0;
    int64_t scale_stream_nb = 0;
    if (!ggml_cuda_match_vcache_nvfp4_global_scale_layout(
                src0, scale, rows, kv_heads, kv_streams, scale_stream_nb)) {
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

    if (kv_size <= 0 || kv_size % QK_NVFP4 != 0) {
        return false;
    }

    if (src0->nb[0] != (int64_t) sizeof(block_nvfp4) || scale->nb[0] != (int64_t) sizeof(float) ||
            src1->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }

    const int64_t r2 = q_heads / kv_heads;
    const int64_t r3 = q_streams / kv_streams;
    ggml_cuda_nvfp4_log_vcache_fp4_pv_once();

    if (ggml_cuda_nvfp4_vcache_batched_enabled()) {
        if (ggml_cuda_mul_mat_vcache_nvfp4_batched(ctx, src0, src1, dst)) {
            ggml_cuda_nvfp4_log_vcache_matmul_path_once("batched-native-dynamic-p-global-scale");
            return true;
        }
        ggml_cuda_nvfp4_log_vcache_batched_fallback_once();
    }

    const bool native_result = ggml_cuda_vcache_nvfp4_matmul_global_native_slices(
            ctx,
            src0,
            src1,
            dst,
            scale,
            kv_size,
            rows,
            cols,
            kv_heads,
            q_heads,
            kv_streams,
            q_streams,
            scale_stream_nb,
            r2,
            r3);
    if (native_result) {
        ggml_cuda_nvfp4_log_vcache_matmul_path_once("native-slice-dynamic-p-global-scale");
    } else {
        ggml_cuda_nvfp4_log_vcache_native_slice_failure_once(
                rows, cols, kv_size, q_heads, q_streams);
    }
    return native_result;
}
