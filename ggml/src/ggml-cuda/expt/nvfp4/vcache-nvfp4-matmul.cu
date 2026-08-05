#include "vcache-nvfp4-matmul.cuh"

#include "nvfp4-fp4mulmat.cuh"
#include "nvfp4-log.cuh"
#include "nvfp4-matmul.cuh"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

template<typename T>
static T * ggml_cuda_vcache_nvfp4_alloc_or_reuse(ggml_cuda_pool_alloc<T> & alloc, size_t size) {
    if (alloc.get() == nullptr) {
        return alloc.alloc(size);
    }
    GGML_ASSERT(alloc.actual_size >= size * sizeof(T));
    return alloc.get();
}

struct ggml_cuda_vcache_nvfp4_parallel_events {
    cudaEvent_t main_ready = nullptr;
    cudaEvent_t done[GGML_CUDA_MAX_STREAMS] = { nullptr };
    int n_streams = 1;

    ~ggml_cuda_vcache_nvfp4_parallel_events() {
        if (main_ready != nullptr) {
            CUDA_CHECK(cudaEventDestroy(main_ready));
        }
        for (int i = 1; i < n_streams; ++i) {
            if (done[i] != nullptr) {
                CUDA_CHECK(cudaEventDestroy(done[i]));
            }
        }
    }
};

struct ggml_cuda_vcache_nvfp4_slice_lane {
    ggml_cuda_pool_alloc<char> v_contig;
    ggml_cuda_nvfp4_native_scratch native_scratch;

    explicit ggml_cuda_vcache_nvfp4_slice_lane(ggml_cuda_pool & pool) :
        v_contig(pool),
        native_scratch(pool) {
    }
};

static void ggml_cuda_vcache_nvfp4_begin_parallel_streams(
        ggml_backend_cuda_context & ctx,
        int64_t slices,
        cudaStream_t main_stream,
        ggml_cuda_vcache_nvfp4_parallel_events & events) {
    events.n_streams = (int) std::min<int64_t>(std::max<int64_t>(slices, 1), GGML_CUDA_MAX_STREAMS);
    if (events.n_streams <= 1) {
        return;
    }

    CUDA_CHECK(cudaEventCreateWithFlags(&events.main_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(events.main_ready, main_stream));
    for (int i = 1; i < events.n_streams; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&events.done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(ctx.device, i), events.main_ready, 0));
    }
}

static void ggml_cuda_vcache_nvfp4_end_parallel_streams(
        ggml_backend_cuda_context & ctx,
        ggml_cuda_vcache_nvfp4_parallel_events & events,
        cudaStream_t main_stream) {
    for (int i = 1; i < events.n_streams; ++i) {
        cudaStream_t stream = ctx.stream(ctx.device, i);
        CUDA_CHECK(cudaEventRecord(events.done[i], stream));
        CUDA_CHECK(cudaStreamWaitEvent(main_stream, events.done[i], 0));
    }
}

} // namespace

static bool ggml_cuda_nvfp4_vcache_cublaslt_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_VCACHE_CUBLASLT_TRACE");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_vcache_cublaslt_trace_switch_once(env, cached != 0);
    }
    return cached != 0;
}

static bool ggml_cuda_nvfp4_vcache_mm_standalone_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_VCACHE_MM_STANDALONE");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_vcache_mm_standalone_switch_once(env, cached != 0);
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

    ggml_cuda_vcache_nvfp4_alloc_or_reuse(storage, (size_t) row_bytes * (size_t) rows);
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

    cudaStream_t main_stream = ctx.stream();
    const int64_t total_slices = q_heads * q_streams;
    ggml_cuda_vcache_nvfp4_parallel_events events = {};
    ggml_cuda_vcache_nvfp4_begin_parallel_streams(ctx, total_slices, main_stream, events);
    std::vector<std::unique_ptr<ggml_cuda_vcache_nvfp4_slice_lane>> lanes;
    if (events.n_streams > 1) {
        lanes.reserve((size_t) events.n_streams);
        for (int i = 0; i < events.n_streams; ++i) {
            lanes.emplace_back(new ggml_cuda_vcache_nvfp4_slice_lane(ctx.pool()));
        }
    }

    bool ok = true;
    for (int64_t q_stream = 0, slice = 0; q_stream < q_streams && ok; ++q_stream) {
        const int64_t kv_stream = q_stream / r3;
        if (kv_stream >= kv_streams) {
            ok = false;
            break;
        }

        for (int64_t q_head = 0; q_head < q_heads; ++q_head, ++slice) {
            const int64_t kv_head = q_head / r2;
            if (kv_head >= kv_heads) {
                ok = false;
                break;
            }
            const int stream_idx = events.n_streams > 1 ? (int) (slice % events.n_streams) : 0;
            cudaStream_t slice_stream = stream_idx == 0 ? main_stream : ctx.stream(ctx.device, stream_idx);

            void * v_ptr = (char *) src0->data + kv_head * src0->nb[2] + kv_stream * src0->nb[3];
            void * p_ptr = (char *) src1->data + q_head * src1->nb[2] + q_stream * src1->nb[3];
            float * dst_ptr = (float *) ((char *) dst->data + q_head * dst->nb[2] + q_stream * dst->nb[3]);
            ggml_cuda_pool_alloc<char> v_contig(ctx.pool());
            ggml_cuda_pool_alloc<char> & v_contig_ref = events.n_streams > 1 ? lanes[(size_t) stream_idx]->v_contig : v_contig;
            void * v_slice_ptr = nullptr;
            ggml_cuda_vcache_nvfp4_materialize_v_slice(
                    ctx,
                    v_ptr,
                    (int64_t) src0->nb[1],
                    v_row_bytes,
                    rows,
                    v_contig_ref,
                    slice_stream,
                    v_slice_ptr);

            ggml_tensor v_slice = ggml_cuda_vcache_nvfp4_make_temp_tensor_2d(GGML_TYPE_NVFP4, v_slice_ptr, kv_size, rows);
            ggml_tensor p_slice = ggml_cuda_vcache_nvfp4_make_temp_tensor_2d(GGML_TYPE_F32, p_ptr, kv_size, cols);
            ggml_tensor out_slice = ggml_cuda_vcache_nvfp4_make_temp_mul_mat_dst(dst_ptr, rows, cols);
            ggml_set_name(&v_slice, "nvfp4-vcache-native-v");
            ggml_set_name(&p_slice, "nvfp4-vcache-native-p");
            ggml_set_name(&out_slice, "nvfp4-vcache-native-pv");

            const float * scale_ptr = (const float *) ((const char *) scale->data + kv_stream * scale_stream_nb);
            ggml_cuda_nvfp4_native_scratch * native_scratch_ptr = events.n_streams > 1 ? &lanes[(size_t) stream_idx]->native_scratch : nullptr;
            if (!ggml_cuda_mul_mat_nvfp4_native_device_weight_scale_stream(
                        ctx, &v_slice, &p_slice, &out_slice, scale_ptr, true, slice_stream, native_scratch_ptr)) {
                ok = false;
                break;
            }
        }
    }
    ggml_cuda_vcache_nvfp4_end_parallel_streams(ctx, events, main_stream);
    for (int i = (int) lanes.size() - 1; i >= 0; --i) {
        lanes[(size_t) i].reset();
    }
    if (!ok) {
        return false;
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

    ggml_cuda_nvfp4_log_vcache_fp4_pv_once();

    if (ggml_cuda_nvfp4_vcache_mm_standalone_enabled()) {
        if (ggml_cuda_mul_mat_vcache_nvfp4_mm_standalone(ctx, src0, src1, dst)) {
            ggml_cuda_nvfp4_log_vcache_matmul_path_once("mm-standalone");
            return true;
        }
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
        GGML_ABORT(
                "%s: native-slice NVFP4 V-cache p*v matmul failed; aborting before generic NVFP4 fallback "
                "rows=%lld cols=%lld kv_size=%lld q_heads=%lld q_streams=%lld",
                __func__,
                (long long) rows,
                (long long) cols,
                (long long) kv_size,
                (long long) q_heads,
                (long long) q_streams);
    }
    return native_result;
}
