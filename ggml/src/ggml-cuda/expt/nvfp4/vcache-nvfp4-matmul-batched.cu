#include "vcache-nvfp4-matmul.cuh"

#include "kcache-outlier.cuh"
#include "nvfp4-fp4mulmat.cuh"
#include "nvfp4-log.cuh"
#include "nvfp4-quantize.cuh"

#include <algorithm>
#include <cstdlib>
#include <vector>

// The implementation started from the detached all-head V-cache path. The experiment keeps
// its low-overhead scheduling structure while matching the current native-slice scale semantics.

#if defined(CUBLAS_VERSION)
#define GGML_CUDA_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VERSION >= 130000)
#elif defined(CUBLAS_VER_MAJOR)
#define GGML_CUDA_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VER_MAJOR >= 13)
#else
#define GGML_CUDA_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS 0
#endif

bool ggml_cuda_nvfp4_vcache_batched_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = std::getenv("GGML_CUDA_NVFP4_VCACHE_BATCHED");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_vcache_batched_switch_once(env, cached != 0);
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_vcache_parallel_lt_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = std::getenv("GGML_CUDA_NVFP4_VCACHE_PARALLEL_LT");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
        ggml_cuda_nvfp4_log_vcache_parallel_lt_switch_once(env, cached != 0);
    }
    return cached != 0;
}

static bool ggml_cuda_nvfp4_vcache_batched_conflicting_experiment_enabled() {
    if (ggml_cuda_nvfp4_fp4mulmat_enabled() ||
            (ggml_cuda_nvfp4_bf16_quant_enabled() && ggml_cuda_nvfp4_bf16_quant_trunc_nn_enabled()) ||
            ggml_cuda_nvfp4_trunc_bf16_input_enabled()) {
        return true;
    }

    const char * row_split = std::getenv("GGML_CUDA_NVFP4_NATIVE_ROW_SPLIT");
    const char * linear_layout = std::getenv("GGML_CUDA_NVFP4_SCALE_LINEAR_LAYOUT");
    const char * native_debug = std::getenv("GGML_CUDA_NVFP4_NATIVE_DEBUG");
    const char * native_validate = std::getenv("GGML_CUDA_NVFP4_NATIVE_VALIDATE");
    return (row_split != nullptr && row_split[0] != '\0' && row_split[0] != '0') ||
            (linear_layout != nullptr && linear_layout[0] != '\0' && linear_layout[0] != '0') ||
            (native_debug != nullptr && native_debug[0] != '\0' && native_debug[0] != '0') ||
            (native_validate != nullptr && native_validate[0] != '\0' && native_validate[0] != '0');
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

static __global__ void k_prepare_vcache_nvfp4_p_scales(
        const float * __restrict__ p_amax,
        const float * __restrict__ v_global_scales,
        float * __restrict__ p_scales,
        int64_t p_rows,
        int64_t cols,
        int64_t q_heads,
        int64_t q_streams,
        int64_t kv_streams,
        int64_t scale_stream_nb,
        int64_t r3) {
    const int64_t p_row = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (p_row >= p_rows) {
        return;
    }

    const int64_t q_stream = p_row / (cols * q_heads);
    if (q_stream >= q_streams) {
        return;
    }
    const int64_t kv_stream = q_stream / r3;
    if (kv_stream >= kv_streams) {
        return;
    }

    const float raw_v_scale = *(const float *) ((const char *) v_global_scales + kv_stream * scale_stream_nb);
    const float v_out_scale = (raw_v_scale > 0.0f && isfinite(raw_v_scale)) ? 1.0f / raw_v_scale : 0.0f;
    p_scales[p_row] = ggml_cuda_nvfp4_kcache_outlier_q_input_scale(p_amax[p_row], v_out_scale);
}

#if GGML_CUDA_HAS_CUBLASLT
static __global__ void k_stage_vcache_nvfp4_all_v_for_lt(
        const block_nvfp4 * __restrict__ v_data,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        int64_t kv_size,
        int64_t rows,
        int64_t kv_heads,
        int64_t kv_streams,
        int64_t v_nb0,
        int64_t v_nb1,
        int64_t v_nb2,
        int64_t v_nb3,
        int64_t row_data_bytes,
        int64_t data_slice_bytes,
        int64_t scale_inner_padded,
        int64_t scale_slice_bytes) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_blocks = kv_size / QK_NVFP4;
    const int64_t elems_per_slice = rows * n_blocks;
    const int64_t total = elems_per_slice * kv_heads * kv_streams;
    if (idx >= total) {
        return;
    }

    const int64_t slice = idx / elems_per_slice;
    const int64_t rem = idx - slice * elems_per_slice;
    const int64_t kv_head = slice % kv_heads;
    const int64_t kv_stream = slice / kv_heads;
    const int64_t row = rem / n_blocks;
    const int64_t block = rem - row * n_blocks;
    const char * v_base = (const char *) v_data + row * v_nb1 + kv_head * v_nb2 + kv_stream * v_nb3;
    const block_nvfp4 vb = *(const block_nvfp4 *) (v_base + block * v_nb0);
    uint8_t * data_dst = out_data + slice * data_slice_bytes + row * row_data_bytes + block * (QK_NVFP4 / 2);
#pragma unroll
    for (int i = 0; i < QK_NVFP4 / 2; ++i) {
        data_dst[i] = vb.qs[i];
    }

    // Match the current native-slice path: keep the V tensor global scale out of the
    // UE4M3 A scale channel and fold its reciprocal into the per-P-row F32 post scale.
    const float block_scale = ggml_cuda_e4m3_to_fp32(vb.e);
    const int64_t scale_idx = ggml_cuda_nvfp4_scale_tiled_index(row, block, scale_inner_padded);
    out_scale[slice * scale_slice_bytes + scale_idx] = ggml_cuda_nvfp4_lt_scale_from_f32(block_scale);
}

static __global__ void k_stage_vcache_nvfp4_all_p_for_lt(
        const block_nvfp4 * __restrict__ p_q,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        int64_t kv_size,
        int64_t cols,
        int64_t q_heads,
        int64_t q_streams,
        int64_t row_data_bytes,
        int64_t data_slice_bytes,
        int64_t scale_inner_padded,
        int64_t scale_slice_bytes) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_blocks = kv_size / QK_NVFP4;
    const int64_t elems_per_slice = cols * n_blocks;
    const int64_t total = elems_per_slice * q_heads * q_streams;
    if (idx >= total) {
        return;
    }

    const int64_t slice = idx / elems_per_slice;
    const int64_t rem = idx - slice * elems_per_slice;
    const int64_t col = rem / n_blocks;
    const int64_t block = rem - col * n_blocks;
    const int64_t p_row = slice * cols + col;
    const block_nvfp4 pb = p_q[p_row * n_blocks + block];

    uint8_t * data_dst = out_data + slice * data_slice_bytes + col * row_data_bytes + block * (QK_NVFP4 / 2);
#pragma unroll
    for (int i = 0; i < QK_NVFP4 / 2; ++i) {
        data_dst[i] = pb.qs[i];
    }

    const float block_scale = ggml_cuda_e4m3_to_fp32(pb.e);
    const int64_t scale_idx = ggml_cuda_nvfp4_scale_tiled_index(col, block, scale_inner_padded);
    out_scale[slice * scale_slice_bytes + scale_idx] = ggml_cuda_nvfp4_lt_scale_from_f32(block_scale);
}

static __global__ void k_store_vcache_nvfp4_lt_all_results(
        const float * __restrict__ lt_data,
        const float * __restrict__ p_scale,
        float * __restrict__ dst_data,
        int64_t rows,
        int64_t cols,
        int64_t lt_cols,
        int64_t q_heads,
        int64_t q_streams,
        int64_t dst_nb1,
        int64_t dst_nb2,
        int64_t dst_nb3) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t elems_per_slice = rows * cols;
    const int64_t slices = q_heads * q_streams;
    const int64_t total = elems_per_slice * slices;
    if (idx >= total) {
        return;
    }

    const int64_t slice = idx / elems_per_slice;
    const int64_t rem = idx - slice * elems_per_slice;
    const int64_t col = rem / rows;
    const int64_t row = rem - col * rows;
    const int64_t q_head = slice % q_heads;
    const int64_t q_stream = slice / q_heads;
    const int64_t p_row = slice * cols + col;
    const size_t lt_off = ((size_t) slice * (size_t) lt_cols + (size_t) col) * (size_t) rows + (size_t) row;
    const float v = lt_data[lt_off] * p_scale[p_row];
    char * dst_ptr = (char *) dst_data + row * sizeof(float) + col * dst_nb1 + q_head * dst_nb2 + q_stream * dst_nb3;
    *(float *) dst_ptr = v;
}

static bool ggml_cuda_vcache_nvfp4_matmul_fp4_p_lt(
        ggml_backend_cuda_context & ctx,
        const block_nvfp4 * v_data,
        const block_nvfp4 * p_q,
        const float * p_scale,
        float * dst_data,
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
        int64_t dst_nb1,
        int64_t dst_nb2,
        int64_t dst_nb3,
        int64_t r2,
        int64_t r3) {
#if GGML_CUDA_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS
    if (kv_size % QK_NVFP4 != 0 || rows <= 0 || cols <= 0 ||
            kv_heads <= 0 || q_heads <= 0 || kv_streams <= 0 || q_streams <= 0) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t n_blocks = kv_size / QK_NVFP4;
    const int64_t lt_k = ggml_cuda_nvfp4_pad_i64(kv_size, 32);
    const int64_t lt_blocks = lt_k / QK_NVFP4;
    const int64_t lt_cols = (cols + 15) & ~15LL;
    const int64_t row_data_bytes = lt_k / 2;
    const int64_t scale_inner_padded = ggml_cuda_nvfp4_pad_i64(lt_blocks, 4);
    const int64_t a_scale_outer_padded = ggml_cuda_nvfp4_pad_i64(rows, 128);
    const int64_t b_scale_outer_padded = ggml_cuda_nvfp4_pad_i64(lt_cols, 128);
    const int64_t a_data_nbytes = rows * row_data_bytes;
    const int64_t b_data_nbytes = lt_cols * row_data_bytes;
    const int64_t a_scale_nbytes = a_scale_outer_padded * scale_inner_padded;
    const int64_t b_scale_nbytes = b_scale_outer_padded * scale_inner_padded;

    const int64_t kv_slices = kv_heads * kv_streams;
    const int64_t lt_slices = q_heads * q_streams;
    ggml_cuda_pool_alloc<uint8_t> a_data(ctx.pool(), (size_t) a_data_nbytes * (size_t) kv_slices);
    ggml_cuda_pool_alloc<uint8_t> a_scale(ctx.pool(), (size_t) a_scale_nbytes * (size_t) kv_slices);
    ggml_cuda_pool_alloc<uint8_t> b_data(ctx.pool(), (size_t) b_data_nbytes * (size_t) lt_slices);
    ggml_cuda_pool_alloc<uint8_t> b_scale(ctx.pool(), (size_t) b_scale_nbytes * (size_t) lt_slices);
    ggml_cuda_pool_alloc<float> lt_dst(ctx.pool(), (size_t) rows * (size_t) lt_cols * (size_t) lt_slices);

    CUDA_CHECK(cudaMemsetAsync(a_data.get(), 0, (size_t) a_data_nbytes * (size_t) kv_slices, stream));
    CUDA_CHECK(cudaMemsetAsync(a_scale.get(), 0, (size_t) a_scale_nbytes * (size_t) kv_slices, stream));
    CUDA_CHECK(cudaMemsetAsync(b_data.get(), 0, (size_t) b_data_nbytes * (size_t) lt_slices, stream));
    CUDA_CHECK(cudaMemsetAsync(b_scale.get(), 0, (size_t) b_scale_nbytes * (size_t) lt_slices, stream));

    const int block_size = 256;
    const int64_t a_total = kv_slices * rows * n_blocks;
    const int a_grid = (int) ((a_total + block_size - 1) / block_size);
    k_stage_vcache_nvfp4_all_v_for_lt<<<a_grid, block_size, 0, stream>>>(
            v_data, a_data.get(), a_scale.get(),
            kv_size, rows, kv_heads, kv_streams,
            v_nb0, v_nb1, v_nb2, v_nb3,
            row_data_bytes, a_data_nbytes, scale_inner_padded, a_scale_nbytes);
    CUDA_CHECK(cudaGetLastError());

    const int64_t b_total = lt_slices * cols * n_blocks;
    const int b_grid = (int) ((b_total + block_size - 1) / block_size);
    k_stage_vcache_nvfp4_all_p_for_lt<<<b_grid, block_size, 0, stream>>>(
            p_q, b_data.get(), b_scale.get(),
            kv_size, cols, q_heads, q_streams,
            row_data_bytes, b_data_nbytes, scale_inner_padded, b_scale_nbytes);
    CUDA_CHECK(cudaGetLastError());

    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    const char * stage = "layout_create_a";
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    if (st == CUBLAS_STATUS_SUCCESS) {
        st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, (uint64_t) lt_k, (uint64_t) rows, (int64_t) lt_k);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        stage = "layout_set_order_a";
        st = cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_b";
        st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, (uint64_t) lt_k, (uint64_t) lt_cols, (int64_t) lt_k);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        stage = "layout_set_order_b";
        st = cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_c";
        st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, (uint64_t) rows, (uint64_t) lt_cols, (int64_t) rows);
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        stage = "layout_set_order_c";
        st = cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    }

    const bool parallel_lt = ggml_cuda_nvfp4_vcache_parallel_lt_enabled();
    const int64_t lt_stream_count = parallel_lt ? std::min<int64_t>(GGML_CUDA_MAX_STREAMS, lt_slices) : 1;
    std::vector<cudaStream_t> lt_streams((size_t) lt_stream_count);
    std::vector<cublasLtMatmulDesc_t> slice_descs((size_t) lt_slices, nullptr);
    std::vector<cudaEvent_t> done_events;
    cudaEvent_t staging_done = nullptr;

    if (st == CUBLAS_STATUS_SUCCESS) {
        for (int64_t i = 0; i < lt_stream_count; ++i) {
            lt_streams[(size_t) i] = i == 0 ? stream : ctx.stream(ctx.device, (int) i);
        }
        for (int64_t i = 0; i < lt_slices && st == CUBLAS_STATUS_SUCCESS; ++i) {
            stage = "slice_matmul_desc_create";
            st = cublasLtMatmulDescCreate(&slice_descs[(size_t) i], CUBLAS_COMPUTE_32F, CUDA_R_32F);
            if (st != CUBLAS_STATUS_SUCCESS) {
                break;
            }
            const cudaDataType_t scale_type = CUDA_R_32F;
            stage = "slice_matmul_desc_set_scale_type";
            st = cublasLtMatmulDescSetAttribute(
                    slice_descs[(size_t) i], CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type));
            if (st != CUBLAS_STATUS_SUCCESS) {
                break;
            }
            const cublasOperation_t op_t = CUBLAS_OP_T;
            stage = "slice_matmul_desc_set_transa";
            st = cublasLtMatmulDescSetAttribute(
                    slice_descs[(size_t) i], CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
            if (st != CUBLAS_STATUS_SUCCESS) {
                break;
            }
            const cublasOperation_t op_n = CUBLAS_OP_N;
            stage = "slice_matmul_desc_set_transb";
            st = cublasLtMatmulDescSetAttribute(
                    slice_descs[(size_t) i], CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
            if (st != CUBLAS_STATUS_SUCCESS) {
                break;
            }
            const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            stage = "slice_matmul_desc_set_a_scale_mode";
            st = cublasLtMatmulDescSetAttribute(
                    slice_descs[(size_t) i], CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
            if (st != CUBLAS_STATUS_SUCCESS) {
                break;
            }
            stage = "slice_matmul_desc_set_b_scale_mode";
            st = cublasLtMatmulDescSetAttribute(
                    slice_descs[(size_t) i], CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
        }
    }

    if (st == CUBLAS_STATUS_SUCCESS && lt_stream_count > 1) {
        CUDA_CHECK(cudaEventCreateWithFlags(&staging_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(staging_done, stream));
        for (int64_t i = 1; i < lt_stream_count; ++i) {
            CUDA_CHECK(cudaStreamWaitEvent(lt_streams[(size_t) i], staging_done, 0));
        }
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        for (int64_t q_stream = 0; q_stream < q_streams && st == CUBLAS_STATUS_SUCCESS; ++q_stream) {
            const int64_t kv_stream = q_stream / r3;
            for (int64_t q_head = 0; q_head < q_heads && st == CUBLAS_STATUS_SUCCESS; ++q_head) {
                const int64_t kv_head = q_head / r2;
                const int64_t kv_slice = kv_stream * kv_heads + kv_head;
                const int64_t q_slice = q_stream * q_heads + q_head;
                cublasLtMatmulDesc_t slice_desc = slice_descs[(size_t) q_slice];
                const void * a_scale_ptr = a_scale.get() + (size_t) kv_slice * (size_t) a_scale_nbytes;
                const void * b_scale_ptr = b_scale.get() + (size_t) q_slice * (size_t) b_scale_nbytes;
                stage = "matmul_desc_set_a_scale_ptr_slice";
                st = cublasLtMatmulDescSetAttribute(
                        slice_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
                if (st != CUBLAS_STATUS_SUCCESS) {
                    break;
                }
                stage = "matmul_desc_set_b_scale_ptr_slice";
                st = cublasLtMatmulDescSetAttribute(
                        slice_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));
                if (st != CUBLAS_STATUS_SUCCESS) {
                    break;
                }

                const float alpha = 1.0f;
                const float beta = 0.0f;
                cudaStream_t lt_stream = lt_streams[(size_t) (q_slice % lt_stream_count)];
                stage = "matmul";
                st = cublasLtMatmul(
                        ctx.cublaslt_handle(),
                        slice_desc,
                        &alpha,
                        a_data.get() + (size_t) kv_slice * (size_t) a_data_nbytes, a_desc,
                        b_data.get() + (size_t) q_slice * (size_t) b_data_nbytes, b_desc,
                        &beta,
                        lt_dst.get() + (size_t) q_slice * (size_t) rows * (size_t) lt_cols, c_desc,
                        lt_dst.get() + (size_t) q_slice * (size_t) rows * (size_t) lt_cols, c_desc,
                        nullptr,
                        nullptr, 0,
                        lt_stream);
            }
        }
    }

    if (lt_stream_count > 1) {
        done_events.resize((size_t) lt_stream_count - 1, nullptr);
        for (int64_t i = 1; i < lt_stream_count; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&done_events[(size_t) i - 1], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(done_events[(size_t) i - 1], lt_streams[(size_t) i]));
            CUDA_CHECK(cudaStreamWaitEvent(stream, done_events[(size_t) i - 1], 0));
        }
    }

    if (st == CUBLAS_STATUS_SUCCESS) {
        const int block_size = 256;
        const int64_t total = rows * cols * lt_slices;
        const int grid = (int) ((total + block_size - 1) / block_size);
        k_store_vcache_nvfp4_lt_all_results<<<grid, block_size, 0, stream>>>(
                lt_dst.get(), p_scale, dst_data, rows, cols, lt_cols, q_heads, q_streams,
                dst_nb1, dst_nb2, dst_nb3);
        CUDA_CHECK(cudaGetLastError());
    }

    for (cudaEvent_t event : done_events) {
        if (event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(event));
        }
    }
    if (staging_done != nullptr) {
        CUDA_CHECK(cudaEventDestroy(staging_done));
    }
    for (cublasLtMatmulDesc_t desc : slice_descs) {
        if (desc != nullptr) {
            cublasLtMatmulDescDestroy(desc);
        }
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
    if (st != CUBLAS_STATUS_SUCCESS) {
        ggml_cuda_nvfp4_log_vcache_lt_failure_once(stage, (int) st, cublas_get_error_str(st));
        return false;
    }

    ggml_cuda_nvfp4_log_vcache_lt_active_once(rows, cols, lt_cols, kv_size, q_heads, q_streams);

    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(v_data);
    GGML_UNUSED(p_q);
    GGML_UNUSED(p_scale);
    GGML_UNUSED(dst_data);
    GGML_UNUSED(kv_size);
    GGML_UNUSED(rows);
    GGML_UNUSED(cols);
    GGML_UNUSED(kv_heads);
    GGML_UNUSED(q_heads);
    GGML_UNUSED(kv_streams);
    GGML_UNUSED(q_streams);
    GGML_UNUSED(v_nb0);
    GGML_UNUSED(v_nb1);
    GGML_UNUSED(v_nb2);
    GGML_UNUSED(v_nb3);
    GGML_UNUSED(dst_nb1);
    GGML_UNUSED(dst_nb2);
    GGML_UNUSED(dst_nb3);
    GGML_UNUSED(r2);
    GGML_UNUSED(r3);
    ggml_cuda_nvfp4_log_vcache_lt_scale_attrs_unavailable_once();
    return false;
#endif
}
#endif

bool ggml_cuda_mul_mat_vcache_nvfp4_batched(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (ggml_cuda_nvfp4_vcache_batched_conflicting_experiment_enabled()) {
        return false;
    }

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
    if (scale->data == nullptr) {
        return false;
    }

    const int64_t kv_size = src0->ne[0];
    const int64_t cols = src1->ne[1];
    const int64_t q_heads = src1->ne[2];
    const int64_t q_streams = src1->ne[3];

    if (src1->ne[0] != kv_size || cols <= 0 || q_heads <= 0 || q_streams <= 0) {
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

    if (src0->nb[0] != (int64_t) sizeof(block_nvfp4) || scale->nb[0] != (int64_t) sizeof(float) || src1->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }

    // Flatten all P columns/heads/streams into one row batch. The real unified
    // V-cache graph uses this dense layout; unusual views keep the existing path.
    if (src1->nb[1] != sizeof(float) * (size_t) kv_size ||
            src1->nb[2] != src1->nb[1] * cols ||
            src1->nb[3] != src1->nb[2] * q_heads) {
        return false;
    }

    const int64_t r2 = q_heads / kv_heads;
    const int64_t r3 = q_streams / kv_streams;

    const int64_t n_blocks = kv_size / QK_NVFP4;
    const int64_t p_rows = cols * q_heads * q_streams;
    ggml_cuda_pool_alloc<block_nvfp4> p_q(ctx.pool(), (size_t) p_rows * (size_t) n_blocks);
    ggml_cuda_pool_alloc<float> p_amax(ctx.pool(), (size_t) p_rows);
    ggml_cuda_pool_alloc<float> p_scale(ctx.pool(), (size_t) p_rows);
    ggml_cuda_nvfp4_abs_max_rows_f32(
            (const float *) src1->data,
            p_amax.get(),
            kv_size,
            p_rows,
            kv_size,
            false,
            ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    const int scale_threads = 256;
    const int scale_grid = (int) ((p_rows + scale_threads - 1) / scale_threads);
    k_prepare_vcache_nvfp4_p_scales<<<scale_grid, scale_threads, 0, ctx.stream()>>>(
            p_amax.get(),
            (const float *) scale->data,
            p_scale.get(),
            p_rows,
            cols,
            q_heads,
            q_streams,
            kv_streams,
            scale_stream_nb,
            r3);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_nvfp4_quantize_rows_dynamic_f32(
            (const float *) src1->data,
            p_q.get(),
            kv_size,
            kv_size,
            p_rows,
            p_amax.get(),
            false,
            false,
            ctx.stream());
    CUDA_CHECK(cudaGetLastError());

#if GGML_CUDA_HAS_CUBLASLT
    if (ggml_cuda_vcache_nvfp4_matmul_fp4_p_lt(
                ctx,
                (const block_nvfp4 *) src0->data,
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
                dst->nb[1],
                dst->nb[2],
                dst->nb[3],
                r2,
                r3)) {
        return true;
    }
#endif

    return false;
}
