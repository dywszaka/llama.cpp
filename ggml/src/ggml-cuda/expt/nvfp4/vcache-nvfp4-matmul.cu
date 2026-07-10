#include "vcache-nvfp4-matmul.cuh"

#include "nvfp4-fp4mulmat.cuh"
#include "nvfp4-log.cuh"
#include "nvfp4-quantize-core.cuh"

static constexpr int64_t GGML_CUDA_VCACHE_NVFP4_FP4_P_AMAX_PREPASS_MIN_KV = 2048;
static constexpr int64_t GGML_CUDA_VCACHE_NVFP4_FP4_PV_LT_PAD_K = 512;

#if defined(CUBLAS_VERSION)
#define GGML_CUDA_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VERSION >= 130000)
#elif defined(CUBLAS_VER_MAJOR)
#define GGML_CUDA_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS (CUBLAS_VER_MAJOR >= 13)
#else
#define GGML_CUDA_VCACHE_NVFP4_HAS_LT_SCALE_CHANNEL_ATTRS 0
#endif

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

static bool ggml_cuda_match_vcache_nvfp4_scale_layout(
        const ggml_tensor * src0,
        const ggml_tensor * scale,
        int64_t & blocks,
        int64_t & rows,
        int64_t & heads,
        int64_t & streams,
        int64_t & scale_row_nb,
        int64_t & scale_head_nb,
        int64_t & scale_stream_nb,
        bool & scale_is_global) {
    blocks = src0->ne[0] / QK_NVFP4;
    rows = src0->ne[1];
    heads = src0->ne[2];
    streams = src0->ne[3];
    scale_is_global = false;

    if (ggml_nelements(scale) == streams) {
        scale_row_nb = 0;
        scale_head_nb = 0;
        scale_stream_nb = streams > 1 ? scale->nb[0] : 0;
        scale_is_global = true;
        return true;
    }

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
            (GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / row_amax) : 0.0f;
    if (lane == 0) {
        p_scale[p_row] = global_scale != 0.0f ? (1.0f / global_scale) : 0.0f;
    }

    ggml_cuda_nvfp4_core_quantize_block_f32(
            x, active, global_scale, p_q + p_row * (kv_size / QK_NVFP4) + block);
}

#if GGML_CUDA_HAS_CUBLASLT
static __global__ void k_stage_vcache_nvfp4_v_for_lt(
        const block_nvfp4 * __restrict__ v_data,
        const float * __restrict__ v_scale,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        int64_t kv_size,
        int64_t rows,
        int64_t kv_head,
        int64_t kv_stream,
        int64_t v_nb0,
        int64_t v_nb1,
        int64_t v_nb2,
        int64_t v_nb3,
        int64_t scale_nb0,
        int64_t scale_row_nb,
        int64_t scale_head_nb,
        int64_t scale_stream_nb,
        bool scale_is_global,
        int64_t row_data_bytes,
        int64_t scale_inner_padded) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_blocks = kv_size / QK_NVFP4;
    const int64_t total = rows * n_blocks;
    if (idx >= total) {
        return;
    }

    const int64_t row = idx / n_blocks;
    const int64_t block = idx - row * n_blocks;
    const char * v_base = (const char *) v_data + row * v_nb1 + kv_head * v_nb2 + kv_stream * v_nb3;
    const char * scale_base = (const char *) v_scale + row * scale_row_nb + kv_head * scale_head_nb + kv_stream * scale_stream_nb;
    const float v_global_scale = scale_is_global ? *(const float *) ((const char *) v_scale + kv_stream * scale_stream_nb) : 0.0f;

    const block_nvfp4 vb = *(const block_nvfp4 *) (v_base + block * v_nb0);
    uint8_t * data_dst = out_data + row * row_data_bytes + block * (QK_NVFP4 / 2);
#pragma unroll
    for (int i = 0; i < QK_NVFP4 / 2; ++i) {
        data_dst[i] = vb.qs[i];
    }

    const float block_scale = scale_is_global ?
        (v_global_scale > 0.0f ? ggml_cuda_e4m3_to_fp32(vb.e) / v_global_scale : 0.0f) :
        ggml_cuda_e4m3_to_fp32(vb.e) * (*(const float *) (scale_base + block * scale_nb0));
    const int64_t scale_idx = ggml_cuda_nvfp4_scale_tiled_index(row, block, scale_inner_padded);
    out_scale[scale_idx] = ggml_cuda_nvfp4_lt_scale_from_f32(block_scale);
}

static __global__ void k_stage_vcache_nvfp4_p_for_lt(
        const block_nvfp4 * __restrict__ p_q,
        uint8_t * __restrict__ out_data,
        uint8_t * __restrict__ out_scale,
        int64_t kv_size,
        int64_t cols,
        int64_t q_head,
        int64_t q_stream,
        int64_t q_heads,
        int64_t row_data_bytes,
        int64_t scale_inner_padded) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_blocks = kv_size / QK_NVFP4;
    const int64_t total = cols * n_blocks;
    if (idx >= total) {
        return;
    }

    const int64_t col = idx / n_blocks;
    const int64_t block = idx - col * n_blocks;
    const int64_t p_row = (q_stream * q_heads + q_head) * cols + col;
    const block_nvfp4 pb = p_q[p_row * n_blocks + block];

    uint8_t * data_dst = out_data + col * row_data_bytes + block * (QK_NVFP4 / 2);
#pragma unroll
    for (int i = 0; i < QK_NVFP4 / 2; ++i) {
        data_dst[i] = pb.qs[i];
    }

    const float block_scale = ggml_cuda_e4m3_to_fp32(pb.e);
    const int64_t scale_idx = ggml_cuda_nvfp4_scale_tiled_index(col, block, scale_inner_padded);
    out_scale[scale_idx] = ggml_cuda_nvfp4_lt_scale_from_f32(block_scale);
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
        const float * v_scale,
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
        int64_t scale_nb0,
        int64_t scale_row_nb,
        int64_t scale_head_nb,
        int64_t scale_stream_nb,
        bool scale_is_global,
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
    const int64_t lt_k = std::max(kv_size, GGML_CUDA_VCACHE_NVFP4_FP4_PV_LT_PAD_K);
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

    ggml_cuda_pool_alloc<uint8_t> a_data(ctx.pool(), (size_t) a_data_nbytes);
    ggml_cuda_pool_alloc<uint8_t> a_scale(ctx.pool(), (size_t) a_scale_nbytes);
    ggml_cuda_pool_alloc<uint8_t> b_data(ctx.pool(), (size_t) b_data_nbytes);
    ggml_cuda_pool_alloc<uint8_t> b_scale(ctx.pool(), (size_t) b_scale_nbytes);
    const int64_t lt_slices = q_heads * q_streams;
    ggml_cuda_pool_alloc<float> lt_dst(ctx.pool(), (size_t) rows * (size_t) lt_cols * (size_t) lt_slices);

    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    const char * stage = "matmul_desc_create";
    cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cudaDataType_t scale_type = CUDA_R_32F;
        stage = "matmul_desc_set_scale_type";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasOperation_t op_t = CUBLAS_OP_T;
        stage = "matmul_desc_set_transa";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasOperation_t op_n = CUBLAS_OP_N;
        stage = "matmul_desc_set_transb";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        stage = "matmul_desc_set_a_scale_mode";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        stage = "matmul_desc_set_b_scale_mode";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const void * a_scale_ptr = a_scale.get();
        stage = "matmul_desc_set_a_scale_ptr";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale_ptr, sizeof(a_scale_ptr));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        const void * b_scale_ptr = b_scale.get();
        stage = "matmul_desc_set_b_scale_ptr";
        st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale_ptr, sizeof(b_scale_ptr));
    }
    if (st == CUBLAS_STATUS_SUCCESS) {
        stage = "layout_create_a";
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

    if (st == CUBLAS_STATUS_SUCCESS) {
        const int block_size = 256;
        for (int64_t q_stream = 0; q_stream < q_streams && st == CUBLAS_STATUS_SUCCESS; ++q_stream) {
            const int64_t kv_stream = q_stream / r3;
            for (int64_t q_head = 0; q_head < q_heads && st == CUBLAS_STATUS_SUCCESS; ++q_head) {
                const int64_t kv_head = q_head / r2;
                CUDA_CHECK(cudaMemsetAsync(a_data.get(), 0, (size_t) a_data_nbytes, stream));
                CUDA_CHECK(cudaMemsetAsync(a_scale.get(), 0, (size_t) a_scale_nbytes, stream));
                CUDA_CHECK(cudaMemsetAsync(b_data.get(), 0, (size_t) b_data_nbytes, stream));
                CUDA_CHECK(cudaMemsetAsync(b_scale.get(), 0, (size_t) b_scale_nbytes, stream));

                const int a_grid = (int) ((rows * n_blocks + block_size - 1) / block_size);
                k_stage_vcache_nvfp4_v_for_lt<<<a_grid, block_size, 0, stream>>>(
                        v_data, v_scale, a_data.get(), a_scale.get(),
                        kv_size, rows, kv_head, kv_stream,
                        v_nb0, v_nb1, v_nb2, v_nb3,
                        scale_nb0, scale_row_nb, scale_head_nb, scale_stream_nb, scale_is_global,
                        row_data_bytes, scale_inner_padded);
                CUDA_CHECK(cudaGetLastError());

                const int b_grid = (int) ((cols * n_blocks + block_size - 1) / block_size);
                k_stage_vcache_nvfp4_p_for_lt<<<b_grid, block_size, 0, stream>>>(
                        p_q, b_data.get(), b_scale.get(),
                        kv_size, cols, q_head, q_stream, q_heads,
                        row_data_bytes, scale_inner_padded);
                CUDA_CHECK(cudaGetLastError());

                const float alpha = 1.0f;
                const float beta = 0.0f;
                stage = "matmul";
                st = cublasLtMatmul(
                        ctx.cublaslt_handle(),
                        op_desc,
                        &alpha,
                        a_data.get(), a_desc,
                        b_data.get(), b_desc,
                        &beta,
                        lt_dst.get() + (size_t) (q_stream * q_heads + q_head) * (size_t) rows * (size_t) lt_cols, c_desc,
                        lt_dst.get() + (size_t) (q_stream * q_heads + q_head) * (size_t) rows * (size_t) lt_cols, c_desc,
                        nullptr,
                        nullptr, 0,
                        stream);
            }
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
        ggml_cuda_nvfp4_log_vcache_lt_failure_once(stage, (int) st, cublas_get_error_str(st));
        return false;
    }

    ggml_cuda_nvfp4_log_vcache_lt_active_once(rows, cols, lt_cols, kv_size, q_heads, q_streams);

    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(v_data);
    GGML_UNUSED(v_scale);
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
    GGML_UNUSED(scale_nb0);
    GGML_UNUSED(scale_row_nb);
    GGML_UNUSED(scale_head_nb);
    GGML_UNUSED(scale_stream_nb);
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
    int64_t blocks = 0;
    int64_t rows = 0;
    int64_t kv_heads = 0;
    int64_t kv_streams = 0;
    int64_t scale_row_nb = 0;
    int64_t scale_head_nb = 0;
    int64_t scale_stream_nb = 0;
    bool scale_is_global = false;
    if (!ggml_cuda_match_vcache_nvfp4_scale_layout(src0, scale, blocks, rows, kv_heads, kv_streams, scale_row_nb, scale_head_nb, scale_stream_nb, scale_is_global)) {
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

    if (src0->nb[0] != (int64_t) sizeof(block_nvfp4) || scale->nb[0] != (int64_t) sizeof(float) || src1->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }

    const int64_t r2 = q_heads / kv_heads;
    const int64_t r3 = q_streams / kv_streams;
    ggml_cuda_nvfp4_log_vcache_fp4_pv_once();

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

    const bool force_fp4mulmat = ggml_cuda_nvfp4_fp4mulmat_enabled();
    if (force_fp4mulmat) {
        ggml_cuda_nvfp4_log_vcache_fp4mulmat_forced_once();
    }

#if GGML_CUDA_HAS_CUBLASLT
    if (!force_fp4mulmat && ggml_cuda_vcache_nvfp4_matmul_fp4_p_lt(
                ctx,
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
                scale_is_global,
                dst->nb[1],
                dst->nb[2],
                dst->nb[3],
                r2,
                r3)) {
        ggml_cuda_nvfp4_log_vcache_matmul_path_once("cublasLt-fp4");
        return true;
    }
#endif

    ggml_cuda_nvfp4_log_vcache_matmul_path_once(force_fp4mulmat ? "fp4_mulmat-derived-custom-cuda-fp4" : "custom-cuda-fp4");
    ggml_cuda_nvfp4_fp4mulmat_vcache_cuda(
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
            scale_is_global,
            dst->nb[1],
            dst->nb[2],
            dst->nb[3],
            r2,
            r3,
            ctx.stream());
    return true;
}
