#include "nvfp4.cuh"

#include "quantize.cuh"

#include <array>

#if CUDART_VERSION >= 12080

static_assert(sizeof(block_nvfp4) == 36, "unexpected block_nvfp4 size");

template<typename T>
static __device__ __forceinline__ T ggml_cuda_nvfp4_zero() {
    return T{};
}

static __global__ void ggml_cuda_nvfp4_split_block(
    const block_nvfp4 * __restrict__ src,
    uint8_t * __restrict__ qdata,
    uint8_t * __restrict__ scales,
    const int64_t rows,
    const int64_t cols,
    const int64_t src_blocks_per_row,
    const int64_t blocks_per_row_pad) {

    const int64_t ib64 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ib64 >= rows * src_blocks_per_row) {
        return;
    }

    const int64_t row = ib64 / src_blocks_per_row;
    const int64_t block64 = ib64 % src_blocks_per_row;

    const block_nvfp4 & in = src[ib64];

    const int64_t base16 = block64 * 4;
    const int64_t base_q = row * blocks_per_row_pad * 8 + base16 * 8;

#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int64_t block16 = base16 + s;
        const size_t scale_off = ggml_cuda_nvfp4_scale_offset(row, block16, blocks_per_row_pad);
        scales[scale_off] = in.d[s];

        const uint8_t * src_q = in.qs + s * 8;
        uint8_t * dst_q = qdata + base_q + s * 8;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst_q[i] = src_q[i];
        }
    }

    GGML_UNUSED(cols);
}

static __global__ void ggml_cuda_nvfp4_quantize_row(
    const float * __restrict__ src,
    uint8_t * __restrict__ qdata,
    uint8_t * __restrict__ scales,
    const int64_t rows,
    const int64_t cols,
    const int64_t blocks_per_row_pad) {

    const int64_t block16 = blockIdx.x;
    const int64_t row = blockIdx.y;
    const int lane = threadIdx.x;

    if (lane >= 16 || row >= rows) {
        return;
    }

    const int64_t col0 = block16 * 16;
    const int64_t row_off = row * cols;

    float x = 0.0f;
    if (col0 + lane < cols) {
        x = src[row_off + col0 + lane];
    }

    float amax = fabsf(x);
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 16));
    }

    __shared__ float   inv_scale;

    if (lane == 0) {
        const uint8_t s = ggml_cuda_float_to_ue4m3(amax / 6.0f);
        inv_scale = (amax == 0.0f) ? 0.0f : 1.0f / ggml_cuda_ue4m3_to_fp32(s);
        scales[ggml_cuda_nvfp4_scale_offset(row, block16, blocks_per_row_pad)] = s;
    }
    __syncthreads();

    const uint8_t q = ggml_cuda_float_to_fp4_e2m1(x, inv_scale);
    const int64_t q_block = row * blocks_per_row_pad + block16;

    if (lane < 8) {
        const uint8_t q0 = q;
        const uint8_t q1 = __shfl_sync(0xFFFFFFFF, q, lane + 8, 16);
        qdata[q_block * 8 + lane] = q0 | (q1 << 4);
    }
}

void ggml_cuda_nvfp4_destroy_weight_cache(ggml_cuda_nvfp4_weight_cache * cache) {
    if (cache == nullptr) {
        return;
    }

    ggml_cuda_set_device(cache->device);
    if (cache->qdata != nullptr) {
        CUDA_CHECK(cudaFree(cache->qdata));
    }
    if (cache->scales != nullptr) {
        CUDA_CHECK(cudaFree(cache->scales));
    }
    delete cache;
}

void ggml_cuda_nvfp4_destroy_plan(ggml_cuda_nvfp4_plan * plan) {
    if (plan == nullptr) {
        return;
    }

    if (plan->a_layout != nullptr) {
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan->a_layout));
    }
    if (plan->b_layout != nullptr) {
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan->b_layout));
    }
    if (plan->c_layout != nullptr) {
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan->c_layout));
    }
    if (plan->d_layout != nullptr) {
        CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(plan->d_layout));
    }
    delete plan;
}

static ggml_cuda_nvfp4_weight_cache * ggml_cuda_nvfp4_get_weight_cache(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, int device) {

    auto & cache_map = ctx.nvfp4_weight_cache[device];
    auto it = cache_map.find(src0);
    if (it != cache_map.end()) {
        ggml_cuda_nvfp4_weight_cache * cache = it->second;
        if (cache->rows == src0->ne[1] && cache->cols == src0->ne[0] && cache->src0_data == src0->data) {
            return cache;
        }
        ggml_cuda_nvfp4_destroy_weight_cache(cache);
        cache_map.erase(it);
    }

    if (src0->type != GGML_TYPE_NVFP4 || !ggml_is_contiguous(src0)) {
        return nullptr;
    }

    const int64_t rows = src0->ne[1];
    const int64_t cols = src0->ne[0];
    if (cols % QK_NVFP4 != 0) {
        return nullptr;
    }

    auto * cache = new ggml_cuda_nvfp4_weight_cache();
    cache->device = device;
    cache->rows = rows;
    cache->cols = cols;
    cache->rows_pad = ggml_cuda_nvfp4_pad_rows(rows);
    cache->blocks_per_row = cols / QK_NVFP4_SUB;
    cache->blocks_per_row_pad = ggml_cuda_nvfp4_pad_blocks(cache->blocks_per_row);
    cache->qdata_size = (size_t) cache->rows_pad * (size_t) cache->blocks_per_row_pad * 8;
    cache->scales_size = (size_t) cache->rows_pad * (size_t) cache->blocks_per_row_pad;
    cache->qdata = nullptr;
    cache->scales = nullptr;
    cache->src0_data = src0->data;

    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMalloc(&cache->qdata, cache->qdata_size));
    CUDA_CHECK(cudaMalloc(&cache->scales, cache->scales_size));
    CUDA_CHECK(cudaMemset(cache->qdata, 0, cache->qdata_size));
    CUDA_CHECK(cudaMemset(cache->scales, 0, cache->scales_size));

    const int64_t src_blocks_per_row = cols / QK_NVFP4;
    const int64_t n_src_blocks = rows * src_blocks_per_row;
    const dim3 block_size(128, 1, 1);
    const dim3 grid_size((n_src_blocks + block_size.x - 1) / block_size.x, 1, 1);

    ggml_cuda_nvfp4_split_block<<<grid_size, block_size, 0, ctx.stream(device, 0)>>>(
        (const block_nvfp4 *) src0->data,
        (uint8_t *) cache->qdata,
        (uint8_t *) cache->scales,
        rows,
        cols,
        src_blocks_per_row,
        cache->blocks_per_row_pad);
    CUDA_CHECK(cudaGetLastError());

    cache_map.emplace(src0, cache);
    return cache;
}

static ggml_cuda_nvfp4_plan * ggml_cuda_nvfp4_get_plan(
    ggml_backend_cuda_context & ctx,
    int device,
    int64_t rows_pad,
    int64_t cols_pad,
    int64_t k_pad,
    const void * a_scale,
    const void * b_scale,
    const void * c_ptr,
    const void * d_ptr) {

    auto & cache_map = ctx.nvfp4_plan_cache[device];
    const uint64_t key = ggml_cuda_nvfp4_plan_key(rows_pad, cols_pad, k_pad);
    auto it = cache_map.find(key);
    if (it != cache_map.end()) {
        return it->second;
    }

    auto * plan = new ggml_cuda_nvfp4_plan();
    plan->rows_pad = rows_pad;
    plan->cols_pad = cols_pad;
    plan->k_pad = k_pad;

    ggml_cuda_set_device(device);

    // Use the same operand orientation as the existing cublasGemmEx path:
    // activations are A and weights are B, with the result written as M x N.
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->a_layout, CUDA_R_4F_E2M1, cols_pad, k_pad, k_pad));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->b_layout, CUDA_R_4F_E2M1, k_pad, rows_pad, k_pad));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->c_layout, CUDA_R_32F, cols_pad, rows_pad, cols_pad));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->d_layout, CUDA_R_32F, cols_pad, rows_pad, cols_pad));

    const cublasLtOrder_t order_row = CUBLASLT_ORDER_ROW;
    const int64_t ld_a = k_pad;
    const int64_t ld_b = k_pad;
    const int64_t ld_c = cols_pad;
    const int64_t ld_d = cols_pad;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan->a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)));
    const cublasLtOrder_t order_col = CUBLASLT_ORDER_COL;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan->b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan->c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan->d_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_col, sizeof(order_col)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan->a_layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld_a, sizeof(ld_a)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan->b_layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld_b, sizeof(ld_b)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan->c_layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld_c, sizeof(ld_c)));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(plan->d_layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld_d, sizeof(ld_d)));

    cublasLtMatmulDesc_t matmul_desc = nullptr;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    const cublasOperation_t transa = CUBLAS_OP_N;
    const cublasOperation_t transb = CUBLAS_OP_T;
    const int scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    cublasLtMatmulPreference_t pref = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    const size_t max_workspace = 16ull * 1024ull * 1024ull;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int algo_count = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        ctx.cublaslt_handle(device), matmul_desc, plan->a_layout, plan->b_layout, plan->c_layout, plan->d_layout, pref, 1, &heuristic, &algo_count);

    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmul_desc));

    if (status != CUBLAS_STATUS_SUCCESS || algo_count <= 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
        ggml_cuda_nvfp4_destroy_plan(plan);
        return nullptr;
    }

    plan->algo = heuristic.algo;
    plan->workspace_size = heuristic.workspaceSize;

    GGML_UNUSED(c_ptr);
    GGML_UNUSED(d_ptr);

    cache_map.emplace(key, plan);
    return plan;
}

bool ggml_cuda_should_use_nvfp4(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (!blackwell_nvfp4_available(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->view_src != nullptr || src1->view_src != nullptr || dst->view_src != nullptr) {
        return false;
    }

    const int64_t rows = src0->ne[1];
    const int64_t cols = src0->ne[0];
    const int64_t nrows = src1->ne[1];

    if (rows <= 0 || cols <= 0 || nrows <= 0) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (cols % QK_NVFP4 != 0) {
        return false;
    }
    return false;
}

static void ggml_cuda_nvfp4_matmul(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    cudaStream_t stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!ggml_cuda_should_use_nvfp4(src0, src1, dst, cc)) {
        GGML_ABORT("fatal error");
    }

    const int64_t rows = row_high - row_low;
    const int64_t cols = src0->ne[0];
    const int64_t cols_pad = ggml_cuda_nvfp4_pad_blocks(cols / QK_NVFP4_SUB) * QK_NVFP4_SUB;
    const int64_t rows_pad = ggml_cuda_nvfp4_pad_rows(rows);
    const int64_t n_pad = ggml_cuda_nvfp4_pad_rows(src1_ncols);
    const int64_t k_pad = cols_pad;

    ggml_cuda_nvfp4_weight_cache * weight_cache = ggml_cuda_nvfp4_get_weight_cache(ctx, src0, device);
    if (weight_cache == nullptr) {
        GGML_ABORT("fatal error");
    }

    ggml_cuda_pool_alloc<uint8_t> src1_qdata(ctx.pool(device));
    ggml_cuda_pool_alloc<uint8_t> src1_scales(ctx.pool(device));

    const size_t src1_qdata_size = (size_t) n_pad * (size_t) (cols_pad / 16) * 8;
    const size_t src1_scales_size = (size_t) n_pad * (size_t) ggml_cuda_nvfp4_pad_blocks(cols_pad / 16);

    src1_qdata.alloc(src1_qdata_size);
    src1_scales.alloc(src1_scales_size);
    CUDA_CHECK(cudaMemsetAsync(src1_qdata.get(), 0, src1_qdata_size, stream));
    CUDA_CHECK(cudaMemsetAsync(src1_scales.get(), 0, src1_scales_size, stream));

    const dim3 quant_blocks((cols_pad / 16), n_pad, 1);
    const dim3 quant_threads(16, 1, 1);
    ggml_cuda_nvfp4_quantize_row<<<quant_blocks, quant_threads, 0, stream>>>(
        src1_ddf_i,
        src1_qdata.get(),
        src1_scales.get(),
        src1_ncols,
        cols_pad,
        ggml_cuda_nvfp4_pad_blocks(cols_pad / 16));
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_nvfp4_plan * plan = ggml_cuda_nvfp4_get_plan(
        ctx, device, rows, n_pad, k_pad,
        weight_cache->scales,
        src1_scales.get(),
        dst_dd_i,
        dst_dd_i);
    if (plan == nullptr) {
        GGML_ABORT("fatal error");
    }

    cublasLtMatmulDesc_t matmul_desc = nullptr;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    const cublasOperation_t transa = CUBLAS_OP_N;
    const cublasOperation_t transb = CUBLAS_OP_T;
    const int scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
    const void * a_scale = weight_cache->scales;
    const void * b_scale = src1_scales.get();
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    ggml_cuda_pool_alloc<float> dst_tmp(ctx.pool(device));
    dst_tmp.alloc((size_t) rows * (size_t) n_pad);

    void * workspace = nullptr;
    ggml_cuda_pool_alloc<uint8_t> workspace_alloc(ctx.pool(device));
    if (plan->workspace_size > 0) {
        workspace_alloc.alloc(plan->workspace_size);
        workspace = workspace_alloc.get();
    }

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(device), stream));
    CUBLAS_CHECK(cublasLtMatmul(
        ctx.cublaslt_handle(device), matmul_desc,
        &alpha,
        src1_qdata.get(), plan->a_layout,
        weight_cache->qdata, plan->b_layout,
        &beta,
        dst_tmp.get(), plan->c_layout,
        dst_tmp.get(), plan->d_layout,
        &plan->algo,
        workspace,
        plan->workspace_size,
        stream));

    CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmul_desc));

    CUDA_CHECK(cudaMemcpy2DAsync(
        dst_dd_i,
        (size_t) src1_ncols * sizeof(float),
        dst_tmp.get(),
        (size_t) n_pad * sizeof(float),
        (size_t) src1_ncols * sizeof(float),
        (size_t) rows,
        cudaMemcpyDeviceToDevice,
        stream));

    GGML_UNUSED(src0_dd_i);
}

void ggml_cuda_op_mul_mat_cublaslt_nvfp4(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
    ggml_cuda_nvfp4_matmul(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, dst_dd_i, row_low, row_high, src1_ncols, stream);
}

#else

bool ggml_cuda_should_use_nvfp4(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(cc);
    return false;
}

void ggml_cuda_nvfp4_destroy_weight_cache(ggml_cuda_nvfp4_weight_cache * cache) {
    GGML_UNUSED(cache);
}

void ggml_cuda_nvfp4_destroy_plan(ggml_cuda_nvfp4_plan * plan) {
    GGML_UNUSED(plan);
}

void ggml_cuda_op_mul_mat_cublaslt_nvfp4(
    ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src0_dd_i);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(dst_dd_i);
    GGML_UNUSED(row_low);
    GGML_UNUSED(row_high);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(stream);
    GGML_ABORT("NVFP4 CUDA path requires CUDA 12.8 or newer");
}

#endif // CUDART_VERSION >= 12080
