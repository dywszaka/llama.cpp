#include "vcache-nvfp4-matmul.cuh"

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

    if (scale->ne[0] != src0->ne[0] / QK_NVFP4 || scale->ne[1] != src0->ne[1]) {
        return false;
    }

    return true;
}

static __global__ void k_vcache_nvfp4_matmul(
        const block_nvfp4 * __restrict__ v_blocks,
        const float * __restrict__ v_scales,
        const float * __restrict__ p,
        float * __restrict__ dst,
        int64_t kv_size,
        int64_t n_rows,
        int64_t n_cols) {
    const int64_t row = blockIdx.x;
    const int64_t col = blockIdx.y;
    const int64_t lane = threadIdx.x;

    if (row >= n_rows || col >= n_cols || lane >= kv_size) {
        return;
    }

    const int64_t block = lane / QK_NVFP4;
    const int64_t in_block = lane % QK_NVFP4;
    const block_nvfp4 vb = v_blocks[row * (kv_size / QK_NVFP4) + block];
    const float input_scale = v_scales[row * (kv_size / QK_NVFP4) + block];
    const float d = ggml_cuda_e4m3_to_fp32_half(vb.e) * input_scale;
    const uint8_t packed = vb.qs[in_block / 2];
    const uint8_t q = (in_block & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
    const float v = d * (float) kvalues_nvfp4[q];
    const float prod = v * p[col * kv_size + lane];

    __shared__ float sum[256];
    sum[threadIdx.x] = prod;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dst[col * n_rows + row] = sum[0];
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

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    const int64_t kv_size = src0->ne[0];
    const int64_t n_rows = src0->ne[1];
    const int64_t n_cols = src1->ne[1];

    if (src1->ne[0] != kv_size || dst->ne[0] != n_rows || dst->ne[1] != n_cols) {
        return false;
    }

    if (kv_size > 256 || (kv_size & (kv_size - 1)) != 0) {
        return false;
    }

    const dim3 grid((uint32_t) n_rows, (uint32_t) n_cols, 1);
    const dim3 block((uint32_t) kv_size, 1, 1);
    k_vcache_nvfp4_matmul<<<grid, block, 0, ctx.stream()>>>(
            (const block_nvfp4 *) src0->data,
            (const float *) ggml_tensor_get_nvfp4_scale(src0)->data,
            (const float *) src1->data,
            (float *) dst->data,
            kv_size,
            n_rows,
            n_cols);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
