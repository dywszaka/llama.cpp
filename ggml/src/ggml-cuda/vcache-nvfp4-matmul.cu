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
        int64_t & scale_stream_nb) {
    blocks = src0->ne[0] / QK_NVFP4;
    rows = src0->ne[1];
    heads = src0->ne[2];
    streams = src0->ne[3];

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

static __global__ void k_vcache_nvfp4_matmul_4d(
        const block_nvfp4 * __restrict__ v_data,
        const float * __restrict__ v_scale,
        const float * __restrict__ p_data,
        float * __restrict__ dst_data,
        int64_t kv_size,
        int64_t rows,
        int64_t cols,
        int64_t heads,
        int64_t streams,
        int64_t v_nb0,
        int64_t v_nb1,
        int64_t v_nb2,
        int64_t v_nb3,
        int64_t scale_nb0,
        int64_t scale_row_nb,
        int64_t scale_head_nb,
        int64_t scale_stream_nb,
        int64_t p_nb1,
        int64_t p_nb2,
        int64_t p_nb3,
        int64_t dst_nb1,
        int64_t dst_nb2,
        int64_t dst_nb3) {
    const int64_t row = blockIdx.x;
    const int64_t col = blockIdx.y;
    const int64_t head = blockIdx.z % heads;
    const int64_t stream = blockIdx.z / heads;
    const int lane = threadIdx.x;

    if (row >= rows || col >= cols || head >= heads || stream >= streams || lane >= kv_size) {
        return;
    }

    const int64_t block = lane / QK_NVFP4;
    const int64_t in_block = lane % QK_NVFP4;

    const char * v_base = (const char *) v_data + row * v_nb1 + head * v_nb2 + stream * v_nb3;
    const block_nvfp4 * v_block_ptr = (const block_nvfp4 *) (v_base + block * v_nb0);

    const char * scale_base = (const char *) v_scale + row * scale_row_nb + head * scale_head_nb + stream * scale_stream_nb;
    const float input_scale = *(const float *) (scale_base + block * scale_nb0);

    const block_nvfp4 vb = *v_block_ptr;
    const float d = ggml_cuda_e4m3_to_fp32_half(vb.e) * input_scale;
    const uint8_t packed = vb.qs[in_block / 2];
    const uint8_t q = (in_block & 1) == 0 ? (packed & 0x0F) : (packed >> 4);
    const float v = d * (float) kvalues_nvfp4[q];

    const char * p_ptr = (const char *) p_data + lane * sizeof(float) + col * p_nb1 + head * p_nb2 + stream * p_nb3;
    const float p = *(const float *) p_ptr;
    const float prod = v * p;

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
        char * dst_ptr = (char *) dst_data + row * sizeof(float) + col * dst_nb1 + head * dst_nb2 + stream * dst_nb3;
        *(float *) dst_ptr = sum[0];
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

    const ggml_tensor * scale = ggml_tensor_get_nvfp4_scale(src0);
    int64_t blocks = 0;
    int64_t rows = 0;
    int64_t heads = 0;
    int64_t streams = 0;
    int64_t scale_row_nb = 0;
    int64_t scale_head_nb = 0;
    int64_t scale_stream_nb = 0;
    if (!ggml_cuda_match_vcache_nvfp4_scale_layout(src0, scale, blocks, rows, heads, streams, scale_row_nb, scale_head_nb, scale_stream_nb)) {
        return false;
    }

    const int64_t kv_size = src0->ne[0];
    const int64_t cols = src1->ne[1];

    if (src1->ne[0] != kv_size || src1->ne[2] != heads || src1->ne[3] != streams) {
        return false;
    }

    if (dst->ne[0] != rows || dst->ne[1] != cols || dst->ne[2] != heads || dst->ne[3] != streams) {
        return false;
    }

    if (kv_size > 256 || (kv_size & (kv_size - 1)) != 0) {
        return false;
    }

    if (src0->nb[0] != (int64_t) sizeof(block_nvfp4) || scale->nb[0] != (int64_t) sizeof(float) || src1->nb[0] != (int64_t) sizeof(float) || dst->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }

    const dim3 grid((uint32_t) rows, (uint32_t) cols, (uint32_t) (heads * streams));
    const dim3 block((uint32_t) kv_size, 1, 1);
    k_vcache_nvfp4_matmul_4d<<<grid, block, 0, ctx.stream()>>>(
            (const block_nvfp4 *) src0->data,
            (const float *) scale->data,
            (const float *) src1->data,
            (float *) dst->data,
            kv_size,
            rows,
            cols,
            heads,
            streams,
            src0->nb[0],
            src0->nb[1],
            src0->nb[2],
            src0->nb[3],
            scale->nb[0],
            scale_row_nb,
            scale_head_nb,
            scale_stream_nb,
            src1->nb[1],
            src1->nb[2],
            src1->nb[3],
            dst->nb[1],
            dst->nb[2],
            dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
