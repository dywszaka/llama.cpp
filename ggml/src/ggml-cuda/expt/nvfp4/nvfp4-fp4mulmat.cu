#include "nvfp4-fp4mulmat.cuh"

#include "nvfp4-log.cuh"

#include <cstdlib>

bool ggml_cuda_nvfp4_fp4mulmat_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_FP4MULMAT");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

bool ggml_cuda_nvfp4_fp4mulmat_log_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_CUDA_NVFP4_FP4MULMAT_LOG");
        cached = (env != nullptr && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static __global__ void ggml_cuda_nvfp4_fp4mulmat_kernel(
        const block_nvfp4 * __restrict__ src0,
        const block_nvfp4 * __restrict__ src1_q,
        const float * __restrict__ dynamic_input_scales,
        char * __restrict__ dst,
        const int64_t ne01,
        const int64_t ne11,
        const int64_t nblk_k,
        const int64_t dst_nb0,
        const int64_t dst_nb1,
        const float static_scale,
        const int32_t used_dynamic_scale) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = ne01 * ne11;
    if (idx >= total) {
        return;
    }

    const int64_t row = idx % ne01;
    const int64_t col = idx / ne01;
    const block_nvfp4 * w_row = src0 + row * nblk_k;
    const block_nvfp4 * x_row = src1_q + col * nblk_k;
    const float column_scale = used_dynamic_scale ? dynamic_input_scales[col] : static_scale;

    ggml_cuda_nvfp4_fp4mulmat_accumulator state = { 0, 0 };
    for (int64_t ib = 0; ib < nblk_k; ++ib) {
        ggml_cuda_nvfp4_fp4mulmat_accumulate_block(w_row[ib], x_row[ib], &state);
    }

    *(float *) (dst + row * dst_nb0 + col * dst_nb1) = ggml_cuda_nvfp4_fp4mulmat_accumulator_to_f32(state) * column_scale;
}

void ggml_cuda_nvfp4_fp4mulmat_cuda(
        const block_nvfp4 * src0,
        const block_nvfp4 * src1_q,
        const float * dynamic_input_scales,
        void * dst,
        int64_t ne01,
        int64_t ne11,
        int64_t nblk_k,
        int64_t dst_nb0,
        int64_t dst_nb1,
        float static_scale,
        bool used_dynamic_scale,
        cudaStream_t stream) {
    const int block_size = 256;
    const int64_t total = ne01 * ne11;
    const int grid_size = (int) ((total + block_size - 1) / block_size);
    ggml_cuda_nvfp4_fp4mulmat_kernel<<<grid_size, block_size, 0, stream>>>(
            src0,
            src1_q,
            dynamic_input_scales,
            (char *) dst,
            ne01,
            ne11,
            nblk_k,
            dst_nb0,
            dst_nb1,
            static_scale,
            used_dynamic_scale ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ggml_cuda_nvfp4_fp4mulmat_vcache_kernel(
        const block_nvfp4 * __restrict__ v_data,
        const float * __restrict__ v_scale,
        const block_nvfp4 * __restrict__ p_q,
        const float * __restrict__ p_scale,
        float * __restrict__ dst_data,
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
    const int64_t row = blockIdx.x;
    const int64_t col = blockIdx.y;
    const int64_t head = blockIdx.z % q_heads;
    const int64_t stream = blockIdx.z / q_heads;
    const int64_t kv_head = head / r2;
    const int64_t kv_stream = stream / r3;

    if (row >= rows || col >= cols || head >= q_heads || stream >= q_streams || kv_head >= kv_heads || kv_stream >= kv_streams) {
        return;
    }

    const int64_t n_blocks = kv_size / QK_NVFP4;
    const int64_t p_row = (stream * q_heads + head) * cols + col;
    const char * v_base = (const char *) v_data + row * v_nb1 + kv_head * v_nb2 + kv_stream * v_nb3;
    const char * scale_base = (const char *) v_scale + row * scale_row_nb + kv_head * scale_head_nb + kv_stream * scale_stream_nb;
    const float v_global_scale = scale_is_global ? *(const float *) ((const char *) v_scale + kv_stream * scale_stream_nb) : 0.0f;
    const block_nvfp4 * p_row_q = p_q + p_row * n_blocks;
    const float p_row_scale = p_scale[p_row];

    float thread_sum = 0.0f;
    for (int64_t block = threadIdx.x; block < n_blocks; block += blockDim.x) {
        const block_nvfp4 * v_block_ptr = (const block_nvfp4 *) (v_base + block * v_nb0);
        const block_nvfp4 vb = *v_block_ptr;
        const block_nvfp4 pb = p_row_q[block];
        const float v_external_scale = scale_is_global ?
            (v_global_scale > 0.0f ? 1.0f / v_global_scale : 0.0f) :
            (*(const float *) (scale_base + block * scale_nb0));
        thread_sum += ggml_cuda_nvfp4_fp4mulmat_block_dot_f32(vb, pb) * v_external_scale * p_row_scale;
    }

    __shared__ float sum[256];
    sum[threadIdx.x] = thread_sum;
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

void ggml_cuda_nvfp4_fp4mulmat_vcache_cuda(
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
        int64_t r3,
        cudaStream_t stream) {
    const int64_t n_blocks = kv_size / QK_NVFP4;
    int fp4_block_threads = 16;
    while (fp4_block_threads < n_blocks && fp4_block_threads < 256) {
        fp4_block_threads *= 2;
    }

    ggml_cuda_nvfp4_log_fp4mulmat_vcache_kernel_once(
            "ggml_cuda_nvfp4_fp4mulmat_vcache_kernel", rows, cols, kv_size, q_heads, q_streams);

    const dim3 grid((uint32_t) rows, (uint32_t) cols, (uint32_t) (q_heads * q_streams));
    ggml_cuda_nvfp4_fp4mulmat_vcache_kernel<<<grid, dim3((uint32_t) fp4_block_threads, 1, 1), 0, stream>>>(
            v_data,
            v_scale,
            p_q,
            p_scale,
            dst_data,
            kv_size,
            rows,
            cols,
            kv_heads,
            q_heads,
            kv_streams,
            q_streams,
            v_nb0,
            v_nb1,
            v_nb2,
            v_nb3,
            scale_nb0,
            scale_row_nb,
            scale_head_nb,
            scale_stream_nb,
            scale_is_global,
            dst_nb1,
            dst_nb2,
            dst_nb3,
            r2,
            r3);
    CUDA_CHECK(cudaGetLastError());
}
