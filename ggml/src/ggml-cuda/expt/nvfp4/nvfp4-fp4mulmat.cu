#include "nvfp4-fp4mulmat.cuh"

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
