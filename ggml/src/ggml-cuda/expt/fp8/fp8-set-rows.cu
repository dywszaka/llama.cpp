#include "fp8-set-rows.cuh"

#include "../../cpy-utils.cuh"

#include <cstdlib>

static bool ggml_cuda_fp8_env_flag_enabled(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && atoi(env) != 0;
}

bool ggml_cuda_fp8_e4m3_e8m0_32_e4m2_experiment_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_cuda_fp8_env_flag_enabled("GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2") ? 1 : 0;
    }
    return cached != 0;
}

static __device__ void quantize_f32_fp8_e4m3_e8m0_32_e4m2_block(const float * __restrict__ x, block_fp8_e4m3_e8m0_32 * __restrict__ y) {
    quantize_f32_fp8_e4m3_e8m0_32_block(x, y, true);
}

static __device__ void quantize_f32_fp8_e4m3_e8m0_32_e4m3_block(const float * __restrict__ x, block_fp8_e4m3_e8m0_32 * __restrict__ y) {
    quantize_f32_fp8_e4m3_e8m0_32_block(x, y, false);
}

template<typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static __global__ void k_set_rows_fp8_quant(
        const float * __restrict__ src0, const int64_t * __restrict__ src1, block_type * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1, const int64_t s2, const int64_t s3) {

    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    const int64_t i03 = i_base / (ne00 * ne01 * ne02);
    const int64_t i02 = (i_base - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t i01 = (i_base - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    const int64_t i00 = i_base - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_fp8_cuda_quant(const ggml_cuda_set_rows_params & p, block_type * dst_d) {
    GGML_ASSERT(p.ne00 % qk == 0);
    const int64_t ne_total = (p.ne00 * p.ne01 * p.ne02 * p.ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = p.nb01/sizeof(float);
    const int64_t s02 = p.nb02/sizeof(float);
    const int64_t s03 = p.nb03/sizeof(float);
    const int64_t s10 = p.nb10/sizeof(int64_t);
    const int64_t s11 = p.nb11/sizeof(int64_t);
    const int64_t s12 = p.nb12/sizeof(int64_t);
    const int64_t s1  = p.nb1;
    const int64_t s2  = p.nb2;
    const int64_t s3  = p.nb3;

    if (ne_total > 0) {
        k_set_rows_fp8_quant<block_type, qk, quantize_func><<<grid_size, block_size, 0, p.stream>>>(
            p.src0_d, p.src1_d, dst_d,
            p.ne00, p.ne01, p.ne02, p.ne03,
            p.ne10, p.ne11, p.ne12, p.ne13,
            s01, s02, s03,
            s10, s11, s12,
            s1, s2, s3);
    }
}

void ggml_cuda_set_rows_fp8_e4m3_e8m0_32(
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst,
        bool e4m2_experiment) {
    if (e4m2_experiment) {
        set_rows_fp8_cuda_quant<block_fp8_e4m3_e8m0_32, QK_FP8_E4M3_E8M0_32, quantize_f32_fp8_e4m3_e8m0_32_e4m2_block>(
                p, (block_fp8_e4m3_e8m0_32 *) dst->data);
    } else {
        set_rows_fp8_cuda_quant<block_fp8_e4m3_e8m0_32, QK_FP8_E4M3_E8M0_32, quantize_f32_fp8_e4m3_e8m0_32_e4m3_block>(
                p, (block_fp8_e4m3_e8m0_32 *) dst->data);
    }
}

void ggml_cuda_set_rows_fp8_e4m3_e8m0_16(
        const ggml_cuda_set_rows_params & p,
        ggml_tensor * dst) {
    set_rows_fp8_cuda_quant<block_fp8_e4m3_e8m0_16, QK_FP8_E4M3_E8M0_16, quantize_f32_fp8_e4m3_e8m0_16_block>(
            p, (block_fp8_e4m3_e8m0_16 *) dst->data);
}
