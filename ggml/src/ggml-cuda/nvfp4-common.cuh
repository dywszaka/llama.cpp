#pragma once

#include "common.cuh"

#include <cuda_fp8.h>

static constexpr float GGML_CUDA_NVFP4_FP4_MAX = 6.0f;
static constexpr float GGML_CUDA_NVFP4_E4M3_HALF_MAX = 224.0f;
static constexpr float GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX =
        GGML_CUDA_NVFP4_FP4_MAX * GGML_CUDA_NVFP4_E4M3_HALF_MAX;

static inline int64_t ggml_cuda_nvfp4_pad_i64(int64_t x, int64_t a) {
    GGML_ASSERT(a > 0);
    return ((x + a - 1) / a) * a;
}

static __host__ __device__ __forceinline__ int64_t ggml_cuda_nvfp4_scale_tiled_index(
        int64_t outer,
        int64_t inner,
        int64_t n_inner_padded) {
    // cuBLASLt VEC16_UE4M3 scale tiling: [outer, inner] -> 128x4 tiled order.
    const int64_t outer_tile = outer / 128;
    const int64_t outer_in_tile = outer % 128;
    const int64_t inner_tile = inner / 4;
    const int64_t inner_in_tile = inner % 4;

    const int64_t tiles_per_outer_block = n_inner_padded / 4;
    const int64_t tile_base = (outer_tile * tiles_per_outer_block + inner_tile) * 512;
    const int64_t tile_offset = (outer_in_tile % 32) * 16 + (outer_in_tile / 32) * 4 + inner_in_tile;
    return tile_base + tile_offset;
}

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_lt_scale_from_f32(float scale_f) {
    if (!(scale_f > 0.0f) || !isfinite(scale_f)) {
        return 0;
    }

    return (uint8_t) __nv_cvt_float_to_fp8(scale_f, __NV_SATFINITE, __NV_E4M3);
}
