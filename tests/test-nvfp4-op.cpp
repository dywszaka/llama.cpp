// Unit test for ggml_nvfp4_to_f32_op

#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-nvfp4.h"

#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#undef GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-impl.h"

#undef NDEBUG
#include <assert.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static float nvfp4_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static uint8_t nvfp4_fp32_to_e4m3(float x) {
    struct e4m3_table {
        float vals[256];
        uint8_t valid[256];
    };
    static const e4m3_table table = []() {
        e4m3_table t{};
        for (int i = 0; i < 256; ++i) {
            const float v = GGML_E4M3_TO_FP32((uint8_t) i);
            t.vals[i] = v;
            t.valid[i] = std::isfinite(v) ? 1 : 0;
        }
        return t;
    }();

    if (!std::isfinite(x)) {
        return 0;
    }

    float best_err = std::numeric_limits<float>::infinity();
    uint8_t best_i = 0;
    for (int i = 0; i < 256; ++i) {
        if (!table.valid[i]) {
            continue;
        }
        const float err = fabsf(table.vals[i] - x);
        if (err < best_err) {
            best_err = err;
            best_i = (uint8_t) i;
        }
    }

    return best_i;
}

static float nvfp4_fp4_quantize(float x) {
    static const float kvalues_fp4[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,
        2.0f,  3.0f,  4.0f,  6.0f,
        0.0f, -0.5f, -1.0f, -1.5f,
       -2.0f, -3.0f, -4.0f, -6.0f,
    };

    float best = kvalues_fp4[0];
    float best_err = fabsf(x - best);
    for (int i = 1; i < 16; ++i) {
        const float err = fabsf(x - kvalues_fp4[i]);
        if (err < best_err) {
            best_err = err;
            best = kvalues_fp4[i];
        }
    }
    return best;
}

static void nvfp4_quantize_row_ref(
        const float * x,
        float * q,
        float * scales,
        int64_t ncol,
        float global_scale) {
    static const int64_t qk = QK_NVFP4;
    static const float k_fp4_max = 6.0f;

    assert(ncol % qk == 0);

    const int64_t nblocks = ncol / qk;
    for (int64_t ib = 0; ib < nblocks; ++ib) {
        const int64_t base = ib * qk;
        float vmax = 0.0f;
        for (int64_t i = 0; i < qk; ++i) {
            vmax = std::max(vmax, fabsf(x[base + i]));
        }

        float scale = global_scale * (vmax / k_fp4_max);
        scale = nvfp4_clampf(scale, -448.0f, 448.0f);
        const uint8_t scale_q = nvfp4_fp32_to_e4m3(scale);
        const float scale_f = GGML_E4M3_TO_FP32(scale_q);
        scales[ib] = scale_f;

        const float inv_scale = (global_scale != 0.0f && scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;
        for (int64_t i = 0; i < qk; ++i) {
            const float scaled = x[base + i] * inv_scale;
            const float clipped = nvfp4_clampf(scaled, -k_fp4_max, k_fp4_max);
            q[base + i] = nvfp4_fp4_quantize(clipped);
        }
    }
}

static void nvfp4_dequantize_row_ref(
        const float * q,
        const float * scales,
        float * y,
        int64_t ncol,
        float global_scale) {
    static const int64_t qk = QK_NVFP4;

    assert(ncol % qk == 0);

    const int64_t nblocks = ncol / qk;
    for (int64_t ib = 0; ib < nblocks; ++ib) {
        const float scale_f = scales[ib];
        const float out_scale = (global_scale != 0.0f) ? (scale_f / global_scale) : 0.0f;
        const int64_t base = ib * qk;
        for (int64_t i = 0; i < qk; ++i) {
            y[base + i] = q[base + i] * out_scale;
        }
    }
}

static void nvfp4_to_f32_ref(
        const float * x,
        float * y,
        int64_t ncol,
        float global_scale) {
    std::vector<float> q(ncol);
    std::vector<float> scales(ncol / QK_NVFP4);
    nvfp4_quantize_row_ref(x, q.data(), scales.data(), ncol, global_scale);
    nvfp4_dequantize_row_ref(q.data(), scales.data(), y, ncol, global_scale);
}

static bool nvfp4_scale_is_recip() {
    const char * env = std::getenv("LLAMA_NVFP4_INPUT_SCALE_RECIP");
    return env && std::strcmp(env, "0") != 0;
}

static float max_abs_diff(const float * a, const float * b, size_t n) {
    float max = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max = std::max(max, fabsf(a[i] - b[i]));
    }
    return max;
}

int main() {
    const size_t ctx_size = 1024 * 1024;
    void * ctx_buf = malloc(ctx_size);
    assert(ctx_buf != nullptr);

    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ctx_buf,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    const int64_t ncols = 32;
    const int64_t nrows = 2;
    const size_t n = ncols * nrows;

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = 0.25f + 0.75f * sinf((float) i * 0.31f);
    }
    memcpy(src->data, input.data(), n * sizeof(float));

    float scale_value = 0.5f;
    memcpy(scale->data, &scale_value, sizeof(scale_value));

    ggml_nvfp4_to_f32_op(dst, src, scale, 0, 1, nullptr);

    float ref_scale = scale_value;
    if (nvfp4_scale_is_recip()) {
        ref_scale = ref_scale != 0.0f ? 1.0f / ref_scale : 0.0f;
    }

    std::vector<float> expected(n);
    for (int64_t r = 0; r < nrows; ++r) {
        nvfp4_to_f32_ref(
                input.data() + r * ncols,
                expected.data() + r * ncols,
                ncols,
                ref_scale);
    }

    const float err = max_abs_diff(expected.data(), (float *) dst->data, n);
    if (err > 1e-6f) {
        fprintf(stderr, "nvfp4_to_f32_op mismatch: max_err=%.6g\n", err);
        return 1;
    }

    // global_scale == 0 should copy input
    const float zero_scale = 0.0f;
    memcpy(scale->data, &zero_scale, sizeof(zero_scale));
    ggml_nvfp4_to_f32_op(dst, src, scale, 0, 1, nullptr);
    const float copy_err = max_abs_diff(input.data(), (float *) dst->data, n);
    if (copy_err != 0.0f) {
        fprintf(stderr, "nvfp4_to_f32_op copy mismatch: max_err=%.6g\n", copy_err);
        return 1;
    }

    // non-multiple of QK_NVFP4 should copy input
    ggml_tensor * src_odd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 20, 1);
    ggml_tensor * dst_odd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 20, 1);
    std::vector<float> odd_in(20, 0.123f);
    memcpy(src_odd->data, odd_in.data(), odd_in.size() * sizeof(float));
    memcpy(scale->data, &scale_value, sizeof(scale_value));
    ggml_nvfp4_to_f32_op(dst_odd, src_odd, scale, 0, 1, nullptr);
    const float odd_err = max_abs_diff(odd_in.data(), (float *) dst_odd->data, odd_in.size());
    if (odd_err != 0.0f) {
        fprintf(stderr, "nvfp4_to_f32_op odd-shape mismatch: max_err=%.6g\n", odd_err);
        return 1;
    }

    ggml_free(ctx);
    free(ctx_buf);

    return 0;
}
