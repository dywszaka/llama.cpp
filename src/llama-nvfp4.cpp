#include "llama-nvfp4.h"

#include "llama-log.h"

#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#undef GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace {

static void nvfp4_roundtrip_f32_ref(
        const float * x,
        float * y,
        int64_t ncol,
        float global_scale,
        std::vector<block_nvfp4> & q_buf) {
    q_buf.resize(ncol / QK_NVFP4);
    quantize_row_nvfp4_ref(x, q_buf.data(), ncol, global_scale);
    dequantize_row_nvfp4(q_buf.data(), y, ncol, global_scale);
}

static bool nvfp4_fetch_scalar_f32(const ggml_tensor * t, float & out) {
    if (!t || !t->data) {
        return false;
    }
    const void * src = t->data;
    switch (t->type) {
        case GGML_TYPE_F32: {
            float v = 0.0f;
            memcpy(&v, src, sizeof(v));
            out = v;
            return true;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t v = 0;
            memcpy(&v, src, sizeof(v));
            out = ggml_fp16_to_fp32(v);
            return true;
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t v = { 0 };
            memcpy(&v, src, sizeof(v));
            out = ggml_bf16_to_fp32(v);
            return true;
        }
        default:
            return false;
    }
}

static void nvfp4_act_roundtrip_op_impl(
        ggml_tensor * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        int ith,
        int nth,
        void * userdata) {
    GGML_UNUSED(userdata);

    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(dst));

    float global_scale = 0.0f;
    if (!nvfp4_fetch_scalar_f32(b, global_scale)) {
        memcpy(dst->data, a->data, ggml_nbytes(a));
        return;
    }

    global_scale = (global_scale != 0.0f) ? (1.0f / global_scale) : 0.0f;

    const int64_t ncols = a->ne[0];
    const int64_t nrows = a->ne[1];
    if (a->ne[2] != 1 || a->ne[3] != 1 || ncols % QK_NVFP4 != 0 || global_scale == 0.0f) {
        memcpy(dst->data, a->data, ggml_nbytes(a));
        return;
    }

    std::vector<block_nvfp4> q_buf;

    const int64_t row_start = (nrows * ith) / nth;
    const int64_t row_end   = (nrows * (ith + 1)) / nth;

    const size_t row_stride = a->nb[1] / sizeof(float);
    for (int64_t r = row_start; r < row_end; ++r) {
        const float * x = (const float *) a->data + r * row_stride;
        float * y = (float *) dst->data + r * row_stride;
        const bool log_once = llama_log::nvfp4_dequant_debug() &&
                              (ith == 0) && (r == row_start);
        float before_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int64_t log_cnt = std::min<int64_t>(4, ncols);
        if (log_once) {
            for (int64_t i = 0; i < log_cnt; ++i) {
                before_vals[i] = x[i];
            }
        }

        nvfp4_roundtrip_f32_ref(x, y, ncols, global_scale, q_buf);

        if (log_once) {
            float after_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int64_t i = 0; i < log_cnt; ++i) {
                after_vals[i] = y[i];
            }
            fprintf(
                stderr,
                "[NVFP4-DEQUANT] input before: x[:4]=[%.6f, %.6f, %.6f, %.6f], input_global_scale=%.6f\n",
                before_vals[0], before_vals[1], before_vals[2], before_vals[3],
                global_scale);
            fprintf(
                stderr,
                "[NVFP4-DEQUANT]input after: x[:4]=[%.6f, %.6f, %.6f, %.6f]\n",
                after_vals[0], after_vals[1], after_vals[2], after_vals[3]);
        }
    }
}

} // namespace

extern "C" void ggml_nvfp4_act_roundtrip_op(
        ggml_tensor * dst,
        const ggml_tensor * a,
        const ggml_tensor * b,
        int ith,
        int nth,
        void * userdata) {
    nvfp4_act_roundtrip_op_impl(dst, a, b, ith, nth, userdata);
}
