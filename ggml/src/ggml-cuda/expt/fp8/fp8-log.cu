#include "fp8-log.cuh"

#include <cstdlib>
#include <cstring>

void ggml_cuda_fp8_log_e4m3_e8m0_32_e4m2_cpy_once(
        const char * path,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        bool enabled) {
    static int logged_f32_q_enabled = 0;
    static int logged_f32_q_disabled = 0;
    static int logged_repack_enabled = 0;
    static int logged_repack_disabled = 0;

    const bool is_repack = std::strcmp(path, "transpose_permute_repack") == 0;
    int * logged = is_repack
            ? (enabled ? &logged_repack_enabled : &logged_repack_disabled)
            : (enabled ? &logged_f32_q_enabled : &logged_f32_q_disabled);

    if (*logged != 0) {
        return;
    }
    *logged = 1;

    const char * env = getenv("GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2");
    GGML_LOG_INFO(
            "%s: path=%s GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2=%s -> %s; src=%s type=%s dst=%s type=%s dst_ne=[%lld,%lld,%lld,%lld]\n",
            __func__,
            path,
            env != nullptr ? env : "(unset)",
            enabled ? "enabled, CUDA cpy will mask FP8 mantissa low bit (E4M2 experiment)"
                    : "disabled, CUDA cpy keeps FP8 E4M3",
            ggml_get_name(src0),
            ggml_type_name(src0->type),
            ggml_get_name(src1),
            ggml_type_name(src1->type),
            (long long) src1->ne[0], (long long) src1->ne[1],
            (long long) src1->ne[2], (long long) src1->ne[3]);
}

void ggml_cuda_fp8_log_e4m3_e8m0_32_e4m2_set_rows_once(
        const ggml_tensor * dst,
        bool enabled) {
    static int logged_enabled = 0;
    static int logged_disabled = 0;

    int * logged = enabled ? &logged_enabled : &logged_disabled;
    if (*logged != 0) {
        return;
    }
    *logged = 1;

    const char * env = getenv("GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2");
    GGML_LOG_INFO(
            "%s: GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2=%s -> %s; dst=%s type=%s ne=[%lld,%lld,%lld,%lld]\n",
            __func__,
            env != nullptr ? env : "(unset)",
            enabled ? "enabled, CUDA set_rows will mask FP8 mantissa low bit (E4M2 experiment)"
                    : "disabled, CUDA set_rows keeps FP8 E4M3",
            ggml_get_name(dst),
            ggml_type_name(dst->type),
            (long long) dst->ne[0], (long long) dst->ne[1],
            (long long) dst->ne[2], (long long) dst->ne[3]);
}
