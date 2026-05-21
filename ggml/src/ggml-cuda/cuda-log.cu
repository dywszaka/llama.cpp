#include "cuda-log.cuh"

#include "mmq.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char * ggml_cuda_tensor_name(const ggml_tensor * t) {
    return t != nullptr ? ggml_get_name(t) : "";
}

static bool ggml_cuda_name_contains(const ggml_tensor * t, const char * needle) {
    return std::strstr(ggml_cuda_tensor_name(t), needle) != nullptr;
}

static int ggml_cuda_mul_mat_kqvp_kind(const ggml_tensor * dst) {
    if (ggml_cuda_name_contains(dst, "kqv") || ggml_cuda_name_contains(dst, "KQV")) {
        return 2;
    }
    if (ggml_cuda_name_contains(dst, "kq") || ggml_cuda_name_contains(dst, "KQ")) {
        return 1;
    }
    return 0;
}

static const char * ggml_cuda_mmq_layout_name(mmq_q8_1_ds_layout layout) {
    switch (layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            return "D4";
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            return "DS4";
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            return "D2S6";
        default:
            return "unknown";
    }
}

void ggml_cuda_log_fattn_tensor_brief_once(
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V,
        const ggml_tensor * dst) {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_INFO(
            "%s: Q{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "K{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "V{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "dst{name=%s type=%s ne=[%lld,%lld,%lld,%lld]}\n",
            __func__,
            Q != nullptr ? ggml_get_name(Q) : "(null)",
            Q != nullptr ? ggml_type_name(Q->type) : "(null)",
            Q != nullptr ? (long long) Q->ne[0] : 0, Q != nullptr ? (long long) Q->ne[1] : 0,
            Q != nullptr ? (long long) Q->ne[2] : 0, Q != nullptr ? (long long) Q->ne[3] : 0,
            K != nullptr ? ggml_get_name(K) : "(null)",
            K != nullptr ? ggml_type_name(K->type) : "(null)",
            K != nullptr ? (long long) K->ne[0] : 0, K != nullptr ? (long long) K->ne[1] : 0,
            K != nullptr ? (long long) K->ne[2] : 0, K != nullptr ? (long long) K->ne[3] : 0,
            V != nullptr ? ggml_get_name(V) : "(null)",
            V != nullptr ? ggml_type_name(V->type) : "(null)",
            V != nullptr ? (long long) V->ne[0] : 0, V != nullptr ? (long long) V->ne[1] : 0,
            V != nullptr ? (long long) V->ne[2] : 0, V != nullptr ? (long long) V->ne[3] : 0,
            dst != nullptr ? ggml_get_name(dst) : "(null)",
            dst != nullptr ? ggml_type_name(dst->type) : "(null)",
            dst != nullptr ? (long long) dst->ne[0] : 0, dst != nullptr ? (long long) dst->ne[1] : 0,
            dst != nullptr ? (long long) dst->ne[2] : 0, dst != nullptr ? (long long) dst->ne[3] : 0);
}

void ggml_cuda_log_mul_mat_kqvp_once(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    static std::atomic<bool> qk_logged(false);
    static std::atomic<bool> pv_logged(false);

    const int kind = ggml_cuda_mul_mat_kqvp_kind(dst);
    if (kind == 0) {
        return;
    }

    const char * label = kind == 1 ? "q*k" : "p*v";
    std::atomic<bool> & logged = kind == 1 ? qk_logged : pv_logged;
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_INFO(
            "%s: %s src0{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "src1{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "dst{name=%s type=%s ne=[%lld,%lld,%lld,%lld]}\n",
            __func__,
            label,
            src0 != nullptr ? ggml_get_name(src0) : "(null)",
            src0 != nullptr ? ggml_type_name(src0->type) : "(null)",
            src0 != nullptr ? (long long) src0->ne[0] : 0, src0 != nullptr ? (long long) src0->ne[1] : 0,
            src0 != nullptr ? (long long) src0->ne[2] : 0, src0 != nullptr ? (long long) src0->ne[3] : 0,
            src1 != nullptr ? ggml_get_name(src1) : "(null)",
            src1 != nullptr ? ggml_type_name(src1->type) : "(null)",
            src1 != nullptr ? (long long) src1->ne[0] : 0, src1 != nullptr ? (long long) src1->ne[1] : 0,
            src1 != nullptr ? (long long) src1->ne[2] : 0, src1 != nullptr ? (long long) src1->ne[3] : 0,
            dst != nullptr ? ggml_get_name(dst) : "(null)",
            dst != nullptr ? ggml_type_name(dst->type) : "(null)",
            dst != nullptr ? (long long) dst->ne[0] : 0, dst != nullptr ? (long long) dst->ne[1] : 0,
            dst != nullptr ? (long long) dst->ne[2] : 0, dst != nullptr ? (long long) dst->ne[3] : 0);
}

void ggml_cuda_log_fp8_e4m3_e8m0_32_e4m2_cpy_once(
        const char * path,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        bool enabled) {
    static int logged_f32_q_enabled = 0;
    static int logged_f32_q_disabled = 0;
    static int logged_repack_enabled = 0;
    static int logged_repack_disabled = 0;

    const bool is_repack = strcmp(path, "transpose_permute_repack") == 0;
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

void ggml_cuda_log_fp8_e4m3_e8m0_32_e4m2_set_rows_once(
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

void ggml_cuda_log_nvfp4_vcache_fast_update_once(bool enabled) {
    static int logged = 0;
    if (logged != 0) {
        return;
    }
    logged = 1;

    const char * env = getenv("LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE");
    GGML_LOG_INFO(
            "%s: LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=%s -> %s\n",
            __func__,
            env != nullptr ? env : "(unset)",
            enabled ? "enabled, CUDA NVFP4 V-cache set_rows may patch single-token updates without requantizing the block"
                    : "disabled");
}

void ggml_cuda_log_nvfp4_block(
        const block_nvfp4 & block,
        const ggml_tensor * dst) {
    char buf[512];
    int off = std::snprintf(buf, sizeof(buf), "%s: dst=%s nvfp4 scale_u8=%u vals:",
            __func__, dst ? ggml_get_name(dst) : "unknown", (unsigned) block.e);
    for (int i = 0; i < QK_NVFP4/2 && off > 0 && off < (int) sizeof(buf); ++i) {
        const uint8_t q = block.qs[i];
        off += std::snprintf(buf + off, sizeof(buf) - off, " %u %u",
                (unsigned) (q & 0x0F), (unsigned) (q >> 4));
    }
    GGML_LOG_INFO("%s\n", buf);
}

void ggml_cuda_log_f32_first4(
        const char * label,
        const float vals[4],
        const ggml_tensor * dst) {
    GGML_UNUSED(label);
    GGML_UNUSED(vals);
    GGML_UNUSED(dst);
    // GGML_LOG_INFO("%s: dst=%s %s first4=%.9g %.9g %.9g %.9g\n",
    //         __func__, dst ? ggml_get_name(dst) : "unknown", label,
    //         vals[0], vals[1], vals[2], vals[3]);
}

void ggml_cuda_log_block_q8_1_mmq(
        const block_q8_1_mmq & block,
        ggml_type type_x,
        const ggml_tensor * dst) {
    const mmq_q8_1_ds_layout layout = mmq_get_q8_1_ds_layout(type_x);
    GGML_LOG_INFO("%s: dst=%s block_q8_1_mmq layout=%s\n",
            __func__, dst ? ggml_get_name(dst) : "unknown", ggml_cuda_mmq_layout_name(layout));

    switch (layout) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            GGML_LOG_INFO("%s: dst=%s d4=%.9g %.9g %.9g %.9g\n",
                    __func__, dst ? ggml_get_name(dst) : "unknown",
                    block.d4[0], block.d4[1], block.d4[2], block.d4[3]);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4: {
            const uint16_t * u16 = reinterpret_cast<const uint16_t *>(block.ds4);
            char buf[256];
            int off = std::snprintf(buf, sizeof(buf), "%s: dst=%s ds4_u16:",
                    __func__, dst ? ggml_get_name(dst) : "unknown");
            for (int i = 0; i < 8 && off > 0 && off < (int) sizeof(buf); ++i) {
                off += std::snprintf(buf + off, sizeof(buf) - off, " %u", (unsigned) u16[i]);
            }
            GGML_LOG_INFO("%s\n", buf);
        } break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6: {
            const uint16_t * u16 = reinterpret_cast<const uint16_t *>(block.d2s6);
            char buf[256];
            int off = std::snprintf(buf, sizeof(buf), "%s: dst=%s d2s6_u16:",
                    __func__, dst ? ggml_get_name(dst) : "unknown");
            for (int i = 0; i < 8 && off > 0 && off < (int) sizeof(buf); ++i) {
                off += std::snprintf(buf + off, sizeof(buf) - off, " %u", (unsigned) u16[i]);
            }
            GGML_LOG_INFO("%s\n", buf);
        } break;
        default:
            break;
    }

    for (int i = 0; i < 4*QK8_1; i += 32) {
        char buf[512];
        int off = std::snprintf(buf, sizeof(buf), "%s: dst=%s qs[%d..%d]:",
                __func__, dst ? ggml_get_name(dst) : "unknown", i, i + 31);
        for (int j = i; j < i + 32 && off > 0 && off < (int) sizeof(buf); ++j) {
            off += std::snprintf(buf + off, sizeof(buf) - off, " %d", (int) block.qs[j]);
        }
        GGML_LOG_INFO("%s\n", buf);
    }
}
