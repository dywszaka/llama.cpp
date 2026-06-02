#include "nvfp4-log.cuh"

#include "nvfp4-common.cuh"
#include "../../../ggml-quants.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

bool ggml_cuda_nvfp4_log_can_copy_from_stream(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const cudaError_t err = cudaStreamIsCapturing(stream, &status);
    return err == cudaSuccess && status == cudaStreamCaptureStatusNone;
}

void ggml_cuda_nvfp4_log_bf16_quant_once(const char * env, bool enabled) {
    GGML_LOG_INFO("%s: GGML_CUDA_NVFP4_BF16_QUANT=%s -> %s\n",
            __func__,
            env != nullptr ? env : "(unset)",
            enabled ? "enabled, F32 activations round to BF16 before NVFP4 quantization"
                    : "disabled, using FP32 nearest-neighbor NVFP4 quantization");
}

void ggml_cuda_nvfp4_log_kcache_outlier_counts(
        const char * caller,
        const char * target,
        const int64_t * dst_rows,
        const int32_t * counts,
        const int32_t * offsets,
        const int32_t * cursor,
        int64_t ne01,
        int64_t dst_rows_stride,
        int64_t sidecar_rows,
        int64_t capacity_limit,
        int64_t compact_capacity,
        float threshold,
        cudaStream_t stream) {
    GGML_UNUSED(dst_rows_stride);

    std::vector<int64_t> dst_rows_h((size_t) ne01);
    std::vector<int32_t> counts_h((size_t) sidecar_rows);
    std::vector<int32_t> offsets_h;
    int32_t cursor_h = 0;
    CUDA_CHECK(cudaMemcpyAsync(dst_rows_h.data(), dst_rows, (size_t) ne01 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(counts_h.data(), counts, (size_t) sidecar_rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    if (offsets != nullptr) {
        offsets_h.resize((size_t) sidecar_rows);
        CUDA_CHECK(cudaMemcpyAsync(offsets_h.data(), offsets, (size_t) sidecar_rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        if (cursor != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(&cursor_h, cursor, sizeof(cursor_h), cudaMemcpyDeviceToHost, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int64_t total = 0;
    int32_t max_count = 0;
    int64_t overflow_rows = 0;
    for (int64_t i = 0; i < ne01; ++i) {
        const int64_t dst_row = dst_rows_h[(size_t) i];
        if (dst_row < 0 || dst_row >= sidecar_rows) {
            continue;
        }
        const int32_t c = counts_h[(size_t) dst_row];
        total += c;
        max_count = std::max(max_count, c);
        if (offsets != nullptr) {
            const int32_t offset = offsets_h[(size_t) dst_row];
            overflow_rows += c > 0 && (offset < 0 || (int64_t) offset + c > compact_capacity) ? 1 : 0;
        } else {
            overflow_rows += c > capacity_limit ? 1 : 0;
        }
    }

    GGML_LOG_INFO(
            "%s: target=%s rows=%lld threshold=%g stored_max=%lld compact_capacity=%lld compact_used=%lld total_outliers=%lld max_row_outliers=%d overflow_rows=%lld\n",
            caller,
            target != nullptr ? target : "(unknown)",
            (long long) ne01,
            (double) threshold,
            (long long) capacity_limit,
            (long long) compact_capacity,
            (long long) (offsets != nullptr ? cursor_h : 0),
            (long long) total,
            max_count,
            (long long) overflow_rows);
}

void ggml_cuda_nvfp4_log_vcache_fast_update_once(bool enabled) {
    static int logged = 0;
    if (logged != 0) {
        return;
    }
    logged = 1;

    const char * env = getenv("LLAMA_NVFP4_VCACHE_FAST_UPDATE");
    GGML_LOG_INFO(
            "%s: LLAMA_NVFP4_VCACHE_FAST_UPDATE=%s -> %s\n",
            __func__,
            env != nullptr ? env : "(unset)",
            enabled ? "enabled, CUDA NVFP4 V-cache set_rows may patch single-token updates without requantizing the block"
                    : "disabled");
}

void ggml_cuda_nvfp4_log_vcache_fp4_pv_once() {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_INFO(
            "%s: CUDA NVFP4 V-cache p*v quantizes P to dynamic NVFP4 by default; using cuBLASLt FP4 when available, otherwise custom CUDA dot kernel\n",
            __func__);
}

void ggml_cuda_nvfp4_log_vcache_matmul_path_once(const char * path) {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_INFO("%s: CUDA NVFP4 V-cache p*v matmul path=%s\n", __func__, path);
}

void ggml_cuda_nvfp4_log_vcache_lt_failure_once(const char * stage, int status, const char * status_str) {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_WARN(
            "%s: cuBLASLt FP4 P*V path failed at %s status=%d (%s); falling back to custom CUDA kernel\n",
            __func__, stage, status, status_str);
}

void ggml_cuda_nvfp4_log_vcache_lt_active_once(
        int64_t rows,
        int64_t cols,
        int64_t lt_cols,
        int64_t kv_size,
        int64_t q_heads,
        int64_t q_streams) {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_INFO(
            "%s: cuBLASLt FP4 P*V path active rows=%lld cols=%lld padded_cols=%lld kv_size=%lld q_heads=%lld q_streams=%lld\n",
            __func__,
            (long long) rows,
            (long long) cols,
            (long long) lt_cols,
            (long long) kv_size,
            (long long) q_heads,
            (long long) q_streams);
}

void ggml_cuda_nvfp4_log_vcache_lt_scale_attrs_unavailable_once() {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_WARN("%s: cuBLASLt FP4 scale-channel attrs unavailable; falling back to custom CUDA kernel\n", __func__);
}

void ggml_cuda_nvfp4_log_fattn_tensor_brief_once(
        const char * label,
        const ggml_tensor * a,
        const ggml_tensor * b,
        const ggml_tensor * dst,
        bool qk) {
    static std::atomic<bool> qk_logged(false);
    static std::atomic<bool> pv_logged(false);
    std::atomic<bool> & logged = qk ? qk_logged : pv_logged;
    if (logged.exchange(true)) {
        return;
    }

    GGML_LOG_INFO(
            "%s: %s A{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "B{name=%s type=%s ne=[%lld,%lld,%lld,%lld]} "
            "dst{name=%s type=%s ne=[%lld,%lld,%lld,%lld]}\n",
            __func__,
            label,
            a != nullptr ? ggml_get_name(a) : "(null)",
            a != nullptr ? ggml_type_name(a->type) : "(null)",
            a != nullptr ? (long long) a->ne[0] : 0, a != nullptr ? (long long) a->ne[1] : 0,
            a != nullptr ? (long long) a->ne[2] : 0, a != nullptr ? (long long) a->ne[3] : 0,
            b != nullptr ? ggml_get_name(b) : "(null)",
            b != nullptr ? ggml_type_name(b->type) : "(null)",
            b != nullptr ? (long long) b->ne[0] : 0, b != nullptr ? (long long) b->ne[1] : 0,
            b != nullptr ? (long long) b->ne[2] : 0, b != nullptr ? (long long) b->ne[3] : 0,
            dst != nullptr ? ggml_get_name(dst) : "(null)",
            dst != nullptr ? ggml_type_name(dst->type) : "(null)",
            dst != nullptr ? (long long) dst->ne[0] : 0, dst != nullptr ? (long long) dst->ne[1] : 0,
            dst != nullptr ? (long long) dst->ne[2] : 0, dst != nullptr ? (long long) dst->ne[3] : 0);
}

void ggml_cuda_nvfp4_log_fattn_quantization(
        int group_size,
        double q_global_scale,
        double k_global_scale,
        double v_global_scale,
        bool p_direct,
        bool q_dynamic,
        bool no_q_smooth,
        bool no_k_smooth,
        int64_t batch,
        int64_t q_heads,
        int64_t kv_heads,
        int64_t q_len,
        int64_t kv_len,
        int64_t head_dim) {
    GGML_LOG_INFO(
            "%s: NVFP4 FATTN native quantization: "
            "Q/K group_dim=head_dim group_size=%d tensor_global_scale_inv=[q=%g k=%g] "
            "V group_dim=kv_len group_size=%d tensor_global_scale_inv=%g "
            "P format=%s "
            "Q quant=%s "
            "smooth=[q=%s k=%s] "
            "shape=[batch=%lld q_heads=%lld kv_heads=%lld q_len=%lld kv_len=%lld head_dim=%lld]\n",
            __func__,
            group_size,
            q_global_scale,
            k_global_scale,
            group_size,
            v_global_scale,
            p_direct ?
                "nvfp4_direct group_dim=kv_len first_level=none second_level=NVFP4(global_scale_inv=1)" :
                "nvfp4_twolevel group_dim=kv_len first_level=row_max/(448*6) second_level=NVFP4(global_scale_inv=1)",
            q_dynamic ? "dynamic_per_row" : "static_global",
            no_q_smooth ? "off" : "on",
            no_k_smooth ? "off" : "on",
            (long long) batch,
            (long long) q_heads,
            (long long) kv_heads,
            (long long) q_len,
            (long long) kv_len,
            (long long) head_dim);
}

void ggml_cuda_nvfp4_log_fattn_qk_requested(
        bool k_nvfp4_cache,
        int64_t k,
        int64_t m,
        double weight_scale,
        int64_t n,
        bool q_dynamic,
        double input_scale) {
    GGML_LOG_INFO(
            "%s: QK matmul requested: backend=cublasLt tensor_core=FP4 lt_type=CUDA_R_4F_E2M1 "
            "A=%s[NVFP4,k=%lld,m=%lld,weight_scale=%g%s] "
            "B=Q_centered[F32->NVFP4,k=%lld,n=%lld,%s=%g] "
            "C=F32[%lld,%lld]\n",
            __func__,
            k_nvfp4_cache ? "K_cache_direct" : "K_centered",
            (long long) k,
            (long long) m,
            weight_scale,
            k_nvfp4_cache ? ",row_scale_after_matmul" : "",
            (long long) k,
            (long long) n,
            q_dynamic ? "dynamic_per_row_scale_placeholder" : "input_scale=1/q_global_scale_inv",
            input_scale,
            (long long) m,
            (long long) n);
}

void ggml_cuda_nvfp4_log_fattn_vp_requested(
        int64_t k,
        int64_t m,
        double weight_scale,
        bool p_direct,
        int64_t n) {
    GGML_LOG_INFO(
            "%s: VP matmul requested: backend=cublasLt tensor_core=FP4 lt_type=CUDA_R_4F_E2M1 "
            "A=V[NVFP4,k=%lld,m=%lld,weight_scale=1/v_global_scale_inv=%g] "
            "B=%s[F32->NVFP4,k=%lld,n=%lld,input_scale=1,%s] "
            "C=F32[%lld,%lld]\n",
            __func__,
            (long long) k,
            (long long) m,
            weight_scale,
            p_direct ? "P_raw" : "P_scaled",
            (long long) k,
            (long long) n,
            p_direct ? "no_first_scale" : "twolevel_first_scale_applied_after_matmul",
            (long long) m,
            (long long) n);
}

void ggml_cuda_nvfp4_log_fattn_native_unavailable(const char * label) {
    GGML_LOG_WARN("%s: %s matmul did not use native Tensor Core FP4 path; NVFP4 FATTN native path unavailable\n",
            __func__, label);
}

void ggml_cuda_nvfp4_log_fattn_native_active(const char * label) {
    GGML_LOG_INFO("%s: %s matmul active: native Tensor Core FP4 path confirmed (cublasLt CUDA_R_4F_E2M1)\n",
            __func__, label);
}

static void ggml_cuda_nvfp4_push_unique_sample(std::vector<int64_t> & values, int64_t v, int64_t upper) {
    if (v < 0 || v >= upper) {
        return;
    }
    for (int64_t x : values) {
        if (x == v) {
            return;
        }
    }
    if (values.size() < 4) {
        values.push_back(v);
    }
}

static void ggml_cuda_nvfp4_log_probe_matrix(
        const ggml_tensor * dst,
        const char * tag,
        const block_nvfp4 * src_blocks,
        const void * repacked_data,
        const void * repacked_scale,
        int64_t outer_valid,
        int64_t ne10,
        int64_t nblk_k,
        int64_t scale_inner_padded,
        bool linear_scale_layout,
        cudaStream_t stream) {
    if (outer_valid <= 0 || nblk_k <= 0) {
        return;
    }

    std::vector<int64_t> outer_samples;
    ggml_cuda_nvfp4_push_unique_sample(outer_samples, 0, outer_valid);
    ggml_cuda_nvfp4_push_unique_sample(outer_samples, 1, outer_valid);
    ggml_cuda_nvfp4_push_unique_sample(outer_samples, 31, outer_valid);
    ggml_cuda_nvfp4_push_unique_sample(outer_samples, 32, outer_valid);
    ggml_cuda_nvfp4_push_unique_sample(outer_samples, 127, outer_valid);
    ggml_cuda_nvfp4_push_unique_sample(outer_samples, 128, outer_valid);
    ggml_cuda_nvfp4_push_unique_sample(outer_samples, outer_valid - 1, outer_valid);
    if (outer_samples.size() < 4) {
        ggml_cuda_nvfp4_push_unique_sample(outer_samples, (outer_valid - 1) / 2, outer_valid);
    }

    std::vector<int64_t> inner_samples;
    ggml_cuda_nvfp4_push_unique_sample(inner_samples, 0, nblk_k);
    ggml_cuda_nvfp4_push_unique_sample(inner_samples, 1, nblk_k);
    ggml_cuda_nvfp4_push_unique_sample(inner_samples, 3, nblk_k);
    ggml_cuda_nvfp4_push_unique_sample(inner_samples, 4, nblk_k);
    ggml_cuda_nvfp4_push_unique_sample(inner_samples, nblk_k / 2, nblk_k);
    ggml_cuda_nvfp4_push_unique_sample(inner_samples, nblk_k - 1, nblk_k);
    if (inner_samples.size() < 4) {
        ggml_cuda_nvfp4_push_unique_sample(inner_samples, 2, nblk_k);
    }

    const int64_t row_data_bytes = ne10 / 2;
    int samples = 0;
    int scale_mismatch = 0;
    int data_mismatch = 0;

    for (int64_t outer : outer_samples) {
        for (int64_t inner : inner_samples) {
            const int64_t src_idx = outer * nblk_k + inner;
            const int64_t scale_idx = linear_scale_layout
                    ? (outer * scale_inner_padded + inner)
                    : ggml_cuda_nvfp4_scale_tiled_index(outer, inner, scale_inner_padded);
            const int64_t data_off = outer * row_data_bytes + inner * (QK_NVFP4 / 2);

            block_nvfp4 src_block = {};
            uint8_t rep_scale = 0;
            uint8_t rep_qs[QK_NVFP4 / 2] = { 0 };

            CUDA_CHECK(cudaMemcpyAsync(&src_block, src_blocks + src_idx, sizeof(src_block), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&rep_scale, (const uint8_t *) repacked_scale + scale_idx, sizeof(rep_scale), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(rep_qs, (const uint8_t *) repacked_data + data_off, sizeof(rep_qs), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            const bool scale_ok = (src_block.e == rep_scale);
            const bool data_ok = (memcmp(src_block.qs, rep_qs, sizeof(rep_qs)) == 0);
            samples += 1;
            scale_mismatch += scale_ok ? 0 : 1;
            data_mismatch += data_ok ? 0 : 1;

            GGML_LOG_INFO(
                    "%s: deep-probe %s %s o=%lld i=%lld "
                    "scale[src=%u rep=%u ok=%d idx=%lld] "
                    "data[src0=%u src7=%u rep0=%u rep7=%u ok=%d off=%lld]\n",
                    __func__,
                    ggml_get_name(dst),
                    tag,
                    (long long) outer,
                    (long long) inner,
                    (unsigned) src_block.e,
                    (unsigned) rep_scale,
                    scale_ok ? 1 : 0,
                    (long long) scale_idx,
                    (unsigned) src_block.qs[0],
                    (unsigned) src_block.qs[QK_NVFP4 / 2 - 1],
                    (unsigned) rep_qs[0],
                    (unsigned) rep_qs[QK_NVFP4 / 2 - 1],
                    data_ok ? 1 : 0,
                    (long long) data_off);
        }
    }

    GGML_LOG_INFO(
            "%s: deep-probe summary %s %s samples=%d scale_mismatch=%d data_mismatch=%d\n",
            __func__,
            ggml_get_name(dst),
            tag,
            samples,
            scale_mismatch,
            data_mismatch);
}

void ggml_cuda_nvfp4_log_native_repack_debug(
        const ggml_tensor * dst,
        const block_nvfp4 * src0_blocks,
        const block_nvfp4 * src1_blocks,
        const void * src0_repacked_data,
        const void * src0_repacked_scale,
        size_t src0_data_nbytes,
        size_t src0_scale_nbytes,
        int64_t src0_scale_outer_padded,
        int64_t src0_scale_inner_padded,
        const void * src1_repacked_data,
        const void * src1_repacked_scale,
        size_t src1_data_nbytes,
        size_t src1_scale_nbytes,
        int64_t src1_scale_outer_padded,
        int64_t src1_scale_inner_padded,
        int64_t ne10,
        int64_t ne01,
        int64_t ne11,
        int64_t nblk_k,
        bool linear_scale_layout,
        bool used_dynamic_scale,
        cudaStream_t stream) {
    GGML_LOG_INFO(
            "%s: native channel layout for %s: "
            "A[data=%zu,scale=%zu,scale_shape=(outer=%lld,inner=%lld)] "
            "B[data=%zu,scale=%zu,scale_shape=(outer=%lld,inner=%lld)]\n",
            __func__, ggml_get_name(dst),
            src0_data_nbytes,
            src0_scale_nbytes,
            (long long) src0_scale_outer_padded,
            (long long) src0_scale_inner_padded,
            src1_data_nbytes,
            src1_scale_nbytes,
            (long long) src1_scale_outer_padded,
            (long long) src1_scale_inner_padded);
    GGML_LOG_INFO("%s: scale layout mode for %s: %s\n",
            __func__, ggml_get_name(dst), linear_scale_layout ? "linear" : "tiled-128x4");
    GGML_LOG_INFO("%s: alpha mode for %s: out_scale/global_scale (%s)\n",
            __func__, ggml_get_name(dst), used_dynamic_scale ? "dynamic-rhs" : "bound-scale");

    const int64_t dump_blocks = std::min<int64_t>(nblk_k, 4);
    if (dump_blocks > 0) {
        std::vector<block_nvfp4> a_blocks((size_t) dump_blocks);
        std::vector<block_nvfp4> b_blocks((size_t) dump_blocks);
        std::vector<uint8_t> a_scale_src((size_t) dump_blocks, 0);
        std::vector<uint8_t> b_scale_src((size_t) dump_blocks, 0);
        std::vector<uint8_t> a_scale_repacked((size_t) dump_blocks, 0);
        std::vector<uint8_t> b_scale_repacked((size_t) dump_blocks, 0);

        CUDA_CHECK(cudaMemcpyAsync(a_blocks.data(), src0_blocks, (size_t) dump_blocks * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(b_blocks.data(), src1_blocks, (size_t) dump_blocks * sizeof(block_nvfp4), cudaMemcpyDeviceToHost, stream));

        for (int64_t i = 0; i < dump_blocks; ++i) {
            const int64_t scale_idx = linear_scale_layout ?
                    i : ggml_cuda_nvfp4_scale_tiled_index(0, i, src1_scale_inner_padded);
            CUDA_CHECK(cudaMemcpyAsync(&a_scale_repacked[(size_t) i], (const uint8_t *) src0_repacked_scale + scale_idx, sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&b_scale_repacked[(size_t) i], (const uint8_t *) src1_repacked_scale + scale_idx, sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int64_t i = 0; i < dump_blocks; ++i) {
            a_scale_src[(size_t) i] = a_blocks[(size_t) i].e;
            b_scale_src[(size_t) i] = b_blocks[(size_t) i].e;
        }

        GGML_LOG_INFO(
                "%s: scale-byte probe %s row0 first4 A[src=%u,%u,%u,%u repacked=%u,%u,%u,%u] "
                "B[src=%u,%u,%u,%u repacked=%u,%u,%u,%u]\n",
                __func__,
                ggml_get_name(dst),
                (unsigned) (dump_blocks > 0 ? a_scale_src[0] : 0),
                (unsigned) (dump_blocks > 1 ? a_scale_src[1] : 0),
                (unsigned) (dump_blocks > 2 ? a_scale_src[2] : 0),
                (unsigned) (dump_blocks > 3 ? a_scale_src[3] : 0),
                (unsigned) (dump_blocks > 0 ? a_scale_repacked[0] : 0),
                (unsigned) (dump_blocks > 1 ? a_scale_repacked[1] : 0),
                (unsigned) (dump_blocks > 2 ? a_scale_repacked[2] : 0),
                (unsigned) (dump_blocks > 3 ? a_scale_repacked[3] : 0),
                (unsigned) (dump_blocks > 0 ? b_scale_src[0] : 0),
                (unsigned) (dump_blocks > 1 ? b_scale_src[1] : 0),
                (unsigned) (dump_blocks > 2 ? b_scale_src[2] : 0),
                (unsigned) (dump_blocks > 3 ? b_scale_src[3] : 0),
                (unsigned) (dump_blocks > 0 ? b_scale_repacked[0] : 0),
                (unsigned) (dump_blocks > 1 ? b_scale_repacked[1] : 0),
                (unsigned) (dump_blocks > 2 ? b_scale_repacked[2] : 0),
                (unsigned) (dump_blocks > 3 ? b_scale_repacked[3] : 0));
    }

    static std::atomic<bool> deep_probe_logged(false);
    if (deep_probe_logged.exchange(true)) {
        return;
    }

    ggml_cuda_nvfp4_log_probe_matrix(
            dst, "A", src0_blocks, src0_repacked_data, src0_repacked_scale,
            ne01, ne10, nblk_k, src0_scale_inner_padded, linear_scale_layout, stream);
    ggml_cuda_nvfp4_log_probe_matrix(
            dst, "B", src1_blocks, src1_repacked_data, src1_repacked_scale,
            ne11, ne10, nblk_k, src1_scale_inner_padded, linear_scale_layout, stream);
}

void ggml_cuda_nvfp4_log_validate_sampled_row(
        const ggml_tensor * dst,
        const char * samples,
        double max_abs,
        int64_t max_col) {
    GGML_LOG_WARN(
            "%s: native validate %s sampled row0: %s (max_abs=%g at_col=%lld)\n",
            __func__,
            ggml_get_name(dst),
            samples,
            max_abs,
            (long long) max_col);
}

void ggml_cuda_nvfp4_log_append_validate_sample(
        char * buf,
        size_t buf_size,
        int & off,
        int64_t col,
        double native_value,
        double ref,
        double abs_err) {
    if (off > 0 && off < (int) buf_size) {
        off += std::snprintf(buf + off, buf_size - off, " | ");
    }
    if (off >= 0 && off < (int) buf_size) {
        off += std::snprintf(
                buf + off,
                buf_size - off,
                "c%lld native=%g ref=%g abs=%g",
                (long long) col,
                native_value,
                ref,
                abs_err);
    }
}

const char * ggml_cuda_nvfp4_log_scale_probe_mode_name(int mode) {
    switch (mode) {
        case 0:  return "cur";
        case 1:  return "lin";
        case 2:  return "tlin";
        case 3:  return "ttile";
        default: return "unknown";
    }
}

void ggml_cuda_nvfp4_log_src0_focus(
        const ggml_tensor * dst,
        const char * probe_tag,
        int64_t row,
        int64_t out_col,
        double actual,
        double ref_src,
        double ref_cur,
        double ref_lin,
        double ref_tlin,
        double ref_ttile,
        double weight_max_abs_cur,
        double weight_max_abs_lin,
        double weight_max_abs_tlin,
        double weight_max_abs_ttile) {
    GGML_LOG_WARN(
            "%s: src0-focus %s %s r=%lld c=%lld actual=%g "
            "ref_src=%g abs_src=%g | cur=%g abs=%g w=%g | lin=%g abs=%g w=%g | "
            "tlin=%g abs=%g w=%g | ttile=%g abs=%g w=%g\n",
            __func__,
            ggml_get_name(dst),
            probe_tag,
            (long long) row,
            (long long) out_col,
            actual,
            ref_src,
            fabs(actual - ref_src),
            ref_cur,
            fabs(actual - ref_cur),
            weight_max_abs_cur,
            ref_lin,
            fabs(actual - ref_lin),
            weight_max_abs_lin,
            ref_tlin,
            fabs(actual - ref_tlin),
            weight_max_abs_tlin,
            ref_ttile,
            fabs(actual - ref_ttile),
            weight_max_abs_ttile);
}

void ggml_cuda_nvfp4_log_src0_focus_groups(
        const ggml_tensor * dst,
        const char * probe_tag,
        int64_t row,
        int64_t out_col,
        int cur_group,
        int64_t tile_pos,
        const char * groups) {
    GGML_LOG_WARN(
            "%s: src0-focus-groups %s %s r=%lld c=%lld cur_group=%d pos=%lld %s\n",
            __func__,
            ggml_get_name(dst),
            probe_tag,
            (long long) row,
            (long long) out_col,
            cur_group,
            (long long) tile_pos,
            groups);
}

void ggml_cuda_nvfp4_log_append_src0_focus_group(
        char * buf,
        size_t buf_size,
        int & off,
        int group,
        bool current,
        double ref,
        double abs_err,
        double weight_max_abs) {
    if (off >= (int) buf_size) {
        return;
    }
    off += std::snprintf(
            buf + off,
            buf_size - off,
            "%sg%d%s=%g abs=%g w=%g",
            off > 0 ? " | " : "",
            group,
            current ? "*" : "",
            ref,
            abs_err,
            weight_max_abs);
}

void ggml_cuda_nvfp4_log_append_top_ref(
        char * buf,
        size_t buf_size,
        int & off,
        int64_t ib,
        double ref,
        uint8_t scale) {
    if (off >= (int) buf_size) {
        return;
    }
    off += std::snprintf(
            buf + off,
            buf_size - off,
            "%sib%lld=%g(e=%u)",
            off > 0 ? " | " : "",
            (long long) ib,
            ref,
            (unsigned) scale);
}

void ggml_cuda_nvfp4_log_append_top_missing_a(
        char * buf,
        size_t buf_size,
        int & off,
        int64_t ib,
        double missing_a,
        double ref,
        double no_a,
        uint8_t scale) {
    if (off >= (int) buf_size) {
        return;
    }
    off += std::snprintf(
            buf + off,
            buf_size - off,
            "%sib%lld=d%g(ref=%g noA=%g e=%u)",
            off > 0 ? " | " : "",
            (long long) ib,
            missing_a,
            ref,
            no_a,
            (unsigned) scale);
}

void ggml_cuda_nvfp4_log_append_selective(
        char * buf,
        size_t buf_size,
        int & off,
        int64_t ib,
        double missing_a_out,
        double missing_a_abs,
        int best_group,
        double best_group_out,
        double best_group_abs,
        int64_t best_inner_src,
        double best_inner_out,
        double best_inner_abs,
        uint8_t best_e_byte,
        double best_e_out,
        double best_e_abs,
        double best_e_ratio,
        double block_ref,
        double best_group_block,
        double best_inner_block,
        double best_e_block,
        uint8_t src_scale,
        uint8_t best_inner_scale) {
    if (off >= (int) buf_size) {
        return;
    }
    off += std::snprintf(
            buf + off,
            buf_size - off,
            "%sib%lld missA=%g abs=%g best_g%d=%g abs=%g best_inner_ib%lld=%g abs=%g "
            "best_e=%u out=%g abs=%g ratio=%g block_ref=%g block_g=%g block_inner=%g block_e=%g "
            "e=%u inner_e=%u",
            off > 0 ? " | " : "",
            (long long) ib,
            missing_a_out,
            missing_a_abs,
            best_group,
            best_group_out,
            best_group_abs,
            (long long) best_inner_src,
            best_inner_out,
            best_inner_abs,
            (unsigned) best_e_byte,
            best_e_out,
            best_e_abs,
            best_e_ratio,
            block_ref,
            best_group_block,
            best_inner_block,
            best_e_block,
            (unsigned) src_scale,
            (unsigned) best_inner_scale);
}

void ggml_cuda_nvfp4_log_append_attenuation(
        char * buf,
        size_t buf_size,
        int & off,
        int index,
        double fit,
        double no_a_ratio) {
    if (off >= (int) buf_size) {
        return;
    }
    off += std::snprintf(
            buf + off,
            buf_size - off,
            "%stop%d fit=%g noA_ratio=%g",
            off > 0 ? " | " : "",
            index,
            fit,
            no_a_ratio);
}

void ggml_cuda_nvfp4_log_src0_block_focus(
        const ggml_tensor * dst,
        const char * probe_tag,
        int64_t row,
        int64_t out_col,
        double actual,
        double ref_total,
        double deficit,
        int top_pos_needed,
        double top_pos_cum,
        const double top_missing_a_ref[3],
        const double top_sign_flip_ref[3],
        const char * attenuation,
        const char * top_ref,
        const char * top_missing_a,
        const char * selective) {
    GGML_LOG_WARN(
            "%s: src0-block-focus %s %s r=%lld c=%lld actual=%g ref=%g deficit=%g "
            "top_pos_needed=%d top_pos_cum=%g\n",
            __func__,
            ggml_get_name(dst),
            probe_tag,
            (long long) row,
            (long long) out_col,
            actual,
            ref_total,
            deficit,
            top_pos_needed,
            top_pos_cum);
    GGML_LOG_WARN(
            "%s: src0-block-focus %s %s missingA combos "
            "top1=%g abs=%g | top2=%g abs=%g | top3=%g abs=%g\n",
            __func__,
            ggml_get_name(dst),
            probe_tag,
            top_missing_a_ref[0],
            fabs(actual - top_missing_a_ref[0]),
            top_missing_a_ref[1],
            fabs(actual - top_missing_a_ref[1]),
            top_missing_a_ref[2],
            fabs(actual - top_missing_a_ref[2]));
    GGML_LOG_WARN(
            "%s: src0-block-focus %s %s signFlip combos "
            "top1=%g abs=%g | top2=%g abs=%g | top3=%g abs=%g\n",
            __func__,
            ggml_get_name(dst),
            probe_tag,
            top_sign_flip_ref[0],
            fabs(actual - top_sign_flip_ref[0]),
            top_sign_flip_ref[1],
            fabs(actual - top_sign_flip_ref[1]),
            top_sign_flip_ref[2],
            fabs(actual - top_sign_flip_ref[2]));
    GGML_LOG_WARN("%s: src0-block-focus %s %s attenuation-fit %s\n",
            __func__, ggml_get_name(dst), probe_tag, attenuation);
    GGML_LOG_WARN("%s: src0-block-focus %s %s top_ref=%s\n",
            __func__, ggml_get_name(dst), probe_tag, top_ref);
    GGML_LOG_WARN("%s: src0-block-focus %s %s top_missing_a=%s\n",
            __func__, ggml_get_name(dst), probe_tag, top_missing_a);
    GGML_LOG_WARN("%s: src0-block-focus %s %s selective=%s\n",
            __func__, ggml_get_name(dst), probe_tag, selective);
}
