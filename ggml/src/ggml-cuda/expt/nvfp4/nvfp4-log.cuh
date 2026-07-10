#pragma once

#include "../../common.cuh"

bool ggml_cuda_nvfp4_log_can_copy_from_stream(cudaStream_t stream);

void ggml_cuda_nvfp4_log_bf16_quant_once(const char * env, bool enabled);
void ggml_cuda_nvfp4_log_bf16_quant_trunc_nn_once(const char * env, bool enabled);
void ggml_cuda_nvfp4_log_bf16_quant_bf16_internal_once(const char * env, bool enabled);
void ggml_cuda_nvfp4_log_bf16_quant_bf16_block_scale_once(const char * env, bool enabled);
void ggml_cuda_nvfp4_log_trunc_bf16_input_once(const char * env, bool enabled);

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
        cudaStream_t stream);

void ggml_cuda_nvfp4_log_kcache_outlier_overflow_if_any(
        const char * caller,
        const char * target,
        const int64_t * dst_rows,
        const int32_t * counts,
        const int32_t * offsets,
        const int32_t * cursor,
        int64_t ne01,
        int64_t dst_rows_stride,
        int64_t sidecar_rows,
        int64_t compact_capacity,
        float threshold,
        cudaStream_t stream);

void ggml_cuda_nvfp4_log_kcache_outlier_fingerprint(
        const char * caller,
        const char * target,
        const float * src,
        const int64_t * dst_rows,
        const int32_t * counts,
        const int32_t * offsets,
        const int32_t * cursor,
        const int32_t * indices,
        const float * values,
        const float * residual_amax,
        int64_t ne00,
        int64_t ne01,
        int64_t src_stride,
        int64_t dst_rows_stride,
        int64_t sidecar_rows,
        int64_t compact_capacity,
        float threshold,
        cudaStream_t stream);

void ggml_cuda_nvfp4_log_vcache_fast_update_once(bool enabled);
void ggml_cuda_nvfp4_log_vcache_fp4_pv_once();
void ggml_cuda_nvfp4_log_vcache_matmul_path_once(const char * path);
void ggml_cuda_nvfp4_log_vcache_fp4mulmat_forced_once();
void ggml_cuda_nvfp4_log_vcache_lt_failure_once(const char * stage, int status, const char * status_str);
void ggml_cuda_nvfp4_log_vcache_lt_active_once(
        int64_t rows,
        int64_t cols,
        int64_t lt_cols,
        int64_t kv_size,
        int64_t q_heads,
        int64_t q_streams);
void ggml_cuda_nvfp4_log_vcache_lt_scale_attrs_unavailable_once();

void ggml_cuda_nvfp4_log_fattn_tensor_brief_once(
        const char * label,
        const ggml_tensor * a,
        const ggml_tensor * b,
        const ggml_tensor * dst,
        bool qk);

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
        int64_t head_dim);

void ggml_cuda_nvfp4_log_fattn_qk_requested(
        bool k_nvfp4_cache,
        int64_t k,
        int64_t m,
        double weight_scale,
        int64_t n,
        bool q_dynamic,
        double input_scale);
void ggml_cuda_nvfp4_log_fattn_vp_requested(
        int64_t k,
        int64_t m,
        double weight_scale,
        bool p_direct,
        int64_t n);
void ggml_cuda_nvfp4_log_fattn_native_unavailable(const char * label);
void ggml_cuda_nvfp4_log_fattn_native_active(const char * label);

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
        cudaStream_t stream);

void ggml_cuda_nvfp4_log_fp4mulmat_native_path(
        const char * caller,
        const ggml_tensor * dst,
        int64_t ne01,
        int64_t ne11,
        int64_t ne10,
        bool used_dynamic_scale,
        bool verbose);

void ggml_cuda_nvfp4_log_validate_sampled_row(
        const ggml_tensor * dst,
        const char * samples,
        double max_abs,
        int64_t max_col);

void ggml_cuda_nvfp4_log_append_validate_sample(
        char * buf,
        size_t buf_size,
        int & off,
        int64_t col,
        double native_value,
        double ref,
        double abs_err);

const char * ggml_cuda_nvfp4_log_scale_probe_mode_name(int mode);

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
        double weight_max_abs_ttile);

void ggml_cuda_nvfp4_log_src0_focus_groups(
        const ggml_tensor * dst,
        const char * probe_tag,
        int64_t row,
        int64_t out_col,
        int cur_group,
        int64_t tile_pos,
        const char * groups);

void ggml_cuda_nvfp4_log_append_src0_focus_group(
        char * buf,
        size_t buf_size,
        int & off,
        int group,
        bool current,
        double ref,
        double abs_err,
        double weight_max_abs);

void ggml_cuda_nvfp4_log_append_top_ref(
        char * buf,
        size_t buf_size,
        int & off,
        int64_t ib,
        double ref,
        uint8_t scale);

void ggml_cuda_nvfp4_log_append_top_missing_a(
        char * buf,
        size_t buf_size,
        int & off,
        int64_t ib,
        double missing_a,
        double ref,
        double no_a,
        uint8_t scale);

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
        uint8_t best_inner_scale);

void ggml_cuda_nvfp4_log_append_attenuation(
        char * buf,
        size_t buf_size,
        int & off,
        int index,
        double fit,
        double no_a_ratio);

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
        const char * selective);
