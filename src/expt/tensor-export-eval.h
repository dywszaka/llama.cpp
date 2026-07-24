#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llama_expt {

struct tensor_record {
    std::string name;
    std::string kind;
    std::string dtype;
    int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    size_t  nb[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    std::string path;
    size_t byte_size = 0;
};

struct tensor_error_metrics {
    double mae  = 0.0;
    double mse  = 0.0;
    double rmse = 0.0;
    size_t n    = 0;
};

struct eval_record_report {
    tensor_record record;
    tensor_error_metrics metrics;
};

struct eval_report {
    std::vector<eval_record_report> records;
    std::map<std::string, tensor_error_metrics> by_kind;
    float global_scale = 1.0f;
};

enum class k_channel_sort_basis {
    FIRST_ROW_ABS,
    ABS_MEAN,
};

struct k_channel_sort_eval_record_report {
    tensor_record record;
    tensor_error_metrics baseline_metrics;
    tensor_error_metrics sorted_metrics;
    tensor_error_metrics delta_metrics;
    std::vector<size_t> channel_order;
    std::string sort_basis;
    size_t channel_count = 0;
    size_t row_count = 0;
};

struct k_channel_sort_eval_aggregate_report {
    tensor_error_metrics baseline_metrics;
    tensor_error_metrics sorted_metrics;
    tensor_error_metrics delta_metrics;
};

struct k_channel_sort_eval_report {
    std::vector<k_channel_sort_eval_record_report> records;
    std::map<std::string, k_channel_sort_eval_aggregate_report> by_kind;
    k_channel_sort_basis sort_basis = k_channel_sort_basis::FIRST_ROW_ABS;
    float global_scale = 1.0f;
};

struct tensor_export_observer;

bool tensor_export_enabled();
bool tensor_export_maybe_log_config();
bool tensor_export_maybe_retain_graph(ggml_cgraph * gf);
tensor_export_observer * tensor_export_observer_create(
        ggml_cgraph * gf,
        bool is_prefill,
        ggml_backend_sched_eval_callback user_callback,
        void * user_data);
bool tensor_export_observer_callback(ggml_tensor * tensor, bool ask, void * user_data);
void tensor_export_observer_free(tensor_export_observer * observer);
bool tensor_export_maybe_bind_nvfp4_mul_mat_capture(
        ggml_context * ctx,
        ggml_tensor * tensor,
        bool is_prefill);
bool tensor_export_graph(
        ggml_backend_sched_t sched,
        ggml_cgraph * gf,
        bool is_prefill,
        const tensor_export_observer * observer = nullptr);

tensor_error_metrics compute_error_metrics(const std::vector<float> & reference, const std::vector<float> & actual);
std::vector<size_t> make_k_channel_order_from_first_row(const std::vector<float> & values, size_t row_size);
std::vector<size_t> make_k_channel_order_from_abs_mean(const std::vector<float> & values, size_t row_size);
std::vector<tensor_record> load_manifest_records(const std::string & manifest_path);
eval_report evaluate_manifest(const std::string & manifest_path, float global_scale = 1.0f);
k_channel_sort_eval_report evaluate_manifest_k_channel_sort(
        const std::string & manifest_path,
        k_channel_sort_basis sort_basis = k_channel_sort_basis::FIRST_ROW_ABS,
        float global_scale = 1.0f);
std::string format_eval_report_json(const eval_report & report);
std::string format_k_channel_sort_eval_report_json(const k_channel_sort_eval_report & report);

} // namespace llama_expt
