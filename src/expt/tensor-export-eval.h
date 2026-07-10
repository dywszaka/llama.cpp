#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llama_expt {

class attention_quant_round_algo;

struct tensor_record {
    std::string name;
    std::string kind;
    std::string dtype;
    int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    size_t  nb[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    std::string path;
    size_t byte_size = 0;
    std::map<std::string, std::string> meta;
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
    double nmse = 0.0;
    double max_abs_err = 0.0;
};

struct eval_report {
    std::vector<eval_record_report> records;
    std::map<std::string, tensor_error_metrics> by_kind;
    float global_scale = 1.0f;
};

struct attention_replay_report {
    tensor_record k_record;
    tensor_record q_record;
    tensor_record kq_record;
    tensor_record softmax_record;
    tensor_error_metrics kq_metrics;
    tensor_error_metrics softmax_metrics;
    double max_abs_err_kq = 0.0;
    double max_abs_err_softmax = 0.0;
    double kq_nmse = 0.0;
    double softmax_nmse = 0.0;
    float kq_scale = 1.0f;
    float max_bias = 0.0f;
};

struct attention_replay_eval_report {
    std::vector<attention_replay_report> records;
};

struct quant_round_tensor_metadata {
    std::string mode;
    std::map<std::string, std::string> string_fields;
    std::map<std::string, double> number_fields;
    std::map<std::string, uint64_t> integer_fields;
};

struct attention_replay_nvfp4_outlier_report : attention_replay_report {
    tensor_error_metrics k_quant_metrics;
    tensor_error_metrics q_quant_metrics;
    double softmax_kld = 0.0;
    double kld_epsilon = 0.0;
    float k_threshold = 0.0f;
    float k_global_scale = 0.0f;
    size_t k_outlier_count = 0;
    std::string k_quantization_mode;
    std::string q_quantization_mode;
    std::string quant_round_algorithm;
    quant_round_tensor_metadata k_quant_round;
    quant_round_tensor_metadata q_quant_round;
};

struct attention_replay_nvfp4_outlier_eval_report {
    std::string algorithm = "attention_replay_nvfp4_outlier";
    std::string quant_round_algorithm;
    std::vector<attention_replay_nvfp4_outlier_report> records;
};

bool tensor_export_enabled();
bool tensor_export_maybe_log_config();
void tensor_export_pin_named_tensor(ggml_tensor * tensor);
bool tensor_export_graph(ggml_backend_sched_t sched, ggml_cgraph * gf);

tensor_error_metrics compute_error_metrics(const std::vector<float> & reference, const std::vector<float> & actual);
double compute_nmse(const std::vector<float> & reference, const std::vector<float> & actual);
std::vector<tensor_record> load_manifest_records(const std::string & manifest_path);
eval_report evaluate_manifest(const std::string & manifest_path, float global_scale = 1.0f);
attention_replay_eval_report evaluate_manifest_attention_replay(const std::string & manifest_path);
attention_replay_nvfp4_outlier_eval_report evaluate_manifest_attention_replay_nvfp4_outlier(const std::string & manifest_path);
attention_replay_nvfp4_outlier_eval_report evaluate_manifest_attention_replay_fp8_e4m3_e8m0(const std::string & manifest_path);
attention_replay_nvfp4_outlier_eval_report evaluate_manifest_attention_replay_quant_round(
        const std::string & manifest_path,
        const attention_quant_round_algo & quant_round_algo);
std::string format_eval_report_json(const eval_report & report);
std::string format_attention_replay_eval_report_json(const attention_replay_eval_report & report);
std::string format_attention_replay_nvfp4_outlier_eval_report_json(const attention_replay_nvfp4_outlier_eval_report & report);

} // namespace llama_expt
