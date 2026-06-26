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

bool tensor_export_enabled();
bool tensor_export_maybe_log_config();
bool tensor_export_graph(ggml_backend_sched_t sched, ggml_cgraph * gf);

tensor_error_metrics compute_error_metrics(const std::vector<float> & reference, const std::vector<float> & actual);
std::vector<tensor_record> load_manifest_records(const std::string & manifest_path);
eval_report evaluate_manifest(const std::string & manifest_path, float global_scale = 1.0f);
std::string format_eval_report_json(const eval_report & report);

} // namespace llama_expt
