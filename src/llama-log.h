#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <limits>
#include <string>
#include <vector>

struct llama_model;

struct llama_tensor_stats {
    int64_t n = 0;
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    int nan = 0;
    int inf = 0;
    int finite = 0;
};

namespace llama_log {

bool nvfp4_enabled();
bool nvfp4_log_all();
bool nvfp4_log_src();
bool nvfp4_log_buf();
bool nvfp4_logits_debug();
bool nvfp4_dequant_debug();
bool nvfp4_tensor_pin_enabled();

const std::vector<std::string> & nvfp4_debug_patterns();

bool compute_tensor_stats(const ggml_tensor * tensor, llama_tensor_stats & stats);
void log_tensor_stats(const ggml_tensor * tensor, const llama_tensor_stats & stats);

void debug_nvfp4_graph_tensors(ggml_backend_sched_t sched, ggml_cgraph * gf);
void debug_nvfp4_norm_weights(const struct llama_model & model);
void nvfp4_pin_tensor_if_match(struct ggml_tensor * tensor);
void nvfp4_log_logits_if_enabled(int idx, int64_t j, int n_vocab, const float * logits);

} // namespace llama_log
