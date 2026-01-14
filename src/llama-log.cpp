#include "llama-log.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

std::string trim_copy(const std::string & value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::vector<std::string> split_patterns(const std::string & value) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= value.size()) {
        size_t pos = value.find(',', start);
        if (pos == std::string::npos) {
            pos = value.size();
        }
        std::string token = trim_copy(value.substr(start, pos - start));
        if (!token.empty()) {
            out.push_back(std::move(token));
        }
        start = pos + 1;
    }
    return out;
}

bool match_pattern(const char * name, const std::string & pattern) {
    if (!name) {
        return false;
    }
    if (pattern.empty()) {
        return false;
    }
    if (pattern == name) {
        return true;
    }
    if (pattern.back() == '*') {
        const std::string prefix = pattern.substr(0, pattern.size() - 1);
        return std::strncmp(name, prefix.c_str(), prefix.size()) == 0;
    }
    return false;
}

} // namespace

namespace llama_log {

static bool env_enabled_cached = false;
static bool env_enabled = false;
static bool env_log_all = false;
static bool env_log_src = false;
static bool env_log_buf = false;
static bool env_logits_debug = false;
static bool env_tensor_pin = false;

static void ensure_env_cached() {
    if (env_enabled_cached) {
        return;
    }
    env_enabled_cached = true;
    const char * env_debug = getenv("LLAMA_NVFP4_TENSOR_DEBUG");
    env_enabled = env_debug && env_debug[0] != '\0';
    env_log_all = getenv("LLAMA_NVFP4_TENSOR_DEBUG_ALL") != nullptr;
    env_log_src = getenv("LLAMA_NVFP4_TENSOR_DEBUG_SRC") != nullptr;
    env_log_buf = getenv("LLAMA_NVFP4_TENSOR_DEBUG_BUF") != nullptr;
    env_logits_debug = getenv("LLAMA_NVFP4_LOGITS_DEBUG") != nullptr;
    env_tensor_pin = getenv("LLAMA_NVFP4_TENSOR_DEBUG_PIN") != nullptr;
}

bool nvfp4_enabled() {
    ensure_env_cached();
    return env_enabled;
}

bool nvfp4_log_all() {
    ensure_env_cached();
    return env_log_all;
}

bool nvfp4_log_src() {
    ensure_env_cached();
    return env_log_src;
}

bool nvfp4_log_buf() {
    ensure_env_cached();
    return env_log_buf;
}

bool nvfp4_logits_debug() {
    ensure_env_cached();
    return env_logits_debug;
}

bool nvfp4_tensor_pin_enabled() {
    ensure_env_cached();
    return env_tensor_pin;
}

const std::vector<std::string> & nvfp4_debug_patterns() {
    static std::vector<std::string> patterns;
    static bool initialized = false;
    if (initialized) {
        return patterns;
    }
    initialized = true;

    if (!nvfp4_enabled()) {
        return patterns;
    }

    const char * env = getenv("LLAMA_NVFP4_TENSOR_DEBUG");
    if (!env || env[0] == '\0') {
        return patterns;
    }

    const std::string env_value = trim_copy(env);
    if (env_value == "1" || env_value == "default") {
        patterns = {
            "inp_embd",
            "norm-0",
            "attn_norm-0",
            "Qcur-scaled-0",
            "Kcur-scaled-0",
            "Vcur-scaled-0",
            "Qcur_normed-0",
            "Kcur_normed-0",
            "kqv_out-0",
            "ffn_norm-0",
            "ffn_up-0",
            "ffn_gate-0",
            "ffn_swiglu-0",
            "ffn_down-0",
            "result_norm",
            "result_output",
        };
        return patterns;
    }

    patterns = split_patterns(env_value);
    return patterns;
}

bool compute_tensor_stats(const ggml_tensor * tensor, llama_tensor_stats & stats) {
    if (!nvfp4_enabled()) {
        return false;
    }

    stats = {};
    if (!tensor) {
        return false;
    }

    const int64_t n = ggml_nelements(tensor);
    if (n <= 0) {
        return false;
    }

    stats.n = n;

    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> buf(n);
        ggml_backend_tensor_get(tensor, buf.data(), 0, buf.size() * sizeof(float));
        for (int64_t i = 0; i < n; ++i) {
            const float v = buf[i];
            if (std::isnan(v)) {
                stats.nan++;
                continue;
            }
            if (std::isinf(v)) {
                stats.inf++;
                continue;
            }
            stats.min = std::min(stats.min, v);
            stats.max = std::max(stats.max, v);
            stats.finite++;
        }
        return true;
    }

    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        ggml_backend_tensor_get(tensor, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; ++i) {
            const float v = ggml_fp16_to_fp32(buf[i]);
            if (std::isnan(v)) {
                stats.nan++;
                continue;
            }
            if (std::isinf(v)) {
                stats.inf++;
                continue;
            }
            stats.min = std::min(stats.min, v);
            stats.max = std::max(stats.max, v);
            stats.finite++;
        }
        return true;
    }

    if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> buf(n);
        ggml_backend_tensor_get(tensor, buf.data(), 0, buf.size() * sizeof(ggml_bf16_t));
        for (int64_t i = 0; i < n; ++i) {
            const float v = ggml_bf16_to_fp32(buf[i]);
            if (std::isnan(v)) {
                stats.nan++;
                continue;
            }
            if (std::isinf(v)) {
                stats.inf++;
                continue;
            }
            stats.min = std::min(stats.min, v);
            stats.max = std::max(stats.max, v);
            stats.finite++;
        }
        return true;
    }

    return false;
}

void log_tensor_stats(const ggml_tensor * tensor, const llama_tensor_stats & stats) {
    if (!nvfp4_enabled()) {
        return;
    }

    const float min_out = stats.finite ? stats.min : 0.0f;
    const float max_out = stats.finite ? stats.max : 0.0f;
    if (nvfp4_log_buf()) {
        const char * buf_name = (tensor && tensor->buffer) ? ggml_backend_buffer_name(tensor->buffer) : "none";
        const int host = (tensor && tensor->buffer) ? (int) ggml_backend_buffer_is_host(tensor->buffer) : -1;
        LLAMA_LOG_WARN("%s: tensor=%s type=%s n=%" PRId64 " min=%.6f max=%.6f nan=%d inf=%d buf=%s host=%d data=%p\n",
                __func__, ggml_get_name(tensor), ggml_type_name(tensor->type),
                stats.n, min_out, max_out, stats.nan, stats.inf, buf_name, host,
                tensor ? tensor->data : nullptr);
    } else {
        LLAMA_LOG_WARN("%s: tensor=%s type=%s n=%" PRId64 " min=%.6f max=%.6f nan=%d inf=%d\n",
                __func__, ggml_get_name(tensor), ggml_type_name(tensor->type),
                stats.n, min_out, max_out, stats.nan, stats.inf);
    }
}

void debug_nvfp4_graph_tensors(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    if (!nvfp4_enabled()) {
        return;
    }

    const auto & patterns = nvfp4_debug_patterns();
    if (patterns.empty() || !gf) {
        return;
    }

    const bool log_all = nvfp4_log_all();
    const bool log_src = nvfp4_log_src();

    ggml_backend_sched_synchronize(sched);

    std::unordered_set<const ggml_tensor *> seen;

    for (const auto & pattern : patterns) {
        if (pattern.empty()) {
            continue;
        }

        if (pattern.back() != '*') {
            ggml_tensor * t = ggml_graph_get_tensor(gf, pattern.c_str());
            if (t && seen.insert(t).second) {
                llama_tensor_stats stats;
                if (compute_tensor_stats(t, stats)) {
                    if (log_all || stats.nan > 0 || stats.inf > 0) {
                        log_tensor_stats(t, stats);
                        if (log_src) {
                            for (int si = 0; si < GGML_MAX_SRC; ++si) {
                                const ggml_tensor * src = t->src[si];
                                if (!src) {
                                    continue;
                                }
                                llama_tensor_stats src_stats;
                                if (compute_tensor_stats(src, src_stats)) {
                                    LLAMA_LOG_WARN("%s: tensor=%s src%d=%s\n",
                                            __func__, ggml_get_name(t), si, ggml_get_name(src));
                                    log_tensor_stats(src, src_stats);
                                }
                            }
                        }
                    }
                } else {
                    LLAMA_LOG_WARN("%s: tensor=%s type=%s unsupported for stats\n",
                            __func__, ggml_get_name(t), ggml_type_name(t->type));
                }
            } else if (!t) {
                LLAMA_LOG_WARN("%s: tensor=%s not found in graph\n", __func__, pattern.c_str());
            }
            continue;
        }

        bool matched = false;
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (!t) {
                continue;
            }
            if (!match_pattern(ggml_get_name(t), pattern)) {
                continue;
            }
            matched = true;
            if (!seen.insert(t).second) {
                continue;
            }

            llama_tensor_stats stats;
            if (compute_tensor_stats(t, stats)) {
                if (log_all || stats.nan > 0 || stats.inf > 0) {
                    log_tensor_stats(t, stats);
                    if (log_src) {
                        for (int si = 0; si < GGML_MAX_SRC; ++si) {
                            const ggml_tensor * src = t->src[si];
                            if (!src) {
                                continue;
                            }
                            llama_tensor_stats src_stats;
                            if (compute_tensor_stats(src, src_stats)) {
                                LLAMA_LOG_WARN("%s: tensor=%s src%d=%s\n",
                                        __func__, ggml_get_name(t), si, ggml_get_name(src));
                                log_tensor_stats(src, src_stats);
                            }
                        }
                    }
                }
            } else {
                LLAMA_LOG_WARN("%s: tensor=%s type=%s unsupported for stats\n",
                        __func__, ggml_get_name(t), ggml_type_name(t->type));
            }
        }
        if (!matched) {
            LLAMA_LOG_WARN("%s: tensor pattern=%s not found in graph\n", __func__, pattern.c_str());
        }
    }
}

void debug_nvfp4_norm_weights(const llama_model & model) {
    if (!nvfp4_enabled()) {
        return;
    }

    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;

    const bool log_all = nvfp4_log_all();

    if (model.layers.empty()) {
        return;
    }

    auto log_weight = [&](const char * name, const ggml_tensor * t) {
        if (!t) {
            LLAMA_LOG_WARN("%s: weight=%s missing\n", __func__, name);
            return;
        }
        llama_tensor_stats stats;
        if (compute_tensor_stats(t, stats)) {
            if (log_all || stats.nan > 0 || stats.inf > 0) {
                log_tensor_stats(t, stats);
            }
        } else {
            LLAMA_LOG_WARN("%s: weight=%s type=%s unsupported for stats\n",
                    __func__, name, ggml_type_name(t->type));
        }
    };

    log_weight("layer0.attn_norm", model.layers[0].attn_norm);
    log_weight("layer0.ffn_norm", model.layers[0].ffn_norm);
    log_weight("output_norm", model.output_norm);
}

void nvfp4_pin_tensor_if_match(struct ggml_tensor * tensor) {
    if (!nvfp4_tensor_pin_enabled() || !tensor) {
        return;
    }

    const auto & patterns = nvfp4_debug_patterns();
    for (const auto & pattern : patterns) {
        if (pattern.empty()) {
            continue;
        }
        if (match_pattern(ggml_get_name(tensor), pattern)) {
            ggml_set_output(tensor);
            break;
        }
    }
}

void nvfp4_log_logits_if_enabled(int idx, int64_t j, int n_vocab, const float * logits) {
    if (!nvfp4_logits_debug() || !logits || n_vocab <= 0) {
        return;
    }

    float min_logit = std::numeric_limits<float>::infinity();
    float max_logit = -std::numeric_limits<float>::infinity();
    int nan_count = 0;
    int inf_count = 0;
    int finite_count = 0;

    for (int32_t t = 0; t < n_vocab; ++t) {
        const float v = logits[t];
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }
        if (v < min_logit) {
            min_logit = v;
        }
        if (v > max_logit) {
            max_logit = v;
        }
        finite_count++;
    }

    if (nan_count > 0 || inf_count > 0) {
        const float min_out = finite_count ? min_logit : 0.0f;
        const float max_out = finite_count ? max_logit : 0.0f;
        LLAMA_LOG_WARN("%s: logits stats: idx=%d j=%" PRId64 " size=%d min=%.6f max=%.6f nan=%d inf=%d\n",
                __func__, idx, j, n_vocab, min_out, max_out, nan_count, inf_count);
    } else {
        static bool logged_once = false;
        if (!logged_once) {
            LLAMA_LOG_INFO("%s: logits stats: idx=%d j=%" PRId64 " size=%d min=%.6f max=%.6f\n",
                    __func__, idx, j, n_vocab, min_logit, max_logit);
            logged_once = true;
        }
    }
}

} // namespace llama_log
