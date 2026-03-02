#include "llama-log.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cinttypes>
#include <cstdio>
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
static bool env_log_key_nodes = false;
static bool env_logits_debug = false;
static bool env_dequant_debug = false;
static bool env_tensor_pin = false;

static void ensure_env_cached() {
    if (env_enabled_cached) {
        return;
    }
    env_enabled_cached = true;
    const char * env_debug = getenv("LLAMA_NVFP4_TENSOR_DEBUG");
    env_log_key_nodes = getenv("LLAMA_NVFP4_TENSOR_DEBUG_KEY_NODES") != nullptr;
    env_enabled = (env_debug && env_debug[0] != '\0') || env_log_key_nodes;
    env_log_all = getenv("LLAMA_NVFP4_TENSOR_DEBUG_ALL") != nullptr;
    env_log_src = getenv("LLAMA_NVFP4_TENSOR_DEBUG_SRC") != nullptr;
    env_log_buf = getenv("LLAMA_NVFP4_TENSOR_DEBUG_BUF") != nullptr;
    env_logits_debug = getenv("LLAMA_NVFP4_LOGITS_DEBUG") != nullptr;
    env_dequant_debug = getenv("LLAMA_NVFP4_DEQUANT_DEBUG") != nullptr;
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

static bool nvfp4_log_key_nodes_enabled() {
    ensure_env_cached();
    return env_log_key_nodes;
}

bool nvfp4_logits_debug() {
    ensure_env_cached();
    return env_logits_debug;
}

bool nvfp4_dequant_debug() {
    ensure_env_cached();
    return env_dequant_debug;
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

static bool read_tensor_f32_flat(const ggml_tensor * tensor, int64_t flat_idx, float & out);

static void log_tensor_first4_f32(const ggml_tensor * tensor) {
    if (!nvfp4_enabled() || !tensor || tensor->type != GGML_TYPE_F32 || ggml_nelements(tensor) <= 0) {
        return;
    }

    const int64_t n = ggml_nelements(tensor);
    const size_t nread = std::min<int64_t>(4, n);
    float buf[4] = {};
    const bool contiguous = ggml_is_contiguous(tensor);

    for (size_t i = 0; i < nread; ++i) {
        read_tensor_f32_flat(tensor, (int64_t) i, buf[i]);
    }

    char values[128];
    int off = snprintf(values, sizeof(values), "%g", buf[0]);
    for (size_t i = 1; i < nread && off < (int) sizeof(values); ++i) {
        off += snprintf(values + off, sizeof(values) - off, " %g", buf[i]);
    }

    int64_t sample_idx[3] = { -1, -1, n - 1 };
    if (tensor->ne[1] > 1) {
        sample_idx[0] = tensor->ne[0];
        sample_idx[1] = (tensor->ne[1] - 1) * tensor->ne[0];
    } else if (tensor->ne[2] > 1) {
        sample_idx[0] = tensor->ne[0] * tensor->ne[1];
        sample_idx[1] = (tensor->ne[2] - 1) * tensor->ne[0] * tensor->ne[1];
    }

    char samples[192];
    samples[0] = '\0';
    int sample_off = 0;
    bool first_sample = true;
    for (int si = 0; si < 3; ++si) {
        const int64_t idx = sample_idx[si];
        if (idx < 0 || idx >= n) {
            continue;
        }
        bool duplicate = false;
        for (int sj = 0; sj < si; ++sj) {
            if (sample_idx[sj] == idx) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        float sample_v = 0.0f;
        if (!read_tensor_f32_flat(tensor, idx, sample_v)) {
            continue;
        }
        if (!first_sample && sample_off < (int) sizeof(samples)) {
            sample_off += snprintf(samples + sample_off, sizeof(samples) - sample_off, " ");
        }
        if (sample_off < (int) sizeof(samples)) {
            sample_off += snprintf(samples + sample_off, sizeof(samples) - sample_off, "%" PRId64 ":%g", idx, sample_v);
        }
        first_sample = false;
    }

    const char * contig_tag = contiguous ? "" : " (non-contig)";
    if (samples[0] != '\0') {
        LLAMA_LOG_WARN("%s: tensor=%s%s first%zu=%s samples=%s\n",
                __func__, ggml_get_name(tensor), contig_tag, nread, values, samples);
    } else {
        LLAMA_LOG_WARN("%s: tensor=%s%s first%zu=%s\n",
                __func__, ggml_get_name(tensor), contig_tag, nread, values);
    }
}

static bool starts_with(const char * name, const char * prefix) {
    if (!name || !prefix) {
        return false;
    }
    const size_t prefix_len = std::strlen(prefix);
    return std::strncmp(name, prefix, prefix_len) == 0;
}

static bool read_scalar_tensor_f32(const ggml_tensor * tensor, float & out) {
    if (!tensor || !ggml_is_scalar(tensor)) {
        return false;
    }

    switch (tensor->type) {
        case GGML_TYPE_F32:
            ggml_backend_tensor_get(tensor, &out, 0, sizeof(out));
            return true;
        case GGML_TYPE_F16: {
            ggml_fp16_t v = 0;
            ggml_backend_tensor_get(tensor, &v, 0, sizeof(v));
            out = ggml_fp16_to_fp32(v);
            return true;
        }
        case GGML_TYPE_BF16: {
            ggml_bf16_t v = { 0 };
            ggml_backend_tensor_get(tensor, &v, 0, sizeof(v));
            out = ggml_bf16_to_fp32(v);
            return true;
        }
        default:
            return false;
    }
}

static bool read_tensor_f32_row(const ggml_tensor * tensor, int64_t row, std::vector<float> & out) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || row < 0 || row >= tensor->ne[1]) {
        return false;
    }
    out.resize((size_t) tensor->ne[0]);
    ggml_backend_tensor_get(tensor, out.data(), (size_t) row * tensor->nb[1], (size_t) tensor->ne[0] * sizeof(float));
    return true;
}

static bool read_tensor_f32_2d(const ggml_tensor * tensor, int64_t i0, int64_t i1, float & out) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || i0 < 0 || i0 >= tensor->ne[0] || i1 < 0 || i1 >= tensor->ne[1]) {
        return false;
    }
    const size_t offset = (size_t) (i0 * tensor->nb[0] + i1 * tensor->nb[1]);
    ggml_backend_tensor_get(tensor, &out, offset, sizeof(out));
    return true;
}

static bool read_tensor_f32_flat(const ggml_tensor * tensor, int64_t flat_idx, float & out) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n = ggml_nelements(tensor);
    if (flat_idx < 0 || flat_idx >= n) {
        return false;
    }

    if (ggml_is_contiguous(tensor)) {
        ggml_backend_tensor_get(tensor, &out, (size_t) flat_idx * sizeof(float), sizeof(out));
        return true;
    }

    int64_t i0 = 0;
    int64_t i1 = 0;
    int64_t i2 = 0;
    int64_t i3 = 0;
    ggml_unravel_index(tensor, flat_idx, &i0, &i1, &i2, &i3);
    const size_t offset = (size_t) (i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]);
    ggml_backend_tensor_get(tensor, &out, offset, sizeof(out));
    return true;
}

static bool find_tensor_f32_extrema_flat(const ggml_tensor * tensor, float & min_v, int64_t & min_idx, float & max_v, int64_t & max_idx) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n = ggml_nelements(tensor);
    if (n <= 0) {
        return false;
    }

    bool found = false;
    for (int64_t i = 0; i < n; ++i) {
        float v = 0.0f;
        if (!read_tensor_f32_flat(tensor, i, v) || !std::isfinite(v)) {
            continue;
        }
        if (!found) {
            min_v = max_v = v;
            min_idx = max_idx = i;
            found = true;
            continue;
        }
        if (v < min_v) {
            min_v = v;
            min_idx = i;
        }
        if (v > max_v) {
            max_v = v;
            max_idx = i;
        }
    }

    return found;
}

static void log_tensor_src_samples_if_supported(const char * tag, const ggml_tensor * owner, int si, const ggml_tensor * src) {
    if (!src) {
        return;
    }

    if (src->type == GGML_TYPE_F32) {
        log_tensor_first4_f32(src);
        return;
    }

    if (src->type != GGML_TYPE_NVFP4 || src->ne[0] <= 0 || src->ne[1] <= 0 || src->ne[0] % QK_NVFP4 != 0) {
        return;
    }

    const int64_t k = src->ne[0];
    const int64_t nblk = k / QK_NVFP4;
    std::vector<block_nvfp4> row_blocks((size_t) nblk);
    std::vector<float> row_deq((size_t) k);

    auto log_row = [&](int64_t row, const char * row_tag) {
        if (row < 0 || row >= src->ne[1]) {
            return;
        }
        ggml_backend_tensor_get(src, row_blocks.data(), (size_t) row * src->nb[1], (size_t) nblk * sizeof(block_nvfp4));
        dequantize_row_nvfp4(row_blocks.data(), row_deq.data(), k, 1.0f);

        const size_t nread = std::min<int64_t>(4, k);
        char values[128];
        int off = std::snprintf(values, sizeof(values), "%g", row_deq[0]);
        for (size_t i = 1; i < nread && off > 0 && off < (int) sizeof(values); ++i) {
            off += std::snprintf(values + off, sizeof(values) - off, " %g", row_deq[i]);
        }
        LLAMA_LOG_WARN(
                "%s: tensor=%s src%d=%s %s deq-first%zu=%s\n",
                tag,
                ggml_get_name(owner),
                si,
                ggml_get_name(src),
                row_tag,
                nread,
                values);
    };

    log_row(0, "row0");
    if (src->ne[1] > 1) {
        log_row(src->ne[1] - 1, "row_last");
    }
}

static void log_nvfp4_mul_mat_reference_if_applicable(const char * tag, const ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }

    const ggml_tensor * compare = tensor;
    const ggml_tensor * mm = nullptr;
    float out_scale = 1.0f;

    if (tensor->op == GGML_OP_MUL_MAT && tensor->src[0] && tensor->src[0]->type == GGML_TYPE_NVFP4) {
        mm = tensor;
        float bound_weight_scale = 0.0f;
        if (const ggml_tensor * w_scale = ggml_mul_mat_get_nvfp4_weight_scale(tensor)) {
            if (read_scalar_tensor_f32(w_scale, bound_weight_scale) && std::isfinite(bound_weight_scale)) {
                out_scale = bound_weight_scale;
            }
        }
    } else if (
            tensor->op == GGML_OP_MUL &&
            tensor->src[0] &&
            tensor->src[0]->op == GGML_OP_MUL_MAT &&
            tensor->src[0]->src[0] &&
            tensor->src[0]->src[0]->type == GGML_TYPE_NVFP4) {
        mm = tensor->src[0];
        float mul_scale = 0.0f;
        if (!read_scalar_tensor_f32(tensor->src[1], mul_scale) || !std::isfinite(mul_scale)) {
            return;
        }
        out_scale = mul_scale;
    } else {
        return;
    }

    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x_direct = mm->src[1];
    if (!w || !x_direct || w->type != GGML_TYPE_NVFP4) {
        return;
    }

    const int64_t k = w->ne[0];
    const int64_t n_out = compare->ne[0];
    const int64_t n_rows = compare->ne[1];
    if (k <= 0 || n_out <= 0 || n_rows <= 0 || k % QK_NVFP4 != 0) {
        return;
    }

    const ggml_tensor * x_source = x_direct;
    bool use_roundtrip = false;
    float input_scale = 0.0f;

    if (const ggml_tensor * in_scale = ggml_mul_mat_get_nvfp4_input_scale(mm)) {
        if (read_scalar_tensor_f32(in_scale, input_scale) && std::isfinite(input_scale) && input_scale != 0.0f) {
            use_roundtrip = true;
        }
    } else if (
            x_direct->op == GGML_OP_MAP_CUSTOM2 &&
            x_direct->src[0] &&
            x_direct->src[1]) {
        float inferred_scale = 0.0f;
        if (read_scalar_tensor_f32(x_direct->src[1], inferred_scale) && std::isfinite(inferred_scale) && inferred_scale != 0.0f) {
            x_source = x_direct->src[0];
            input_scale = inferred_scale;
            use_roundtrip = true;
        }
    }

    if (x_source->type != GGML_TYPE_F32) {
        return;
    }

    const int64_t nblk = k / QK_NVFP4;
    std::vector<float> x_row_src;
    std::vector<float> x_row_used;
    std::vector<block_nvfp4> x_q((size_t) nblk);
    std::vector<block_nvfp4> w_row((size_t) nblk);
    std::vector<float> w_deq((size_t) k);

    std::vector<int64_t> sample_rows;
    auto append_unique_row = [&](int64_t row) {
        if (row < 0 || row >= n_rows) {
            return;
        }
        for (int64_t existing : sample_rows) {
            if (existing == row) {
                return;
            }
        }
        sample_rows.push_back(row);
    };
    append_unique_row(0);
    append_unique_row(1);
    append_unique_row(n_rows - 1);

    std::vector<int64_t> sample_cols;
    auto append_unique_col = [&](int64_t col) {
        if (col < 0 || col >= n_out) {
            return;
        }
        for (int64_t existing : sample_cols) {
            if (existing == col) {
                return;
            }
        }
        sample_cols.push_back(col);
    };
    append_unique_col(0);
    append_unique_col(1);
    append_unique_col(127);
    append_unique_col(n_out / 2);
    append_unique_col(n_out - 1);

    auto format_first4 = [&](const std::vector<float> & values, char * buf, size_t buf_size) {
        if (values.empty() || buf_size == 0) {
            return;
        }
        const size_t nread = std::min<size_t>(4, values.size());
        int off = std::snprintf(buf, buf_size, "%g", values[0]);
        for (size_t i = 1; i < nread && off > 0 && off < (int) buf_size; ++i) {
            off += std::snprintf(buf + off, buf_size - off, " %g", values[i]);
        }
    };

    char w_buf[512];
    w_buf[0] = '\0';
    int w_off = 0;
    std::vector<int64_t> weight_log_cols;
    append_unique_col(0);
    append_unique_col(n_out / 2);
    append_unique_col(n_out - 1);
    weight_log_cols.push_back(sample_cols.front());
    if (sample_cols.size() > 2) {
        weight_log_cols.push_back(sample_cols[sample_cols.size() / 2]);
    }
    if (sample_cols.size() > 1) {
        weight_log_cols.push_back(sample_cols.back());
    }
    for (size_t wi = 0; wi < weight_log_cols.size(); ++wi) {
        const int64_t col = weight_log_cols[wi];
        bool duplicate = false;
        for (size_t wj = 0; wj < wi; ++wj) {
            if (weight_log_cols[wj] == col) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        ggml_backend_tensor_get(w, w_row.data(), (size_t) (col * w->nb[1]), (size_t) nblk * sizeof(block_nvfp4));
        dequantize_row_nvfp4(w_row.data(), w_deq.data(), k, 1.0f);
        char values[128];
        format_first4(w_deq, values, sizeof(values));
        if (w_off > 0 && w_off < (int) sizeof(w_buf)) {
            w_off += std::snprintf(w_buf + w_off, sizeof(w_buf) - w_off, " | ");
        }
        if (w_off >= 0 && w_off < (int) sizeof(w_buf)) {
            w_off += std::snprintf(w_buf + w_off, sizeof(w_buf) - w_off, "c%" PRId64 "=%s", col, values);
        }
    }

    float min_v = 0.0f;
    float max_v = 0.0f;
    int64_t min_idx = -1;
    int64_t max_idx = -1;
    const bool have_extrema = compare->ne[2] == 1 && compare->ne[3] == 1 && find_tensor_f32_extrema_flat(compare, min_v, min_idx, max_v, max_idx);
    if (have_extrema) {
        const int64_t min_row = min_idx / n_out;
        const int64_t min_col = min_idx % n_out;
        const int64_t max_row = max_idx / n_out;
        const int64_t max_col2 = max_idx % n_out;
        append_unique_row(min_row);
        append_unique_row(max_row);
        append_unique_col(min_col);
        append_unique_col(max_col2);
        LLAMA_LOG_WARN(
                "%s: tensor=%s extrema: min=%g@flat=%" PRId64 "(r%" PRId64 "c%" PRId64 ") max=%g@flat=%" PRId64 "(r%" PRId64 "c%" PRId64 ")\n",
                tag,
                ggml_get_name(compare),
                min_v, min_idx, min_row, min_col,
                max_v, max_idx, max_row, max_col2);
    }

    char sample_buf[4096];
    sample_buf[0] = '\0';
    int sample_off = 0;
    double max_abs = 0.0;
    int64_t max_col = -1;
    int64_t max_row = -1;
    int sample_count = 0;
    char x_src_buf[512];
    x_src_buf[0] = '\0';
    int x_src_off = 0;
    char x_used_buf[512];
    x_used_buf[0] = '\0';
    int x_used_off = 0;

    for (const int64_t row : sample_rows) {
        if (!read_tensor_f32_row(x_source, row, x_row_src)) {
            continue;
        }

        if (use_roundtrip) {
            x_row_used.resize((size_t) k);
            const float global_scale = 1.0f / input_scale;
            quantize_row_nvfp4_ref(x_row_src.data(), x_q.data(), k, global_scale);
            dequantize_row_nvfp4(x_q.data(), x_row_used.data(), k, global_scale);
        } else {
            x_row_used = x_row_src;
        }

        char x_src_values[128];
        char x_used_values[128];
        format_first4(x_row_src, x_src_values, sizeof(x_src_values));
        format_first4(x_row_used, x_used_values, sizeof(x_used_values));
        if (x_src_off > 0 && x_src_off < (int) sizeof(x_src_buf)) {
            x_src_off += std::snprintf(x_src_buf + x_src_off, sizeof(x_src_buf) - x_src_off, " | ");
        }
        if (x_src_off >= 0 && x_src_off < (int) sizeof(x_src_buf)) {
            x_src_off += std::snprintf(x_src_buf + x_src_off, sizeof(x_src_buf) - x_src_off, "r%" PRId64 "=%s", row, x_src_values);
        }
        if (x_used_off > 0 && x_used_off < (int) sizeof(x_used_buf)) {
            x_used_off += std::snprintf(x_used_buf + x_used_off, sizeof(x_used_buf) - x_used_off, " | ");
        }
        if (x_used_off >= 0 && x_used_off < (int) sizeof(x_used_buf)) {
            x_used_off += std::snprintf(x_used_buf + x_used_off, sizeof(x_used_buf) - x_used_off, "r%" PRId64 "=%s", row, x_used_values);
        }

        for (const int64_t col : sample_cols) {
            ggml_backend_tensor_get(w, w_row.data(), (size_t) (col * w->nb[1]), (size_t) nblk * sizeof(block_nvfp4));
            dequantize_row_nvfp4(w_row.data(), w_deq.data(), k, 1.0f);

            double ref = 0.0;
            for (int64_t i = 0; i < k; ++i) {
                ref += (double) w_deq[(size_t) i] * (double) x_row_used[(size_t) i];
            }
            ref *= (double) out_scale;

            float actual = 0.0f;
            if (!read_tensor_f32_2d(compare, col, row, actual)) {
                continue;
            }

            const double abs_err = fabs((double) actual - ref);
            if (sample_count == 0 || abs_err > max_abs) {
                max_abs = abs_err;
                max_col = col;
                max_row = row;
            }
            sample_count++;

            if (sample_off > 0 && sample_off < (int) sizeof(sample_buf)) {
                sample_off += snprintf(sample_buf + sample_off, sizeof(sample_buf) - sample_off, " | ");
            }
            if (sample_off >= 0 && sample_off < (int) sizeof(sample_buf)) {
                sample_off += snprintf(
                        sample_buf + sample_off,
                        sizeof(sample_buf) - sample_off,
                        "r%" PRId64 "c%" PRId64 " out=%g ref=%g abs=%g",
                        row,
                        col,
                        (double) actual,
                        ref,
                        abs_err);
            }
        }
    }

    if (sample_count == 0) {
        return;
    }

    LLAMA_LOG_WARN(
            "%s: tensor=%s nvfp4-ref samples: %s (max_abs=%g at=r%" PRId64 "c%" PRId64 ", out_scale=%g, roundtrip=%d)\n",
            tag,
            ggml_get_name(compare),
            sample_buf,
            max_abs,
            max_row,
            max_col,
            (double) out_scale,
            use_roundtrip ? 1 : 0);
    LLAMA_LOG_WARN(
            "%s: tensor=%s nvfp4-input x-src=%s | x-used=%s | w-deq=%s | input_scale=%g\n",
            tag,
            ggml_get_name(compare),
            x_src_buf,
            x_used_buf,
            w_buf,
            (double) input_scale);
}

static void log_tensor_with_sources(const char * tag, const ggml_tensor * tensor, bool log_src) {
    if (!tensor) {
        return;
    }

    LLAMA_LOG_WARN("%s: tensor=%s op=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
            tag,
            ggml_get_name(tensor),
            ggml_op_name(tensor->op),
            ggml_type_name(tensor->type),
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    llama_tensor_stats stats;
    if (!compute_tensor_stats(tensor, stats)) {
        LLAMA_LOG_WARN("%s: tensor=%s type=%s unsupported for stats\n",
                tag, ggml_get_name(tensor), ggml_type_name(tensor->type));
        return;
    }

    log_tensor_first4_f32(tensor);
    log_tensor_stats(tensor, stats);
    log_nvfp4_mul_mat_reference_if_applicable(tag, tensor);

    if (!log_src) {
        return;
    }

    for (int si = 0; si < GGML_MAX_SRC; ++si) {
        const ggml_tensor * src = tensor->src[si];
        if (!src) {
            continue;
        }
        llama_tensor_stats src_stats;
        if (compute_tensor_stats(src, src_stats)) {
            LLAMA_LOG_WARN("%s: tensor=%s src%d=%s op=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                    tag,
                    ggml_get_name(tensor),
                    si,
                    ggml_get_name(src),
                    ggml_op_name(src->op),
                    ggml_type_name(src->type),
                    src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
            log_tensor_src_samples_if_supported(tag, tensor, si, src);
            log_tensor_stats(src, src_stats);
        } else {
            LLAMA_LOG_WARN("%s: tensor=%s src%d=%s op=%s type=%s stats=unsupported\n",
                    tag,
                    ggml_get_name(tensor),
                    si,
                    ggml_get_name(src),
                    ggml_op_name(src->op),
                    ggml_type_name(src->type));
            log_tensor_src_samples_if_supported(tag, tensor, si, src);
        }
    }
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

static bool tensor_preview_enabled() {
    static bool cached = false;
    static bool enabled = false;
    if (!cached) {
        cached = true;
        enabled = getenv("LLAMA_TENSOR_PREVIEW_DEBUG") != nullptr;
    }
    return enabled;
}

void log_tensor_preview(const ggml_tensor * tensor, const void * data_ptr, size_t available) {
    if (!tensor_preview_enabled() || tensor == nullptr || data_ptr == nullptr || available == 0) {
        return;
    }
    const char * tensor_name = ggml_get_name(tensor);
    const char * type_name   = ggml_type_name(tensor->type);

    std::printf("\n[TENSOR LOAD] %s | weight precision: %s\n", tensor_name, type_name);

    if (tensor->type == GGML_TYPE_NVFP4 && available >= ggml_type_size(tensor->type)) {
        const uint8_t * raw_block = static_cast<const uint8_t *>(data_ptr);
        const uint8_t scale = raw_block[0];
        const size_t qk = (size_t) ggml_blck_size(tensor->type); // number of 4-bit values, expected 16
        const size_t qs_bytes = std::min<size_t>(available > 1 ? available - 1 : 0, qk / 2); // expect 8 bytes
        const uint8_t * qs    = raw_block + 1; // skip scale byte

        std::printf("  nvfp4 first block: scale=%u | qs (%zu bytes):", scale, qs_bytes);
        for (size_t i = 0; i < qs_bytes; ++i) {
            std::printf(" %u", qs[i]);
        }
        std::printf("\n");
    } else if (tensor->type == GGML_TYPE_F32 && available >= sizeof(float)) {
        const float * vals = static_cast<const float *>(data_ptr);
        std::printf("  f32 first value: %.9g\n", vals[0]);
    } else {
        const size_t count       = std::min<size_t>(available, (size_t) 10);
        const uint8_t * bytes    = static_cast<const uint8_t *>(data_ptr);

        std::printf("  first %zu uint8 values:", count);
        for (size_t i = 0; i < count; ++i) {
            std::printf(" %u", bytes[i]);
        }
        std::printf("\n");
    }

    std::fflush(stdout);
}

void debug_nvfp4_graph_tensors(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    if (!nvfp4_enabled()) {
        return;
    }

    const bool log_key_nodes = nvfp4_log_key_nodes_enabled();
    const auto & patterns = nvfp4_debug_patterns();
    if ((!log_key_nodes && patterns.empty()) || !gf) {
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
                        log_tensor_first4_f32(t);
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
                    log_tensor_first4_f32(t);
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

    if (!log_key_nodes) {
        return;
    }

    LLAMA_LOG_WARN("%s: key-node scan enabled, n_nodes=%d\n", __func__, ggml_graph_n_nodes(gf));

    struct key_slot {
        const char * prefix = nullptr;
        ggml_tensor * first = nullptr;
        ggml_tensor * last = nullptr;
    };

    key_slot key_slots[] = {
        {"attn_norm-",   nullptr, nullptr},
        {"Qcur-scaled-", nullptr, nullptr},
        {"Kcur-scaled-", nullptr, nullptr},
        {"Vcur-scaled-", nullptr, nullptr},
        {"Qcur_normed-", nullptr, nullptr},
        {"Kcur_normed-", nullptr, nullptr},
        {"ffn_inp-",     nullptr, nullptr},
        {"ffn_norm-",    nullptr, nullptr},
        {"ffn_up-",      nullptr, nullptr},
        {"ffn_gate-",    nullptr, nullptr},
        {"ffn_swiglu-",  nullptr, nullptr},
        {"ffn_down-",    nullptr, nullptr},
        {"ffn_out-",     nullptr, nullptr},
        {"l_out-",       nullptr, nullptr},
    };

    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!t) {
            continue;
        }
        const char * name = ggml_get_name(t);
        for (auto & slot : key_slots) {
            if (!starts_with(name, slot.prefix)) {
                continue;
            }
            if (!slot.first) {
                slot.first = t;
            }
            slot.last = t;
        }
    }

    auto log_key_tensor_once = [&](const char * tag, ggml_tensor * t) {
        if (!t) {
            return;
        }
        if (!seen.insert(t).second) {
            return;
        }
        log_tensor_with_sources(tag, t, log_src);
    };

    for (const auto & slot : key_slots) {
        std::string first_tag = std::string(__func__) + ": key-first(" + slot.prefix + ")";
        log_key_tensor_once(first_tag.c_str(), slot.first);
        if (slot.last && slot.last != slot.first) {
            std::string last_tag = std::string(__func__) + ": key-last(" + slot.prefix + ")";
            log_key_tensor_once(last_tag.c_str(), slot.last);
        }
    }

    log_key_tensor_once((std::string(__func__) + ": key(result_norm)").c_str(), ggml_graph_get_tensor(gf, "result_norm"));
    log_key_tensor_once((std::string(__func__) + ": key(result_output_no_bias)").c_str(), ggml_graph_get_tensor(gf, "result_output_no_bias"));
    log_key_tensor_once((std::string(__func__) + ": key(result_output)").c_str(), ggml_graph_get_tensor(gf, "result_output"));
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

void nvfp4_pin_tensor_if_match(struct ggml_tensor * tensor, const char * base_name, int il, int n_layer) {
    nvfp4_pin_tensor_if_match(tensor);

    if (!nvfp4_tensor_pin_enabled() || !tensor || !base_name || !nvfp4_log_key_nodes_enabled()) {
        return;
    }

    static const char * key_names[] = {
        "attn_norm",
        "Qcur-scaled",
        "Kcur-scaled",
        "Vcur-scaled",
        "Qcur_normed",
        "Kcur_normed",
        "ffn_inp",
        "ffn_norm",
        "ffn_up",
        "ffn_gate",
        "ffn_swiglu",
        "ffn_down",
        "ffn_out",
        "l_out",
    };

    const bool is_first = il == 0;
    const bool is_last  = il >= 0 && n_layer > 0 && il == n_layer - 1;

    if (il >= 0 && (is_first || is_last)) {
        for (const char * key_name : key_names) {
            if (std::strcmp(base_name, key_name) == 0) {
                ggml_set_output(tensor);
                return;
            }
        }
    }

    if (il < 0) {
        if (std::strcmp(base_name, "result_norm") == 0 ||
            std::strcmp(base_name, "result_output_no_bias") == 0 ||
            std::strcmp(base_name, "result_output") == 0) {
            ggml_set_output(tensor);
        }
    }
}

void nvfp4_log_logits_if_enabled(int idx, int64_t j, int n_vocab, const float * logits) {
    if (!nvfp4_logits_debug() || !logits || n_vocab <= 0) {
        return;
    }

    struct top_logit_entry {
        int id = -1;
        float logit = -std::numeric_limits<float>::infinity();
    };

    constexpr int TOPK = 8;
    top_logit_entry topk[TOPK];
    float min_logit = std::numeric_limits<float>::infinity();
    float max_logit = -std::numeric_limits<float>::infinity();
    float sum_exp = 0.0f;
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

        for (int k = 0; k < TOPK; ++k) {
            if (v > topk[k].logit) {
                for (int shift = TOPK - 1; shift > k; --shift) {
                    topk[shift] = topk[shift - 1];
                }
                topk[k].id = t;
                topk[k].logit = v;
                break;
            }
        }
    }

    if (finite_count > 0) {
        for (int32_t t = 0; t < n_vocab; ++t) {
            const float v = logits[t];
            if (!std::isfinite(v)) {
                continue;
            }
            sum_exp += std::exp(v - max_logit);
        }
    }

    if (nan_count > 0 || inf_count > 0) {
        const float min_out = finite_count ? min_logit : 0.0f;
        const float max_out = finite_count ? max_logit : 0.0f;
        LLAMA_LOG_WARN("%s: logits stats: idx=%d j=%" PRId64 " size=%d min=%.6f max=%.6f nan=%d inf=%d\n",
                __func__, idx, j, n_vocab, min_out, max_out, nan_count, inf_count);
    } else {
        static bool logged_once = false;
        if (!logged_once) {
            const float logsumexp = max_logit + std::log(sum_exp);
            const float top1_logit = topk[0].logit;
            const float top1_logprob = top1_logit - logsumexp;
            char topk_buf[512];
            int off = std::snprintf(
                    topk_buf,
                    sizeof(topk_buf),
                    "top%d=",
                    TOPK);
            for (int k = 0; k < TOPK && off > 0 && off < (int) sizeof(topk_buf); ++k) {
                if (topk[k].id < 0 || !std::isfinite(topk[k].logit)) {
                    break;
                }
                off += std::snprintf(
                        topk_buf + off,
                        sizeof(topk_buf) - (size_t) off,
                        "%s%d:%.6f",
                        (k == 0) ? "" : ",",
                        topk[k].id,
                        topk[k].logit);
            }
            LLAMA_LOG_INFO("%s: logits stats: idx=%d j=%" PRId64 " size=%d min=%.6f max=%.6f\n",
                    __func__, idx, j, n_vocab, min_logit, max_logit);
            LLAMA_LOG_INFO(
                    "%s: logits detail: idx=%d j=%" PRId64 " top1_id=%d top1_logit=%.6f logsumexp=%.6f top1_logprob=%.6f %s\n",
                    __func__,
                    idx,
                    j,
                    topk[0].id,
                    top1_logit,
                    logsumexp,
                    top1_logprob,
                    topk_buf);
            logged_once = true;
        }
    }
}

} // namespace llama_log
