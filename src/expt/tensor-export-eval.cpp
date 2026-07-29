#include "tensor-export-eval.h"

#include "llama-impl.h"

#include "../../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "../../vendor/nlohmann/json.hpp"

namespace llama_expt {

struct tensor_export_observed_node {
    int node_index = -1;
    ggml_tensor * dst = nullptr;
    std::unordered_map<const ggml_tensor *, std::vector<uint8_t>> snapshots;
};

struct tensor_export_observer {
    ggml_backend_sched_eval_callback user_callback = nullptr;
    void * user_data = nullptr;
    std::vector<tensor_export_observed_node> nodes;
    std::unordered_map<const ggml_tensor *, size_t> node_lookup;
    std::unordered_map<const ggml_tensor *, std::vector<size_t>> source_lookup;
    std::unordered_map<const ggml_tensor *, uint8_t> pending;
};

namespace {

using json = nlohmann::ordered_json;

constexpr const char * ENV_DIR    = "LLAMA_EXPT_TENSOR_EXPORT_DIR";
constexpr const char * ENV_KINDS  = "LLAMA_EXPT_TENSOR_EXPORT_KINDS";
constexpr const char * ENV_OP     = "LLAMA_EXPT_TENSOR_EXPORT_OP";
constexpr const char * ENV_NAME   = "LLAMA_EXPT_TENSOR_EXPORT_NAME";
constexpr const char * ENV_TYPE   = "LLAMA_EXPT_TENSOR_EXPORT_TYPE";
constexpr const char * ENV_LAYER  = "LLAMA_EXPT_TENSOR_EXPORT_LAYER";

std::unordered_set<std::string> completed_op_exports;

std::string env_str(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "";
}

std::string sanitize_filename(std::string name) {
    for (char & ch : name) {
        const bool ok = (ch >= 'a' && ch <= 'z') ||
                        (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') ||
                        ch == '-' || ch == '_' || ch == '.';
        if (!ok) {
            ch = '_';
        }
    }
    if (name.empty()) {
        return "tensor";
    }
    return name;
}

std::string normalize_op_name(std::string name) {
    name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }), name.end());
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        if (c == '-') {
            return '_';
        }
        return (char) std::toupper(c);
    });
    constexpr const char * prefix = "GGML_OP_";
    const size_t prefix_len = std::strlen(prefix);
    if (name.size() >= prefix_len && name.compare(0, prefix_len, prefix) == 0) {
        name.erase(0, prefix_len);
    }
    return name;
}

std::string selected_export_name() {
    std::string name = env_str(ENV_NAME);
    while (!name.empty() && std::isspace((unsigned char) name.front())) {
        name.erase(name.begin());
    }
    while (!name.empty() && std::isspace((unsigned char) name.back())) {
        name.pop_back();
    }
    return name;
}

std::string selected_export_type() {
    std::string type = env_str(ENV_TYPE);
    type.erase(std::remove_if(type.begin(), type.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }), type.end());
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return type.empty() ? "decode" : type;
}

bool export_type_matches(bool is_prefill, const std::string & type) {
    return (type == "prefill" && is_prefill) || (type == "decode" && !is_prefill);
}

int selected_export_layer() {
    const std::string raw = env_str(ENV_LAYER);
    if (raw.empty()) {
        return -1;
    }

    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(raw.c_str(), &end, 10);
    if (errno != 0 || end == raw.c_str() || *end != '\0' || value < 0 || value > INT_MAX) {
        return -2;
    }
    return (int) value;
}

std::string op_export_key(
        const std::filesystem::path & dir,
        const std::string & type,
        const std::string & requested_name,
        const std::string & requested_op,
        int layer) {
    return dir.lexically_normal().string() + "\n" + type + "\n" +
            requested_name + "\n" + requested_op + "\n" + std::to_string(layer);
}

int parse_layer_number(const std::string & name, size_t start) {
    if (start >= name.size() || !std::isdigit((unsigned char) name[start])) {
        return -1;
    }
    size_t end = start;
    long value = 0;
    while (end < name.size() && std::isdigit((unsigned char) name[end])) {
        value = value * 10 + (name[end] - '0');
        if (value > INT_MAX) {
            return -1;
        }
        ++end;
    }
    if (end != name.size() && name[end] != ' ' && name[end] != '(' && name[end] != '[' && name[end] != '.') {
        return -1;
    }
    return (int) value;
}

int trailing_dash_layer(const std::string & name) {
    const size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash + 1 >= name.size()) {
        return -1;
    }
    for (size_t i = dash + 1; i < name.size(); ++i) {
        if (!std::isdigit((unsigned char) name[i])) {
            return -1;
        }
    }
    return parse_layer_number(name, dash + 1);
}

bool tensor_name_matches_selection(const ggml_tensor * tensor, const std::string & requested_name, int layer) {
    if (!tensor || requested_name.empty()) {
        return false;
    }

    const std::string actual_name = ggml_get_name(tensor);
    const int requested_layer = trailing_dash_layer(requested_name);
    if (layer >= 0) {
        if (requested_layer >= 0 && requested_layer != layer) {
            return false;
        }
        const std::string resolved_name = requested_layer >= 0
                ? requested_name
                : requested_name + "-" + std::to_string(layer);
        return actual_name == resolved_name;
    }

    if (actual_name == requested_name) {
        return true;
    }
    if (requested_layer >= 0 || actual_name.size() <= requested_name.size() + 1 ||
            actual_name.compare(0, requested_name.size(), requested_name) != 0 ||
            actual_name[requested_name.size()] != '-') {
        return false;
    }
    for (size_t i = requested_name.size() + 1; i < actual_name.size(); ++i) {
        if (!std::isdigit((unsigned char) actual_name[i])) {
            return false;
        }
    }
    return true;
}

int tensor_name_layer(const ggml_tensor * tensor) {
    if (!tensor) {
        return -1;
    }
    const std::string name = ggml_get_name(tensor);
    for (size_t pos = 0; pos < name.size(); ++pos) {
        if (name[pos] == '-') {
            const int layer = parse_layer_number(name, pos + 1);
            if (layer >= 0) {
                return layer;
            }
        }
        if (name.compare(pos, 4, "blk.") == 0) {
            const int layer = parse_layer_number(name, pos + 4);
            if (layer >= 0) {
                return layer;
            }
        }
        if (name.compare(pos, 2, "_l") == 0) {
            const int layer = parse_layer_number(name, pos + 2);
            if (layer >= 0) {
                return layer;
            }
        }
    }
    return -1;
}

bool op_node_matches_layer(const ggml_tensor * dst, int layer) {
    if (layer < 0) {
        return true;
    }
    const int dst_layer = tensor_name_layer(dst);
    if (dst_layer >= 0) {
        return dst_layer == layer;
    }
    return tensor_name_layer(dst->src[0]) == layer ||
           tensor_name_layer(dst->src[1]) == layer;
}

bool export_node_matches(
        const ggml_tensor * dst,
        const std::string & requested_name,
        const std::string & requested_op,
        int layer) {
    if (!requested_name.empty()) {
        return tensor_name_matches_selection(dst, requested_name, layer);
    }
    return !requested_op.empty() &&
           normalize_op_name(ggml_op_name(dst->op)) == requested_op &&
           op_node_matches_layer(dst, layer);
}

bool has_prefix(const std::string & text, const char * prefix) {
    const size_t n = std::strlen(prefix);
    return text.size() >= n && text.compare(0, n, prefix) == 0;
}

std::string tensor_kind(const char * raw_name) {
    std::string name = raw_name ? raw_name : "";
    const size_t dash = name.find('-');
    const std::string base = dash == std::string::npos ? name : name.substr(0, dash);

    if (base == "kqv" || base == "kqv_out" || has_prefix(name, "kqv-")) {
        return "kqv";
    }
    if (base == "kq" || has_prefix(name, "kq-")) {
        return "kq";
    }
    if (base == "q" || has_prefix(name, "q-") || has_prefix(name, "Qcur")) {
        return "q";
    }
    if (base == "k" || has_prefix(name, "k-") || has_prefix(name, "Kcur") || has_prefix(name, "cache_k_l")) {
        return "k";
    }
    if (base == "v" || has_prefix(name, "v-") || has_prefix(name, "Vcur") || has_prefix(name, "cache_v_l")) {
        return "v";
    }
    return "";
}

std::set<std::string> selected_kinds() {
    const std::string raw = env_str(ENV_KINDS);
    if (raw.empty()) {
        return { "k", "q", "v", "kq", "kqv" };
    }

    std::set<std::string> out;
    size_t start = 0;
    while (start <= raw.size()) {
        size_t comma = raw.find(',', start);
        std::string item = raw.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c) != 0; }), item.end());
        std::transform(item.begin(), item.end(), item.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        if (!item.empty()) {
            out.insert(item);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

size_t tensor_f32_byte_size(const ggml_tensor * t) {
    return (size_t) ggml_nelements(t) * sizeof(float);
}

json record_to_json(const tensor_record & rec) {
    json obj;
    obj["name"] = rec.name;
    obj["kind"] = rec.kind;
    obj["dtype"] = rec.dtype;
    obj["ne"] = { rec.ne[0], rec.ne[1], rec.ne[2], rec.ne[3] };
    obj["nb"] = { rec.nb[0], rec.nb[1], rec.nb[2], rec.nb[3] };
    obj["path"] = rec.path;
    obj["byte_size"] = rec.byte_size;
    return obj;
}

json metrics_to_json(const tensor_error_metrics & metrics) {
    return {
        { "mae", metrics.mae },
        { "mse", metrics.mse },
        { "rmse", metrics.rmse },
        { "n", metrics.n },
    };
}

tensor_record record_from_json(const json & obj) {
    tensor_record rec;
    rec.name = obj.at("name").get<std::string>();
    rec.kind = obj.value("kind", tensor_kind(rec.name.c_str()));
    rec.dtype = obj.at("dtype").get<std::string>();
    rec.path = obj.at("path").get<std::string>();
    rec.byte_size = obj.at("byte_size").get<size_t>();

    const auto & ne = obj.at("ne");
    const auto & nb = obj.at("nb");
    if (!ne.is_array() || ne.size() != GGML_MAX_DIMS || !nb.is_array() || nb.size() != GGML_MAX_DIMS) {
        throw std::runtime_error("manifest record '" + rec.name + "' must contain 4-element ne and nb arrays");
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        rec.ne[i] = ne.at(i).get<int64_t>();
        rec.nb[i] = nb.at(i).get<size_t>();
    }
    return rec;
}

std::filesystem::path manifest_dir(const std::string & manifest_path) {
    std::filesystem::path path(manifest_path);
    if (path.has_parent_path()) {
        return path.parent_path();
    }
    return std::filesystem::current_path();
}

std::vector<float> load_record_f32(const std::filesystem::path & base_dir, const tensor_record & rec) {
    if (rec.dtype != "f32") {
        throw std::runtime_error("record '" + rec.name + "' has incompatible dtype '" + rec.dtype + "', expected f32");
    }

    int64_t n = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (rec.ne[i] <= 0) {
            throw std::runtime_error("record '" + rec.name + "' has invalid shape");
        }
        n *= rec.ne[i];
    }
    if (n <= 0) {
        throw std::runtime_error("record '" + rec.name + "' has empty shape");
    }
    if (rec.ne[0] % QK_NVFP4 != 0) {
        throw std::runtime_error("record '" + rec.name + "' row shape is not divisible by NVFP4 block size");
    }
    const size_t expected = (size_t) n * sizeof(float);
    if (rec.byte_size != expected) {
        throw std::runtime_error("record '" + rec.name + "' byte_size mismatch: manifest=" +
                std::to_string(rec.byte_size) + " expected=" + std::to_string(expected));
    }

    const std::filesystem::path path = base_dir / rec.path;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open raw tensor '" + path.string() + "'");
    }
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (file_size != (std::streamoff) expected) {
        throw std::runtime_error("record '" + rec.name + "' raw byte_size mismatch: file=" +
                std::to_string((long long) file_size) + " expected=" + std::to_string(expected));
    }

    std::vector<float> values((size_t) n);
    in.read(reinterpret_cast<char *>(values.data()), (std::streamsize) expected);
    if (!in) {
        throw std::runtime_error("failed to read raw tensor '" + path.string() + "'");
    }
    return values;
}

std::vector<float> nvfp4_roundtrip(const std::vector<float> & input, float global_scale) {
    if (input.size() % QK_NVFP4 != 0) {
        throw std::runtime_error("NVFP4 baseline requires element count divisible by block size");
    }

    std::vector<block_nvfp4> quantized(input.size() / QK_NVFP4);
    std::vector<float> output(input.size());
    quantize_row_nvfp4_ref(input.data(), quantized.data(), (int64_t) input.size(), global_scale);
    dequantize_row_nvfp4(quantized.data(), output.data(), (int64_t) output.size(), global_scale);
    return output;
}

tensor_error_metrics metric_delta(const tensor_error_metrics & sorted, const tensor_error_metrics & baseline) {
    tensor_error_metrics out;
    out.mae  = sorted.mae  - baseline.mae;
    out.mse  = sorted.mse  - baseline.mse;
    out.rmse = sorted.rmse - baseline.rmse;
    out.n    = sorted.n;
    return out;
}

std::vector<float> apply_channel_order_by_row(const std::vector<float> & values, size_t row_size, const std::vector<size_t> & order) {
    if (row_size == 0 || values.size() % row_size != 0) {
        throw std::runtime_error("K channel sort requires non-empty contiguous rows");
    }
    if (order.size() != row_size) {
        throw std::runtime_error("K channel sort order size does not match row size");
    }

    std::vector<float> out(values.size());
    const size_t rows = values.size() / row_size;
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * row_size;
        for (size_t j = 0; j < row_size; ++j) {
            out[offset + j] = values[offset + order[j]];
        }
    }
    return out;
}

std::vector<float> restore_channel_order_by_row(const std::vector<float> & values, size_t row_size, const std::vector<size_t> & order) {
    if (row_size == 0 || values.size() % row_size != 0) {
        throw std::runtime_error("K channel sort requires non-empty contiguous rows");
    }
    if (order.size() != row_size) {
        throw std::runtime_error("K channel sort order size does not match row size");
    }

    std::vector<float> out(values.size());
    const size_t rows = values.size() / row_size;
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * row_size;
        for (size_t j = 0; j < row_size; ++j) {
            out[offset + order[j]] = values[offset + j];
        }
    }
    return out;
}

void accumulate_metrics(
        std::map<std::string, double> & sum_abs,
        std::map<std::string, double> & sum_sq,
        std::map<std::string, size_t> & count,
        const std::string & kind,
        const tensor_error_metrics & metrics) {
    sum_abs[kind] += metrics.mae * (double) metrics.n;
    sum_sq[kind]  += metrics.mse * (double) metrics.n;
    count[kind]   += metrics.n;
}

std::map<std::string, tensor_error_metrics> make_aggregate_metrics(
        const std::map<std::string, double> & sum_abs,
        const std::map<std::string, double> & sum_sq,
        const std::map<std::string, size_t> & count) {
    std::map<std::string, tensor_error_metrics> out;
    for (const auto & kv : count) {
        tensor_error_metrics metrics;
        metrics.n = kv.second;
        metrics.mae = sum_abs.at(kv.first) / (double) metrics.n;
        metrics.mse = sum_sq.at(kv.first) / (double) metrics.n;
        metrics.rmse = std::sqrt(metrics.mse);
        out[kv.first] = metrics;
    }
    return out;
}

const char * k_channel_sort_basis_name(k_channel_sort_basis basis) {
    switch (basis) {
        case k_channel_sort_basis::FIRST_ROW_ABS:
            return "first_row_abs";
        case k_channel_sort_basis::ABS_MEAN:
            return "abs_mean";
    }
    return "unknown";
}

const char * k_channel_sort_algorithm_name(k_channel_sort_basis basis) {
    switch (basis) {
        case k_channel_sort_basis::FIRST_ROW_ABS:
            return "nvfp4_k_channel_sort";
        case k_channel_sort_basis::ABS_MEAN:
            return "nvfp4_k_channel_mean_sort";
    }
    return "unknown";
}

void write_manifest(const std::filesystem::path & dir, const std::vector<tensor_record> & records) {
    json manifest;
    manifest["format"] = "llama_expt_tensor_export_v1";
    manifest["records"] = json::array();
    for (const tensor_record & rec : records) {
        manifest["records"].push_back(record_to_json(rec));
    }

    std::ofstream out(dir / "manifest.json", std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open manifest for writing");
    }
    out << manifest.dump(2) << "\n";
}

bool write_op_tensor(
        const std::filesystem::path & dir,
        const ggml_tensor * tensor,
        int node_index,
        const std::string & op,
        const char * role,
        size_t record_index,
        json & records,
        const json & extra = json::object(),
        const std::vector<uint8_t> * snapshot = nullptr) {
    if (!tensor) {
        return false;
    }

    const size_t byte_size = ggml_nbytes(tensor);
    std::vector<uint8_t> bytes;
    if (snapshot) {
        if (snapshot->size() != byte_size) {
            LLAMA_LOG_ERROR("%s: snapshot byte size mismatch for node=%d role=%s tensor='%s': got=%zu expected=%zu\n",
                    __func__, node_index, role, ggml_get_name(tensor), snapshot->size(), byte_size);
            return false;
        }
        bytes = *snapshot;
    } else {
        const ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!buffer || !tensor->data) {
            LLAMA_LOG_WARN("%s: skipping node=%d op=%s role=%s tensor='%s' because it has no backend storage\n",
                    __func__, node_index, op.c_str(), role, ggml_get_name(tensor));
            return false;
        }
        bytes.resize(byte_size);
        ggml_backend_tensor_get(tensor, bytes.data(), 0, byte_size);
    }

    const std::string name = ggml_get_name(tensor);
    const std::string path = std::to_string(record_index) + "-node" + std::to_string(node_index) + "-" +
            role + "-" + sanitize_filename(name) + ".bin";
    std::ofstream out(dir / path, std::ios::binary);
    if (!out) {
        LLAMA_LOG_ERROR("%s: failed to write node=%d op=%s role=%s tensor='%s'\n",
                __func__, node_index, op.c_str(), role, name.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char *>(bytes.data()), (std::streamsize) bytes.size());
    if (!out) {
        LLAMA_LOG_ERROR("%s: failed while writing node=%d op=%s role=%s tensor='%s'\n",
                __func__, node_index, op.c_str(), role, name.c_str());
        return false;
    }

    json rec;
    rec["node_index"] = node_index;
    rec["op"] = op;
    rec["role"] = role;
    rec["name"] = name;
    rec["dtype"] = ggml_type_name(tensor->type);
    rec["ne"] = { tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3] };
    rec["nb"] = { tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3] };
    rec["path"] = path;
    rec["byte_size"] = byte_size;
    rec["contiguous"] = ggml_is_contiguous(tensor);
    rec["view_offset"] = tensor->view_offs;
    rec.update(extra);
    records.push_back(std::move(rec));
    return true;
}

json op_dst_metadata(const ggml_tensor * dst) {
    if (!dst) {
        return json::object();
    }

    if (dst->op == GGML_OP_SOFT_MAX) {
        float scale = 1.0f;
        float max_bias = 0.0f;
        std::memcpy(&scale, (const float *) dst->op_params + 0, sizeof(scale));
        std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(max_bias));
        return {
            { "op_params", {
                { "scale", scale },
                { "max_bias", max_bias },
            } },
        };
    }

    if (dst->op == GGML_OP_ROPE) {
        const int32_t * params = (const int32_t *) dst->op_params;
        float freq_base = 0.0f;
        float freq_scale = 0.0f;
        float ext_factor = 0.0f;
        float attn_factor = 0.0f;
        float beta_fast = 0.0f;
        float beta_slow = 0.0f;
        std::memcpy(&freq_base,   params +  5, sizeof(freq_base));
        std::memcpy(&freq_scale,  params +  6, sizeof(freq_scale));
        std::memcpy(&ext_factor,  params +  7, sizeof(ext_factor));
        std::memcpy(&attn_factor, params +  8, sizeof(attn_factor));
        std::memcpy(&beta_fast,   params +  9, sizeof(beta_fast));
        std::memcpy(&beta_slow,   params + 10, sizeof(beta_slow));
        return {
            { "op_params", {
                { "n_dims", params[1] },
                { "mode", params[2] },
                { "n_ctx_orig", params[4] },
                { "freq_base", freq_base },
                { "freq_scale", freq_scale },
                { "ext_factor", ext_factor },
                { "attn_factor", attn_factor },
                { "beta_fast", beta_fast },
                { "beta_slow", beta_slow },
                { "sections", { params[11], params[12], params[13], params[14] } },
            } },
        };
    }

    return json::object();
}

bool read_contiguous_tensor_f32(
        const ggml_tensor * tensor,
        std::vector<float> & values,
        const std::vector<uint8_t> * snapshot = nullptr) {
    if (!tensor || !tensor->data || !ggml_is_contiguous(tensor)) {
        return false;
    }
    const ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer) {
        return false;
    }

    const size_t n = (size_t) ggml_nelements(tensor);
    values.resize(n);
    switch (tensor->type) {
        case GGML_TYPE_F32:
            if (ggml_nbytes(tensor) != n * sizeof(float)) {
                return false;
            }
            if (snapshot) {
                if (snapshot->size() != n * sizeof(float)) {
                    return false;
                }
                std::memcpy(values.data(), snapshot->data(), snapshot->size());
            } else {
                ggml_backend_tensor_get(tensor, values.data(), 0, n * sizeof(float));
            }
            return true;
        case GGML_TYPE_F16: {
            if (ggml_nbytes(tensor) != n * sizeof(ggml_fp16_t)) {
                return false;
            }
            std::vector<ggml_fp16_t> raw(n);
            if (snapshot) {
                if (snapshot->size() != raw.size() * sizeof(raw[0])) {
                    return false;
                }
                std::memcpy(raw.data(), snapshot->data(), snapshot->size());
            } else {
                ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(raw[0]));
            }
            for (size_t i = 0; i < n; ++i) {
                values[i] = ggml_fp16_to_fp32(raw[i]);
            }
            return true;
        }
        case GGML_TYPE_BF16: {
            if (ggml_nbytes(tensor) != n * sizeof(ggml_bf16_t)) {
                return false;
            }
            std::vector<ggml_bf16_t> raw(n);
            if (snapshot) {
                if (snapshot->size() != raw.size() * sizeof(raw[0])) {
                    return false;
                }
                std::memcpy(raw.data(), snapshot->data(), snapshot->size());
            } else {
                ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size() * sizeof(raw[0]));
            }
            for (size_t i = 0; i < n; ++i) {
                values[i] = ggml_bf16_to_fp32(raw[i]);
            }
            return true;
        }
        default:
            return false;
    }
}

bool write_derived_f32_tensor(
        const std::filesystem::path & dir,
        const ggml_tensor * shape_source,
        const std::vector<float> & values,
        int node_index,
        const std::string & op,
        const char * role,
        size_t record_index,
        json & records,
        const json & extra = json::object()) {
    if (!shape_source || values.size() != (size_t) ggml_nelements(shape_source)) {
        return false;
    }

    const std::string name = std::string(ggml_get_name(shape_source)) + ".canonical_global_scale";
    const std::string path = std::to_string(record_index) + "-node" + std::to_string(node_index) + "-" +
            role + "-" + sanitize_filename(name) + ".bin";
    std::ofstream out(dir / path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(values.data()), (std::streamsize) (values.size() * sizeof(float)));
    if (!out) {
        return false;
    }

    json rec;
    rec["node_index"] = node_index;
    rec["op"] = op;
    rec["role"] = role;
    rec["name"] = name;
    rec["dtype"] = "f32";
    rec["ne"] = { shape_source->ne[0], shape_source->ne[1], shape_source->ne[2], shape_source->ne[3] };
    const size_t nb0 = sizeof(float);
    const size_t nb1 = nb0 * (size_t) shape_source->ne[0];
    const size_t nb2 = nb1 * (size_t) shape_source->ne[1];
    const size_t nb3 = nb2 * (size_t) shape_source->ne[2];
    rec["nb"] = { nb0, nb1, nb2, nb3 };
    rec["path"] = path;
    rec["byte_size"] = values.size() * sizeof(float);
    rec["contiguous"] = true;
    rec["view_offset"] = 0;
    rec["derived"] = true;
    rec.update(extra);
    records.push_back(std::move(rec));
    return true;
}

const char * capture_status_name(uint32_t flags) {
    if ((flags & GGML_NVFP4_MUL_MAT_CAPTURE_REQUESTED) == 0) {
        return "not_requested";
    }
    if ((flags & GGML_NVFP4_MUL_MAT_CAPTURE_VALID) == 0) {
        return "native_nvfp4_not_used";
    }
    return "native_nvfp4_valid";
}

const char * capture_variant_name(uint32_t flags) {
    if ((flags & GGML_NVFP4_MUL_MAT_CAPTURE_FP4MULMAT) != 0) {
        return "fp4mulmat";
    }
    if ((flags & GGML_NVFP4_MUL_MAT_CAPTURE_CUBLASLT) != 0) {
        return "cublaslt";
    }
    return "none";
}

const tensor_export_observed_node * observed_node(
        const tensor_export_observer * observer,
        int node_index,
        const ggml_tensor * dst) {
    if (!observer) {
        return nullptr;
    }
    for (const tensor_export_observed_node & node : observer->nodes) {
        if (node.node_index == node_index && node.dst == dst) {
            return &node;
        }
    }
    return nullptr;
}

const std::vector<uint8_t> * observed_tensor(
        const tensor_export_observed_node * node,
        const ggml_tensor * tensor) {
    if (!node || !tensor) {
        return nullptr;
    }
    const auto it = node->snapshots.find(tensor);
    return it == node->snapshots.end() ? nullptr : &it->second;
}

bool export_op_graph(
        ggml_backend_sched_t sched,
        ggml_cgraph * gf,
        bool is_prefill,
        const std::filesystem::path & dir,
        const tensor_export_observer * observer) {
    const std::string requested_name = selected_export_name();
    const std::string requested_op = normalize_op_name(env_str(ENV_OP));
    const std::string type = selected_export_type();
    const int layer = selected_export_layer();
    if (requested_name.empty() && requested_op.empty()) {
        return false;
    }
    if (type != "decode" && type != "prefill") {
        LLAMA_LOG_ERROR("%s: invalid %s='%s'; expected decode or prefill\n", __func__, ENV_TYPE, type.c_str());
        return false;
    }
    if (layer == -2) {
        LLAMA_LOG_ERROR("%s: invalid %s='%s'; expected a non-negative integer\n",
                __func__, ENV_LAYER, env_str(ENV_LAYER).c_str());
        return false;
    }
    if (!export_type_matches(is_prefill, type)) {
        return false;
    }

    const int requested_name_layer = trailing_dash_layer(requested_name);
    if (!requested_name.empty() && layer >= 0 && requested_name_layer >= 0 && requested_name_layer != layer) {
        LLAMA_LOG_ERROR("%s: conflicting %s='%s' and %s=%d\n",
                __func__, ENV_NAME, requested_name.c_str(), ENV_LAYER, layer);
        return false;
    }

    std::vector<std::pair<int, ggml_tensor *>> matched;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * dst = ggml_graph_node(gf, i);
        if (dst && export_node_matches(dst, requested_name, requested_op, layer)) {
            matched.emplace_back(i, dst);
        }
    }
    if (matched.empty()) {
        return false;
    }

    const std::string export_key = op_export_key(dir, type, requested_name, requested_op, layer);
    if (!completed_op_exports.insert(export_key).second) {
        return false;
    }

    ggml_backend_sched_synchronize(sched);

    json manifest;
    manifest["format"] = "llama_expt_op_tensor_export_v2";
    manifest["type"] = type;
    manifest["op"] = requested_op;
    manifest["layer"] = layer;
    manifest["snapshot_timing"] = observer ? "source_producer_and_node_completion" : "post_graph";
    manifest["selection"] = {
        { "priority", requested_name.empty() ? "op" : "tensor_name" },
        { "requested_name", requested_name },
        { "requested_op", requested_op },
        { "requested_layer", layer },
    };
    if (!requested_name.empty() && !requested_op.empty()) {
        manifest["selection"]["ignored_op"] = requested_op;
    }
    manifest["records"] = json::array();
    manifest["captures"] = json::array();

    size_t record_index = 0;
    for (const auto & item : matched) {
        const int i = item.first;
        ggml_tensor * dst = item.second;
        const tensor_export_observed_node * observed = observed_node(observer, i, dst);
        if (observer && !observed) {
            LLAMA_LOG_ERROR("%s: missing node-completion snapshot for node=%d tensor='%s'\n",
                    __func__, i, ggml_get_name(dst));
            return false;
        }
        if (observer) {
            for (const ggml_tensor * tensor : { dst, dst->src[0], dst->src[1], dst->src[2] }) {
                if (tensor && !observed_tensor(observed, tensor)) {
                    LLAMA_LOG_ERROR("%s: missing tensor snapshot for node=%d tensor='%s'\n",
                            __func__, i, ggml_get_name(tensor));
                    return false;
                }
            }
        }
        const std::string actual_op = normalize_op_name(ggml_op_name(dst->op));
        if (write_op_tensor(dir, dst, i, actual_op, "dst", record_index, manifest["records"],
                    op_dst_metadata(dst), observed_tensor(observed, dst))) {
            ++record_index;
        }
        if (write_op_tensor(dir, dst->src[0], i, actual_op, "src0", record_index, manifest["records"],
                    { { "effective_role", dst->src[0] && dst->src[0]->type == GGML_TYPE_NVFP4 ? "a_nvfp4" : "" } },
                    observed_tensor(observed, dst->src[0]))) {
            ++record_index;
        }
        if (write_op_tensor(dir, dst->src[1], i, actual_op, "src1", record_index, manifest["records"],
                    { { "effective_role", "b_original" } }, observed_tensor(observed, dst->src[1]))) {
            ++record_index;
        }
        if (write_op_tensor(dir, dst->src[2], i, actual_op, "src2", record_index, manifest["records"],
                    json::object(), observed_tensor(observed, dst->src[2]))) {
            ++record_index;
        }

        if (dst->op != GGML_OP_MUL_MAT || !dst->src[0] || dst->src[0]->type != GGML_TYPE_NVFP4) {
            continue;
        }

        json capture;
        const uint32_t flags = ggml_mul_mat_get_nvfp4_capture_flags(dst);
        capture["node_index"] = i;
        capture["name"] = ggml_get_name(dst);
        capture["actual_op"] = actual_op;
        capture["status"] = capture_status_name(flags);
        capture["native_variant"] = capture_variant_name(flags);
        capture["scale_mode"] = (flags & GGML_NVFP4_MUL_MAT_CAPTURE_DYNAMIC) == 0
                ? "static"
                : ((flags & GGML_NVFP4_MUL_MAT_CAPTURE_PER_TENSOR) != 0 ? "dynamic_per_tensor" : "dynamic_per_row");
        const bool captures_final_scale =
                (flags & GGML_NVFP4_MUL_MAT_CAPTURE_FINAL_SCALE) != 0;

        if (!captures_final_scale) {
            const ggml_tensor * a_scale_raw = ggml_tensor_get_nvfp4_scale(dst->src[0]);
            const char * a_scale_source = "src0_nvfp4_scale";
            if (a_scale_raw == nullptr) {
                a_scale_raw = ggml_mul_mat_get_nvfp4_weight_scale(dst);
                a_scale_source = "mul_mat_weight_scale";
            }
            if (a_scale_raw != nullptr) {
                if (observer && !observed_tensor(observed, a_scale_raw)) {
                    LLAMA_LOG_ERROR("%s: missing A-scale snapshot for node=%d tensor='%s'\n",
                            __func__, i, ggml_get_name(a_scale_raw));
                    return false;
                }
                if (write_op_tensor(dir, a_scale_raw, i, actual_op, "src0_scale_raw", record_index,
                            manifest["records"], {
                                { "scale_source", a_scale_source },
                                { "scale_encoding", "inverse_global_scale" },
                            }, observed_tensor(observed, a_scale_raw))) {
                    ++record_index;
                }

                std::vector<float> inverse_scales;
                if (read_contiguous_tensor_f32(a_scale_raw, inverse_scales, observed_tensor(observed, a_scale_raw))) {
                    std::vector<float> global_scales(inverse_scales.size());
                    for (size_t j = 0; j < inverse_scales.size(); ++j) {
                        const float v = inverse_scales[j];
                        global_scales[j] = std::isfinite(v) && v != 0.0f ? 1.0f / v : 0.0f;
                    }
                    if (write_derived_f32_tensor(dir, a_scale_raw, global_scales, i, actual_op,
                                "src0_global_scale", record_index, manifest["records"], {
                                    { "scale_source", a_scale_source },
                                    { "derived_from_encoding", "inverse_global_scale" },
                                })) {
                        ++record_index;
                        capture["a_global_scale_role"] = "src0_global_scale";
                    }
                } else {
                    LLAMA_LOG_WARN("%s: unable to canonicalize A global scale for tensor '%s'\n",
                            __func__, ggml_get_name(dst));
                }
            } else {
                capture["a_global_scale"] = 1.0f;
                capture["a_global_scale_source"] = "implicit_unit";
            }
        }

        if ((flags & GGML_NVFP4_MUL_MAT_CAPTURE_VALID) != 0) {
            const ggml_tensor * b_nvfp4 = ggml_mul_mat_get_nvfp4_rhs_capture(dst);
            const ggml_tensor * b_scale_capture = ggml_mul_mat_get_nvfp4_rhs_global_scale_capture(dst);
            if (observer && (!observed_tensor(observed, b_nvfp4) ||
                    !observed_tensor(observed, b_scale_capture))) {
                LLAMA_LOG_ERROR("%s: missing effective RHS snapshot for node=%d tensor='%s'\n",
                        __func__, i, ggml_get_name(dst));
                return false;
            }
            if (write_op_tensor(dir, b_nvfp4, i, actual_op, "src1_nvfp4", record_index,
                        manifest["records"], { { "effective_role", "b_nvfp4" } },
                        observed_tensor(observed, b_nvfp4))) {
                ++record_index;
            }
            const char * scale_role = captures_final_scale ? "matmul_scale" : "src1_global_scale";
            if (write_op_tensor(dir, b_scale_capture, i, actual_op, scale_role, record_index,
                        manifest["records"], captures_final_scale ? json {
                            { "scale_encoding", "f32" },
                            { "scale_semantics", "final_output_multiplier" },
                            { "operand_rounding", "bf16_rne" },
                            { "scale_axis", (flags & GGML_NVFP4_MUL_MAT_CAPTURE_DYNAMIC) != 0 ? 1 : -1 },
                        } : json {
                            { "scale_encoding", "global_scale" },
                            { "scale_axis", (flags & GGML_NVFP4_MUL_MAT_CAPTURE_DYNAMIC) != 0 ? 1 : -1 },
                        }, observed_tensor(observed, b_scale_capture))) {
                ++record_index;
            }
            if (captures_final_scale) {
                capture["effective_srcs"] = {
                    { "a", { { "tensor_role", "src0" } } },
                    { "b", { { "tensor_role", "src1_nvfp4" } } },
                    { "matmul_scale_role", "matmul_scale" },
                };
            } else {
                capture["effective_srcs"] = {
                    { "a", {
                        { "tensor_role", "src0" },
                        { "global_scale_role", capture.value("a_global_scale_role", "") },
                    } },
                    { "b", {
                        { "tensor_role", "src1_nvfp4" },
                        { "global_scale_role", "src1_global_scale" },
                    } },
                };
            }
        }
        manifest["captures"].push_back(std::move(capture));
    }
    manifest["matched_nodes"] = matched.size();

    std::ofstream out(dir / "manifest.json", std::ios::binary);
    if (!out) {
        LLAMA_LOG_ERROR("%s: failed to open op export manifest in '%s'\n", __func__, dir.string().c_str());
        return false;
    }
    out << manifest.dump(2) << "\n";
    if (!out) {
        LLAMA_LOG_ERROR("%s: failed to write op export manifest in '%s'\n", __func__, dir.string().c_str());
        return false;
    }

    LLAMA_LOG_INFO("%s: exported type=%s name=%s op=%s layer=%d matched_nodes=%zu tensor_records=%zu to '%s'\n",
            __func__, type.c_str(), requested_name.c_str(), requested_op.c_str(), layer,
            matched.size(), record_index, dir.string().c_str());
    return true;
}

} // namespace

bool tensor_export_enabled() {
    return !env_str(ENV_DIR).empty();
}

bool tensor_export_maybe_retain_graph(ggml_cgraph * gf) {
    if (!tensor_export_enabled() || !gf) {
        return false;
    }

    const std::string requested_name = selected_export_name();
    const std::string requested_op = normalize_op_name(env_str(ENV_OP));
    const std::string type = selected_export_type();
    const int layer = selected_export_layer();
    if ((type != "decode" && type != "prefill") || layer == -2) {
        return false;
    }

    const int requested_name_layer = trailing_dash_layer(requested_name);
    if (!requested_name.empty() && layer >= 0 && requested_name_layer >= 0 && requested_name_layer != layer) {
        return false;
    }

    std::unordered_set<ggml_tensor *> retained;
    auto retain = [&](ggml_tensor * tensor) {
        if (!tensor) {
            return;
        }
        if (retained.insert(tensor).second) {
            ggml_set_output(tensor);
        }
        for (ggml_tensor * view_src = tensor->view_src; view_src; view_src = view_src->view_src) {
            if (retained.insert(view_src).second) {
                ggml_set_output(view_src);
            }
        }
    };

    size_t matched_nodes = 0;
    const int n_nodes = ggml_graph_n_nodes(gf);
    if (!requested_name.empty() || !requested_op.empty()) {
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * dst = ggml_graph_node(gf, i);
            if (!dst || !export_node_matches(dst, requested_name, requested_op, layer)) {
                continue;
            }
            ++matched_nodes;
            retain(dst);
            retain(dst->src[0]);
            retain(dst->src[1]);
            retain(dst->src[2]);
        }
    } else {
        const auto kinds = selected_kinds();
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * tensor = ggml_graph_node(gf, i);
            if (!tensor) {
                continue;
            }
            const std::string kind = tensor_kind(ggml_get_name(tensor));
            if (!kind.empty() && kinds.count(kind) != 0) {
                ++matched_nodes;
                retain(tensor);
            }
        }
    }

    static bool logged = false;
    if (!logged && matched_nodes > 0) {
        logged = true;
        LLAMA_LOG_INFO("%s: retained %zu matched nodes (%zu tensors) before graph allocation\n",
                __func__, matched_nodes, retained.size());
    }
    return matched_nodes > 0;
}

tensor_export_observer * tensor_export_observer_create(
        ggml_cgraph * gf,
        bool is_prefill,
        ggml_backend_sched_eval_callback user_callback,
        void * user_data) {
    if (!tensor_export_enabled() || !gf) {
        return nullptr;
    }

    const std::string requested_name = selected_export_name();
    const std::string requested_op = normalize_op_name(env_str(ENV_OP));
    const std::string type = selected_export_type();
    const int layer = selected_export_layer();
    if ((requested_name.empty() && requested_op.empty()) ||
            (type != "decode" && type != "prefill") || layer == -2 ||
            !export_type_matches(is_prefill, type)) {
        return nullptr;
    }

    const int requested_name_layer = trailing_dash_layer(requested_name);
    if (!requested_name.empty() && layer >= 0 && requested_name_layer >= 0 && requested_name_layer != layer) {
        return nullptr;
    }

    const std::filesystem::path dir(env_str(ENV_DIR));
    if (completed_op_exports.count(op_export_key(dir, type, requested_name, requested_op, layer)) != 0) {
        return nullptr;
    }

    tensor_export_observer * observer = new tensor_export_observer;
    observer->user_callback = user_callback;
    observer->user_data = user_data;

    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * dst = ggml_graph_node(gf, i);
        if (!dst || !export_node_matches(dst, requested_name, requested_op, layer)) {
            continue;
        }
        tensor_export_observed_node node;
        node.node_index = i;
        node.dst = dst;
        const size_t observed_index = observer->nodes.size();
        observer->node_lookup.emplace(dst, observed_index);
        observer->nodes.push_back(std::move(node));
        for (const ggml_tensor * src : { dst->src[0], dst->src[1], dst->src[2] }) {
            if (src) {
                observer->source_lookup[src].push_back(observed_index);
            }
        }
    }

    if (observer->nodes.empty()) {
        delete observer;
        return nullptr;
    }
    return observer;
}

bool tensor_export_observer_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    tensor_export_observer * observer = static_cast<tensor_export_observer *>(user_data);
    if (!observer || !tensor) {
        return true;
    }

    if (ask) {
        const bool observes_tensor = observer->node_lookup.count(tensor) != 0 ||
                observer->source_lookup.count(tensor) != 0;
        uint8_t pending = observes_tensor ? UINT8_C(1) : UINT8_C(0);
        if (observer->user_callback && observer->user_callback(tensor, true, observer->user_data)) {
            pending |= UINT8_C(2);
        }
        if (pending != 0) {
            observer->pending[tensor] = pending;
        }
        return pending != 0;
    }

    const auto pending_it = observer->pending.find(tensor);
    const uint8_t pending = pending_it == observer->pending.end() ? UINT8_C(0) : pending_it->second;
    if ((pending & UINT8_C(1)) != 0) {
        auto snapshot = [&](tensor_export_observed_node & node, const ggml_tensor * value) {
            if (!value || node.snapshots.count(value) != 0) {
                return;
            }
            const ggml_backend_buffer_t buffer = value->view_src ? value->view_src->buffer : value->buffer;
            if (!buffer || !value->data) {
                LLAMA_LOG_WARN("%s: unable to snapshot node=%d tensor='%s' without backend storage\n",
                        __func__, node.node_index, ggml_get_name(value));
                return;
            }
            std::vector<uint8_t> bytes(ggml_nbytes(value));
            ggml_backend_tensor_get(value, bytes.data(), 0, bytes.size());
            node.snapshots.emplace(value, std::move(bytes));
        };

        const auto source_it = observer->source_lookup.find(tensor);
        if (source_it != observer->source_lookup.end()) {
            for (size_t observed_index : source_it->second) {
                snapshot(observer->nodes.at(observed_index), tensor);
            }
        }

        const auto node_it = observer->node_lookup.find(tensor);
        if (node_it != observer->node_lookup.end()) {
            tensor_export_observed_node & node = observer->nodes.at(node_it->second);
            snapshot(node, node.dst);
            snapshot(node, node.dst->src[0]);
            snapshot(node, node.dst->src[1]);
            snapshot(node, node.dst->src[2]);
            if (node.dst->op == GGML_OP_MUL_MAT && node.dst->src[0] &&
                    node.dst->src[0]->type == GGML_TYPE_NVFP4) {
                const ggml_tensor * a_scale = ggml_tensor_get_nvfp4_scale(node.dst->src[0]);
                if (!a_scale) {
                    a_scale = ggml_mul_mat_get_nvfp4_weight_scale(node.dst);
                }
                snapshot(node, a_scale);
                snapshot(node, ggml_mul_mat_get_nvfp4_rhs_capture(node.dst));
                snapshot(node, ggml_mul_mat_get_nvfp4_rhs_global_scale_capture(node.dst));
            }
        }
    }

    bool keep_going = true;
    if ((pending & UINT8_C(2)) != 0 && observer->user_callback) {
        keep_going = observer->user_callback(tensor, false, observer->user_data);
    }
    if (pending_it != observer->pending.end()) {
        observer->pending.erase(pending_it);
    }
    return keep_going;
}

void tensor_export_observer_free(tensor_export_observer * observer) {
    delete observer;
}

bool tensor_export_maybe_bind_nvfp4_mul_mat_capture(
        ggml_context * ctx,
        ggml_tensor * tensor,
        bool is_prefill) {
    if (!ctx || !tensor_export_enabled() || !tensor) {
        return false;
    }

    const std::string requested_name = selected_export_name();
    const std::string requested_op = normalize_op_name(env_str(ENV_OP));
    const std::string type = selected_export_type();
    const int layer = selected_export_layer();
    if ((requested_name.empty() && requested_op.empty()) ||
            (type != "decode" && type != "prefill") ||
            layer == -2) {
        return false;
    }
    GGML_UNUSED(is_prefill);

    const int requested_name_layer = trailing_dash_layer(requested_name);
    if (!requested_name.empty() && layer >= 0 && requested_name_layer >= 0 && requested_name_layer != layer) {
        static bool logged_name_layer_conflict = false;
        if (!logged_name_layer_conflict) {
            logged_name_layer_conflict = true;
            LLAMA_LOG_ERROR("%s: conflicting %s='%s' and %s=%d\n",
                    __func__, ENV_NAME, requested_name.c_str(), ENV_LAYER, layer);
        }
        return false;
    }

    if (!export_node_matches(tensor, requested_name, requested_op, layer) ||
            tensor->op != GGML_OP_MUL_MAT ||
            tensor->src[0] == nullptr || tensor->src[1] == nullptr ||
            tensor->src[0]->type != GGML_TYPE_NVFP4 ||
            tensor->src[1]->type != GGML_TYPE_F32 ||
            tensor->type != GGML_TYPE_F32 ||
            tensor->src[1]->ne[0] % QK_NVFP4 != 0) {
        return false;
    }

    if (ggml_mul_mat_get_nvfp4_rhs_capture(tensor) != nullptr) {
        return true;
    }

    ggml_tensor * rhs_nvfp4 = ggml_new_tensor(
            ctx, GGML_TYPE_NVFP4, GGML_MAX_DIMS, tensor->src[1]->ne);
    ggml_tensor * rhs_scale_capture = nullptr;
    if (ggml_mul_mat_get_nvfp4_input_scale(tensor) != nullptr) {
        rhs_scale_capture = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    } else {
        rhs_scale_capture = ggml_new_tensor_3d(
                ctx, GGML_TYPE_F32,
                tensor->src[1]->ne[1],
                tensor->src[1]->ne[2],
                tensor->src[1]->ne[3]);
    }

    ggml_format_name(rhs_nvfp4, "%s.src1_nvfp4", ggml_get_name(tensor));
    ggml_format_name(rhs_scale_capture, "%s.src1_scale_capture", ggml_get_name(tensor));
    ggml_set_output(rhs_nvfp4);
    ggml_set_output(rhs_scale_capture);
    ggml_mul_mat_set_nvfp4_rhs_capture(tensor, rhs_nvfp4, rhs_scale_capture);
    return true;
}

bool tensor_export_maybe_log_config() {
    static bool logged = false;
    if (logged) {
        return tensor_export_enabled();
    }
    logged = true;

    const bool enabled = tensor_export_enabled();
    if (enabled) {
        LLAMA_LOG_INFO("%s: enabled %s='%s' %s='%s' %s='%s' %s='%s' %s='%s' %s='%s'\n",
                __func__, ENV_DIR, env_str(ENV_DIR).c_str(), ENV_KINDS, env_str(ENV_KINDS).c_str(),
                ENV_OP, env_str(ENV_OP).c_str(), ENV_NAME, env_str(ENV_NAME).c_str(), ENV_TYPE, env_str(ENV_TYPE).c_str(),
                ENV_LAYER, env_str(ENV_LAYER).c_str());
    } else {
        LLAMA_LOG_INFO("%s: disabled; %s is unset\n", __func__, ENV_DIR);
    }
    return enabled;
}

bool tensor_export_graph(
        ggml_backend_sched_t sched,
        ggml_cgraph * gf,
        bool is_prefill,
        const tensor_export_observer * observer) {
    if (!tensor_export_maybe_log_config()) {
        return false;
    }
    if (!sched || !gf) {
        return false;
    }

    const std::filesystem::path dir(env_str(ENV_DIR));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LLAMA_LOG_ERROR("%s: failed to create export dir '%s': %s\n", __func__, dir.string().c_str(), ec.message().c_str());
        return false;
    }

    if (!env_str(ENV_NAME).empty() || !env_str(ENV_OP).empty()) {
        return export_op_graph(sched, gf, is_prefill, dir, observer);
    }

    const auto kinds = selected_kinds();
    std::vector<tensor_record> records;
    std::unordered_set<const ggml_tensor *> seen;

    ggml_backend_sched_synchronize(sched);

    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!t || !seen.insert(t).second) {
            continue;
        }

        const std::string kind = tensor_kind(ggml_get_name(t));
        if (kind.empty() || kinds.count(kind) == 0) {
            continue;
        }
        if (t->type != GGML_TYPE_F32) {
            LLAMA_LOG_WARN("%s: skipping tensor '%s' kind=%s dtype=%s, only f32 export is supported\n",
                    __func__, ggml_get_name(t), kind.c_str(), ggml_type_name(t->type));
            continue;
        }
        if (!ggml_is_contiguous(t)) {
            LLAMA_LOG_WARN("%s: skipping tensor '%s' kind=%s because only contiguous f32 export is supported\n",
                    __func__, ggml_get_name(t), kind.c_str());
            continue;
        }

        const size_t byte_size = tensor_f32_byte_size(t);
        std::vector<uint8_t> bytes(byte_size);
        ggml_backend_tensor_get(t, bytes.data(), 0, byte_size);

        tensor_record rec;
        rec.name = ggml_get_name(t);
        rec.kind = kind;
        rec.dtype = "f32";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            rec.ne[d] = t->ne[d];
            rec.nb[d] = t->nb[d];
        }
        rec.byte_size = byte_size;
        rec.path = std::to_string(records.size()) + "-" + sanitize_filename(rec.name) + ".bin";

        std::ofstream out(dir / rec.path, std::ios::binary);
        if (!out) {
            LLAMA_LOG_ERROR("%s: failed to write tensor '%s'\n", __func__, rec.name.c_str());
            continue;
        }
        out.write(reinterpret_cast<const char *>(bytes.data()), (std::streamsize) bytes.size());
        if (!out) {
            LLAMA_LOG_ERROR("%s: failed while writing tensor '%s'\n", __func__, rec.name.c_str());
            continue;
        }
        records.push_back(rec);
    }

    try {
        write_manifest(dir, records);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }

    LLAMA_LOG_INFO("%s: exported %zu tensor records to '%s'\n", __func__, records.size(), dir.string().c_str());
    return true;
}

tensor_error_metrics compute_error_metrics(const std::vector<float> & reference, const std::vector<float> & actual) {
    if (reference.size() != actual.size()) {
        throw std::runtime_error("metric input size mismatch");
    }
    if (reference.empty()) {
        throw std::runtime_error("metric input is empty");
    }

    tensor_error_metrics out;
    out.n = reference.size();
    for (size_t i = 0; i < reference.size(); ++i) {
        const double diff = (double) actual[i] - (double) reference[i];
        out.mae += std::fabs(diff);
        out.mse += diff * diff;
    }
    out.mae /= (double) out.n;
    out.mse /= (double) out.n;
    out.rmse = std::sqrt(out.mse);
    return out;
}

std::vector<size_t> make_k_channel_order_from_first_row(const std::vector<float> & values, size_t row_size) {
    if (row_size == 0 || values.size() < row_size) {
        throw std::runtime_error("K channel sort requires a non-empty first row");
    }

    std::vector<size_t> order(row_size);
    for (size_t i = 0; i < row_size; ++i) {
        order[i] = i;
    }

    std::stable_sort(order.begin(), order.end(), [&values](size_t lhs, size_t rhs) {
        const float lhs_abs = std::fabs(values[lhs]);
        const float rhs_abs = std::fabs(values[rhs]);
        if (lhs_abs == rhs_abs) {
            return lhs < rhs;
        }
        return lhs_abs > rhs_abs;
    });
    return order;
}

std::vector<size_t> make_k_channel_order_from_abs_mean(const std::vector<float> & values, size_t row_size) {
    if (row_size == 0 || values.empty() || values.size() % row_size != 0) {
        throw std::runtime_error("K channel mean sort requires non-empty contiguous rows");
    }

    const size_t rows = values.size() / row_size;
    std::vector<double> means(row_size, 0.0);
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * row_size;
        for (size_t channel = 0; channel < row_size; ++channel) {
            means[channel] += (double) values[offset + channel];
        }
    }
    for (double & mean : means) {
        mean /= (double) rows;
    }

    std::vector<size_t> order(row_size);
    for (size_t i = 0; i < row_size; ++i) {
        order[i] = i;
    }

    std::stable_sort(order.begin(), order.end(), [&means](size_t lhs, size_t rhs) {
        const double lhs_abs = std::fabs(means[lhs]);
        const double rhs_abs = std::fabs(means[rhs]);
        if (lhs_abs == rhs_abs) {
            return lhs < rhs;
        }
        return lhs_abs > rhs_abs;
    });
    return order;
}

std::vector<tensor_record> load_manifest_records(const std::string & manifest_path) {
    std::ifstream in(manifest_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open manifest '" + manifest_path + "'");
    }
    json manifest = json::parse(in);
    if (!manifest.contains("records") || !manifest["records"].is_array()) {
        throw std::runtime_error("manifest must contain a records array");
    }

    std::vector<tensor_record> records;
    for (const auto & item : manifest["records"]) {
        records.push_back(record_from_json(item));
    }
    return records;
}

eval_report evaluate_manifest(const std::string & manifest_path, float global_scale) {
    eval_report report;
    report.global_scale = global_scale;
    const std::filesystem::path base_dir = manifest_dir(manifest_path);
    const std::vector<tensor_record> records = load_manifest_records(manifest_path);

    std::map<std::string, double> sum_abs;
    std::map<std::string, double> sum_sq;
    std::map<std::string, size_t> count;

    for (const tensor_record & rec : records) {
        std::vector<float> values = load_record_f32(base_dir, rec);
        std::vector<float> roundtrip = nvfp4_roundtrip(values, global_scale);
        tensor_error_metrics metrics = compute_error_metrics(values, roundtrip);

        report.records.push_back({ rec, metrics });
        accumulate_metrics(sum_abs, sum_sq, count, rec.kind, metrics);
    }

    report.by_kind = make_aggregate_metrics(sum_abs, sum_sq, count);

    return report;
}

k_channel_sort_eval_report evaluate_manifest_k_channel_sort(
        const std::string & manifest_path,
        k_channel_sort_basis sort_basis,
        float global_scale) {
    k_channel_sort_eval_report report;
    report.global_scale = global_scale;
    report.sort_basis = sort_basis;
    const std::filesystem::path base_dir = manifest_dir(manifest_path);
    const std::vector<tensor_record> records = load_manifest_records(manifest_path);

    std::map<std::string, double> baseline_sum_abs;
    std::map<std::string, double> baseline_sum_sq;
    std::map<std::string, size_t> baseline_count;
    std::map<std::string, double> sorted_sum_abs;
    std::map<std::string, double> sorted_sum_sq;
    std::map<std::string, size_t> sorted_count;

    for (const tensor_record & rec : records) {
        if (rec.kind != "k") {
            throw std::runtime_error("K channel sort requires kind 'k', got kind '" + rec.kind + "' for record '" + rec.name + "'");
        }

        std::vector<float> values = load_record_f32(base_dir, rec);
        const size_t row_size = (size_t) rec.ne[0];
        if (row_size == 0 || values.size() % row_size != 0) {
            throw std::runtime_error("record '" + rec.name + "' has invalid K row layout");
        }

        std::vector<size_t> order;
        switch (sort_basis) {
            case k_channel_sort_basis::FIRST_ROW_ABS:
                order = make_k_channel_order_from_first_row(values, row_size);
                break;
            case k_channel_sort_basis::ABS_MEAN:
                order = make_k_channel_order_from_abs_mean(values, row_size);
                break;
        }

        const std::vector<float> baseline_roundtrip = nvfp4_roundtrip(values, global_scale);
        const tensor_error_metrics baseline_metrics = compute_error_metrics(values, baseline_roundtrip);

        const std::vector<float> sorted_values = apply_channel_order_by_row(values, row_size, order);
        const std::vector<float> sorted_roundtrip = nvfp4_roundtrip(sorted_values, global_scale);
        const std::vector<float> restored_roundtrip = restore_channel_order_by_row(sorted_roundtrip, row_size, order);
        const tensor_error_metrics sorted_metrics = compute_error_metrics(values, restored_roundtrip);

        k_channel_sort_eval_record_report rr;
        rr.record = rec;
        rr.baseline_metrics = baseline_metrics;
        rr.sorted_metrics = sorted_metrics;
        rr.delta_metrics = metric_delta(sorted_metrics, baseline_metrics);
        rr.channel_order = order;
        rr.sort_basis = k_channel_sort_basis_name(sort_basis);
        rr.channel_count = row_size;
        rr.row_count = values.size() / row_size;
        report.records.push_back(std::move(rr));

        accumulate_metrics(baseline_sum_abs, baseline_sum_sq, baseline_count, rec.kind, baseline_metrics);
        accumulate_metrics(sorted_sum_abs, sorted_sum_sq, sorted_count, rec.kind, sorted_metrics);
    }

    const std::map<std::string, tensor_error_metrics> baseline_by_kind =
        make_aggregate_metrics(baseline_sum_abs, baseline_sum_sq, baseline_count);
    const std::map<std::string, tensor_error_metrics> sorted_by_kind =
        make_aggregate_metrics(sorted_sum_abs, sorted_sum_sq, sorted_count);
    for (const auto & kv : baseline_by_kind) {
        k_channel_sort_eval_aggregate_report aggregate;
        aggregate.baseline_metrics = kv.second;
        aggregate.sorted_metrics = sorted_by_kind.at(kv.first);
        aggregate.delta_metrics = metric_delta(aggregate.sorted_metrics, aggregate.baseline_metrics);
        report.by_kind[kv.first] = aggregate;
    }

    return report;
}

std::string format_eval_report_json(const eval_report & report) {
    json root;
    root["algorithm"] = "nvfp4_ref";
    root["global_scale"] = report.global_scale;
    root["records"] = json::array();
    for (const eval_record_report & rr : report.records) {
        json item = record_to_json(rr.record);
        item["metrics"] = metrics_to_json(rr.metrics);
        root["records"].push_back(item);
    }

    root["aggregate_by_kind"] = json::object();
    for (const auto & kv : report.by_kind) {
        root["aggregate_by_kind"][kv.first] = metrics_to_json(kv.second);
    }
    return root.dump(2);
}

std::string format_k_channel_sort_eval_report_json(const k_channel_sort_eval_report & report) {
    json root;
    root["algorithm"] = k_channel_sort_algorithm_name(report.sort_basis);
    root["global_scale"] = report.global_scale;
    root["sort_basis"] = k_channel_sort_basis_name(report.sort_basis);
    root["records"] = json::array();
    for (const k_channel_sort_eval_record_report & rr : report.records) {
        json item = record_to_json(rr.record);
        item["channel_count"] = rr.channel_count;
        item["row_count"] = rr.row_count;
        item["sort_basis"] = rr.sort_basis;
        item["channel_order"] = rr.channel_order;
        item["baseline_metrics"] = metrics_to_json(rr.baseline_metrics);
        item["sorted_metrics"] = metrics_to_json(rr.sorted_metrics);
        item["delta_metrics"] = metrics_to_json(rr.delta_metrics);
        root["records"].push_back(item);
    }

    root["aggregate_by_kind"] = json::object();
    for (const auto & kv : report.by_kind) {
        root["aggregate_by_kind"][kv.first] = {
            { "baseline_metrics", metrics_to_json(kv.second.baseline_metrics) },
            { "sorted_metrics", metrics_to_json(kv.second.sorted_metrics) },
            { "delta_metrics", metrics_to_json(kv.second.delta_metrics) },
        };
    }
    return root.dump(2);
}

} // namespace llama_expt
