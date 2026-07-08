#include "tensor-export-eval.h"

#include "llama-impl.h"

#include "../../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "../../vendor/nlohmann/json.hpp"

namespace llama_expt {
namespace {

using json = nlohmann::ordered_json;

constexpr const char * ENV_DIR    = "LLAMA_EXPT_TENSOR_EXPORT_DIR";
constexpr const char * ENV_KINDS  = "LLAMA_EXPT_TENSOR_EXPORT_KINDS";

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

std::string json_scalar_to_string(const json & value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? "true" : "false";
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<long long>());
    }
    if (value.is_number_unsigned()) {
        return std::to_string(value.get<unsigned long long>());
    }
    if (value.is_number_float()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.9g", value.get<double>());
        return buf;
    }
    return value.dump();
}

bool has_prefix(const std::string & text, const char * prefix) {
    const size_t n = std::strlen(prefix);
    return text.size() >= n && text.compare(0, n, prefix) == 0;
}

std::string tensor_kind(const char * raw_name) {
    std::string name = raw_name ? raw_name : "";
    const size_t dash = name.find('-');
    const std::string base = dash == std::string::npos ? name : name.substr(0, dash);

    if (base == "kq_softmax" || has_prefix(name, "kq-softmax-")) {
        return "kq_softmax";
    }
    if (base == "kq_mask" || has_prefix(name, "kq-mask-")) {
        return "kq_mask";
    }
    if (base == "k_attn" || has_prefix(name, "k-attn-")) {
        return "k_attn";
    }
    if (base == "q_attn" || has_prefix(name, "q-attn-")) {
        return "q_attn";
    }
    if (name.find("cache_k_l") != std::string::npos) {
        return "k_attn";
    }
    if (name.find("(permuted)") != std::string::npos && has_prefix(name, "Qcur-")) {
        return "q_attn";
    }
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
        return { "k", "q", "v", "kq", "kqv", "kq_softmax", "kq_mask", "k_attn", "q_attn" };
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
    if (!rec.meta.empty()) {
        obj["meta"] = json::object();
        for (const auto & kv : rec.meta) {
            obj["meta"][kv.first] = kv.second;
        }
    }
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
    if (obj.contains("meta")) {
        const auto & meta = obj.at("meta");
        if (!meta.is_object()) {
            throw std::runtime_error("manifest record '" + rec.name + "' meta must be an object");
        }
        for (auto it = meta.begin(); it != meta.end(); ++it) {
            rec.meta[it.key()] = json_scalar_to_string(it.value());
        }
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

std::vector<float> load_record_f32(const std::filesystem::path & base_dir, const tensor_record & rec, bool require_nvfp4_row_shape = true) {
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
    if (require_nvfp4_row_shape && rec.ne[0] % QK_NVFP4 != 0) {
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

int64_t tensor_record_nelements(const tensor_record & rec) {
    int64_t n = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (rec.ne[i] <= 0) {
            throw std::runtime_error("record '" + rec.name + "' has invalid shape");
        }
        n *= rec.ne[i];
    }
    return n;
}

double compute_max_abs_err(const std::vector<float> & reference, const std::vector<float> & actual) {
    if (reference.size() != actual.size()) {
        throw std::runtime_error("max_abs_err input size mismatch");
    }

    double out = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        out = std::max(out, std::fabs((double) actual[i] - (double) reference[i]));
    }
    return out;
}

bool tensor_name_is_layer0_attention(const char * name, const char * prefix) {
    if (!name || !prefix) {
        return false;
    }
    const size_t prefix_len = std::strlen(prefix);
    return std::strncmp(name, prefix, prefix_len) == 0 && std::strcmp(name + prefix_len, "0") == 0;
}

bool tensor_name_is_softmax_prob(const char * name) {
    return tensor_name_is_layer0_attention(name, "kq-softmax-");
}

bool tensor_name_is_presoftmax_kq(const char * name) {
    return tensor_name_is_layer0_attention(name, "kq-");
}

bool tensor_name_is_layer0_q(const char * name) {
    return tensor_name_is_layer0_attention(name, "Qcur-");
}

bool tensor_name_is_layer0_k(const char * name) {
    return tensor_name_is_layer0_attention(name, "Kcur-");
}

bool tensor_name_is_layer0_k_mask(const char * name) {
    return tensor_name_is_layer0_attention(name, "kq-mask-");
}

bool parse_meta_f32(const tensor_record & rec, const char * key, float & out) {
    const auto it = rec.meta.find(key);
    if (it == rec.meta.end()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const float value = std::strtof(it->second.c_str(), &end);
    if (errno != 0 || end == it->second.c_str() || (end && *end != '\0')) {
        throw std::runtime_error("record '" + rec.name + "' has invalid float meta '" + key + "'");
    }
    out = value;
    return true;
}

std::string require_meta_str(const tensor_record & rec, const char * key) {
    const auto it = rec.meta.find(key);
    if (it == rec.meta.end() || it->second.empty()) {
        throw std::runtime_error("record '" + rec.name + "' is missing meta '" + key + "'");
    }
    return it->second;
}

void replay_attention_scores_and_probs(
        const std::vector<float> & k_values,
        const tensor_record & k_record,
        const std::vector<float> & q_values,
        const tensor_record & q_record,
        const std::vector<float> & mask_values,
        const tensor_record & mask_record,
        float kq_scale,
        float max_bias,
        std::vector<float> & out_kq,
        std::vector<float> & out_softmax) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 64u * 1024u * 1024u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    std::unique_ptr<ggml_context, void (*)(ggml_context *)> ctx(ggml_init(params), ggml_free);
    if (!ctx) {
        throw std::runtime_error("failed to initialize ggml replay context");
    }

    ggml_tensor * k_base = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16,
            k_record.ne[0], k_record.ne[1], k_record.ne[2], k_record.ne[3]);
    ggml_tensor * q_base = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
            q_record.ne[0], q_record.ne[1], q_record.ne[2], q_record.ne[3]);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32,
            mask_record.ne[0], mask_record.ne[1], mask_record.ne[2], mask_record.ne[3]);

    ggml_tensor * k = k_base;
    ggml_tensor * q = q_base;
    const int64_t n_stream = k->ne[3];
    q = ggml_reshape_4d(ctx.get(), q, q->ne[0], q->ne[1], q->ne[2] / n_stream, n_stream);
    q = ggml_permute(ctx.get(), q, 0, 2, 1, 3);
    k = ggml_permute(ctx.get(), k, 0, 2, 1, 3);

    ggml_tensor * kq = ggml_mul_mat(ctx.get(), k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * probs = ggml_soft_max_ext(ctx.get(), kq, mask, kq_scale, max_bias);

    ggml_cgraph * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, kq);
    ggml_build_forward_expand(gf, probs);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize ggml CPU backend for attention replay");
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
    if (buf == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to allocate ggml replay tensors");
    }

    std::vector<ggml_fp16_t> k_values_f16(k_values.size());
    ggml_fp32_to_fp16_row(k_values.data(), k_values_f16.data(), (int64_t) k_values.size());

    ggml_backend_tensor_set(k_base, k_values_f16.data(), 0, k_values_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(q_base, q_values.data(), 0, q_values.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_values.data(), 0, mask_values.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        throw std::runtime_error("ggml attention replay graph compute failed");
    }

    out_kq.resize((size_t) ggml_nelements(kq));
    out_softmax.resize((size_t) ggml_nelements(probs));
    ggml_backend_tensor_get(kq, out_kq.data(), 0, out_kq.size() * sizeof(float));
    ggml_backend_tensor_get(probs, out_softmax.data(), 0, out_softmax.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
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

void maybe_fill_attention_record_meta(ggml_tensor * t, tensor_record & rec) {
    if (!t || t->op != GGML_OP_SOFT_MAX || rec.kind != "kq_softmax") {
        return;
    }
    const char * name = ggml_get_name(t);
    if (!tensor_name_is_softmax_prob(name)) {
        return;
    }

    if (!t->src[0] || !t->src[1]) {
        return;
    }

    float kq_scale = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&kq_scale, (const float *) t->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) t->op_params + 1, sizeof(float));

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", (double) kq_scale);
    rec.meta["kq_scale"] = buf;
    std::snprintf(buf, sizeof(buf), "%.9g", (double) max_bias);
    rec.meta["max_bias"] = buf;
    rec.meta["src_kq"] = ggml_get_name(t->src[0]) ? ggml_get_name(t->src[0]) : "";
    rec.meta["src_mask"] = tensor_name_is_softmax_prob(name)
            ? "kq-mask-0"
            : (ggml_get_name(t->src[1]) ? ggml_get_name(t->src[1]) : "");
    rec.meta["src_k"] = tensor_name_is_softmax_prob(name) ? "k-attn-0" : "";
    rec.meta["src_q"] = tensor_name_is_softmax_prob(name) ? "q-attn-0" : "";
    if (rec.meta["src_k"].empty() && t->src[0]->src[0]) {
        rec.meta["src_k"] = ggml_get_name(t->src[0]->src[0]) ? ggml_get_name(t->src[0]->src[0]) : "";
    }
    if (rec.meta["src_q"].empty() && t->src[0]->src[1]) {
        rec.meta["src_q"] = ggml_get_name(t->src[0]->src[1]) ? ggml_get_name(t->src[0]->src[1]) : "";
    }
}

std::string export_record_name_for_tensor(const ggml_tensor * t) {
    const char * name = ggml_get_name(t);
    if (t != nullptr && t->op == GGML_OP_NONE && t->type == GGML_TYPE_F32 && name && std::strstr(name, "kq-mask-") != nullptr) {
        return name;
    }
    if (t != nullptr && name && std::strstr(name, "kq-mask-") != nullptr) {
        return name;
    }
    return name ? name : "";
}

} // namespace

bool tensor_export_enabled() {
    return !env_str(ENV_DIR).empty();
}

bool tensor_export_maybe_log_config() {
    static bool logged = false;
    if (logged) {
        return tensor_export_enabled();
    }
    logged = true;

    const bool enabled = tensor_export_enabled();
    if (enabled) {
        LLAMA_LOG_INFO("%s: enabled %s='%s' %s='%s'\n",
                __func__, ENV_DIR, env_str(ENV_DIR).c_str(), ENV_KINDS, env_str(ENV_KINDS).c_str());
    } else {
        LLAMA_LOG_INFO("%s: disabled; %s is unset\n", __func__, ENV_DIR);
    }
    return enabled;
}

void tensor_export_pin_named_tensor(ggml_tensor * tensor) {
    if (!tensor_export_enabled() || tensor == nullptr) {
        return;
    }

    const char * name = ggml_get_name(tensor);
    if (tensor_name_is_softmax_prob(name) ||
            tensor_name_is_presoftmax_kq(name) ||
            tensor_name_is_layer0_q(name) ||
            tensor_name_is_layer0_k(name) ||
            (name && std::strcmp(name, "k-attn-0") == 0) ||
            (name && std::strcmp(name, "q-attn-0") == 0)) {
        ggml_set_output(tensor);
    }
}

bool tensor_export_graph(ggml_backend_sched_t sched, ggml_cgraph * gf) {
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

    const auto kinds = selected_kinds();
    std::vector<tensor_record> records;
    std::unordered_set<const ggml_tensor *> seen;
    struct extra_export_tensor {
        ggml_tensor * tensor;
        std::string forced_name;
    };
    std::vector<extra_export_tensor> extra_tensors;

    ggml_backend_sched_synchronize(sched);

    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (!t || !seen.insert(t).second) {
            continue;
        }

        if (t->op == GGML_OP_SOFT_MAX && tensor_name_is_softmax_prob(ggml_get_name(t)) && t->src[1] != nullptr) {
            extra_tensors.push_back({ t->src[1], "kq-mask-0" });
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
        maybe_fill_attention_record_meta(t, rec);

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

    for (const extra_export_tensor & extra : extra_tensors) {
        ggml_tensor * t = extra.tensor;
        if (!t || !seen.insert(t).second) {
            continue;
        }
        const std::string export_name = extra.forced_name.empty() ? ggml_get_name(t) : extra.forced_name;
        const std::string kind = tensor_kind(export_name.c_str());
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
        rec.name = export_name;
        rec.kind = kind;
        rec.dtype = "f32";
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            rec.ne[d] = t->ne[d];
            rec.nb[d] = t->nb[d];
        }
        rec.byte_size = byte_size;
        rec.path = std::to_string(records.size()) + "-" + sanitize_filename(rec.name) + ".bin";
        maybe_fill_attention_record_meta(t, rec);

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

attention_replay_eval_report evaluate_manifest_attention_replay(const std::string & manifest_path) {
    attention_replay_eval_report report;
    const std::filesystem::path base_dir = manifest_dir(manifest_path);
    const std::vector<tensor_record> records = load_manifest_records(manifest_path);

    std::map<std::string, const tensor_record *> by_name;
    for (const tensor_record & rec : records) {
        by_name[rec.name] = &rec;
    }

    for (const tensor_record & rec : records) {
        if (rec.kind != "kq_softmax" || !tensor_name_is_softmax_prob(rec.name.c_str())) {
            continue;
        }

        const std::string k_name = require_meta_str(rec, "src_k");
        const std::string q_name = require_meta_str(rec, "src_q");
        const std::string kq_name = require_meta_str(rec, "src_kq");
        const std::string mask_name = require_meta_str(rec, "src_mask");

        if (by_name.count(k_name) == 0 || by_name.count(q_name) == 0 || by_name.count(kq_name) == 0 || by_name.count(mask_name) == 0) {
            throw std::runtime_error("attention replay inputs are missing from manifest for '" + rec.name + "'");
        }

        const tensor_record & k_rec = *by_name.at(k_name);
        const tensor_record & q_rec = *by_name.at(q_name);
        const tensor_record & kq_rec = *by_name.at(kq_name);
        const tensor_record & mask_rec = *by_name.at(mask_name);

        float kq_scale = 1.0f;
        float max_bias = 0.0f;
        (void) parse_meta_f32(rec, "kq_scale", kq_scale);
        (void) parse_meta_f32(rec, "max_bias", max_bias);

        const std::vector<float> k_values = load_record_f32(base_dir, k_rec, false);
        const std::vector<float> q_values = load_record_f32(base_dir, q_rec, false);
        const std::vector<float> kq_values = load_record_f32(base_dir, kq_rec, false);
        const std::vector<float> softmax_values = load_record_f32(base_dir, rec, false);
        const std::vector<float> mask_values = load_record_f32(base_dir, mask_rec, false);

        std::vector<float> replay_kq;
        std::vector<float> replay_softmax;
        replay_attention_scores_and_probs(
                k_values, k_rec,
                q_values, q_rec,
                mask_values, mask_rec,
                kq_scale, max_bias,
                replay_kq, replay_softmax);

        attention_replay_report rr;
        rr.k_record = k_rec;
        rr.q_record = q_rec;
        rr.kq_record = kq_rec;
        rr.softmax_record = rec;
        rr.kq_metrics = compute_error_metrics(kq_values, replay_kq);
        rr.softmax_metrics = compute_error_metrics(softmax_values, replay_softmax);
        rr.max_abs_err_kq = compute_max_abs_err(kq_values, replay_kq);
        rr.max_abs_err_softmax = compute_max_abs_err(softmax_values, replay_softmax);
        rr.kq_scale = kq_scale;
        rr.max_bias = max_bias;
        report.records.push_back(std::move(rr));
    }

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

std::string format_attention_replay_eval_report_json(const attention_replay_eval_report & report) {
    json root;
    root["algorithm"] = "attention_replay";
    root["records"] = json::array();
    for (const attention_replay_report & rr : report.records) {
        json item;
        item["k_record"] = record_to_json(rr.k_record);
        item["q_record"] = record_to_json(rr.q_record);
        item["kq_record"] = record_to_json(rr.kq_record);
        item["softmax_record"] = record_to_json(rr.softmax_record);
        item["kq_scale"] = rr.kq_scale;
        item["max_bias"] = rr.max_bias;
        item["kq_metrics"] = metrics_to_json(rr.kq_metrics);
        item["softmax_metrics"] = metrics_to_json(rr.softmax_metrics);
        item["max_abs_err_kq"] = rr.max_abs_err_kq;
        item["max_abs_err_softmax"] = rr.max_abs_err_softmax;
        root["records"].push_back(std::move(item));
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
