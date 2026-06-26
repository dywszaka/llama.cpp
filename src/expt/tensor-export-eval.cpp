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
        sum_abs[rec.kind] += metrics.mae * (double) metrics.n;
        sum_sq[rec.kind]  += metrics.mse * (double) metrics.n;
        count[rec.kind]   += metrics.n;
    }

    for (const auto & kv : count) {
        tensor_error_metrics metrics;
        metrics.n = kv.second;
        metrics.mae = sum_abs[kv.first] / (double) metrics.n;
        metrics.mse = sum_sq[kv.first] / (double) metrics.n;
        metrics.rmse = std::sqrt(metrics.mse);
        report.by_kind[kv.first] = metrics;
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
        item["metrics"] = {
            { "mae", rr.metrics.mae },
            { "mse", rr.metrics.mse },
            { "rmse", rr.metrics.rmse },
            { "n", rr.metrics.n },
        };
        root["records"].push_back(item);
    }

    root["aggregate_by_kind"] = json::object();
    for (const auto & kv : report.by_kind) {
        root["aggregate_by_kind"][kv.first] = {
            { "mae", kv.second.mae },
            { "mse", kv.second.mse },
            { "rmse", kv.second.rmse },
            { "n", kv.second.n },
        };
    }
    return root.dump(2);
}

} // namespace llama_expt
