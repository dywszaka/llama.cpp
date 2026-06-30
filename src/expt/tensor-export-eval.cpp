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
