#include "../src/expt/tensor-export-eval.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static bool expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "%s\n", msg);
        return false;
    }
    return true;
}

static bool expect_close(double actual, double expected, double tol, const char * msg) {
    if (std::fabs(actual - expected) > tol) {
        std::fprintf(stderr, "%s: actual=%.17g expected=%.17g\n", msg, actual, expected);
        return false;
    }
    return true;
}

static std::string temp_dir() {
    const char * base = std::getenv("TMPDIR");
    if (!base) {
        base = "/tmp";
    }
    return std::string(base) + "/llama-expt-export-eval-test";
}

static void write_file(const std::string & path, const std::string & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open " + path);
    }
    out.write(data.data(), (std::streamsize) data.size());
}

static void write_f32(const std::string & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open " + path);
    }
    out.write(reinterpret_cast<const char *>(values.data()), (std::streamsize) (values.size() * sizeof(float)));
}

static bool test_metrics() {
    const std::vector<float> a = { 1.0f, 2.0f, 4.0f, -1.0f };
    const std::vector<float> b = { 0.0f, 2.0f, 1.0f,  1.0f };
    const llama_expt::tensor_error_metrics m = llama_expt::compute_error_metrics(a, b);
    return expect_close(m.mae, 1.5, 1e-12, "MAE mismatch") &&
           expect_close(m.mse, 3.5, 1e-12, "MSE mismatch") &&
           expect_close(m.rmse, std::sqrt(3.5), 1e-12, "RMSE mismatch") &&
           expect(m.n == 4, "metric count mismatch");
}

static bool test_export_dir_switch_enables_export() {
    const char * old_dir = std::getenv("LLAMA_EXPT_TENSOR_EXPORT_DIR");
    const std::string old_dir_value = old_dir ? old_dir : "";

    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", "/tmp/llama-expt-export-switch-test", 1);

    const bool enabled = llama_expt::tensor_export_enabled();

    if (old_dir) {
        setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", old_dir_value.c_str(), 1);
    } else {
        unsetenv("LLAMA_EXPT_TENSOR_EXPORT_DIR");
    }

    return expect(enabled, "expected LLAMA_EXPT_TENSOR_EXPORT_DIR to enable export");
}

static bool test_manifest_eval_and_rejection() {
    const std::string dir = temp_dir();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::vector<float> k_data(16);
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = (float) i - 8.0f;
    }

    write_f32(dir + "/k0.bin", k_data);
    write_file(dir + "/manifest.json",
            "{\n"
            "  \"records\": [\n"
            "    {\"name\":\"k-0\",\"kind\":\"k\",\"dtype\":\"f32\",\"ne\":[16,1,1,1],\"nb\":[4,64,64,64],\"path\":\"k0.bin\",\"byte_size\":64}\n"
            "  ]\n"
            "}\n");

    llama_expt::eval_report report = llama_expt::evaluate_manifest(dir + "/manifest.json");
    if (!expect(report.records.size() == 1, "expected one record report")) {
        return false;
    }
    if (!expect(report.by_kind.count("k") == 1, "expected aggregate for kind k")) {
        return false;
    }
    if (!expect(report.records[0].metrics.n == k_data.size(), "expected metric count to match values")) {
        return false;
    }
    if (!expect(report.records[0].metrics.mae >= 0.0, "MAE should be non-negative")) {
        return false;
    }

    const llama_expt::eval_record_report first_report = report.records[0];
    const std::vector<float> v_data = {
        -2.0f, -1.75f, -1.5f, -1.25f, -1.0f, -0.75f, -0.5f, -0.25f,
         0.0f,  0.25f,  0.5f,  0.75f,  1.0f,  1.25f,  1.5f,  1.75f,
    };
    write_f32(dir + "/v0.bin", v_data);
    write_file(dir + "/manifest-two-kinds.json",
            "{\n"
            "  \"records\": [\n"
            "    {\"name\":\"k-0\",\"kind\":\"k\",\"dtype\":\"f32\",\"ne\":[16,1,1,1],\"nb\":[4,64,64,64],\"path\":\"k0.bin\",\"byte_size\":64},\n"
            "    {\"name\":\"v-0\",\"kind\":\"v\",\"dtype\":\"f32\",\"ne\":[16,1,1,1],\"nb\":[4,64,64,64],\"path\":\"v0.bin\",\"byte_size\":64}\n"
            "  ]\n"
            "}\n");
    const llama_expt::eval_report two_kind_report = llama_expt::evaluate_manifest(dir + "/manifest-two-kinds.json");
    if (!expect(two_kind_report.records.size() == 2, "expected two record reports")) {
        return false;
    }
    if (!expect(two_kind_report.by_kind.count("k") == 1, "expected aggregate for kind k in two-kind report")) {
        return false;
    }
    if (!expect(two_kind_report.by_kind.count("v") == 1, "expected aggregate for kind v")) {
        return false;
    }
    if (!expect_close(two_kind_report.by_kind.at("k").mae, first_report.metrics.mae, 1e-12,
            "single-record aggregate MAE should match record MAE")) {
        return false;
    }

    write_file(dir + "/manifest-bad-dtype.json",
            "{ \"records\": ["
            "{\"name\":\"bad\",\"kind\":\"q\",\"dtype\":\"f16\",\"ne\":[16,1,1,1],\"nb\":[2,32,32,32],\"path\":\"k0.bin\",\"byte_size\":32}"
            "] }\n");
    bool rejected_dtype = false;
    try {
        (void) llama_expt::evaluate_manifest(dir + "/manifest-bad-dtype.json");
    } catch (const std::exception & e) {
        rejected_dtype = std::string(e.what()).find("dtype") != std::string::npos;
    }
    if (!expect(rejected_dtype, "expected dtype rejection")) {
        return false;
    }

    write_file(dir + "/manifest-bad-size.json",
            "{ \"records\": ["
            "{\"name\":\"bad\",\"kind\":\"q\",\"dtype\":\"f32\",\"ne\":[16,1,1,1],\"nb\":[4,64,64,64],\"path\":\"k0.bin\",\"byte_size\":4}"
            "] }\n");
    bool rejected_size = false;
    try {
        (void) llama_expt::evaluate_manifest(dir + "/manifest-bad-size.json");
    } catch (const std::exception & e) {
        rejected_size = std::string(e.what()).find("byte_size") != std::string::npos;
    }
    if (!expect(rejected_size, "expected byte_size rejection")) {
        return false;
    }

    write_file(dir + "/manifest-bad-shape.json",
            "{ \"records\": ["
            "{\"name\":\"bad\",\"kind\":\"q\",\"dtype\":\"f32\",\"ne\":[15,1,1,1],\"nb\":[4,60,60,60],\"path\":\"k0.bin\",\"byte_size\":60}"
            "] }\n");
    bool rejected_shape = false;
    try {
        (void) llama_expt::evaluate_manifest(dir + "/manifest-bad-shape.json");
    } catch (const std::exception & e) {
        rejected_shape = std::string(e.what()).find("NVFP4") != std::string::npos;
    }
    if (!expect(rejected_shape, "expected NVFP4 shape rejection")) {
        return false;
    }

    write_file(dir + "/manifest-bad-row-shape.json",
            "{ \"records\": ["
            "{\"name\":\"bad-row\",\"kind\":\"q\",\"dtype\":\"f32\",\"ne\":[8,2,1,1],\"nb\":[4,32,64,64],\"path\":\"k0.bin\",\"byte_size\":64}"
            "] }\n");
    bool rejected_row_shape = false;
    try {
        (void) llama_expt::evaluate_manifest(dir + "/manifest-bad-row-shape.json");
    } catch (const std::exception & e) {
        rejected_row_shape = std::string(e.what()).find("NVFP4") != std::string::npos;
    }
    if (!expect(rejected_row_shape, "expected NVFP4 row-shape rejection")) {
        return false;
    }

    return true;
}

static bool test_k_channel_mean_sort_order_uses_abs_mean_desc() {
    const std::vector<float> values = {
        // channel means over 2 rows:
        // c0= 0.0, c1=-3.0, c2= 3.0, c3= 2.0, c4=-2.0, c5=1.0, c6=-1.0, c7=0.0,
        // c8= 0.5, c9=-0.5, c10=0.25, c11=-0.25, c12=0.0, c13=0.0, c14=0.0, c15=0.0
         1.0f, -4.0f,  2.0f,  4.0f, -3.0f,  0.0f, -2.0f,  1.0f,
         0.0f, -1.0f,  0.5f, -0.5f,  3.0f, -3.0f,  0.0f,  0.0f,
        -1.0f, -2.0f,  4.0f,  0.0f, -1.0f,  2.0f,  0.0f, -1.0f,
         1.0f,  0.0f,  0.0f,  0.0f, -3.0f,  3.0f,  0.0f,  0.0f,
    };

    const std::vector<size_t> order = llama_expt::make_k_channel_order_from_abs_mean(values, 16);
    const std::vector<size_t> expected = {
        1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 0, 7, 12, 13, 14, 15,
    };
    if (!expect(order == expected, "expected abs-mean-desc channel order with index tie-break")) {
        return false;
    }

    bool rejected = false;
    try {
        (void) llama_expt::make_k_channel_order_from_abs_mean(values, 0);
    } catch (const std::exception & e) {
        rejected = std::string(e.what()).find("row") != std::string::npos;
    }
    return expect(rejected, "expected invalid row size rejection");
}

static bool test_k_channel_mean_sort_manifest_eval_reports_basis_and_deltas() {
    const std::string dir = temp_dir();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::vector<float> k_data(32);
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = ((int) i - 15) * 0.1875f;
    }
    k_data[1]  = -7.0f;
    k_data[2]  =  9.0f;
    k_data[17] = -5.0f;
    k_data[18] =  7.0f;

    write_f32(dir + "/k-mean-sort.bin", k_data);
    write_file(dir + "/manifest-k-mean-sort.json",
            "{\n"
            "  \"records\": [\n"
            "    {\"name\":\"Kcur-0\",\"kind\":\"k\",\"dtype\":\"f32\",\"ne\":[16,2,1,1],\"nb\":[4,64,128,128],\"path\":\"k-mean-sort.bin\",\"byte_size\":128}\n"
            "  ]\n"
            "}\n");

    const llama_expt::k_channel_sort_eval_report report =
        llama_expt::evaluate_manifest_k_channel_sort(
                dir + "/manifest-k-mean-sort.json",
                llama_expt::k_channel_sort_basis::ABS_MEAN,
                1.0f);
    if (!expect(report.records.size() == 1, "expected one mean-sort record")) {
        return false;
    }

    const llama_expt::k_channel_sort_eval_record_report & rr = report.records[0];
    if (!expect(rr.channel_count == 16, "expected channel count from ne0")) {
        return false;
    }
    if (!expect(rr.row_count == 2, "expected row count from remaining dimensions")) {
        return false;
    }
    if (!expect(rr.sort_basis == "abs_mean", "expected abs_mean sort basis")) {
        return false;
    }
    if (!expect(rr.channel_order[0] == 2 && rr.channel_order[1] == 1,
            "expected mean order prefix from all rows")) {
        return false;
    }
    if (!expect_close(rr.delta_metrics.mae, rr.sorted_metrics.mae - rr.baseline_metrics.mae, 1e-12,
            "mean-sort MAE delta mismatch")) {
        return false;
    }

    const std::string json = llama_expt::format_k_channel_sort_eval_report_json(report);
    if (!expect(json.find("\"algorithm\": \"nvfp4_k_channel_mean_sort\"") != std::string::npos,
            "expected mean-sort algorithm in JSON")) {
        return false;
    }
    if (!expect(json.find("\"sort_basis\": \"abs_mean\"") != std::string::npos,
            "expected abs_mean sort basis in JSON")) {
        return false;
    }
    if (!expect(json.find("\"channel_order\"") != std::string::npos,
            "expected channel order in JSON")) {
        return false;
    }

    return true;
}

int main() {
    if (!test_metrics()) {
        return 1;
    }
    if (!test_export_dir_switch_enables_export()) {
        return 1;
    }
    if (!test_manifest_eval_and_rejection()) {
        return 1;
    }
    if (!test_k_channel_mean_sort_order_uses_abs_mean_desc()) {
        return 1;
    }
    if (!test_k_channel_mean_sort_manifest_eval_reports_basis_and_deltas()) {
        return 1;
    }

    std::puts("test-expt-tensor-export-eval: ok");
    return 0;
}
