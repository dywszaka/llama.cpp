#include "../src/expt/tensor-export-eval.h"

#include <ggml-cpu.h>

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

struct eval_callback_probe {
    ggml_tensor * selected = nullptr;
    int asks = 0;
    int observations = 0;
};

static bool eval_callback_for_probe(ggml_tensor * tensor, bool ask, void * user_data) {
    eval_callback_probe * probe = static_cast<eval_callback_probe *>(user_data);
    if (ask) {
        ++probe->asks;
        return tensor == probe->selected;
    }
    ++probe->observations;
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

static bool test_op_export_retains_selected_node_storage() {
    const char * env_names[] = {
        "LLAMA_EXPT_TENSOR_EXPORT_DIR",
        "LLAMA_EXPT_TENSOR_EXPORT_KINDS",
        "LLAMA_EXPT_TENSOR_EXPORT_NAME",
        "LLAMA_EXPT_TENSOR_EXPORT_OP",
        "LLAMA_EXPT_TENSOR_EXPORT_TYPE",
        "LLAMA_EXPT_TENSOR_EXPORT_LAYER",
    };
    std::vector<std::string> old_values;
    std::vector<bool> old_present;
    for (const char * env_name : env_names) {
        const char * value = std::getenv(env_name);
        old_present.push_back(value != nullptr);
        old_values.emplace_back(value ? value : "");
    }

    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", "/tmp/llama-expt-export-retain-test", 1);
    unsetenv("LLAMA_EXPT_TENSOR_EXPORT_KINDS");
    unsetenv("LLAMA_EXPT_TENSOR_EXPORT_NAME");
    setenv("LLAMA_EXPT_TENSOR_EXPORT_OP", "RMS_NORM", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_TYPE", "decode", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_LAYER", "0", 1);

    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!expect(ctx != nullptr, "expected ggml context for graph retention test")) {
        return false;
    }

    ggml_tensor * input0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_tensor * input1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_set_name(input0, "inp_embd-0");
    ggml_set_name(input1, "inp_embd-1");
    ggml_tensor * norm0 = ggml_rms_norm(ctx, input0, 1.0e-6f);
    ggml_tensor * norm1 = ggml_rms_norm(ctx, input1, 1.0e-6f);
    ggml_set_name(norm0, "norm-0");
    ggml_set_name(norm1, "norm-1");
    ggml_tensor * output = ggml_add(ctx, norm0, norm1);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(gf, output);
    const bool retained = llama_expt::tensor_export_maybe_retain_graph(gf);

    eval_callback_probe probe;
    probe.selected = norm1;
    llama_expt::tensor_export_observer * observer =
        llama_expt::tensor_export_observer_create(gf, false, eval_callback_for_probe, &probe);
    llama_expt::tensor_export_observer * prefill_observer =
        llama_expt::tensor_export_observer_create(gf, true, nullptr, nullptr);

    const bool ok = expect(retained, "expected selected RMS_NORM storage retention") &&
        expect((norm0->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,
                "selected RMS_NORM dst must be retained as a graph output") &&
        expect((input0->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,
                "selected RMS_NORM src0 must be retained as a graph output") &&
        expect((norm1->flags & GGML_TENSOR_FLAG_OUTPUT) == 0,
                "unselected RMS_NORM dst must remain reusable") &&
        expect((input1->flags & GGML_TENSOR_FLAG_OUTPUT) == 0,
                "unselected RMS_NORM src0 must remain reusable") &&
        expect(observer != nullptr, "expected decode observer for selected RMS_NORM") &&
        expect(prefill_observer == nullptr, "decode selection must not observe prefill execution") &&
        expect(llama_expt::tensor_export_observer_callback(norm0, true, observer),
                "export observer must request the selected RMS_NORM node") &&
        expect(llama_expt::tensor_export_observer_callback(norm1, true, observer),
                "export observer must preserve a user callback request") &&
        expect(llama_expt::tensor_export_observer_callback(norm1, false, observer),
                "export observer must preserve the user callback result") &&
        expect(probe.asks == 2 && probe.observations == 1,
                "export observer must chain user callback ask/observe calls");

    llama_expt::tensor_export_observer_free(prefill_observer);
    llama_expt::tensor_export_observer_free(observer);
    ggml_free(ctx);
    for (size_t i = 0; i < old_values.size(); ++i) {
        if (old_present[i]) {
            setenv(env_names[i], old_values[i].c_str(), 1);
        } else {
            unsetenv(env_names[i]);
        }
    }
    return ok;
}

static bool test_tensor_name_priority_binds_nvfp4_capture() {
    const char * env_names[] = {
        "LLAMA_EXPT_TENSOR_EXPORT_DIR",
        "LLAMA_EXPT_TENSOR_EXPORT_NAME",
        "LLAMA_EXPT_TENSOR_EXPORT_OP",
        "LLAMA_EXPT_TENSOR_EXPORT_TYPE",
        "LLAMA_EXPT_TENSOR_EXPORT_LAYER",
    };
    std::vector<std::string> old_values;
    std::vector<bool> old_present;
    for (const char * env_name : env_names) {
        const char * value = std::getenv(env_name);
        old_present.push_back(value != nullptr);
        old_values.emplace_back(value ? value : "");
    }

    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", "/tmp/llama-expt-export-name-test", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_NAME", "kq", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_OP", "RMS_NORM", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_TYPE", "decode", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_LAYER", "0", 1);

    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!expect(ctx != nullptr, "expected ggml context for tensor-name capture test")) {
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, 16, 16);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 2);
    ggml_tensor * dst = ggml_mul_mat(ctx, a, b);
    ggml_set_name(dst, "kq-0");

    const bool bound = llama_expt::tensor_export_maybe_bind_nvfp4_mul_mat_capture(ctx, dst, false);
    const ggml_tensor * rhs_capture = ggml_mul_mat_get_nvfp4_rhs_capture(dst);
    const ggml_tensor * scale_capture = ggml_mul_mat_get_nvfp4_rhs_global_scale_capture(dst);
    const uint32_t flags = ggml_mul_mat_get_nvfp4_capture_flags(dst);

    ggml_tensor * other = ggml_mul_mat(ctx, a, b);
    ggml_set_name(other, "kq-1");
    const bool other_bound = llama_expt::tensor_export_maybe_bind_nvfp4_mul_mat_capture(ctx, other, false);

    bool ok = expect(bound, "tensor name should override mismatched op selection") &&
        expect(rhs_capture != nullptr && rhs_capture->type == GGML_TYPE_NVFP4,
                "expected NVFP4 RHS capture tensor") &&
        expect(rhs_capture->ne[0] == 16 && rhs_capture->ne[1] == 2,
                "RHS capture shape mismatch") &&
        expect(scale_capture != nullptr && scale_capture->type == GGML_TYPE_F32,
                "expected RHS scale capture tensor") &&
        expect(scale_capture->ne[0] == 2 && ggml_nelements(scale_capture) == 2,
                "dynamic RHS scale capture shape mismatch") &&
        expect((rhs_capture->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 &&
               (scale_capture->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,
                "capture tensors must be retained as graph outputs") &&
        expect((flags & GGML_NVFP4_MUL_MAT_CAPTURE_REQUESTED) != 0 &&
               (flags & GGML_NVFP4_MUL_MAT_CAPTURE_VALID) == 0,
                "capture flags should start requested but not valid") &&
        expect(!other_bound && ggml_mul_mat_get_nvfp4_rhs_capture(other) == nullptr,
                "layer-qualified tensor name should not bind kq-1");

    ggml_free(ctx);
    for (size_t i = 0; i < old_values.size(); ++i) {
        if (old_present[i]) {
            setenv(env_names[i], old_values[i].c_str(), 1);
        } else {
            unsetenv(env_names[i]);
        }
    }
    return ok;
}

static bool test_fp4mulmat_export_writes_only_final_scale() {
    const char * env_names[] = {
        "LLAMA_EXPT_TENSOR_EXPORT_DIR",
        "LLAMA_EXPT_TENSOR_EXPORT_NAME",
        "LLAMA_EXPT_TENSOR_EXPORT_OP",
        "LLAMA_EXPT_TENSOR_EXPORT_TYPE",
        "LLAMA_EXPT_TENSOR_EXPORT_LAYER",
    };
    std::vector<std::string> old_values;
    std::vector<bool> old_present;
    for (const char * env_name : env_names) {
        const char * value = std::getenv(env_name);
        old_present.push_back(value != nullptr);
        old_values.emplace_back(value ? value : "");
    }

    const std::string dir = temp_dir() + "-fp4mulmat-final-scale";
    std::filesystem::remove_all(dir);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", dir.c_str(), 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_NAME", "fp4mulmat-export-0", 1);
    unsetenv("LLAMA_EXPT_TENSOR_EXPORT_OP");
    setenv("LLAMA_EXPT_TENSOR_EXPORT_TYPE", "decode", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_LAYER", "0", 1);

    ggml_init_params params = {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!expect(ctx != nullptr, "expected ggml context for FP4MULMAT export test")) {
        return false;
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, 16, 16);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 2);
    ggml_tensor * dst = ggml_mul_mat(ctx, a, b);
    ggml_set_name(dst, "fp4mulmat-export-0");
    ggml_tensor * rhs_capture = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, 16, 2);
    ggml_tensor * final_scale_capture = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_set_output(rhs_capture);
    ggml_set_output(final_scale_capture);
    ggml_mul_mat_set_nvfp4_rhs_capture(dst, rhs_capture, final_scale_capture);
    ggml_mul_mat_set_nvfp4_capture_flags(
            dst,
            GGML_NVFP4_MUL_MAT_CAPTURE_REQUESTED |
            GGML_NVFP4_MUL_MAT_CAPTURE_VALID |
            GGML_NVFP4_MUL_MAT_CAPTURE_DYNAMIC |
            GGML_NVFP4_MUL_MAT_CAPTURE_FP4MULMAT |
            GGML_NVFP4_MUL_MAT_CAPTURE_FINAL_SCALE);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 16, false, true);
    bool ok = expect(backend != nullptr && sched != nullptr, "expected CPU scheduler for FP4MULMAT export test") &&
            expect(ggml_backend_sched_alloc_graph(sched, gf), "expected graph allocation for FP4MULMAT export test");

    if (ok) {
        std::vector<uint8_t> a_data(ggml_nbytes(a), 0);
        std::vector<float> b_data((size_t) ggml_nelements(b), 0.0f);
        std::vector<float> dst_data((size_t) ggml_nelements(dst), 0.0f);
        std::vector<uint8_t> rhs_data(ggml_nbytes(rhs_capture), 0);
        const std::vector<float> final_scales = { 0.125f, 0.25f };
        ggml_backend_tensor_set(a, a_data.data(), 0, a_data.size());
        ggml_backend_tensor_set(b, b_data.data(), 0, b_data.size() * sizeof(float));
        ggml_backend_tensor_set(dst, dst_data.data(), 0, dst_data.size() * sizeof(float));
        ggml_backend_tensor_set(rhs_capture, rhs_data.data(), 0, rhs_data.size());
        ggml_backend_tensor_set(final_scale_capture, final_scales.data(), 0, final_scales.size() * sizeof(float));

        ok = expect(llama_expt::tensor_export_graph(sched, gf, false),
                    "expected FP4MULMAT graph export") && ok;
    }

    if (ok) {
        std::ifstream in(dir + "/manifest.json", std::ios::binary);
        const std::string manifest((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        ok = expect(manifest.find("\"role\": \"matmul_scale\"") != std::string::npos,
                    "FP4MULMAT export must contain matmul_scale") &&
             expect(manifest.find("\"scale_semantics\": \"final_output_multiplier\"") != std::string::npos,
                    "FP4MULMAT export must describe final scale semantics") &&
             expect(manifest.find("\"scale_encoding\": \"f32\"") != std::string::npos,
                    "FP4MULMAT export must preserve the F32 scale") &&
             expect(manifest.find("\"operand_rounding\": \"bf16_rne\"") != std::string::npos,
                    "FP4MULMAT export must describe scale operand rounding") &&
             expect(manifest.find("\"role\": \"src0_scale_raw\"") == std::string::npos,
                    "FP4MULMAT export must omit raw src0 scale") &&
             expect(manifest.find("\"role\": \"src0_global_scale\"") == std::string::npos,
                    "FP4MULMAT export must omit derived src0 global scale") &&
             expect(manifest.find("\"role\": \"src1_global_scale\"") == std::string::npos,
                    "FP4MULMAT export must omit RHS global scale") && ok;
    }

    if (sched) {
        ggml_backend_sched_free(sched);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
    ggml_free(ctx);
    for (size_t i = 0; i < old_values.size(); ++i) {
        if (old_present[i]) {
            setenv(env_names[i], old_values[i].c_str(), 1);
        } else {
            unsetenv(env_names[i]);
        }
    }
    return ok;
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
    if (!test_op_export_retains_selected_node_storage()) {
        return 1;
    }
    if (!test_tensor_name_priority_binds_nvfp4_capture()) {
        return 1;
    }
    if (!test_fp4mulmat_export_writes_only_final_scale()) {
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
