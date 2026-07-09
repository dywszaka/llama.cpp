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

static void compute_attention_reference(
        const std::vector<float> & k_data,
        const std::vector<float> & q_data,
        const std::vector<float> & mask_data,
        int64_t k_ne0,
        int64_t k_ne1,
        int64_t k_ne2,
        int64_t k_ne3,
        int64_t q_ne0,
        int64_t q_ne1,
        int64_t q_ne2,
        int64_t q_ne3,
        int64_t mask_ne0,
        int64_t mask_ne1,
        int64_t mask_ne2,
        int64_t mask_ne3,
        std::vector<float> & kq_data,
        std::vector<float> & softmax_data) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 16u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("failed to init ggml context");
    }

    ggml_tensor * k_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, k_ne0, k_ne1, k_ne2, k_ne3);
    ggml_tensor * q_base = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, q_ne0, q_ne1, q_ne2, q_ne3);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mask_ne0, mask_ne1, mask_ne2, mask_ne3);

    ggml_tensor * k = k_base;
    ggml_tensor * q = q_base;
    const int64_t n_stream = k->ne[3];
    q = ggml_reshape_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2] / n_stream, n_stream);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * probs = ggml_soft_max_ext(ctx, kq, mask, 1.0f, 0.0f);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, kq);
    ggml_build_forward_expand(gf, probs);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        ggml_free(ctx);
        throw std::runtime_error("failed to init ggml backend");
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        throw std::runtime_error("failed to alloc ggml backend buffer");
    }

    std::vector<ggml_fp16_t> k_data_f16(k_data.size());
    ggml_fp32_to_fp16_row(k_data.data(), k_data_f16.data(), (int64_t) k_data.size());

    ggml_backend_tensor_set(k_base, k_data_f16.data(), 0, k_data_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(q_base, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        throw std::runtime_error("failed to compute attention reference");
    }

    kq_data.resize((size_t) ggml_nelements(kq));
    softmax_data.resize((size_t) ggml_nelements(probs));
    ggml_backend_tensor_get(kq, kq_data.data(), 0, kq_data.size() * sizeof(float));
    ggml_backend_tensor_get(probs, softmax_data.data(), 0, softmax_data.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
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

static bool test_attention_export_pin_marks_output_tensors() {
    const char * old_dir = std::getenv("LLAMA_EXPT_TENSOR_EXPORT_DIR");
    const std::string old_dir_value = old_dir ? old_dir : "";
    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", "/tmp/llama-expt-export-pin-test", 1);

    struct ggml_init_params params = {
        /* .mem_size   = */ 1u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("failed to init ggml context");
    }

    ggml_tensor * kq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(kq, "kq-0");
    llama_expt::tensor_export_pin_named_tensor(kq);

    ggml_tensor * kcur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(kcur, "Kcur-0");
    llama_expt::tensor_export_pin_named_tensor(kcur);

    ggml_tensor * softmax = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(softmax, "kq-softmax-0");
    llama_expt::tensor_export_pin_named_tensor(softmax);

    ggml_tensor * other = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(other, "kq-1");
    llama_expt::tensor_export_pin_named_tensor(other);

    const bool ok =
        expect((kq->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected kq-0 to be pinned as output") &&
        expect((kcur->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur-0 to be pinned as output") &&
        expect((softmax->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected kq-softmax-0 to be pinned as output") &&
        expect((other->flags & GGML_TENSOR_FLAG_OUTPUT) == 0, "expected non-layer0 tensor to remain unpinned");

    ggml_free(ctx);
    if (old_dir) {
        setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", old_dir_value.c_str(), 1);
    } else {
        unsetenv("LLAMA_EXPT_TENSOR_EXPORT_DIR");
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

static bool test_attention_replay_manifest_eval_reports_small_error() {
    const std::string dir = temp_dir();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const std::vector<float> k_data = {
         1.0f,  0.0f,  0.5f, -0.5f,
         0.0f,  1.0f, -0.5f,  0.5f,
    };
    const std::vector<float> q_data = {
         1.0f, -1.0f,
         0.5f,  0.5f,
        -0.5f,  1.0f,
         1.0f,  0.0f,
    };
    const std::vector<float> mask_data = {
         0.0f,
        -INFINITY,
    };
    std::vector<float> kq_data;
    std::vector<float> softmax_data;
    compute_attention_reference(
            k_data, q_data, mask_data,
            4, 2, 1, 1,
            4, 2, 1, 1,
            1, 2, 1, 1,
            kq_data, softmax_data);

    write_f32(dir + "/k.bin", k_data);
    write_f32(dir + "/q.bin", q_data);
    write_f32(dir + "/mask.bin", mask_data);
    write_f32(dir + "/kq.bin", kq_data);
    write_f32(dir + "/softmax.bin", softmax_data);

    write_file(dir + "/manifest-attention.json",
            "{\n"
            "  \"records\": [\n"
            "    {\"name\":\"Kcur-0\",\"kind\":\"k\",\"dtype\":\"f32\",\"ne\":[4,2,1,1],\"nb\":[4,16,32,32],\"path\":\"k.bin\",\"byte_size\":32},\n"
            "    {\"name\":\"Qcur-0\",\"kind\":\"q\",\"dtype\":\"f32\",\"ne\":[4,2,1,1],\"nb\":[4,16,32,32],\"path\":\"q.bin\",\"byte_size\":32},\n"
            "    {\"name\":\"kq-0\",\"kind\":\"kq\",\"dtype\":\"f32\",\"ne\":[1,1,2,1],\"nb\":[4,4,4,8],\"path\":\"kq.bin\",\"byte_size\":8},\n"
            "    {\"name\":\"kq-mask-0\",\"kind\":\"kq_mask\",\"dtype\":\"f32\",\"ne\":[1,2,1,1],\"nb\":[4,4,8,8],\"path\":\"mask.bin\",\"byte_size\":8},\n"
            "    {\"name\":\"kq-softmax-0\",\"kind\":\"kq_softmax\",\"dtype\":\"f32\",\"ne\":[1,1,2,1],\"nb\":[4,4,4,8],\"path\":\"softmax.bin\",\"byte_size\":8,\n"
            "     \"meta\":{\"src_k\":\"Kcur-0\",\"src_q\":\"Qcur-0\",\"src_kq\":\"kq-0\",\"src_mask\":\"kq-mask-0\",\"kq_scale\":\"1\",\"max_bias\":\"0\"}}\n"
            "  ]\n"
            "}\n");

    const llama_expt::attention_replay_eval_report report =
        llama_expt::evaluate_manifest_attention_replay(dir + "/manifest-attention.json");
    if (!expect(report.records.size() == 1, "expected one attention replay record")) {
        return false;
    }

    const llama_expt::attention_replay_report & rr = report.records[0];
    if (!expect_close(rr.max_abs_err_kq, 0.0, 1e-6, "expected exact KQ replay")) {
        return false;
    }
    if (!expect_close(rr.max_abs_err_softmax, 0.0, 1e-6, "expected exact softmax replay")) {
        return false;
    }
    if (!expect(rr.softmax_metrics.n == softmax_data.size(), "expected softmax metric count")) {
        return false;
    }

    const std::string json = llama_expt::format_attention_replay_eval_report_json(report);
    if (!expect(json.find("\"algorithm\": \"attention_replay\"") != std::string::npos,
            "expected attention replay algorithm in JSON")) {
        return false;
    }
    if (!expect(json.find("\"max_abs_err_softmax\": 0.0") != std::string::npos ||
                json.find("\"max_abs_err_softmax\": 0") != std::string::npos,
            "expected softmax max_abs_err in JSON")) {
        return false;
    }

    return true;
}

static bool test_attention_replay_nvfp4_outlier_manifest_eval_reports_metrics() {
    const std::string dir = temp_dir();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::vector<float> k_data(32);
    std::vector<float> q_data(32);
    for (size_t i = 0; i < 16; ++i) {
        k_data[i] = (float) i * 0.125f - 0.75f;
        k_data[16 + i] = (float) i * -0.0625f + 0.5f;
        q_data[i] = (float) i * 0.03125f + 0.125f;
        q_data[16 + i] = (float) i * -0.046875f + 0.25f;
    }
    k_data[3] = 300.0f;
    k_data[22] = -320.0f;

    const std::vector<float> mask_data = {
         0.0f,
        -INFINITY,
    };
    std::vector<float> kq_data;
    std::vector<float> softmax_data;
    compute_attention_reference(
            k_data, q_data, mask_data,
            16, 2, 1, 1,
            16, 2, 1, 1,
            1, 2, 1, 1,
            kq_data, softmax_data);

    write_f32(dir + "/k.bin", k_data);
    write_f32(dir + "/q.bin", q_data);
    write_f32(dir + "/mask.bin", mask_data);
    write_f32(dir + "/kq.bin", kq_data);
    write_f32(dir + "/softmax.bin", softmax_data);

    write_file(dir + "/manifest-attention.json",
            "{\n"
            "  \"records\": [\n"
            "    {\"name\":\"Kcur-0\",\"kind\":\"k\",\"dtype\":\"f32\",\"ne\":[16,2,1,1],\"nb\":[4,64,128,128],\"path\":\"k.bin\",\"byte_size\":128},\n"
            "    {\"name\":\"Qcur-0\",\"kind\":\"q\",\"dtype\":\"f32\",\"ne\":[16,2,1,1],\"nb\":[4,64,128,128],\"path\":\"q.bin\",\"byte_size\":128},\n"
            "    {\"name\":\"kq-0\",\"kind\":\"kq\",\"dtype\":\"f32\",\"ne\":[1,1,2,1],\"nb\":[4,4,4,8],\"path\":\"kq.bin\",\"byte_size\":8},\n"
            "    {\"name\":\"kq-mask-0\",\"kind\":\"kq_mask\",\"dtype\":\"f32\",\"ne\":[1,2,1,1],\"nb\":[4,4,8,8],\"path\":\"mask.bin\",\"byte_size\":8},\n"
            "    {\"name\":\"kq-softmax-0\",\"kind\":\"kq_softmax\",\"dtype\":\"f32\",\"ne\":[1,1,2,1],\"nb\":[4,4,4,8],\"path\":\"softmax.bin\",\"byte_size\":8,\n"
            "     \"meta\":{\"src_k\":\"Kcur-0\",\"src_q\":\"Qcur-0\",\"src_kq\":\"kq-0\",\"src_mask\":\"kq-mask-0\",\"kq_scale\":\"1\",\"max_bias\":\"0\"}}\n"
            "  ]\n"
            "}\n");

    const llama_expt::attention_replay_nvfp4_outlier_eval_report report =
        llama_expt::evaluate_manifest_attention_replay_nvfp4_outlier(dir + "/manifest-attention.json");
    if (!expect(report.records.size() == 1, "expected one NVFP4 outlier attention replay record")) {
        return false;
    }

    const llama_expt::attention_replay_nvfp4_outlier_report & rr = report.records[0];
    if (!expect(rr.softmax_metrics.n == softmax_data.size(), "expected softmax metric count")) {
        return false;
    }
    if (!expect(rr.k_outlier_count == 2, "expected K outliers above layer threshold")) {
        return false;
    }
    if (!expect(rr.k_threshold > 0.0f, "expected K threshold metadata")) {
        return false;
    }
    if (!expect(rr.k_global_scale > 0.0f, "expected K global scale metadata")) {
        return false;
    }
    if (!expect(rr.softmax_kld >= 0.0, "expected non-negative softmax KLD")) {
        return false;
    }
    if (!expect_close(rr.kld_epsilon, 1e-12, 0.0, "expected KLD epsilon")) {
        return false;
    }

    const std::string json = llama_expt::format_attention_replay_nvfp4_outlier_eval_report_json(report);
    if (!expect(json.find("\"algorithm\": \"attention_replay_nvfp4_outlier\"") != std::string::npos,
            "expected NVFP4 outlier attention replay algorithm in JSON")) {
        return false;
    }
    if (!expect(json.find("\"mode\": \"nvfp4_outlier_threshold_layer0\"") != std::string::npos,
            "expected K quantization mode in JSON")) {
        return false;
    }
    if (!expect(json.find("\"mode\": \"nvfp4_dynamic_row_amax\"") != std::string::npos,
            "expected Q quantization mode in JSON")) {
        return false;
    }
    if (!expect(json.find("\"threshold\"") != std::string::npos,
            "expected threshold in JSON")) {
        return false;
    }
    if (!expect(json.find("\"softmax_mse\"") != std::string::npos,
            "expected softmax MSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"softmax_kld\"") != std::string::npos,
            "expected softmax KLD in JSON")) {
        return false;
    }
    if (!expect(json.find("\"kld_reference_distribution\": \"exported_softmax\"") != std::string::npos,
            "expected KLD reference distribution in JSON")) {
        return false;
    }
    if (!expect(json.find("clamp reference and actual probabilities to epsilon") != std::string::npos,
            "expected KLD epsilon clamp documentation in JSON")) {
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
    if (!test_attention_export_pin_marks_output_tensors()) {
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
    if (!test_attention_replay_manifest_eval_reports_small_error()) {
        return 1;
    }
    if (!test_attention_replay_nvfp4_outlier_manifest_eval_reports_metrics()) {
        return 1;
    }

    std::puts("test-expt-tensor-export-eval: ok");
    return 0;
}
