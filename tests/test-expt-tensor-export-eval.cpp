#include "../src/expt/tensor-export-eval.h"
#include "../src/expt/quant_algo/attention-quant-round.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
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

static bool read_first_u16(const std::filesystem::path & path, uint16_t & value) {
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    return in.good();
}

class test_identity_attention_quant_round_algo : public llama_expt::attention_quant_round_algo {
public:
    std::string name() const override {
        return "test_identity";
    }

    llama_expt::attention_quant_round_result quant_round(
            const llama_expt::attention_quant_round_input & input) const override {
        llama_expt::attention_quant_round_result result;
        result.k.values = input.k_values;
        result.k.metadata.mode = "identity_k";
        result.k.metadata.integer_fields["layer"] = input.layer;
        result.q.values = input.q_values;
        result.q.metadata.mode = "identity_q";
        result.q.metadata.string_fields["global_scale"] = "none";
        return result;
    }
};

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
           expect_close(llama_expt::compute_nmse(a, b), 14.0 / 22.0, 1e-12, "NMSE mismatch") &&
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
        "LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP",
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
    unsetenv("LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP");

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

static bool test_soft_max_bf16_dump_writes_f32_records_as_bf16() {
    const char * env_names[] = {
        "LLAMA_EXPT_TENSOR_EXPORT_DIR",
        "LLAMA_EXPT_TENSOR_EXPORT_NAME",
        "LLAMA_EXPT_TENSOR_EXPORT_OP",
        "LLAMA_EXPT_TENSOR_EXPORT_TYPE",
        "LLAMA_EXPT_TENSOR_EXPORT_LAYER",
        "LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP",
    };
    std::vector<std::string> old_values;
    std::vector<bool> old_present;
    for (const char * env_name : env_names) {
        const char * value = std::getenv(env_name);
        old_present.push_back(value != nullptr);
        old_values.emplace_back(value ? value : "");
    }

    const std::string dir = temp_dir() + "-soft-max-bf16-dump";
    std::filesystem::remove_all(dir);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", dir.c_str(), 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_NAME", "softmax-export-0", 1);
    unsetenv("LLAMA_EXPT_TENSOR_EXPORT_OP");
    setenv("LLAMA_EXPT_TENSOR_EXPORT_TYPE", "decode", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_LAYER", "0", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP", "1", 1);

    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!expect(ctx != nullptr, "expected ggml context for SOFT_MAX BF16 dump test")) {
        return false;
    }

    ggml_tensor * input_leaf = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1, 2, 1);
    ggml_tensor * input = ggml_scale(ctx, input_leaf, 1.0f);
    ggml_set_name(input, "softmax-input-0");
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 4, 1, 1, 1);
    ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor * dst = ggml_soft_max_ext(ctx, input, mask, 0.125f, 8.0f);
    ggml_soft_max_add_sinks(dst, sinks);
    ggml_set_name(dst, "softmax-export-0");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, dst);
    const bool retained = llama_expt::tensor_export_maybe_retain_graph(gf);
    llama_expt::tensor_export_observer * observer =
            llama_expt::tensor_export_observer_create(gf, false, nullptr, nullptr);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 16, false, true);
    bool ok = expect(retained, "expected SOFT_MAX BF16 dump graph retention") &&
            expect(observer != nullptr, "expected SOFT_MAX BF16 dump observer") &&
            expect(backend != nullptr && sched != nullptr, "expected CPU scheduler for SOFT_MAX BF16 dump test") &&
            expect(ggml_backend_sched_alloc_graph(sched, gf), "expected graph allocation for SOFT_MAX BF16 dump test");

    if (ok) {
        const std::vector<float> input_data = { -1.0f, 0.0f, 1.0f, 2.0f, 2.0f, 1.0f, 0.0f, -1.0f };
        const std::vector<ggml_fp16_t> mask_data = {
            ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(0.0f),
            ggml_fp32_to_fp16(-1.0f), ggml_fp32_to_fp16(-2.0f),
        };
        const std::vector<float> sink_data = { -0.5f, -1.0f };
        const std::vector<float> dst_data((size_t) ggml_nelements(dst), 0.25f);
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(sinks, sink_data.data(), 0, sink_data.size() * sizeof(float));
        ggml_backend_tensor_set(dst, dst_data.data(), 0, dst_data.size() * sizeof(float));
        ok = expect(llama_expt::tensor_export_observer_callback(input, true, observer),
                    "BF16 dump observer must stop after SOFT_MAX src0 producer") &&
             expect(llama_expt::tensor_export_observer_callback(input, false, observer),
                    "BF16 dump observer must snapshot SOFT_MAX src0 before execution") && ok;
        ggml_backend_tensor_set(input, dst_data.data(), 0, dst_data.size() * sizeof(float));
        ok = expect(llama_expt::tensor_export_observer_callback(dst, true, observer),
                    "BF16 dump observer must request the SOFT_MAX dst") &&
             expect(llama_expt::tensor_export_observer_callback(dst, false, observer),
                    "BF16 dump observer must snapshot the SOFT_MAX dst") &&
             expect(llama_expt::tensor_export_graph(sched, gf, false, observer),
                    "expected SOFT_MAX BF16 dump graph export") && ok;
    }

    if (ok) {
        std::ifstream in(dir + "/manifest.json", std::ios::binary);
        const std::string manifest((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        ok = expect(manifest.find("\"bf16_dump\": true") != std::string::npos,
                    "SOFT_MAX BF16 dump manifest must record bf16_dump") &&
             expect(manifest.find("\"dump_conversion\": \"f32_to_bf16_trunc\"") != std::string::npos,
                    "SOFT_MAX BF16 dump manifest must record conversion") &&
             expect(manifest.find("\"dtype\": \"bf16\"") != std::string::npos,
                    "SOFT_MAX BF16 dump manifest must write F32 records as BF16") &&
             expect(manifest.find("\"original_dtype\": \"f32\"") != std::string::npos,
                    "SOFT_MAX BF16 dump manifest must preserve original F32 dtype") &&
             expect(manifest.find("\"dtype\": \"f16\"") != std::string::npos,
                    "SOFT_MAX BF16 dump must leave non-F32 mask storage unchanged") && ok;

        bool found_dst = false;
        bool found_src0 = false;
        bool found_src2 = false;
        bool found_mask_f16 = false;
        uint16_t dst_first = 0;
        uint16_t src0_first = 0;
        uint16_t src2_first = 0;
        for (const auto & entry : std::filesystem::directory_iterator(dir)) {
            const std::string filename = entry.path().filename().string();
            if (filename.find("-dst-") != std::string::npos) {
                found_dst = read_first_u16(entry.path(), dst_first) &&
                        std::filesystem::file_size(entry.path()) == (size_t) ggml_nelements(dst) * sizeof(uint16_t);
            } else if (filename.find("-src0-") != std::string::npos) {
                found_src0 = read_first_u16(entry.path(), src0_first) &&
                        std::filesystem::file_size(entry.path()) == (size_t) ggml_nelements(input) * sizeof(uint16_t);
            } else if (filename.find("-src1-") != std::string::npos) {
                found_mask_f16 = std::filesystem::file_size(entry.path()) == (size_t) ggml_nelements(mask) * sizeof(ggml_fp16_t);
            } else if (filename.find("-src2-") != std::string::npos) {
                found_src2 = read_first_u16(entry.path(), src2_first) &&
                        std::filesystem::file_size(entry.path()) == (size_t) ggml_nelements(sinks) * sizeof(uint16_t);
            }
        }
        ok = expect(found_dst, "SOFT_MAX BF16 dump must write dst as compact BF16") &&
             expect(found_src0, "SOFT_MAX BF16 dump must write src0 as compact BF16") &&
             expect(found_src2, "SOFT_MAX BF16 dump must write src2 as compact BF16") &&
             expect(found_mask_f16, "SOFT_MAX BF16 dump must leave src1 F16 compact") &&
             expect(dst_first == UINT16_C(0x3e80), "SOFT_MAX BF16 dump dst first value must be 0.25 BF16") &&
             expect(src0_first == UINT16_C(0xbf80), "SOFT_MAX BF16 dump src0 first value must be -1.0 BF16") &&
             expect(src2_first == UINT16_C(0xbf00), "SOFT_MAX BF16 dump src2 first value must be -0.5 BF16") && ok;
    }

    llama_expt::tensor_export_observer_free(observer);
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

static bool test_fp4mulmat_export_writes_only_final_scale() {
    const char * env_names[] = {
        "LLAMA_EXPT_TENSOR_EXPORT_DIR",
        "LLAMA_EXPT_TENSOR_EXPORT_NAME",
        "LLAMA_EXPT_TENSOR_EXPORT_OP",
        "LLAMA_EXPT_TENSOR_EXPORT_TYPE",
        "LLAMA_EXPT_TENSOR_EXPORT_LAYER",
        "LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP",
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
    unsetenv("LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP");

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

static bool test_soft_max_export_writes_params_and_sinks() {
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

    const std::string dir = temp_dir() + "-soft-max";
    std::filesystem::remove_all(dir);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", dir.c_str(), 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_NAME", "softmax-export-0", 1);
    unsetenv("LLAMA_EXPT_TENSOR_EXPORT_OP");
    setenv("LLAMA_EXPT_TENSOR_EXPORT_TYPE", "decode", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_LAYER", "0", 1);

    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!expect(ctx != nullptr, "expected ggml context for SOFT_MAX export test")) {
        return false;
    }

    ggml_tensor * input_leaf = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1, 2, 1);
    ggml_tensor * input = ggml_scale(ctx, input_leaf, 1.0f);
    ggml_set_name(input, "softmax-input-0");
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 4, 1, 1, 1);
    ggml_tensor * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_tensor * dst = ggml_soft_max_ext(ctx, input, mask, 0.125f, 8.0f);
    ggml_soft_max_add_sinks(dst, sinks);
    ggml_set_name(dst, "softmax-export-0");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, dst);
    const bool retained = llama_expt::tensor_export_maybe_retain_graph(gf);
    llama_expt::tensor_export_observer * observer =
            llama_expt::tensor_export_observer_create(gf, false, nullptr, nullptr);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 16, false, true);
    bool ok = expect(retained, "expected SOFT_MAX graph retention") &&
            expect((dst->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,
                    "SOFT_MAX dst must be retained") &&
            expect((sinks->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,
                    "SOFT_MAX src2 sinks must be retained") &&
            expect(observer != nullptr, "expected SOFT_MAX export observer") &&
            expect(backend != nullptr && sched != nullptr, "expected CPU scheduler for SOFT_MAX export test") &&
            expect(ggml_backend_sched_alloc_graph(sched, gf), "expected graph allocation for SOFT_MAX export test");

    if (ok) {
        const std::vector<float> input_data = { -1.0f, 0.0f, 1.0f, 2.0f, 2.0f, 1.0f, 0.0f, -1.0f };
        const std::vector<ggml_fp16_t> mask_data = {
            ggml_fp32_to_fp16(0.0f), ggml_fp32_to_fp16(0.0f),
            ggml_fp32_to_fp16(-1.0f), ggml_fp32_to_fp16(-2.0f),
        };
        const std::vector<float> sink_data = { -0.5f, -1.0f };
        const std::vector<float> dst_data((size_t) ggml_nelements(dst), 0.25f);
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(sinks, sink_data.data(), 0, sink_data.size() * sizeof(float));
        ggml_backend_tensor_set(dst, dst_data.data(), 0, dst_data.size() * sizeof(float));
        ok = expect(llama_expt::tensor_export_observer_callback(input, true, observer),
                    "observer must stop after the SOFT_MAX src0 producer") &&
             expect(llama_expt::tensor_export_observer_callback(input, false, observer),
                    "observer must snapshot SOFT_MAX src0 before execution") && ok;
        ggml_backend_tensor_set(input, dst_data.data(), 0, dst_data.size() * sizeof(float));
        ok = expect(llama_expt::tensor_export_observer_callback(dst, true, observer),
                    "observer must request the SOFT_MAX dst") &&
             expect(llama_expt::tensor_export_observer_callback(dst, false, observer),
                    "observer must snapshot the SOFT_MAX dst") &&
             expect(llama_expt::tensor_export_graph(sched, gf, false, observer),
                    "expected SOFT_MAX graph export") && ok;
    }

    if (ok) {
        std::ifstream in(dir + "/manifest.json", std::ios::binary);
        const std::string manifest((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        ok = expect(manifest.find("\"op\": \"SOFT_MAX\"") != std::string::npos,
                    "SOFT_MAX export must record the op") &&
             expect(manifest.find(
                    "\"snapshot_timing\": \"source_producer_and_node_completion\"") != std::string::npos,
                    "SOFT_MAX export must describe producer-time source snapshots") &&
             expect(manifest.find("\"role\": \"src2\"") != std::string::npos,
                    "SOFT_MAX export must contain src2 sinks") &&
             expect(manifest.find("\"scale\": 0.125") != std::string::npos,
                    "SOFT_MAX export must record scale") &&
             expect(manifest.find("\"max_bias\": 8.0") != std::string::npos,
                    "SOFT_MAX export must record max_bias") && ok;

        float exported_src0_first = 0.0f;
        bool found_src0 = false;
        for (const auto & entry : std::filesystem::directory_iterator(dir)) {
            if (entry.path().filename().string().find("-src0-") == std::string::npos) {
                continue;
            }
            std::ifstream raw(entry.path(), std::ios::binary);
            raw.read(reinterpret_cast<char *>(&exported_src0_first), sizeof(exported_src0_first));
            found_src0 = raw.good();
            break;
        }
        ok = expect(found_src0, "SOFT_MAX export must write src0 data") &&
             expect_close(exported_src0_first, -1.0, 0.0,
                    "SOFT_MAX src0 must be captured before in-place overwrite") && ok;
    }

    llama_expt::tensor_export_observer_free(observer);
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

static bool test_rope_export_writes_params_and_positions() {
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

    const std::string dir = temp_dir() + "-rope";
    std::filesystem::remove_all(dir);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", dir.c_str(), 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_NAME", "rope-export", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_OP", "ROPE", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_TYPE", "decode", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_LAYER", "0", 1);

    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!expect(ctx != nullptr, "expected ggml context for ROPE export test")) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 2, 2);
    ggml_set_name(input, "rope-input-0");
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
    ggml_set_name(positions, "rope-pos-0");
    ggml_tensor * dst = ggml_rope_ext(
            ctx, input, positions, nullptr,
            128, GGML_ROPE_TYPE_NEOX, 40960,
            1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_set_name(dst, "rope-export-0");

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, dst);
    const bool retained = llama_expt::tensor_export_maybe_retain_graph(gf);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 16, false, true);
    bool ok = expect(retained, "expected ROPE graph retention") &&
            expect((dst->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,
                    "ROPE dst must be retained") &&
            expect((input->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,
                    "ROPE src0 must be retained") &&
            expect((positions->flags & GGML_TENSOR_FLAG_OUTPUT) != 0,
                    "ROPE src1 positions must be retained") &&
            expect(backend != nullptr && sched != nullptr, "expected CPU scheduler for ROPE export test") &&
            expect(ggml_backend_sched_alloc_graph(sched, gf), "expected graph allocation for ROPE export test");

    if (ok) {
        std::vector<float> input_data((size_t) ggml_nelements(input), 0.0f);
        std::vector<float> dst_data((size_t) ggml_nelements(dst), 0.0f);
        const std::vector<int32_t> position_data = { 17, 18 };
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        ggml_backend_tensor_set(positions, position_data.data(), 0, position_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(dst, dst_data.data(), 0, dst_data.size() * sizeof(float));
        ok = expect(llama_expt::tensor_export_graph(sched, gf, false),
                    "expected ROPE graph export") && ok;
    }

    if (ok) {
        std::ifstream in(dir + "/manifest.json", std::ios::binary);
        const std::string manifest((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        ok = expect(manifest.find("\"op\": \"ROPE\"") != std::string::npos,
                    "ROPE export must record the op") &&
             expect(manifest.find("\"role\": \"src1\"") != std::string::npos,
                    "ROPE export must contain src1 positions") &&
             expect(manifest.find("\"n_dims\": 128") != std::string::npos,
                    "ROPE export must record n_dims") &&
             expect(manifest.find("\"mode\": 2") != std::string::npos,
                    "ROPE export must record mode") &&
             expect(manifest.find("\"n_ctx_orig\": 40960") != std::string::npos,
                    "ROPE export must record n_ctx_orig") &&
             expect(manifest.find("\"freq_base\": 1000000.0") != std::string::npos,
                    "ROPE export must record freq_base") &&
             expect(manifest.find("\"freq_scale\": 1.0") != std::string::npos,
                    "ROPE export must record freq_scale") &&
             expect(manifest.find("\"ext_factor\": 0.0") != std::string::npos,
                    "ROPE export must record ext_factor") &&
             expect(manifest.find("\"attn_factor\": 1.0") != std::string::npos,
                    "ROPE export must record attn_factor") &&
             expect(manifest.find("\"beta_fast\": 32.0") != std::string::npos,
                    "ROPE export must record beta_fast") &&
             expect(manifest.find("\"beta_slow\": 1.0") != std::string::npos,
                    "ROPE export must record beta_slow") &&
             expect(manifest.find("\"sections\": [") != std::string::npos,
                    "ROPE export must record sections") && ok;
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

static bool test_attention_export_pin_marks_output_tensors() {
    const char * old_dir = std::getenv("LLAMA_EXPT_TENSOR_EXPORT_DIR");
    const std::string old_dir_value = old_dir ? old_dir : "";
    const char * old_kinds = std::getenv("LLAMA_EXPT_TENSOR_EXPORT_KINDS");
    const std::string old_kinds_value = old_kinds ? old_kinds : "";
    setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", "/tmp/llama-expt-export-pin-test", 1);
    setenv("LLAMA_EXPT_TENSOR_EXPORT_KINDS", "k", 1);

    struct ggml_init_params params = {
        /* .mem_size   = */ 1u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("failed to init ggml context");
    }

    const auto make_and_pin = [&](const char * name) {
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        ggml_set_name(tensor, name);
        llama_expt::tensor_export_pin_named_tensor(tensor);
        return tensor;
    };

    ggml_tensor * kq = make_and_pin("kq-0");
    ggml_tensor * kcur0 = make_and_pin("Kcur-0");
    ggml_tensor * kcur1 = make_and_pin("Kcur-1");
    ggml_tensor * kcur12 = make_and_pin("Kcur-12");
    ggml_tensor * softmax = make_and_pin("kq-softmax-0");
    ggml_tensor * other = make_and_pin("kq-1");
    ggml_tensor * kcur_mm = make_and_pin("Kcur-mm-0");
    ggml_tensor * kcur_scaled = make_and_pin("Kcur-scaled-0");
    ggml_tensor * kcur_normed = make_and_pin("Kcur_normed-0");
    ggml_tensor * kcur_post_rope = make_and_pin("Kcur-post-rope-0");
    ggml_tensor * kcur_view = make_and_pin("Kcur-0-view");

    setenv("LLAMA_EXPT_TENSOR_EXPORT_KINDS", "q", 1);
    ggml_tensor * kcur_without_k_kind = make_and_pin("Kcur-7");

    const bool ok =
        expect((kq->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected kq-0 to be pinned as output") &&
        expect((kcur0->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur-0 to be pinned as output") &&
        expect((kcur1->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur-1 to be pinned as output") &&
        expect((kcur12->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur-12 to be pinned as output") &&
        expect((softmax->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected kq-softmax-0 to be pinned as output") &&
        expect((other->flags & GGML_TENSOR_FLAG_OUTPUT) == 0, "expected non-layer0 attention tensor to remain unpinned") &&
        expect((kcur_mm->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur-mm tensor to be pinned as output") &&
        expect((kcur_scaled->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur-scaled tensor to be pinned as output") &&
        expect((kcur_normed->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur_normed tensor to be pinned as output") &&
        expect((kcur_post_rope->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur-post-rope tensor to be pinned as output") &&
        expect((kcur_view->flags & GGML_TENSOR_FLAG_OUTPUT) != 0, "expected Kcur view tensor to be pinned as output") &&
        expect((kcur_without_k_kind->flags & GGML_TENSOR_FLAG_OUTPUT) == 0, "expected Kcur tensor to remain unpinned when k export is disabled");

    ggml_free(ctx);
    if (old_dir) {
        setenv("LLAMA_EXPT_TENSOR_EXPORT_DIR", old_dir_value.c_str(), 1);
    } else {
        unsetenv("LLAMA_EXPT_TENSOR_EXPORT_DIR");
    }
    if (old_kinds) {
        setenv("LLAMA_EXPT_TENSOR_EXPORT_KINDS", old_kinds_value.c_str(), 1);
    } else {
        unsetenv("LLAMA_EXPT_TENSOR_EXPORT_KINDS");
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
    if (!expect_close(rr.softmax_nmse, 0.0, 1e-12, "expected exact softmax NMSE")) {
        return false;
    }
    if (!expect_close(rr.kq_nmse, 0.0, 1e-12, "expected exact KQ NMSE")) {
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
    if (!expect(json.find("\"softmax_nmse\"") != std::string::npos,
            "expected softmax NMSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"kq_mse\"") != std::string::npos,
            "expected KQ MSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"kq_nmse\"") != std::string::npos,
            "expected KQ NMSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"kq_max_abs_err\"") != std::string::npos,
            "expected KQ max_abs_err in JSON")) {
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
    if (!expect(rr.softmax_nmse >= 0.0, "expected non-negative softmax NMSE")) {
        return false;
    }
    if (!expect(rr.kq_nmse >= 0.0, "expected non-negative KQ NMSE")) {
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
    if (!expect(json.find("\"softmax_nmse\"") != std::string::npos,
            "expected softmax NMSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"softmax_kld\"") != std::string::npos,
            "expected softmax KLD in JSON")) {
        return false;
    }
    if (!expect(json.find("\"kq_mse\"") != std::string::npos,
            "expected KQ MSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"kq_nmse\"") != std::string::npos,
            "expected KQ NMSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"kq_max_abs_err\"") != std::string::npos,
            "expected KQ max_abs_err in JSON")) {
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

static bool test_attention_replay_quant_round_accepts_custom_algo() {
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

    const test_identity_attention_quant_round_algo algo;
    const llama_expt::attention_replay_nvfp4_outlier_eval_report report =
        llama_expt::evaluate_manifest_attention_replay_quant_round(dir + "/manifest-attention.json", algo);
    if (!expect(report.records.size() == 1, "expected one custom quant round attention replay record")) {
        return false;
    }
    if (!expect(report.quant_round_algorithm == "test_identity", "expected custom quant round algorithm name")) {
        return false;
    }

    const llama_expt::attention_replay_nvfp4_outlier_report & rr = report.records[0];
    if (!expect_close(rr.max_abs_err_kq, 0.0, 1e-6, "expected exact custom KQ replay")) {
        return false;
    }
    if (!expect_close(rr.max_abs_err_softmax, 0.0, 1e-6, "expected exact custom softmax replay")) {
        return false;
    }
    if (!expect(rr.k_quantization_mode == "identity_k", "expected custom K mode")) {
        return false;
    }
    if (!expect(rr.q_quantization_mode == "identity_q", "expected custom Q mode")) {
        return false;
    }

    const std::string json = llama_expt::format_attention_replay_nvfp4_outlier_eval_report_json(report);
    if (!expect(json.find("\"quant_round_algorithm\": \"test_identity\"") != std::string::npos,
            "expected custom quant round algorithm in JSON")) {
        return false;
    }
    if (!expect(json.find("\"k_quant_round\"") != std::string::npos,
            "expected K quant round metadata in JSON")) {
        return false;
    }
    if (!expect(json.find("\"mode\": \"identity_q\"") != std::string::npos,
            "expected custom Q mode in JSON")) {
        return false;
    }

    return true;
}

static bool test_attention_replay_fp8_e4m3_e8m0_reports_metrics() {
    const std::string dir = temp_dir();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::vector<float> k_data(64);
    std::vector<float> q_data(64);
    for (size_t i = 0; i < 32; ++i) {
        k_data[i] = (float) i * 0.0625f - 1.0f;
        k_data[32 + i] = (float) i * -0.03125f + 0.75f;
        q_data[i] = (float) i * 0.046875f + 0.0625f;
        q_data[32 + i] = (float) i * -0.0390625f + 0.5f;
    }

    const std::vector<float> mask_data = {
         0.0f,
        -INFINITY,
    };
    std::vector<float> kq_data;
    std::vector<float> softmax_data;
    compute_attention_reference(
            k_data, q_data, mask_data,
            32, 2, 1, 1,
            32, 2, 1, 1,
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
            "    {\"name\":\"Kcur-0\",\"kind\":\"k\",\"dtype\":\"f32\",\"ne\":[32,2,1,1],\"nb\":[4,128,256,256],\"path\":\"k.bin\",\"byte_size\":256},\n"
            "    {\"name\":\"Qcur-0\",\"kind\":\"q\",\"dtype\":\"f32\",\"ne\":[32,2,1,1],\"nb\":[4,128,256,256],\"path\":\"q.bin\",\"byte_size\":256},\n"
            "    {\"name\":\"kq-0\",\"kind\":\"kq\",\"dtype\":\"f32\",\"ne\":[1,1,2,1],\"nb\":[4,4,4,8],\"path\":\"kq.bin\",\"byte_size\":8},\n"
            "    {\"name\":\"kq-mask-0\",\"kind\":\"kq_mask\",\"dtype\":\"f32\",\"ne\":[1,2,1,1],\"nb\":[4,4,8,8],\"path\":\"mask.bin\",\"byte_size\":8},\n"
            "    {\"name\":\"kq-softmax-0\",\"kind\":\"kq_softmax\",\"dtype\":\"f32\",\"ne\":[1,1,2,1],\"nb\":[4,4,4,8],\"path\":\"softmax.bin\",\"byte_size\":8,\n"
            "     \"meta\":{\"src_k\":\"Kcur-0\",\"src_q\":\"Qcur-0\",\"src_kq\":\"kq-0\",\"src_mask\":\"kq-mask-0\",\"kq_scale\":\"1\",\"max_bias\":\"0\"}}\n"
            "  ]\n"
            "}\n");

    const llama_expt::attention_replay_nvfp4_outlier_eval_report report =
        llama_expt::evaluate_manifest_attention_replay_fp8_e4m3_e8m0(dir + "/manifest-attention.json");
    if (!expect(report.records.size() == 1, "expected one FP8 attention replay record")) {
        return false;
    }
    if (!expect(report.algorithm == "attention_replay_fp8_e4m3_e8m0", "expected FP8 report algorithm")) {
        return false;
    }
    if (!expect(report.quant_round_algorithm == "fp8_e4m3_e8m0_32", "expected FP8 quant round algorithm")) {
        return false;
    }

    const llama_expt::attention_replay_nvfp4_outlier_report & rr = report.records[0];
    if (!expect(rr.softmax_metrics.n == softmax_data.size(), "expected FP8 softmax metric count")) {
        return false;
    }
    if (!expect(rr.softmax_metrics.mse >= 0.0, "expected non-negative FP8 softmax MSE")) {
        return false;
    }
    if (!expect(rr.softmax_nmse >= 0.0, "expected non-negative FP8 softmax NMSE")) {
        return false;
    }
    if (!expect(rr.softmax_kld >= 0.0, "expected non-negative FP8 softmax KLD")) {
        return false;
    }
    if (!expect(rr.k_quantization_mode == "fp8_e4m3_e8m0_32_k", "expected FP8 K mode")) {
        return false;
    }
    if (!expect(rr.q_quantization_mode == "fp8_e4m3_e8m0_32_q", "expected FP8 Q mode")) {
        return false;
    }

    const std::string json = llama_expt::format_attention_replay_nvfp4_outlier_eval_report_json(report);
    if (!expect(json.find("\"algorithm\": \"attention_replay_fp8_e4m3_e8m0\"") != std::string::npos,
            "expected FP8 algorithm in JSON")) {
        return false;
    }
    if (!expect(json.find("\"quant_round_algorithm\": \"fp8_e4m3_e8m0_32\"") != std::string::npos,
            "expected FP8 quant round algorithm in JSON")) {
        return false;
    }
    if (!expect(json.find("\"softmax_mse\"") != std::string::npos,
            "expected FP8 softmax MSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"softmax_nmse\"") != std::string::npos,
            "expected FP8 softmax NMSE in JSON")) {
        return false;
    }
    if (!expect(json.find("\"softmax_kld\"") != std::string::npos,
            "expected FP8 softmax KLD in JSON")) {
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
    if (!test_soft_max_export_writes_params_and_sinks()) {
        return 1;
    }
    if (!test_soft_max_bf16_dump_writes_f32_records_as_bf16()) {
        return 1;
    }
    if (!test_rope_export_writes_params_and_positions()) {
        return 1;
    }
    if (!test_attention_export_pin_marks_output_tensors()) {
        return 1;
    }
    if (!test_manifest_eval_and_rejection()) {
        return 1;
    }
    if (!test_attention_replay_manifest_eval_reports_small_error()) {
        return 1;
    }
    if (!test_attention_replay_nvfp4_outlier_manifest_eval_reports_metrics()) {
        return 1;
    }
    if (!test_attention_replay_quant_round_accepts_custom_algo()) {
        return 1;
    }
    if (!test_attention_replay_fp8_e4m3_e8m0_reports_metrics()) {
        return 1;
    }

    std::puts("test-expt-tensor-export-eval: ok");
    return 0;
}
