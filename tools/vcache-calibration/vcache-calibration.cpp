#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "../../src/llama-vcache-calibration.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f calibration.txt -o vcache-calibration.json -ctv f16 -fa\n", argv[0]);
    LOG("\n");
    LOG("The calibration file should contain one prompt per line.\n");
    LOG("\n");
}

struct vcache_calibration_collector {
    std::map<int, llama_vcache_layer_stats> layers;
    std::vector<uint8_t> scratch;

    bool collect(struct ggml_tensor * t, bool ask) {
        int layer = -1;

        if (ask) {
            return llama_vcache_parse_vcur_layer(t->name, layer);
        }

        if (!llama_vcache_parse_vcur_layer(t->name, layer)) {
            return true;
        }

        if (t->op != GGML_OP_RESHAPE) {
            return true;
        }

        const int64_t n = ggml_nelements(t);
        if (n <= 0) {
            return true;
        }

        auto & stats = layers[layer];

        if (t->type == GGML_TYPE_F32) {
            scratch.resize((size_t) n * sizeof(float));
            ggml_backend_tensor_get(t, scratch.data(), 0, scratch.size());
            stats.add((const float *) scratch.data(), (size_t) n);
            return true;
        }

        if (t->type == GGML_TYPE_F16) {
            scratch.resize((size_t) n * sizeof(ggml_fp16_t));
            ggml_backend_tensor_get(t, scratch.data(), 0, scratch.size());
            std::vector<float> buf((size_t) n);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) scratch.data(), buf.data(), n);
            stats.add(buf.data(), buf.size());
            return true;
        }

        if (t->type == GGML_TYPE_BF16) {
            scratch.resize((size_t) n * sizeof(ggml_bf16_t));
            ggml_backend_tensor_get(t, scratch.data(), 0, scratch.size());
            std::vector<float> buf((size_t) n);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) scratch.data(), buf.data(), n);
            stats.add(buf.data(), buf.size());
            return true;
        }

        LOG_WRN("%s: skipping tensor %s with unsupported type %s\n", __func__, t->name, ggml_type_name(t->type));
        return true;
    }
};

static bool cb_eval_vcache(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * collector = (vcache_calibration_collector *) user_data;
    return collector->collect(t, ask);
}

static std::vector<std::string> load_prompts(const common_params & params) {
    std::vector<std::string> prompts;

    for (const std::string & line : string_split<std::string>(params.prompt, '\n')) {
        const std::string stripped = string_strip(line);
        if (!stripped.empty()) {
            prompts.push_back(stripped);
        }
    }

    if (prompts.empty()) {
        const std::string stripped = string_strip(params.prompt);
        if (!stripped.empty()) {
            prompts.push_back(stripped);
        }
    }

    return prompts;
}

static bool run_prompt(llama_context * ctx, const common_params & params, const std::string & prompt) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, add_bos, params.special);
    if (tokens.empty()) {
        LOG_WRN("%s: skipping empty prompt after tokenization\n", __func__);
        return true;
    }

    const int32_t batch_size = std::max<int32_t>(1, params.n_batch);
    int32_t pos = 0;
    while (pos < (int32_t) tokens.size()) {
        const int32_t n_eval = std::min<int32_t>(batch_size, (int32_t) tokens.size() - pos);
        llama_batch batch = llama_batch_get_one(tokens.data() + pos, n_eval);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed while processing prompt chunk at token %d/%zu\n",
                    __func__, pos, tokens.size());
            return false;
        }

        pos += n_eval;
    }

    return true;
}

static bool write_report(const common_params & params, const std::vector<std::string> & prompts, const vcache_calibration_collector & collector) {
    std::ofstream out(params.out_file);
    if (!out) {
        LOG_ERR("%s: failed to open '%s' for writing\n", __func__, params.out_file.c_str());
        return false;
    }

    out << "{\n";
    out << "  \"prompt_file\": \"" << params.prompt_file << "\",\n";
    out << "  \"n_prompts\": " << prompts.size() << ",\n";
    out << "  \"layers\": [\n";

    bool first = true;
    for (const auto & kv : collector.layers) {
        const int layer = kv.first;
        const auto & stats = kv.second;
        const auto rec = stats.recommendation();

        if (!first) {
            out << ",\n";
        }
        first = false;

        out << "    {\n";
        out << "      \"layer\": " << layer << ",\n";
        out << "      \"sample_count\": " << stats.sample_count << ",\n";
        out << "      \"zero_count\": " << stats.zero_count << ",\n";
        out << "      \"nonfinite_count\": " << stats.nonfinite_count << ",\n";
        out << "      \"abs_max\": " << stats.abs_max << ",\n";
        out << "      \"p99_abs\": " << rec.range_p99 << ",\n";
        out << "      \"p999_abs\": " << rec.range_p999 << ",\n";
        out << "      \"p9999_abs\": " << rec.range_p9999 << ",\n";
        out << "      \"recommended_range_abs\": " << rec.range_p999 << ",\n";
        out << "      \"recommended_range\": [" << -rec.range_p999 << ", " << rec.range_p999 << "],\n";
        out << "      \"scale_p99\": " << rec.scale_p99 << ",\n";
        out << "      \"scale_p999\": " << rec.scale_p999 << ",\n";
        out << "      \"scale_p9999\": " << rec.scale_p9999 << "\n";
        out << "    }";
    }

    out << "\n  ]\n";
    out << "}\n";

    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.out_file = "vcache-calibration.json";
    params.n_predict = 0;
    params.warmup = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("%s: model path is required\n", __func__);
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: calibration prompts are required, pass -f with one prompt per line\n", __func__);
        return 1;
    }

    std::vector<std::string> prompts = load_prompts(params);
    if (prompts.empty()) {
        LOG_ERR("%s: no non-empty calibration prompts found\n", __func__);
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    vcache_calibration_collector collector;
    params.cb_eval = cb_eval_vcache;
    params.cb_eval_user_data = &collector;

    common_init_result llama_init = common_init_from_params(params);
    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    LOG_INF("%s: loaded %zu calibration prompts from %s\n", __func__, prompts.size(), params.prompt_file.c_str());

    for (size_t i = 0; i < prompts.size(); ++i) {
        llama_memory_clear(llama_get_memory(ctx), true);
        LOG_INF("%s: processing prompt %zu/%zu\n", __func__, i + 1, prompts.size());
        if (!run_prompt(ctx, params, prompts[i])) {
            return 1;
        }
    }

    if (!write_report(params, prompts, collector)) {
        return 1;
    }

    LOG_INF("%s: wrote layer calibration report to %s\n", __func__, params.out_file.c_str());
    for (const auto & kv : collector.layers) {
        const auto rec = kv.second.recommendation();
        LOG_INF("layer %3d: abs_max=%8.5f p99=%8.5f p999=%8.5f suggested=[-%8.5f, %8.5f] scale=%8.5f\n",
                kv.first, kv.second.abs_max, rec.range_p99, rec.range_p999, rec.range_p999, rec.range_p999, rec.scale_p999);
    }

    llama_backend_free();
    return 0;
}
