#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f prompt.txt -o kcache-mean.jsonl -n 128 -ctk f16 -ctv f16 --kv-unified\n", argv[0]);
    LOG("\n    --include-prompt    include prompt-token K-cache appends in the running means\n");
    LOG("    --generated-only    record generated-token K-cache appends only (default)\n");
    LOG("    --dump-mean-vectors-every N  include full per-channel running mean vectors at step 1 and every N append steps\n");
    LOG("    --dump-final-mean-vectors    include full per-channel running mean vectors for all layers after the final append\n");
    LOG("    --layers L                   comma-separated cache_k layers to collect\n");
    LOG("    --raw-dump-dir DIR           write raw F32 K rows to DIR/layer_XX.f32.bin\n");
    LOG("    --histogram-json FILE         write aggregate per-layer distribution metrics as JSON\n");
    LOG("\n");
}

static bool parse_cache_k_layer(const char * name, int & layer) {
    const char prefix[] = "cache_k_l";
    const size_t prefix_len = sizeof(prefix) - 1;
    if (std::strncmp(name, prefix, prefix_len) != 0) {
        return false;
    }

    char * end = nullptr;
    const long value = std::strtol(name + prefix_len, &end, 10);
    if (end == name + prefix_len || value < 0) {
        return false;
    }

    layer = (int) value;
    return true;
}

struct layer_running_stats {
    std::vector<double> channel_sums;
    std::vector<uint64_t> abs_bins;
    std::vector<uint64_t> channel_outlier_counts;
    std::ofstream raw;
    int64_t n_seen = 0;
    uint64_t total_values = 0;
    uint64_t total_outliers_gt16 = 0;
    double sum = 0.0;
    double sum_sq = 0.0;
    double abs_sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double abs_max = 0.0;
};

struct kcache_mean_collector {
    static constexpr int N_BINS = 17;

    std::ofstream out;
    std::vector<uint8_t> scratch;
    std::vector<float> values;
    std::vector<double> running_means;
    std::map<int, layer_running_stats> layers;
    std::set<int> layer_filter;

    bool recording = false;
    std::string phase = "generated";
    int64_t step = 0;
    int64_t phase_step = 0;
    int32_t token = -1;
    int64_t dump_mean_vectors_every = 0;
    bool dump_mean_vectors_current_step = false;
    bool include_prompt_batches = false;
    std::string raw_dump_dir;
    std::string histogram_json_path;

    bool open(const std::string & path) {
        out.open(path);
        return (bool) out;
    }

    void begin_token(const char * phase_in, int64_t step_in, int64_t phase_step_in, int32_t token_in) {
        recording = true;
        phase = phase_in;
        step = step_in;
        phase_step = phase_step_in;
        token = token_in;
    }

    void end_token() {
        recording = false;
        token = -1;
    }

    void begin_batch(const char * phase_in, int64_t step_in, int64_t phase_step_in) {
        recording = true;
        phase = phase_in;
        step = step_in;
        phase_step = phase_step_in;
        token = -1;
    }

    bool collect(struct ggml_tensor * t, bool ask) {
        int layer = -1;
        const bool wants = t->op == GGML_OP_SET_ROWS && parse_cache_k_layer(t->name, layer);
        if (ask) {
            return recording && wants && wants_layer(layer);
        }

        if (!recording || !wants || !wants_layer(layer)) {
            return true;
        }

        ggml_tensor * src = t->src[0];
        if (src == nullptr) {
            return true;
        }

        const int64_t n_channels = src->ne[0];
        const int64_t n_tokens = src->ne[1] * src->ne[2] * src->ne[3];
        if (n_channels <= 0 || n_tokens <= 0) {
            LOG_WRN("%s: skipping %s source shape [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                    __func__, t->name, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
            return true;
        }

        if (!read_tensor_as_f32(src, values)) {
            LOG_WRN("%s: skipping %s source type %s\n", __func__, t->name, ggml_type_name(src->type));
            return true;
        }

        auto & stats = layers[layer];
        if (stats.channel_sums.empty()) {
            stats.channel_sums.assign((size_t) n_channels, 0.0);
            stats.channel_outlier_counts.assign((size_t) n_channels, 0);
            stats.abs_bins.assign(N_BINS, 0);
        } else if ((int64_t) stats.channel_sums.size() != n_channels) {
            LOG_ERR("%s: channel count changed for layer %d: have %zu, got %" PRId64 "\n",
                    __func__, layer, stats.channel_sums.size(), n_channels);
            return false;
        }

        if (!raw_dump_dir.empty() && !ensure_raw_open(layer, stats)) {
            return false;
        }

        if (stats.raw.is_open()) {
            stats.raw.write((const char *) values.data(), (std::streamsize) (values.size() * sizeof(float)));
            if (!stats.raw) {
                LOG_ERR("%s: failed writing raw dump for layer %d\n", __func__, layer);
                return false;
            }
        }

        if (raw_dump_dir.empty() && histogram_json_path.empty()) {
            apply_cache_precision(t->type, values);
        }

        double token_sum = 0.0;
        double token_sum_sq = 0.0;
        double token_abs_max = 0.0;
        double token_min = std::numeric_limits<double>::infinity();
        double token_max = -std::numeric_limits<double>::infinity();
        uint64_t bins[N_BINS] = {};
        for (int64_t row = 0; row < n_tokens; ++row) {
            for (int64_t i = 0; i < n_channels; ++i) {
                const float v = values[(size_t) row * (size_t) n_channels + (size_t) i];
                const double vd = (double) v;
                const double av = std::fabs(vd);
                token_sum += vd;
                token_sum_sq += vd * vd;
                token_abs_max = std::max(token_abs_max, av);
                token_min = std::min(token_min, vd);
                token_max = std::max(token_max, vd);
                const int bin = abs_bin(v);
                bins[bin]++;
                stats.abs_bins[(size_t) bin]++;
                stats.channel_sums[(size_t) i] += vd;
                stats.sum += vd;
                stats.sum_sq += vd * vd;
                stats.abs_sum += av;
                stats.min = std::min(stats.min, vd);
                stats.max = std::max(stats.max, vd);
                stats.abs_max = std::max(stats.abs_max, av);
                stats.total_values++;
                if (av > 16.0) {
                    stats.total_outliers_gt16++;
                    stats.channel_outlier_counts[(size_t) i]++;
                }
            }
        }
        stats.n_seen += n_tokens;

        const double token_values = (double) n_channels * (double) n_tokens;
        const double token_mean = token_sum / token_values;
        const double token_var = std::max(0.0, token_sum_sq / token_values - token_mean * token_mean);

        double running_min = std::numeric_limits<double>::infinity();
        double running_max = -std::numeric_limits<double>::infinity();
        double running_abs_sum = 0.0;
        double running_sum_sq = 0.0;
        double delta_abs_sum = 0.0;
        double delta_abs_max = 0.0;
        double delta_sum_sq = 0.0;
        const double n_seen = (double) stats.n_seen;
        const double n_prev = (double) (stats.n_seen - 1);
        const bool dump_mean_vector = dump_mean_vectors_current_step ||
                (dump_mean_vectors_every > 0 && (step == 1 || step % dump_mean_vectors_every == 0));
        if (dump_mean_vector) {
            running_means.resize((size_t) n_channels);
        }
        for (int64_t i = 0; i < n_channels; ++i) {
            const double mean = stats.channel_sums[(size_t) i] / n_seen;
            if (dump_mean_vector) {
                running_means[(size_t) i] = mean;
            }
            running_min = std::min(running_min, mean);
            running_max = std::max(running_max, mean);
            running_abs_sum += std::fabs(mean);
            running_sum_sq += mean * mean;

            double delta = mean;
            if (stats.n_seen > n_tokens) {
                double current_batch_sum = 0.0;
                for (int64_t row = 0; row < n_tokens; ++row) {
                    current_batch_sum += (double) values[(size_t) row * (size_t) n_channels + (size_t) i];
                }
                const double prev_mean = (stats.channel_sums[(size_t) i] - current_batch_sum) / n_prev;
                delta = mean - prev_mean;
            }
            const double abs_delta = std::fabs(delta);
            delta_abs_sum += abs_delta;
            delta_abs_max = std::max(delta_abs_max, abs_delta);
            delta_sum_sq += delta * delta;
        }

        const double running_rms = std::sqrt(running_sum_sq / (double) n_channels);
        const double delta_rms = std::sqrt(delta_sum_sq / (double) n_channels);

        out << "{\"phase\":\"" << phase << "\""
            << ",\"step\":" << step
            << ",\"phase_step\":" << phase_step
            << ",\"token\":" << token
            << ",\"layer\":" << layer
            << ",\"n_channels\":" << n_channels
            << ",\"n_rows\":" << n_tokens
            << ",\"layer_tokens_seen\":" << stats.n_seen
            << ",\"token_min\":" << token_min
            << ",\"token_max\":" << token_max
            << ",\"token_channel_mean\":" << token_mean
            << ",\"token_stddev\":" << std::sqrt(token_var)
            << ",\"token_abs_max\":" << token_abs_max
            << ",\"running_channel_mean_min\":" << running_min
            << ",\"running_channel_mean_max\":" << running_max
            << ",\"running_channel_mean_span\":" << (running_max - running_min)
            << ",\"running_channel_mean_abs_mean\":" << (running_abs_sum / (double) n_channels)
            << ",\"running_channel_mean_rms\":" << running_rms
            << ",\"running_delta_abs_mean\":" << (delta_abs_sum / (double) n_channels)
            << ",\"running_delta_abs_max\":" << delta_abs_max
            << ",\"running_delta_rms\":" << delta_rms
            << ",\"abs_bins\":[";
        for (int i = 0; i < N_BINS; ++i) {
            if (i > 0) {
                out << ",";
            }
            out << bins[i];
        }
        out << "]";
        if (dump_mean_vector) {
            out << ",\"running_channel_means\":[";
            for (int64_t i = 0; i < n_channels; ++i) {
                if (i > 0) {
                    out << ",";
                }
                out << running_means[(size_t) i];
            }
            out << "]";
        }
        out << "}\n";

        return true;
    }

    bool wants_layer(int layer) const {
        return layer_filter.empty() || layer_filter.count(layer) != 0;
    }

    static int abs_bin(float value) {
        const double a = std::fabs((double) value);
        if (a == 0.0) {
            return 0;
        }
        if (a < 0x1p-16) {
            return 1;
        }
        if (a < 0x1p-12) {
            return 2;
        }
        if (a < 0x1p-8) {
            return 3;
        }
        if (a < 0x1p-4) {
            return 4;
        }
        if (a < 0.25) {
            return 5;
        }
        if (a < 0.5) {
            return 6;
        }
        if (a < 1.0) {
            return 7;
        }
        if (a < 2.0) {
            return 8;
        }
        if (a < 4.0) {
            return 9;
        }
        if (a < 8.0) {
            return 10;
        }
        if (a < 16.0) {
            return 11;
        }
        if (a < 24.0) {
            return 12;
        }
        if (a < 32.0) {
            return 13;
        }
        if (a < 48.0) {
            return 14;
        }
        if (a < 64.0) {
            return 15;
        }
        return 16;
    }

    bool read_tensor_as_f32(const ggml_tensor * t, std::vector<float> & out_values) {
        const int64_t n = ggml_nelements(t);
        if (n <= 0) {
            out_values.clear();
            return true;
        }

        if (t->type == GGML_TYPE_F32) {
            out_values.resize((size_t) n);
            ggml_backend_tensor_get(t, out_values.data(), 0, out_values.size() * sizeof(float));
            return true;
        }

        if (t->type == GGML_TYPE_F16) {
            scratch.resize((size_t) n * sizeof(ggml_fp16_t));
            out_values.resize((size_t) n);
            ggml_backend_tensor_get(t, scratch.data(), 0, scratch.size());
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) scratch.data(), out_values.data(), n);
            return true;
        }

        if (t->type == GGML_TYPE_BF16) {
            scratch.resize((size_t) n * sizeof(ggml_bf16_t));
            out_values.resize((size_t) n);
            ggml_backend_tensor_get(t, scratch.data(), 0, scratch.size());
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) scratch.data(), out_values.data(), n);
            return true;
        }

        return false;
    }

    static void apply_cache_precision(enum ggml_type cache_type, std::vector<float> & out_values) {
        if (cache_type == GGML_TYPE_F16) {
            for (float & v : out_values) {
                v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
            }
        } else if (cache_type == GGML_TYPE_BF16) {
            for (float & v : out_values) {
                v = ggml_bf16_to_fp32(ggml_fp32_to_bf16(v));
            }
        }
    }

    static const char * abs_bin_label(int idx) {
        static const char * labels[N_BINS] = {
            "0",
            "(0,2^-16)",
            "[2^-16,2^-12)",
            "[2^-12,2^-8)",
            "[2^-8,2^-4)",
            "[2^-4,0.25)",
            "[0.25,0.5)",
            "[0.5,1)",
            "[1,2)",
            "[2,4)",
            "[4,8)",
            "[8,16)",
            "[16,24)",
            "[24,32)",
            "[32,48)",
            "[48,64)",
            "[64,inf)"
        };
        return labels[idx];
    }

    bool ensure_raw_open(int layer, layer_running_stats & stats) {
        if (stats.raw.is_open()) {
            return true;
        }

        char path[1024];
        std::snprintf(path, sizeof(path), "%s/layer_%02d.f32.bin", raw_dump_dir.c_str(), layer);
        stats.raw.open(path, std::ios::binary);
        if (!stats.raw) {
            LOG_ERR("%s: failed to open raw dump '%s'\n", __func__, path);
            return false;
        }
        return true;
    }

    bool write_histogram_json() {
        if (histogram_json_path.empty()) {
            return true;
        }

        std::ofstream json(histogram_json_path);
        if (!json) {
            LOG_ERR("%s: failed to open histogram JSON '%s'\n", __func__, histogram_json_path.c_str());
            return false;
        }

        json << "{\n";
        json << "  \"abs_bin_labels\": [";
        for (int i = 0; i < N_BINS; ++i) {
            if (i > 0) {
                json << ", ";
            }
            json << "\"" << abs_bin_label(i) << "\"";
        }
        json << "],\n";
        json << "  \"layers\": [\n";

        bool first_layer = true;
        for (const auto & it : layers) {
            const int layer = it.first;
            const layer_running_stats & stats = it.second;
            if (!first_layer) {
                json << ",\n";
            }
            first_layer = false;

            const double n = (double) std::max<uint64_t>(stats.total_values, 1);
            const double mean = stats.sum / n;
            const double variance = std::max(0.0, stats.sum_sq / n - mean * mean);
            json << "    {";
            json << "\"layer\": " << layer;
            json << ", \"rows\": " << stats.n_seen;
            json << ", \"total_values\": " << stats.total_values;
            json << ", \"mean\": " << mean;
            json << ", \"stddev\": " << std::sqrt(variance);
            json << ", \"abs_mean\": " << (stats.abs_sum / n);
            json << ", \"min\": " << stats.min;
            json << ", \"max\": " << stats.max;
            json << ", \"abs_max\": " << stats.abs_max;
            json << ", \"outliers_gt16\": " << stats.total_outliers_gt16;
            json << ", \"outlier_pct_gt16\": " << (100.0 * (double) stats.total_outliers_gt16 / n);
            json << ", \"abs_bins\": [";
            for (size_t i = 0; i < stats.abs_bins.size(); ++i) {
                if (i > 0) {
                    json << ", ";
                }
                json << stats.abs_bins[i];
            }
            json << "]";

            std::vector<std::pair<uint64_t, int>> channel_counts;
            channel_counts.reserve(stats.channel_outlier_counts.size());
            for (size_t i = 0; i < stats.channel_outlier_counts.size(); ++i) {
                channel_counts.emplace_back(stats.channel_outlier_counts[i], (int) i);
            }
            std::sort(channel_counts.begin(), channel_counts.end(),
                    [](const auto & a, const auto & b) {
                        if (a.first != b.first) {
                            return a.first > b.first;
                        }
                        return a.second < b.second;
                    });

            json << ", \"top_channels_gt16\": [";
            const size_t n_top = std::min<size_t>(16, channel_counts.size());
            for (size_t i = 0; i < n_top; ++i) {
                if (i > 0) {
                    json << ", ";
                }
                json << "{\"channel\": " << channel_counts[i].second
                     << ", \"count\": " << channel_counts[i].first << "}";
            }
            json << "]";
            json << "}";
        }

        json << "\n  ]\n";
        json << "}\n";
        return (bool) json;
    }
};

static bool cb_eval_kcache_mean(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * collector = (kcache_mean_collector *) user_data;
    return collector->collect(t, ask);
}

static bool decode_tokens(llama_context * ctx, const llama_token * tokens, int32_t n_tokens, int32_t n_batch) {
    int32_t pos = 0;
    while (pos < n_tokens) {
        const int32_t n_eval = std::min<int32_t>(n_batch, n_tokens - pos);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens + pos), n_eval);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed at token offset %d/%d\n", __func__, pos, n_tokens);
            return false;
        }
        pos += n_eval;
    }
    return true;
}

static bool decode_recorded_token(
        llama_context * ctx,
        kcache_mean_collector & collector,
        const char * phase,
        int64_t step,
        int64_t phase_step,
        llama_token token) {
    collector.begin_token(phase, step, phase_step, token);
    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode failed while appending %s token at step %" PRId64 "\n",
                __func__, phase, step);
        collector.end_token();
        return false;
    }
    collector.end_token();
    return true;
}

static bool decode_recorded_prompt_chunks(
        llama_context * ctx,
        kcache_mean_collector & collector,
        const std::vector<llama_token> & tokens,
        int32_t n_ctx,
        int32_t n_batch) {
    if (n_ctx <= 0 || n_batch <= 0) {
        return false;
    }

    const int32_t n_tokens = (int32_t) tokens.size();
    int64_t step = 0;
    for (int32_t start = 0; start < n_tokens; start += n_ctx) {
        const int32_t end = std::min<int32_t>(start + n_ctx, n_tokens);
        llama_memory_clear(llama_get_memory(ctx), true);

        for (int32_t pos = start; pos < end; pos += n_batch) {
            const int32_t n_eval = std::min<int32_t>(n_batch, end - pos);
            step += n_eval;
            collector.begin_batch("prompt", step, pos + n_eval);
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + pos), n_eval);
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: llama_decode failed at token offset %d/%d\n", __func__, pos, n_tokens);
                collector.end_token();
                return false;
            }
            collector.end_token();
        }
    }

    return true;
}

static bool run(
        llama_context * ctx,
        const common_params & params,
        kcache_mean_collector & collector,
        bool include_prompt,
        bool dump_final_mean_vectors) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, params.prompt, add_bos, params.special);
    if (prompt_tokens.empty()) {
        LOG_ERR("%s: empty prompt after tokenization\n", __func__);
        return false;
    }

    int64_t append_step = 0;
    const int64_t final_append_step = (include_prompt ? (int64_t) prompt_tokens.size() : 0) + params.n_predict;
    if (include_prompt) {
        if (params.n_predict <= 0 && !dump_final_mean_vectors) {
            if (!decode_recorded_prompt_chunks(ctx, collector, prompt_tokens, params.n_ctx, std::max<int32_t>(1, params.n_batch))) {
                return false;
            }
            LOG_INF("%s: prompt_tokens=%zu generated_tokens=0 output=%s\n",
                    __func__, prompt_tokens.size(), params.out_file.c_str());
            return true;
        } else {
            for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                append_step++;
                collector.dump_mean_vectors_current_step = dump_final_mean_vectors && append_step == final_append_step;
                if (!decode_recorded_token(ctx, collector, "prompt", append_step, (int64_t) i + 1, prompt_tokens[i])) {
                    collector.dump_mean_vectors_current_step = false;
                    return false;
                }
                collector.dump_mean_vectors_current_step = false;
            }
        }
    } else {
        if (!decode_tokens(ctx, prompt_tokens.data(), (int32_t) prompt_tokens.size(), std::max<int32_t>(1, params.n_batch))) {
            return false;
        }
    }

    common_sampler * sampler = common_sampler_init(model, params.sampling);
    if (sampler == nullptr) {
        LOG_ERR("%s: failed to initialize sampler\n", __func__);
        return false;
    }

    for (llama_token tok : prompt_tokens) {
        common_sampler_accept(sampler, tok, false);
    }

    std::vector<llama_token> generated;
    generated.reserve((size_t) params.n_predict);
    bool ok = true;

    for (int32_t i = 0; i < params.n_predict; ++i) {
        const llama_token id = common_sampler_sample(sampler, ctx, -1);
        common_sampler_accept(sampler, id, true);
        generated.push_back(id);

        append_step++;
        collector.dump_mean_vectors_current_step = dump_final_mean_vectors && append_step == final_append_step;
        if (!decode_recorded_token(ctx, collector, "generated", append_step, (int64_t) i + 1, generated.back())) {
            collector.dump_mean_vectors_current_step = false;
            ok = false;
            break;
        }
        collector.dump_mean_vectors_current_step = false;
    }

    common_sampler_free(sampler);

    LOG_INF("%s: prompt_tokens=%zu generated_tokens=%zu output=%s\n",
            __func__, prompt_tokens.size(), generated.size(), params.out_file.c_str());

    return ok && (int32_t) generated.size() == params.n_predict;
}

static bool parse_kcache_mean_args(
        int argc,
        char ** argv,
        bool & include_prompt,
        int64_t & dump_mean_vectors_every,
        bool & dump_final_mean_vectors,
        std::set<int> & layer_filter,
        std::string & raw_dump_dir,
        std::string & histogram_json_path,
        std::vector<char *> & filtered_argv) {
    include_prompt = false;
    dump_mean_vectors_every = 0;
    dump_final_mean_vectors = false;
    layer_filter.clear();
    raw_dump_dir.clear();
    histogram_json_path.clear();
    filtered_argv.clear();
    filtered_argv.reserve((size_t) argc);
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--include-prompt") {
            include_prompt = true;
            continue;
        }
        if (arg == "--generated-only") {
            include_prompt = false;
            continue;
        }
        if (arg == "--dump-mean-vectors-every") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: expected value for --dump-mean-vectors-every\n");
                return false;
            }
            dump_mean_vectors_every = std::stoll(argv[++i]);
            if (dump_mean_vectors_every <= 0) {
                fprintf(stderr, "error: --dump-mean-vectors-every must be > 0\n");
                return false;
            }
            continue;
        }
        if (arg == "--dump-final-mean-vectors") {
            dump_final_mean_vectors = true;
            continue;
        }
        if (arg == "--layers") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: expected value for --layers\n");
                return false;
            }
            std::string spec = argv[++i];
            size_t pos = 0;
            while (pos <= spec.size()) {
                const size_t comma = spec.find(',', pos);
                const std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!item.empty()) {
                    char * end = nullptr;
                    const long parsed = std::strtol(item.c_str(), &end, 10);
                    if (end == item.c_str() || parsed < 0) {
                        fprintf(stderr, "error: invalid layer id in --layers: %s\n", item.c_str());
                        return false;
                    }
                    layer_filter.insert((int) parsed);
                }
                if (comma == std::string::npos) {
                    break;
                }
                pos = comma + 1;
            }
            continue;
        }
        if (arg == "--raw-dump-dir") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: expected value for --raw-dump-dir\n");
                return false;
            }
            raw_dump_dir = argv[++i];
            continue;
        }
        if (arg == "--histogram-json") {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: expected value for --histogram-json\n");
                return false;
            }
            histogram_json_path = argv[++i];
            continue;
        }
        filtered_argv.push_back(argv[i]);
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.out_file = "kcache-mean.jsonl";
    params.n_predict = 128;
    params.warmup = false;

    bool include_prompt = false;
    int64_t dump_mean_vectors_every = 0;
    bool dump_final_mean_vectors = false;
    std::set<int> layer_filter;
    std::string raw_dump_dir;
    std::string histogram_json_path;
    std::vector<char *> filtered_argv;
    if (!parse_kcache_mean_args(argc, argv, include_prompt, dump_mean_vectors_every, dump_final_mean_vectors,
                layer_filter, raw_dump_dir, histogram_json_path, filtered_argv)) {
        return 1;
    }

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("%s: model path is required\n", __func__);
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: prompt is required\n", __func__);
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    kcache_mean_collector collector;
    collector.dump_mean_vectors_every = dump_mean_vectors_every;
    collector.include_prompt_batches = include_prompt;
    collector.layer_filter = layer_filter;
    collector.raw_dump_dir = raw_dump_dir;
    collector.histogram_json_path = histogram_json_path;
    if (!collector.open(params.out_file)) {
        LOG_ERR("%s: failed to open output file '%s'\n", __func__, params.out_file.c_str());
        return 1;
    }

    params.cb_eval = cb_eval_kcache_mean;
    params.cb_eval_user_data = &collector;
    params.warmup = false;

    common_init_result llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init.context.get();
    if (llama_init.model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    LOG_INF("%s: collecting %s K-cache channel means\n",
            __func__, include_prompt ? "prompt+generated-token" : "generated-token-only");

    const bool ok = run(ctx, params, collector, include_prompt, dump_final_mean_vectors);
    if (!collector.write_histogram_json()) {
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return ok ? 0 : 1;
}
