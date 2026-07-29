#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct config {
    std::filesystem::path output_dir;
    std::string model;
    int device = 0;
    int64_t context_size = 0;
    int64_t n_dims = 128;
    int mode = GGML_ROPE_TYPE_NEOX;
    int n_ctx_orig = 40960;
    float freq_base = 1000000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
};

[[noreturn]] void fail(const std::string & message) {
    std::fprintf(stderr, "%s\n", message.c_str());
    std::exit(1);
}

void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s OUTPUT_DIR CONTEXT_SIZE [options]\n"
            "\n"
            "Options:\n"
            "  --device N\n"
            "  --model PATH\n"
            "  --n-dims N\n"
            "  --mode N\n"
            "  --n-ctx-orig N\n"
            "  --freq-base F\n"
            "  --freq-scale F\n"
            "  --ext-factor F\n"
            "  --attn-factor F\n"
            "  --beta-fast F\n"
            "  --beta-slow F\n",
            argv0);
}

int64_t parse_i64(const char * text, const char * label) {
    errno = 0;
    char * end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fail(std::string("invalid integer for ") + label + ": " + text);
    }
    return (int64_t) value;
}

int parse_i32(const char * text, const char * label) {
    const int64_t value = parse_i64(text, label);
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        fail(std::string("integer out of range for ") + label + ": " + text);
    }
    return (int) value;
}

float parse_f32(const char * text, const char * label) {
    errno = 0;
    char * end = nullptr;
    const float value = std::strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        fail(std::string("invalid float for ") + label + ": " + text);
    }
    return value;
}

config parse_args(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
        std::exit(2);
    }

    config cfg;
    cfg.output_dir = argv[1];
    cfg.context_size = parse_i64(argv[2], "CONTEXT_SIZE");

    for (int i = 3; i < argc; ++i) {
        const std::string opt = argv[i];
        auto require_value = [&](const char * label) -> const char * {
            if (i + 1 >= argc) {
                fail(std::string("missing value for ") + label);
            }
            return argv[++i];
        };

        if (opt == "--device") {
            cfg.device = parse_i32(require_value("--device"), "--device");
        } else if (opt == "--model") {
            cfg.model = require_value("--model");
        } else if (opt == "--n-dims") {
            cfg.n_dims = parse_i64(require_value("--n-dims"), "--n-dims");
        } else if (opt == "--mode") {
            cfg.mode = parse_i32(require_value("--mode"), "--mode");
        } else if (opt == "--n-ctx-orig") {
            cfg.n_ctx_orig = parse_i32(require_value("--n-ctx-orig"), "--n-ctx-orig");
        } else if (opt == "--freq-base") {
            cfg.freq_base = parse_f32(require_value("--freq-base"), "--freq-base");
        } else if (opt == "--freq-scale") {
            cfg.freq_scale = parse_f32(require_value("--freq-scale"), "--freq-scale");
        } else if (opt == "--ext-factor") {
            cfg.ext_factor = parse_f32(require_value("--ext-factor"), "--ext-factor");
        } else if (opt == "--attn-factor") {
            cfg.attn_factor = parse_f32(require_value("--attn-factor"), "--attn-factor");
        } else if (opt == "--beta-fast") {
            cfg.beta_fast = parse_f32(require_value("--beta-fast"), "--beta-fast");
        } else if (opt == "--beta-slow") {
            cfg.beta_slow = parse_f32(require_value("--beta-slow"), "--beta-slow");
        } else if (opt == "--help" || opt == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            fail("unknown option: " + opt);
        }
    }

    if (cfg.context_size <= 0) {
        fail("CONTEXT_SIZE must be positive");
    }
    if (cfg.n_dims <= 0 || cfg.n_dims % 2 != 0) {
        fail("--n-dims must be a positive even integer");
    }
    if (cfg.context_size > std::numeric_limits<int32_t>::max()) {
        fail("CONTEXT_SIZE exceeds I32 position range");
    }

    return cfg;
}

void write_binary(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail("failed to open output binary: " + path.string());
    }
    out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) {
        fail("failed to write output binary: " + path.string());
    }
}

const char * mode_name(int mode) {
    if (mode == GGML_ROPE_TYPE_VISION) {
        return "vision";
    }
    if ((mode & GGML_ROPE_TYPE_MROPE) != 0) {
        return "mrope";
    }
    if ((mode & GGML_ROPE_TYPE_NEOX) != 0) {
        return "neox";
    }
    return "standard";
}

void write_manifest(const std::filesystem::path & path, const config & cfg) {
    const int64_t channels = cfg.n_dims / 2;
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail("failed to open manifest: " + path.string());
    }
    out << "{\n"
        << "  \"format\": \"llama_cuda_rope_cos_sin_v1\",\n"
        << "  \"source\": \"GGML_OP_ROPE CUDA kernel evaluated with basis input\",\n"
        << "  \"model\": \"" << cfg.model << "\",\n"
        << "  \"context_size\": " << cfg.context_size << ",\n"
        << "  \"position_range\": {\"start\": 0, \"end_exclusive\": " << cfg.context_size << "},\n"
        << "  \"channel_idx_range\": {\"start\": 0, \"end_exclusive\": " << channels << "},\n"
        << "  \"shape\": [" << cfg.context_size << ", " << channels << ", 2],\n"
        << "  \"component_order\": [\"cos\", \"sin\"],\n"
        << "  \"dtype\": \"f32_le\",\n"
        << "  \"layout\": \"position_major_channel_major_component\",\n"
        << "  \"byte_offset\": \"((position * " << channels << " + channel_idx) * 2 + component) * 4\",\n"
        << "  \"data_file\": \"rope-cos-sin-f32.bin\",\n"
        << "  \"rope_params\": {\n"
        << "    \"mode\": " << cfg.mode << ",\n"
        << "    \"mode_name\": \"" << mode_name(cfg.mode) << "\",\n"
        << "    \"n_dims\": " << cfg.n_dims << ",\n"
        << "    \"n_ctx_orig\": " << cfg.n_ctx_orig << ",\n"
        << "    \"freq_base\": " << cfg.freq_base << ",\n"
        << "    \"freq_scale\": " << cfg.freq_scale << ",\n"
        << "    \"ext_factor\": " << cfg.ext_factor << ",\n"
        << "    \"attn_factor\": " << cfg.attn_factor << ",\n"
        << "    \"beta_fast\": " << cfg.beta_fast << ",\n"
        << "    \"beta_slow\": " << cfg.beta_slow << ",\n"
        << "    \"freq_factors\": null\n"
        << "  }\n"
        << "}\n";
    if (!out) {
        fail("failed to write manifest: " + path.string());
    }
}

} // namespace

int main(int argc, char ** argv) {
    const config cfg = parse_args(argc, argv);
    const int64_t channels = cfg.n_dims / 2;

    std::filesystem::create_directories(cfg.output_dir);

    ggml_backend_t backend = ggml_backend_cuda_init(cfg.device);
    if (!backend) {
        fail("failed to initialize CUDA backend " + std::to_string(cfg.device));
    }

    ggml_init_params params = {};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * input = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, cfg.n_dims, 1, cfg.context_size, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, cfg.context_size);
    ggml_tensor * rope = ggml_rope_ext(
            ctx, input, positions, nullptr, cfg.n_dims, cfg.mode, cfg.n_ctx_orig,
            cfg.freq_base, cfg.freq_scale, cfg.ext_factor, cfg.attn_factor,
            cfg.beta_fast, cfg.beta_slow);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, rope);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fail("failed to allocate CUDA tensors");
    }

    std::vector<float> input_values(static_cast<size_t>(cfg.n_dims * cfg.context_size), 0.0f);
    for (int64_t position = 0; position < cfg.context_size; ++position) {
        const size_t row = static_cast<size_t>(position * cfg.n_dims);
        for (int64_t channel = 0; channel < channels; ++channel) {
            input_values[row + static_cast<size_t>(channel)] = 1.0f;
        }
    }
    std::vector<int32_t> position_values(static_cast<size_t>(cfg.context_size));
    for (int64_t position = 0; position < cfg.context_size; ++position) {
        position_values[static_cast<size_t>(position)] = (int32_t) position;
    }

    ggml_backend_tensor_set(input, input_values.data(), 0,
            input_values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0,
            position_values.size() * sizeof(int32_t));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "CUDA graph compute failed: %s\n", ggml_status_to_string(status));
        return 1;
    }

    std::vector<float> rope_values(static_cast<size_t>(cfg.n_dims * cfg.context_size));
    ggml_backend_tensor_get(rope, rope_values.data(), 0,
            rope_values.size() * sizeof(float));

    std::vector<float> interleaved(static_cast<size_t>(cfg.context_size * channels * 2));
    for (int64_t position = 0; position < cfg.context_size; ++position) {
        const size_t source_row = static_cast<size_t>(position * cfg.n_dims);
        for (int64_t channel = 0; channel < channels; ++channel) {
            const size_t target = static_cast<size_t>((position * channels + channel) * 2);
            interleaved[target + 0] = rope_values[source_row + static_cast<size_t>(channel)];
            interleaved[target + 1] = rope_values[source_row + static_cast<size_t>(channels + channel)];
        }
    }

    write_binary(cfg.output_dir / "rope-cos-sin-f32.bin", interleaved);
    write_manifest(cfg.output_dir / "manifest.json", cfg);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::printf("exported context_size=%lld channels=%lld values=%zu to %s\n",
            (long long) cfg.context_size, (long long) channels,
            interleaved.size(), cfg.output_dir.string().c_str());
    return 0;
}
