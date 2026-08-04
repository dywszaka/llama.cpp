#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int64_t CONTEXT_SIZE = 8192;
constexpr int64_t N_DIMS = 128;
constexpr int64_t CHANNELS = N_DIMS / 2;
constexpr int ROPE_MODE = GGML_ROPE_TYPE_NEOX;
constexpr int N_CTX_ORIG = 40960;
constexpr float FREQ_BASE = 1000000.0f;
constexpr float FREQ_SCALE = 1.0f;
constexpr float EXT_FACTOR = 0.0f;
constexpr float ATTN_FACTOR = 1.0f;
constexpr float BETA_FAST = 32.0f;
constexpr float BETA_SLOW = 1.0f;

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

void write_binary(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail("failed to open output binary");
    }
    out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) {
        fail("failed to write output binary");
    }
}

void write_manifest(const std::filesystem::path & path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail("failed to open manifest");
    }
    out << "{\n"
        << "  \"format\": \"llama_cuda_rope_cos_sin_v1\",\n"
        << "  \"source\": \"GGML_OP_ROPE CUDA kernel evaluated with GPT-NeoX basis input\",\n"
        << "  \"model\": \"/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf\",\n"
        << "  \"context_size\": " << CONTEXT_SIZE << ",\n"
        << "  \"position_range\": {\"start\": 0, \"end_exclusive\": " << CONTEXT_SIZE << "},\n"
        << "  \"channel_idx_range\": {\"start\": 0, \"end_exclusive\": " << CHANNELS << "},\n"
        << "  \"shape\": [" << CONTEXT_SIZE << ", " << CHANNELS << ", 2],\n"
        << "  \"component_order\": [\"cos\", \"sin\"],\n"
        << "  \"dtype\": \"f32_le\",\n"
        << "  \"layout\": \"position_major_channel_major_component\",\n"
        << "  \"byte_offset\": \"((position * 64 + channel_idx) * 2 + component) * 4\",\n"
        << "  \"data_file\": \"rope-cos-sin-f32.bin\",\n"
        << "  \"rope_params\": {\n"
        << "    \"mode\": " << ROPE_MODE << ",\n"
        << "    \"mode_name\": \"neox\",\n"
        << "    \"n_dims\": " << N_DIMS << ",\n"
        << "    \"n_ctx_orig\": " << N_CTX_ORIG << ",\n"
        << "    \"freq_base\": " << FREQ_BASE << ",\n"
        << "    \"freq_scale\": " << FREQ_SCALE << ",\n"
        << "    \"ext_factor\": " << EXT_FACTOR << ",\n"
        << "    \"attn_factor\": " << ATTN_FACTOR << ",\n"
        << "    \"beta_fast\": " << BETA_FAST << ",\n"
        << "    \"beta_slow\": " << BETA_SLOW << ",\n"
        << "    \"freq_factors\": null\n"
        << "  }\n"
        << "}\n";
    if (!out) {
        fail("failed to write manifest");
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s OUTPUT_DIR\n", argv[0]);
        return 2;
    }

    const std::filesystem::path output_dir(argv[1]);
    std::filesystem::create_directories(output_dir);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fail("failed to initialize CUDA backend 0");
    }

    ggml_init_params params = {};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("failed to initialize ggml context");
    }

    ggml_tensor * input = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, N_DIMS, 1, CONTEXT_SIZE, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(
            ctx, GGML_TYPE_I32, CONTEXT_SIZE);
    ggml_tensor * rope = ggml_rope_ext(
            ctx, input, positions, nullptr, N_DIMS, ROPE_MODE, N_CTX_ORIG,
            FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR, BETA_FAST, BETA_SLOW);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, rope);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fail("failed to allocate CUDA tensors");
    }

    std::vector<float> input_values(static_cast<size_t>(N_DIMS * CONTEXT_SIZE), 0.0f);
    for (int64_t position = 0; position < CONTEXT_SIZE; ++position) {
        const size_t row = static_cast<size_t>(position * N_DIMS);
        for (int64_t channel = 0; channel < CHANNELS; ++channel) {
            input_values[row + static_cast<size_t>(channel)] = 1.0f;
        }
    }
    std::vector<int32_t> position_values(static_cast<size_t>(CONTEXT_SIZE));
    for (int32_t position = 0; position < CONTEXT_SIZE; ++position) {
        position_values[static_cast<size_t>(position)] = position;
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

    std::vector<float> rope_values(static_cast<size_t>(N_DIMS * CONTEXT_SIZE));
    ggml_backend_tensor_get(rope, rope_values.data(), 0,
            rope_values.size() * sizeof(float));

    std::vector<float> interleaved(static_cast<size_t>(CONTEXT_SIZE * CHANNELS * 2));
    for (int64_t position = 0; position < CONTEXT_SIZE; ++position) {
        const size_t source_row = static_cast<size_t>(position * N_DIMS);
        for (int64_t channel = 0; channel < CHANNELS; ++channel) {
            const size_t target = static_cast<size_t>((position * CHANNELS + channel) * 2);
            interleaved[target + 0] = rope_values[source_row + static_cast<size_t>(channel)];
            interleaved[target + 1] = rope_values[source_row + static_cast<size_t>(CHANNELS + channel)];
        }
    }

    write_binary(output_dir / "rope-cos-sin-f32.bin", interleaved);
    write_manifest(output_dir / "manifest.json");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    std::printf("exported %zu F32 values to %s\n", interleaved.size(), output_dir.string().c_str());
    return 0;
}
