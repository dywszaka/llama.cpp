#include <cuda_runtime.h>

#include "../ggml/src/ggml-cuda/expt/bf16-expp/bf16-expp.cuh"
#include "../c100-sim/ext/riscv-isa-sim-lib-demo/src/top/custom/riscv/custom_expp.hpp"

#include <ggml-backend.h>
#include <ggml-cuda.h>
#include <ggml.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct expp_case {
    uint32_t input;
    uint16_t expected;
};

static constexpr expp_case CASES[] = {
    {0x00000000u, 0x3f80u},
    {0x80000000u, 0x3f80u},
    {0x00010000u, 0x3f80u},
    {0x80010000u, 0x3f80u},
    {0x7fc10000u, 0x7fc0u},
    {0x7f800000u, 0x7f80u},
    {0xff800000u, 0x0000u},
    {0xbf800000u, 0x3ebcu},
    {0xbf000000u, 0x3f1cu},
    {0x3f000000u, 0x3fd3u},
    {0x3f800000u, 0x402eu},
    {0x41200000u, 0x46acu},
    {0x42b10000u, 0x7f4du},
    {0x42b20000u, 0x7f80u},
    {0xc2ae0000u, 0x00b3u},
    {0xc2af0000u, 0x0000u},
    {0x3f80ffffu, 0x402eu},
};

static_assert(ggml_cuda_bf16_expp_bits(0x3f80u) == 0x402eu, "exp(1) mismatch");
static_assert(ggml_cuda_bf16_expp_bits(0xbf80u) == 0x3ebcu, "exp(-1) mismatch");
static_assert(ggml_cuda_bf16_expp_bits(0x42b2u) == 0x7f80u, "overflow mismatch");

__global__ void expp_kernel(const expp_case * cases, uint32_t * output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float input = __uint_as_float(cases[i].input);
        output[i] = __float_as_uint(ggml_cuda_bf16_expp_f32(input));
    }
}

static bool run_exhaustive_host_test() {
    BF16ExppUnit_Model reference;
    for (uint32_t input = 0; input <= 0xffffu; ++input) {
        const uint16_t expected = reference.process(uint16_t(input));
        const uint16_t actual = ggml_cuda_bf16_expp_bits(uint16_t(input));
        if (actual != expected) {
            std::fprintf(stderr, "host case 0x%04x: got 0x%04x, expected 0x%04x\n",
                         input, uint32_t(actual), uint32_t(expected));
            return false;
        }
    }
    return true;
}

static bool run_softmax_switch_test(bool use_bf16_exp) {
#if defined(_WIN32)
    _putenv_s("GGML_CUDA_SOFTMAX_BF16_EXP", use_bf16_exp ? "1" : "0");
#else
    setenv("GGML_CUDA_SOFTMAX_BF16_EXP", use_bf16_exp ? "1" : "0", 1);
#endif

    ggml_init_params params = {
        4 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "failed to initialize CUDA backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    ggml_tensor * output_tensor = ggml_soft_max(ctx, input);
    ggml_tensor * sinks = nullptr;
    if (use_bf16_exp) {
        sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_soft_max_add_sinks(output_tensor, sinks);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, output_tensor);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate CUDA tensors\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    const float input_data[2] = {-1.0f, 0.0f};
    ggml_backend_tensor_set(input, input_data, 0, sizeof(input_data));
    if (sinks != nullptr) {
        const float sink_data = -0.5f;
        ggml_backend_tensor_set(sinks, &sink_data, 0, sizeof(sink_data));
    }
    const ggml_status status = ggml_backend_graph_compute(backend, graph);

    float actual[2] = {};
    ggml_backend_tensor_get(output_tensor, actual, 0, sizeof(actual));

    const float exp_negative_one = use_bf16_exp ? 0.3671875f : std::exp(-1.0f);
    const float exp_sink = use_bf16_exp ? 0.609375f : 0.0f;
    const float sum = exp_negative_one + 1.0f + exp_sink;
    const float expected[2] = {exp_negative_one / sum, 1.0f / sum};
    const bool ok = status == GGML_STATUS_SUCCESS &&
        std::fabs(actual[0] - expected[0]) <= 1e-6f &&
        std::fabs(actual[1] - expected[1]) <= 1e-6f;

    if (!ok) {
        std::fprintf(stderr, "softmax %s: got {%g, %g}, expected {%g, %g}\n",
                     use_bf16_exp ? "BF16 expp" : "default expf",
                     double(actual[0]), double(actual[1]),
                     double(expected[0]), double(expected[1]));
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (!run_exhaustive_host_test()) {
        return 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
        return 0;
    }
    const bool use_bf16_exp = !(argc == 2 && std::strcmp(argv[1], "--default") == 0);

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "CUDA device unavailable; skipping runtime checks\n");
        return 77;
    }

    expp_case * cases_d = nullptr;
    uint32_t * output_d = nullptr;
    uint32_t output[sizeof(CASES) / sizeof(CASES[0])] = {};

    if (cudaMalloc(&cases_d, sizeof(CASES)) != cudaSuccess) {
        std::fprintf(stderr, "CUDA input allocation failed\n");
        return 1;
    }
    if (cudaMalloc(&output_d, sizeof(output)) != cudaSuccess) {
        std::fprintf(stderr, "CUDA output allocation failed\n");
        cudaFree(cases_d);
        return 1;
    }

    cudaMemcpy(cases_d, CASES, sizeof(CASES), cudaMemcpyHostToDevice);
    expp_kernel<<<1, 32>>>(cases_d, output_d, int(sizeof(CASES) / sizeof(CASES[0])));
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpy(output, output_d, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "CUDA BF16 expp kernel failed\n");
        cudaFree(output_d);
        cudaFree(cases_d);
        return 1;
    }

    bool ok = true;
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); ++i) {
        const uint32_t expected = uint32_t(CASES[i].expected) << 16;
        if (output[i] != expected) {
            std::fprintf(stderr, "case %zu: got 0x%08x, expected 0x%08x\n",
                         i, output[i], expected);
            ok = false;
        }
    }

    cudaFree(output_d);
    cudaFree(cases_d);
    if (ok) {
        ok = run_softmax_switch_test(use_bf16_exp);
    }
    return ok ? 0 : 1;
}
