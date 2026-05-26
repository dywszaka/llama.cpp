# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

### Configure and build

Use CMake. The root `Makefile` build is deprecated.

```bash
# CPU release build; binaries go to build/bin/
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)"

# CUDA release build; binaries go to build_cuda/bin/
cmake -B build_cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda --config Release -j "$(nproc)"

# Debug build with warnings as errors
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug -DLLAMA_FATAL_WARNINGS=ON
cmake --build build-debug -j "$(nproc)"

# Build one target
cmake --build build --target llama-cli -j "$(nproc)"
cmake --build build --target test-chat-template -j "$(nproc)"
```

Useful build options:

- Add `-DLLAMA_CURL=OFF` if libcurl development headers are unavailable.
- Add `-DBUILD_SHARED_LIBS=OFF` for a static build.
- For CUDA, add `-DCMAKE_CUDA_ARCHITECTURES="86;89"` (adapt values) if `nvcc` cannot detect the GPU.
- CMake presets exist for common compiler/build combinations, e.g. `cmake --preset x64-linux-gcc-release` then `cmake --build build-x64-linux-gcc-release`.

### Run tests

```bash
# All CTest tests for a build tree
ctest --test-dir build --output-on-failure

# Run one CTest by regex, with verbose output
ctest --test-dir build -R test-chat-template --output-on-failure -V

# List test commands without running them
ctest --test-dir build -R test-tokenizer -V -N

# Run CUDA-labeled tests from a CUDA build
ctest --test-dir build_cuda -L cuda --output-on-failure

# Focused local test/debug helper; add -g to run under gdb
./scripts/debug-test.sh test-tokenizer
./scripts/debug-test.sh -g test-tokenizer
```

CUDA/NVFP4 focused tests are only built with `-DGGML_CUDA=ON`; useful regexes include `test-nvfp4-matmul`, `test-nvfp4-fattn`, `test-vcache-nvfp4`, and `test-vcache-fp8`.

### CI and validation commands

```bash
# Local CI framework
mkdir -p tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

# Performance/regression tools after building
./build/bin/llama-bench -m /path/to/model.gguf
./build/bin/llama-perplexity -m /path/to/model.gguf -f /path/to/corpus.txt
./build/bin/llama-cli -m /path/to/model.gguf -p "Hello"
./build/bin/llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

There is no single repo-wide lint target. Use the checked-in config files directly when needed:

```bash
clang-format -i path/to/file.cpp path/to/file.h
clang-tidy path/to/file.cpp -p build
```

For Python conversion/scripts, install either the top-level requirements or the more specific file under `requirements/` that matches the script being used.

## High-level architecture

- `include/llama.h` is the public C API for `libllama`; `include/llama-cpp.h` provides a small C++ wrapper surface. `src/llama.cpp` exports the C API and delegates to the internal model, context, sampling, and memory components.
- `src/llama-model*.{h,cpp}` owns GGUF model loading, architecture/hyperparameter interpretation, tensor mapping, model saving, and weight placement across backend buffers. `llama_model::build_graph()` dispatches to per-architecture graph builders.
- `src/llama-graph.*` contains shared graph-building primitives for embeddings, attention, FFNs, MoE, recurrent state, pooling, and KV-cache interaction. Model-specific builders in `src/llama-model.cpp` compose these helpers.
- `src/llama-context.*` is the runtime orchestrator: it normalizes context parameters, initializes ggml backends, creates memory/KV modules, reserves/reuses compute graphs, drives scheduler execution, and exposes logits/embeddings state.
- `src/llama-memory*`, `src/llama-kv-cache-unified*`, and related files implement unified KV cache, recurrent memory, hybrid memory, defrag/update graphs, and cache type behavior.
- `ggml/` is the tensor library and backend layer. Public ggml headers are in `ggml/include/`; core tensor/graph/allocation/backend registry code is in `ggml/src/`; backend implementations live in `ggml/src/ggml-cpu`, `ggml/src/ggml-cuda`, `ggml/src/ggml-metal`, `ggml/src/ggml-vulkan`, etc.
- `common/` is shared application glue used by tools, examples, and tests: CLI arg parsing, chat templates/parsing, sampling helpers, logging, JSON/grammar helpers, speculative decoding utilities, and build metadata.
- `tools/` contains the installed programs: `tools/main` builds `llama-cli`, `tools/server` builds the OpenAI-compatible HTTP server, and other subdirectories build quantization, perplexity, benchmarking, tokenization, GGUF, imatrix, RPC, TTS, and calibration tools.
- `examples/` contains smaller API examples and demos; prefer `examples/simple` or `examples/simple-chat` for minimal usage patterns.
- `gguf-py/` and top-level `convert_*.py` scripts handle GGUF conversion and metadata/tensor naming support for Python workflows.
- `tests/` uses CTest via helper functions in `tests/CMakeLists.txt`. Some tests depend on vocab/model fixtures under `models/`; CUDA smoke tests assume CUDA support and, for some cache smoke tests, a local model at `../models/qwen3-8b-nvfp4.gguf`.

## Project-specific working rules

- Treat most behavior-changing or performance-sensitive work here as experimental unless the user says otherwise. New experiments should be switch-gated, default off, and have narrow, auditable switch plumbing.
- Environment-variable experiment switches are tracked in `expt-switch-env.md`; update it when adding or removing a switch.
- Experiment and validation artifacts belong under `experiments/`. For PPL, `llama-server`, and benchmark experiments, start from `expt-baseline.md`, change only the parameter under test, and record scripts, input references, raw logs, metrics, and summary in a dedicated experiment folder.
- For NVFP4/CUDA/KV-cache work, read `AGENTS.md` and the referenced docs before editing. Important local guides include `docs/development/nvfp4-k-cache.md`, `docs/development/ncu-kvcache-profiling.md`, `docs/development/debugging-tests.md`, and `docs/development/experiment-records.md`.
- Keep high-volume diagnostics out of release paths. Tight-loop/per-token/per-kernel logging should be debug-only or behind an explicit diagnostic switch; release logs for experiment switch confirmation should print once.
- Report only validation that was actually run. If CUDA hardware, model files, toolkits, or data are unavailable, say which focused command should be run instead of implying success.

## Coding conventions that matter in this repo

- C is C11; C++ is C++17. Do not bump the required standards without a specific reason.
- Avoid new third-party dependencies and extra headers unless they are essential.
- Follow existing llama.cpp style: 4-space indentation, same-line braces, lowercase dash-separated C/C++ filenames, snake_case symbols, enum values prefixed with the enum name, and project pointer/reference spacing such as `void * ptr` and `int & a`.
- Prefer simple loops and straightforward code over template-heavy or highly modern STL constructs.
- ggml tensor dimensions are row-major; dimension 0 is columns, dimension 1 is rows, dimension 2 is matrices. `ggml_mul_mat(ctx, A, B)` represents `C^T = A B^T`.
- When changing ggml operators or adding backend implementations, update/add `test-backend-ops` coverage and regenerate `docs/ops.md` with `./scripts/create_ops_docs.py` if backend support tables change.
- When adding a new model architecture, follow `docs/development/HOWTO-add-model.md`: update GGUF Python constants/mappings, add the `llm_arch` metadata and tensor mapping in `src/llama-arch.*`, load any special hparams in model loading, and add a graph builder plus dispatch case.
