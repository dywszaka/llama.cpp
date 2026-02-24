# Session Context (NVFP4 CUDA / Release)

## What was fixed
- NVFP4 native CUDA matmul correctness issue was fixed in `ggml/src/ggml-cuda/nvfp4-matmul.cu`.
- Root causes:
  - `quantize_row_nvfp4_kernel` used `__shfl_xor_sync` under a lane predicate; this could corrupt packed high nibble data.
  - `cublasLtMatmul` scaling missed `global_scale` compensation.
- Effective fixes:
  - Always execute the shuffle for all lanes, then conditionally store on even lanes.
  - Use `matmul_alpha = out_scale / global_scale` (when `global_scale != 0`).

## Validation status
- Added/kept CUDA test target `test-nvfp4-matmul` and integration-style cases.
- Test file: `tests/test-nvfp4-matmul.cu`.
- CMake wiring: `tests/CMakeLists.txt`.
- Current result: `build_cuda/bin/test-nvfp4-matmul` passes all cases.

## Release logging policy applied
- In Release builds, these logs were disabled (kept for Debug):
  - `llama_decode begin/end` in `tools/server/server.cpp`
  - `sampled token: tok=...` in `tools/server/server.cpp`
  - `ggml_compute_forward_get_rows_f32 ... firstN=...` in `ggml/src/ggml-cpu/ops.cpp`
  - `NVFP4 layout diagnostic for ...` in `ggml/src/ggml-cuda/nvfp4-matmul.cu`
- Implementation pattern: `#ifndef NDEBUG`.

## Release build outputs
- CUDA Release build dir: `build_cuda_release`.
- Main binaries:
  - `build_cuda_release/bin/llama-server`
  - `build_cuda_release/bin/llama-bench`

## Helper scripts added
- `run-llama-server-nvfp4-cuda.sh`
  - Mirrors NVFP4 CUDA server launch, without debug env toggles.
- `llama_bench.sh`
  - Simple release benchmark runner for local models.

## Relevant commits from this session
- `54ecf5e1` - `cuda: fix nvfp4 native quantization shuffle and alpha scaling`
- `d8e23e47` - `release: quiet debug logs and add release run scripts`
