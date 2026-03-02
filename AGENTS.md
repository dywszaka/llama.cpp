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

## NVFP4 runtime map
- Model graph binding lives in `src/llama-model.cpp`.
  - NVFP4-specific tensors (`*_input_scale`, `*_weight_scale_2`, etc.) are defined in `src/llama-model.h`.
  - During graph build, `build_lora_mm_scaled()` is the key branch:
    - CUDA build: binds `input_scale` and `weight_scale` onto the `GGML_OP_MUL_MAT` node via `ggml_mul_mat_set_nvfp4_input_scale()` and `ggml_mul_mat_set_nvfp4_weight_scale()`.
    - CPU build: applies activation roundtrip first with `ggml_map_custom2(..., ggml_nvfp4_act_roundtrip_op, ...)`, then runs `ggml_mul_mat()`, then applies `weight_scale` as an output multiply.
- CPU execution path:
  - Reference NVFP4 quantize/dequantize logic is in `ggml/src/ggml-quants.c` (`quantize_row_nvfp4_ref()`, `dequantize_row_nvfp4()`).
  - CPU type traits are in `ggml/src/ggml-cpu/ggml-cpu.c`.
    - `GGML_TYPE_NVFP4` uses `vec_dot = ggml_vec_dot_nvfp4_f32` and `vec_dot_type = GGML_TYPE_F32`.
    - This means NVFP4 weights are consumed directly, while activations stay in F32 for the dot kernel.
  - CPU dot kernel is `ggml_vec_dot_nvfp4_f32[_generic]` in `ggml/src/ggml-cpu/quants.c`.
  - CPU-side activation roundtrip helper is in `src/llama-nvfp4.cpp`; it converts the bound input scale into `global_scale = 1 / input_scale`, quantizes to NVFP4 with the reference path, then dequantizes back to F32 before matmul.
- CUDA execution path:
  - Main dispatch is in `ggml/src/ggml-cuda/ggml-cuda.cu`.
  - If `GGML_CUDA_NVFP4_NATIVE` is enabled and tensor types are `src0=NVFP4`, `src1=F32`, `dst=F32`, CUDA first attempts the native path `ggml_cuda_mul_mat_nvfp4_native()`.
  - Native implementation is in `ggml/src/ggml-cuda/nvfp4-matmul.cu`.
    - Reads `input_scale` from the bound mul-mat node and converts it to `global_scale = 1 / input_scale`.
    - Quantizes the F32 activation matrix to temporary NVFP4 on device with `quantize_row_nvfp4_kernel`.
    - Splits packed NVFP4 blocks into separate data and scale channels before calling `cublasLtMatmul`.
    - Reuses a repacked cache for static NVFP4 weights (`src0`) to avoid repeated repacking.
    - Correct scaling is now `matmul_alpha = out_scale / global_scale` when `global_scale != 0`.
  - The nibble packing fix in `quantize_row_nvfp4_kernel` is critical: all lanes must participate in the warp shuffle, and only even lanes store packed bytes.
- CUDA fallback path:
  - If native NVFP4 is not applicable or fails, execution falls back to the general quantized matmul path in `ggml/src/ggml-cuda/mmq.cu`.
  - In that path, the F32 activation is quantized to `Q8_1`, then the kernel uses the NVFP4-specific device dot product `vec_dot_nvfp4_q8_1` from `ggml/src/ggml-cuda/vecdotq.cuh`.
- Important caveat:
  - Generic `to_float` / `get_rows` style dequantization uses only the in-band NVFP4 block scale byte (`e`) and does not know about the extra tensor-wise `global_scale`.
  - For correctness-sensitive debugging, prefer the explicit NVFP4 matmul path (CPU roundtrip or CUDA native/fallback matmul path), not unrelated generic dequant-only paths.

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
