# Project Agent Instructions

## Project Coding Policy
- Treat most code changes in this repository as experimental unless the user explicitly says otherwise.
- Every new experiment must be gated by a switch.
  - New experiment switches must default to off.
  - Switch definitions must be centralized in one clear place.
  - The centralized definition must document what each switch does and what behavior it enables.
- Switch usage should be consolidated behind one helper/function per switch whenever possible.
  - Avoid checking the same switch directly in many unrelated call sites.
  - Keep switch plumbing narrow and easy to audit.
  - Existing historical switches may predate this policy; when touching them, migrate toward centralized helpers and documentation instead of adding more scattered `getenv()` checks.
- Switch effectiveness must be confirmed by logs.
  - Log whether each switch is enabled or disabled.
  - The confirmation log should print only once during service startup or first runtime use.
  - Avoid noisy per-token, per-request, or per-kernel repeated logging.
- After code validation passes, commit the verified change.
- Follow SOLID principles when coding:
  - Keep experiment control separate from core algorithm code.
  - Prefer small functions with one reason to change.
  - Depend on narrow helper APIs instead of scattering environment/config parsing.

## Documentation Hygiene
- Keep this file as durable project guidance, not a dated session log.
- Do not record temporary build directories, one-off validation results, or old commit lists here.
- If a path or runtime description changes, update the relevant map below in the same commit as the code change.

## Experiment Run Records
- 每次运行 PPL 实验或用 `llama-server` 启动验证时，都必须在 `experiments/` 下新建一个实验文件夹。
  - 每个实验或同一组验证 run 使用一个独立文件夹，相关产物都放在该文件夹内。
  - 记录本次实验实际使用的运行脚本或启动脚本。
  - `llama-server` 验证需要记录请求数据、请求返回和验证结果。
  - PPL 实验需要记录运行脚本、输入/config 引用、原始输出日志和汇总后的验证结果。
- 新实验脚本必须参考 `experiments/EXPERI.md` 中的 baseline 参数。
  - 只替换实验明确需要变化的参数。
  - 模型路径、prompt/request 数据、上下文长度、batch 参数、cache 类型、GPU layer 数、线程数、CUDA 设备和 server KV 模式默认都要与 baseline 保持一致。
  - 如果实验必须改变某个 baseline 参数，需要在该实验文件夹内说明原因，并且不要把结果当作直接 baseline A/B 对比。

## NVFP4 Runtime Map
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
    - For statically-bound activation scales, `matmul_alpha` compensates for the activation `global_scale`; check the current code before changing alpha/scale behavior.
  - The nibble packing fix in `quantize_row_nvfp4_kernel` is critical: all lanes must participate in the warp shuffle, and only even lanes store packed bytes.
- CUDA NVFP4 flash-attention experiments are split between graph flags and CUDA implementation.
  - Graph flag selection is in `src/llama-graph.cpp`.
  - CUDA execution lives in `ggml/src/ggml-cuda/fattn-nvfp4.cu`.
  - Current related env switches include `GGML_CUDA_NVFP4_FATTN`, `GGML_CUDA_NVFP4_FATTN_NO_FALLBACK`, `GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH`, `GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH`, `GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC`, `GGML_CUDA_NVFP4_FATTN_P_DIRECT`, and `GGML_CUDA_NVFP4_FATTN_DEBUG`.
- CUDA NVFP4 V-cache p*v experiments live in `ggml/src/ggml-cuda/vcache-nvfp4-matmul.cu`.
  - `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV=1` makes the V-cache p*v matmul dynamically quantize P rows to NVFP4 before dotting with NVFP4 V. It defaults off and logs its enabled/disabled state once.
- CUDA fallback path:
  - If native NVFP4 is not applicable or fails, execution falls back to the general quantized matmul path in `ggml/src/ggml-cuda/mmq.cu`.
  - In that path, the F32 activation is quantized to `Q8_1`, then the kernel uses the NVFP4-specific device dot product `vec_dot_nvfp4_q8_1` from `ggml/src/ggml-cuda/vecdotq.cuh`.
- Important caveat:
  - Generic `to_float` / `get_rows` style dequantization uses only the in-band NVFP4 block scale byte (`e`) and does not know about the extra tensor-wise `global_scale`.
  - For correctness-sensitive debugging, prefer the explicit NVFP4 matmul path (CPU roundtrip or CUDA native/fallback matmul path), not unrelated generic dequant-only paths.

## Logging Policy
- Release builds should not contain high-volume debug logs. These historical noisy logs are expected to stay Debug-only:
  - `llama_decode begin/end` in `tools/server/server.cpp`
  - `sampled token: tok=...` in `tools/server/server.cpp`
  - `ggml_compute_forward_get_rows_f32 ... firstN=...` in `ggml/src/ggml-cpu/ops.cpp`
  - `NVFP4 layout diagnostic for ...` in `ggml/src/ggml-cuda/nvfp4-matmul.cu`
- Implementation pattern: `#ifndef NDEBUG`.
- Experiment switch confirmation logs are allowed in Release only when they print once and are useful for confirming runtime behavior.

## Helper Scripts
- `run-llama-server-nvfp4-cuda.sh`
  - Local NVFP4 CUDA server launch helper.
- `llama_bench.sh`
  - Local benchmark helper.

## Validation Guidance
- For NVFP4 native CUDA matmul changes, prefer running `test-nvfp4-matmul` from the active CUDA build directory when the local GPU/toolkit supports it.
- For NVFP4 flash-attention or KV-cache changes, run the nearest focused CUDA tests first, then a small server or perplexity smoke test if behavior changes.
- Document skipped validation explicitly in the final response when local hardware, toolkit, or build availability prevents a test.
