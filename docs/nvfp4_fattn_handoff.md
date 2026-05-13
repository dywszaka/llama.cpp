# NVFP4 CUDA FATTN Handoff

Created: 2026-05-11

This file compresses the current development context so another agent can continue the NVFP4 CUDA flash-attention work without replaying the full chat.

## Current Goal

Latest requested task:

- Add one experiment with `K smooth` disabled and `--cache-type-k nvfp4` enabled.
- First inspect whether code changes are needed.
- If changes are needed, implement them, then run the PPL experiment.

Status: implemented and run.

The experiment command was based on the existing PPL setup and forced the native NVFP4 FATTN path, with no fallback.

## Repository State

Workspace:

- Repo: `/home/allen/host_workspace/develop/llama.cpp`
- Active area: `ggml/src/ggml-cuda/fattn-nvfp4.cu`
- Additional changed files from the K-cache NVFP4 experiment:
  - `ggml/src/ggml-cuda/fattn-nvfp4.cu`
  - `src/llama-context.cpp`
  - `src/llama-graph.cpp`
  - `tests/test-nvfp4-fattn.cu`
  - `docs/nvfp4_fattn_handoff.md`
  - `docs/nvfp4_fattn_ppl_results.md`
  - `?? run-llama-perplexity-nvfp4-cuda.sh`

The NVFP4 FATTN code changes described below are already present in the current tree relative to HEAD.

## Relevant Files

- `ggml/src/ggml-cuda/fattn-nvfp4.cu`
- `ggml/src/ggml-cuda/fattn-nvfp4.cuh`
- `ggml/src/ggml-cuda/fattn.cu`
- `ggml/src/ggml-cuda/nvfp4-matmul.cu`
- `ggml/src/ggml-cuda/set-rows.cu`
- `ggml/src/ggml-cuda/cpy.cu`
- `src/llama-context.cpp`
- `src/llama-kv-cache-unified.cpp`
- `tests/test-nvfp4-fattn.cu`

## Current FATTN Implementation State

`ggml/src/ggml-cuda/fattn-nvfp4.cu` currently has a unified native path:

- `ggml_cuda_flash_attn_ext_nvfp4_ref()` was removed.
- Separate `ggml_cuda_flash_attn_ext_nvfp4_prefill()` and `ggml_cuda_flash_attn_ext_nvfp4_decode()` wrappers were removed.
- Public entry point now directly calls:
  - `ggml_cuda_flash_attn_ext_nvfp4_gpu_native(ctx, dst)`
- `ggml/src/ggml-cuda/fattn.cu` dispatch no longer distinguishes prefill/decode by `Q->ne[1] == 1`.
- `GGML_CUDA_NVFP4_FATTN_DECODE` was removed.

Decode now shares the same native NVFP4 FATTN route as prefill.

## Current Experiment Switches

Environment switches currently implemented in `fattn-nvfp4.cu`:

- `GGML_CUDA_NVFP4_FATTN=1`
  - Enables NVFP4 FATTN dispatch.
- `GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1`
  - Forces failure instead of silently falling back.
- `GGML_CUDA_NVFP4_FATTN_P_DIRECT=1`
  - Uses raw softmax `P` directly for NVFP4 quantization.
  - Skips the two-level `P` scale path.
- `GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=1`
  - Disables Q smoothing.
- `GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1`
  - Disables K smoothing.
- `GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC=1`
  - Makes QK use `ggml_cuda_mul_mat_nvfp4_native()` dynamic RHS quantization.
  - The QK destination does not bind an NVFP4 input scale, so native matmul computes per-row Q amax/global scale and applies the matching dynamic column scale after matmul.

## Current Kernels and Meaning

Important kernels in `fattn-nvfp4.cu`:

- `q_smooth_kernel`
  - Prefill Q smoothing.
  - Centers Q by subtracting per-row mean.
- `q_smooth_decode_kernel`
  - Decode specialization for `q_len == 1`.
  - Avoids the generic Q smoothing reduction path.
- `q_no_smooth_kernel`
  - Copies Q as-is and writes `q_mean = 0`.
- `k_smooth_kernel`
  - Centers visible K rows and zeroes invisible/padding positions.
- `k_no_smooth_kernel`
  - Copies visible K rows as-is and zeroes invisible/padding positions.
- `copy_k_nvfp4_head_kernel`
  - Experiment path for `--cache-type-k nvfp4` with `GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1`.
  - Copies compressed `block_nvfp4` K-cache blocks for one `(batch, kv_head)` into a contiguous temporary NVFP4 matrix.
  - Does not materialize K as F32 and does not requantize K for QK.
- `qk_apply_k_scale_kernel`
  - Applies the existing K-cache per-token global scale to QK rows after native FP4 matmul.
  - The native matmul consumes the in-band NVFP4 block scales directly; this kernel restores the extra tensor scale.
- `qmean_kcorr_nvfp4_kernel`
  - Computes the Q-mean correction term by reading NVFP4 K-cache blocks directly with the attached scale tensor.
  - Avoids using the F32 `k_centered` workspace for the K-cache NVFP4 path.
- `probs_twolevel_scale_kernel`
  - Default P scaling before NVFP4 quantization.
  - Produces row-level first scale and scaled probabilities.
- `probs_direct_scale_kernel`
  - Experiment path for raw P.
  - Copies P unchanged to `probs_scaled` and writes first scale as `1.0f`.
- `v_by_dim_kernel`
  - Reorders V into a by-dimension layout for the later `P * V` matmul.

## Validation Already Run

These commands were previously run successfully after the unified native path and experiment switches were added:

```bash
cmake --build build_cuda --target test-nvfp4-fattn llama-perplexity -j 8
./build_cuda/bin/test-nvfp4-fattn
```

The unit test passed at that time.

Additional validation after adding K-cache NVFP4 support:

```bash
cmake --build build_cuda --target test-nvfp4-fattn llama-perplexity -j 8
./build_cuda/bin/test-nvfp4-fattn
```

The updated unit test passed. It now includes a no-K-smooth NVFP4 K-cache case that compares FATTN output against a reference path using the same per-row K-cache NVFP4 quantize/dequant scale convention.

## K Cache NVFP4 Implementation

The requested experiment needed code changes and has been implemented.

Previous blocker in `src/llama-context.cpp`:

```cpp
if (params.flash_attn && (params.type_k == GGML_TYPE_NVFP4 || params.type_k == GGML_TYPE_NVFP4_8)) {
    LLAMA_LOG_ERROR("%s: NVFP4 K cache does not support flash_attn yet\n", __func__);
    return nullptr;
}
```

Before this change, `-fa --cache-type-k nvfp4` aborted before graph execution unless this restriction was relaxed.

Current behavior:

- `GGML_TYPE_NVFP4` K cache is allowed with flash attention.
- `GGML_TYPE_NVFP4_8` K cache remains disabled with flash attention.
- NVFP4 V cache remains disabled.

Observed K cache support in `src/llama-kv-cache-unified.cpp`:

```cpp
has_k_scale = type_k == GGML_TYPE_NVFP4 || type_k == GGML_TYPE_NVFP4_8;
```

The K cache can have an attached NVFP4 scale tensor.

Observed CUDA set-rows support in `ggml/src/ggml-cuda/set-rows.cu`:

- `dst->type == GGML_TYPE_NVFP4` is supported.
- It asserts a scale tensor exists:
  - `ggml_tensor_get_nvfp4_scale(dst)`
- Scale tensor type is `GGML_TYPE_F32`.
- It writes per-row `amax`-derived scale through `k_set_rows_scale`.

Observed CUDA copy support in `ggml/src/ggml-cuda/cpy.cu`:

- `GGML_TYPE_NVFP4 -> GGML_TYPE_F32` copy/dequant is supported.
- It reads `ggml_tensor_get_nvfp4_scale(src0)`.
- It detects scale layout by axis:
  - scale axis 1 if scale length matches `src0->ne[1]`
  - scale axis 2 if scale length matches `src0->ne[2]`
  - scale axis 0 if scale length matches `src0->ne[0]`

Previous native FATTN input type check only accepted F16/F32 K and V:

```cpp
if (!((k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32) &&
      (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32))) {
    return false;
}
```

Previous device read helper also only read F32/F16:

```cpp
if (type == GGML_TYPE_F32) {
    return *(const float *) ptr;
}
return __half2float(*(const half *) ptr);
```

Implemented path:

1. Relaxed the `src/llama-context.cpp` guard for `params.flash_attn && type_k == GGML_TYPE_NVFP4`.
2. Kept `GGML_TYPE_NVFP4_8` disabled for flash attention.
3. Allowed native FATTN to accept `k->type == GGML_TYPE_NVFP4` only when `GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1`.
4. Rebound the K NVFP4 scale tensor after `ggml_permute()` in `src/llama-graph.cpp`, because the permuted view did not otherwise retain the scale binding.
5. Skipped `k_centered` allocation, K F32 materialization, K absmax, and K requantization for the NVFP4 K-cache path.
6. Added `copy_k_nvfp4_head_kernel()` to copy the existing compressed K-cache `block_nvfp4` blocks for the current `(batch, kv_head)` into a contiguous temporary matrix consumed by `ggml_cuda_mul_mat_nvfp4_native()`.
7. Kept native matmul `weight_scale = 1.0` for direct K cache so the matmul uses the in-band block scales as stored.
8. Added `qk_apply_k_scale_kernel()` to multiply each QK column by the existing per-token K-cache scale after native matmul.
9. Added `qmean_kcorr_nvfp4_kernel()` so the Q-mean correction reads K directly from NVFP4 cache with the same scale tensor.
10. Added `GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC=1` for the no-Q/no-K-smooth experiment. When enabled, QK passes `nullptr` as the Q input-scale tensor to `ggml_cuda_mul_mat_nvfp4_native()`, triggering its existing dynamic per-row Q quantization path.

This measures the impact of storing K cache as the existing per-row global-token NVFP4 format while directly consuming the cache representation in QK. The K-cache path does not dequantize K to F32 and does not requantize K before QK.

## Suggested First Checks

Run these before editing:

```bash
rg -n "NVFP4 K cache does not support flash_attn|type_k == GGML_TYPE_NVFP4|ggml_tensor_get_nvfp4_scale|GGML_TYPE_NVFP4" \
  src/llama-context.cpp src/llama-kv-cache-unified.cpp ggml/src/ggml-cuda/fattn-nvfp4.cu ggml/src/ggml-cuda/cpy.cu ggml/src/ggml-cuda/set-rows.cu

sed -n '2318,2345p' src/llama-context.cpp
sed -n '1,140p' src/llama-kv-cache-unified.cpp
sed -n '470,535p' ggml/src/ggml-cuda/set-rows.cu
sed -n '620,745p' ggml/src/ggml-cuda/cpy.cu
```

## K Cache NVFP4 PPL Result

Command:

```bash
env CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=0 \
    GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    --cache-type-k nvfp4 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_no_k_smooth_kcache_nvfp4.log
```

Observed:

- `Final estimate: PPL = 10.5143 +/- 0.08132`
- `prompt eval time = 606472.01 ms / 299008 tokens`
- `493.03 tokens per second`
- `graphs reused = 583`

Comparison:

- F16 K cache with no K smooth: `10.4057 +/- 0.08035`
- NVFP4 K cache with no K smooth, direct cache QK: `10.5143 +/- 0.08132`
- NVFP4 K cache with no Q/K smooth, direct cache QK, dynamic Q: `10.5940 +/- 0.08204`
- NVFP4 K cache with no Q/K smooth, direct cache QK, Q input scale: `10.9134 +/- 0.08527`

## K Cache NVFP4 + No Q/K Smooth + Dynamic Q PPL Result

Command:

```bash
env CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=1 \
    GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1 \
    GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC=1 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    --cache-type-k nvfp4 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_no_q_no_k_smooth_kcache_nvfp4_q_dynamic.log
```

Observed:

- `Final estimate: PPL = 10.5940 +/- 0.08204`
- `prompt eval time = 606118.48 ms / 299008 tokens`
- `493.32 tokens per second`
- `graphs reused = 583`

## K Cache NVFP4 + No Q/K Smooth + Q Input Scale PPL Result

This is the same setup as the dynamic-Q experiment, except `GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC` is unset. QK therefore binds the static `q_input_scale` tensor and uses the non-dynamic RHS quantization path in `ggml_cuda_mul_mat_nvfp4_native()`.

Command:

```bash
env -u GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC \
    CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=1 \
    GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    --cache-type-k nvfp4 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_no_q_no_k_smooth_kcache_nvfp4_q_input_scale.log
```

Observed:

- `Final estimate: PPL = 10.9134 +/- 0.08527`
- `prompt eval time = 605513.39 ms / 299008 tokens`
- `493.81 tokens per second`
- `graphs reused = 583`

## Build Targets

Useful build commands:

```bash
cmake --build build_cuda --target test-nvfp4-fattn llama-perplexity -j 8
./build_cuda/bin/test-nvfp4-fattn
```

## Caution

- Do not enable NVFP4 V cache. `src/llama-context.cpp` explicitly disables it.
- Keep no-fallback enabled during experiments so failures are visible.
- Do not rely on generic dequant-only paths as proof of NVFP4 matmul correctness; the project notes say generic `to_float` / `get_rows` style dequant does not always know about tensor-wise global scale.
- The current debug build PPL runs are much slower than release-style runs. Compare accuracy primarily by PPL, not speed, unless build type is controlled.
