# NVFP4 Quantization Unification Plan

## Purpose

This document defines the planned CUDA NVFP4 quantization cleanup for the BF16
NVFP4 work. The immediate problem is that CUDA NVFP4 quantization currently has
several independent implementations. Changing the NVFP4 algorithm in one path
can leave other paths behind.

The concrete known gap is that enabling:

```text
GGML_CUDA_NVFP4_BF16_QUANT=1
GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1
```

selects the BF16 truncation nearest-neighbor quantizer for the native NVFP4
matmul RHS path, but does not select the same algorithm in K-cache
`set_rows` or V-cache update paths.

## Current CUDA Entry Points

### Native matmul RHS quantization

Files:

- `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-quantize.cu`
- `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-quantize-bf16.cu`

Important functions:

- `quantize_row_nvfp4_kernel`
- `quantize_row_nvfp4_dynamic_kernel`
- `quantize_row_nvfp4_bf16_block`

The FP32 path computes block `amax`, quantizes the E4M3 block scale by scanning
finite E4M3 values, then chooses the nearest E2M1 table entry in FP32. The BF16
path truncates inputs to BF16 bit patterns and uses the hardware-style
truncation nearest-neighbor model.

The host side already has switch helpers for the BF16 experiment:

- `ggml_cuda_nvfp4_bf16_quant_enabled()`
- `ggml_cuda_nvfp4_bf16_quant_trunc_nn_enabled()`
- `ggml_cuda_nvfp4_bf16_quant_bf16_internal_enabled()`
- `ggml_cuda_nvfp4_bf16_quant_bf16_block_scale_enabled()`
- `ggml_cuda_nvfp4_trunc_bf16_input_enabled()`

### K-cache set_rows quantization

File:

- `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-set-rows.cu`

Important functions:

- `k_set_rows_nvfp4`
- `ggml_cuda_best_index_nvfp4_set_rows`
- `ggml_cuda_best_index_e4m3_set_rows`

This path has its own FP32 nearest-neighbor block quantization logic. It also
handles K-cache outlier zeroing and tensor scale side data. It does not consult
the BF16 NVFP4 switches, so BF16 algorithm changes do not affect K-cache
storage.

The `GGML_TYPE_NVFP4_8` path uses a different block size and remains out of
scope for the first pass unless explicitly requested.

### V-cache set_rows and fast update

File:

- `ggml/src/ggml-cuda/expt/nvfp4/vcache-nvfp4-set-rows.cu`

Important functions:

- `k_set_rows_nvfp4_vcache`
- `flush_block`
- `ggml_cuda_best_index_nvfp4_vcache_set_rows`
- `ggml_cuda_best_index_e4m3_vcache_set_rows`
- `ggml_cuda_nvfp4_vcache_fast_update_enabled()`

The full V-cache update path dequantizes the current block into a shared
`tile`, applies pending values, recomputes block scale, requantizes all 16
values, and writes the full `block_nvfp4`.

The fast update path is different. When `LLAMA_NVFP4_VCACHE_FAST_UPDATE=1` and
`n_tokens == 1`, lane 0 checks whether the new value fits inside the current
block range. If it fits, it preserves the existing block scale byte and patches
only the target nibble. If it does not fit, execution falls back to the full
block update path.

Because fast update intentionally preserves the existing scale, it cannot be
treated as equivalent to normal full-block quantization.

## Goals

- Make CUDA NVFP4 block quantization use a shared implementation surface.
- Ensure the BF16 NVFP4 switches affect all relevant `GGML_TYPE_NVFP4` CUDA
  quantization users:
  - native matmul RHS quantization;
  - K-cache `k_set_rows_nvfp4`;
  - V-cache full update and fallback flush;
  - V-cache fast-update decision path.
- Preserve current behavior when BF16 switches are disabled.
- Keep experiment implementation code under
  `ggml/src/ggml-cuda/expt/nvfp4/`.
- Keep top-level CUDA integration narrow.
- Preserve K-cache outlier behavior: outlier values are zeroed before
  quantization and the K-cache global scale helper remains the source of
  global scale.
- Preserve existing environment switch names. No new switch is planned for the
  first pass.

## Non-Goals

- Do not change cuBLASLt repacking, scale-channel layout, or matmul dispatch
  behavior.
- Do not change CPU NVFP4 reference quantization in this pass.
- Do not change `GGML_TYPE_NVFP4_8` behavior in the first pass.
- Do not remove `LLAMA_NVFP4_VCACHE_FAST_UPDATE`.
- Do not make PPL numbers directly comparable unless all baseline parameters
  match `expt-baseline.md`.

## Proposed Code Structure

Add a shared CUDA device helper header:

```text
ggml/src/ggml-cuda/expt/nvfp4/nvfp4-quantize-core.cuh
```

The header should contain reusable device-level primitives for producing a
logical `block_nvfp4`:

- E4M3 finite nearest helper used by the FP32 path;
- E2M1 nearest helper used by the FP32 path;
- FP32 block quantization helper for one 16-value block;
- BF16 truncation nearest-neighbor block quantization helper;
- one-value quantization helper that uses an existing block scale, for
  algorithm-aware fast-update decisions.

The existing `nvfp4-quantize-bf16.cu` helpers that are currently file-local
should be moved or wrapped so K-cache and V-cache kernels can call the same
BF16 logic instead of duplicating it.

Host-side dispatch should compute the quantization strategy once and pass
simple booleans to kernels. The strategy fields needed for the first pass are:

```text
use_bf16_trunc_nn
bf16_internal_arith
bf16_block_scale
truncate_bf16_input
```

`use_bf16_trunc_nn` is true only when both
`GGML_CUDA_NVFP4_BF16_QUANT=1` and
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1` are enabled.

## Path-Specific Design

### Matmul RHS

Keep the existing public wrappers and dispatch behavior, but replace duplicated
low-level helper logic with calls into `nvfp4-quantize-core.cuh`.

Expected behavior:

- switches off: byte output remains the current FP32 path output;
- `GGML_CUDA_NVFP4_TRUNC_BF16_INPUT=1`: existing FP32 path with BF16-truncated
  inputs remains available;
- BF16 trunc-NN switches on: output remains the current
  `quantize_row_nvfp4_bf16_block` behavior.

### K-cache set_rows

Update `k_set_rows_nvfp4` to select the shared quantization strategy.

For the default FP32 strategy, the output must match the current
`k_set_rows_nvfp4` behavior.

For BF16 trunc-NN, the path should:

1. load the F32 source value;
2. apply the existing K-cache outlier rule, replacing outliers with zero before
   quantization;
3. use `ggml_cuda_nvfp4_kcache_outlier_k_global_scale()` as today;
4. compute the `block_nvfp4` bytes through the shared BF16 trunc-NN block
   helper.

The scale tensor update performed by `k_set_rows_scale` stays unchanged because
it stores the input scale derived from the K-cache global scale. The block
scale byte inside `block_nvfp4` is the part that changes with the selected
quantization algorithm.

### V-cache full update and fallback flush

Update `flush_block` in `k_set_rows_nvfp4_vcache` to use the shared
quantization strategy when it recomputes a full block.

Expected behavior:

- switches off: output matches the current V-cache full update behavior;
- BF16 trunc-NN switches on: full store and fast-update fallback output match
  the shared BF16 trunc-NN block helper.

The existing scale modes remain:

- local per-block input scale mode stores `1 / global_scale` in the scale
  tensor;
- global scale mode reads `scale[row_global / rows_per_scale]`.

### V-cache fast update

Fast update needs separate treatment because it patches one nibble while
preserving the existing block scale. A full-block BF16 quantizer is not a
drop-in replacement for this operation.

The first implementation should make fast update algorithm-aware with this
conservative rule:

- default FP32 strategy: keep the current nibble-only fast patch behavior, but
  use the shared one-value FP32 helper;
- BF16 trunc-NN strategy: do not perform nibble-only fast patch in the first
  pass; fall through to full-block `flush_block`, which recomputes scale and
  quantizes through the shared BF16 helper.

This rule preserves correctness and prevents a mixed state where BF16 switches
are enabled but a single V-cache element is patched with FP32 nearest-neighbor
semantics. It may reduce the benefit of
`LLAMA_NVFP4_VCACHE_FAST_UPDATE=1` while BF16 trunc-NN is enabled. That
performance trade-off is acceptable for the first correctness pass.

A later optimization can add a BF16 one-value fast patch if tests define exact
semantics for:

- the fit check under BF16 truncation;
- values exactly on quantization thresholds;
- whether the existing E4M3 block scale is interpreted as FP32 half-scale or
  BF16 internal scale.

## Testing Plan

### Unit and focused CUDA tests

Use tests to lock behavior before and after the code move.

Matmul RHS:

```bash
cmake --build build_cuda --target test-nvfp4-matmul -j$(nproc)
./build_cuda/bin/test-nvfp4-matmul
GGML_CUDA_NVFP4_BF16_QUANT=1 GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1 ./build_cuda/bin/test-nvfp4-matmul
```

V-cache:

```bash
cmake --build build_cuda --target test-vcache-nvfp4-store -j$(nproc)
./build_cuda/bin/test-vcache-nvfp4-store
GGML_CUDA_NVFP4_BF16_QUANT=1 GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1 ./build_cuda/bin/test-vcache-nvfp4-store
```

Extend `tests/test-vcache-nvfp4-store.cu` with BF16-switch coverage:

- full store matches BF16 host reference;
- fast update with BF16 switches enabled falls back to full-block BF16
  requantization;
- fast update with BF16 switches disabled preserves the existing nibble-only
  patch behavior.

Add or extend a K-cache set_rows focused test:

- default `k_set_rows_nvfp4` output matches the current FP32 reference;
- BF16 trunc-NN switch output matches the BF16 reference for `GGML_TYPE_NVFP4`;
- K-cache outlier zeroing still happens before quantization.

Host-side BF16 expected bytes should reuse the same BF16 trunc-NN reference
logic already used by `tests/test-nvfp4-matmul.cu`, either by extracting a
small shared test helper or by duplicating only the minimal reference code in
the new focused test.

### PPL comparison

After focused tests pass, run the requested PPL comparison using the baseline
parameters from `expt-baseline.md`.

Required variants:

- baseline with BF16 switches disabled;
- BF16 trunc-NN enabled with:

  ```text
  GGML_CUDA_NVFP4_BF16_QUANT=1
  GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1
  ```

If `LLAMA_NVFP4_VCACHE_FAST_UPDATE=1` is included in the PPL run, record it as
a separate variant so the impact of conservative BF16 fallback is visible.

Create a dedicated experiment folder under `experiments/` for the PPL run. It
must include:

- run script;
- environment variables;
- baseline parameter reference;
- raw output logs;
- parsed metrics;
- summary comparing baseline and BF16 variants.

## Implementation Order

1. Add failing or currently-missing tests for K-cache and V-cache BF16 switch
   coverage.
2. Introduce `nvfp4-quantize-core.cuh` and migrate the existing matmul RHS
   helpers without changing behavior.
3. Route `k_set_rows_nvfp4` through the shared core and pass BF16 strategy
   booleans from host dispatch.
4. Route V-cache `flush_block` through the shared core.
5. Make V-cache fast update strategy-aware:
   - FP32 keeps nibble patch;
   - BF16 trunc-NN falls back to full-block flush.
6. Run focused CUDA tests in default and BF16 switch modes.
7. Run the PPL comparison and write experiment artifacts.

## Review Checklist

- The BF16 switches affect matmul RHS, K-cache set_rows, and V-cache full
  block update.
- V-cache fast update cannot silently use FP32 nibble quantization while BF16
  trunc-NN is enabled.
- Default switch-off output is unchanged.
- `GGML_TYPE_NVFP4_8` is not changed accidentally.
- K-cache outlier extraction and side scale tensor behavior are unchanged.
- No new environment switch is added without updating `expt-switch-env.md`.
- PPL artifacts are stored under `experiments/` before reporting results.
