# Experiment Switch Environment Variables

## FP8 E4M3 E8M0 32 K-Cache

### `--cache-type-k fp8_e4m3_e8m0_32`

Enables the experimental FP8(E4M3+E8M0 block32) K-cache path. Default: off
because the standard K-cache type default remains `f16`.

Initial scope: CUDA non-flash KQ with KQV offload enabled. During KQ, the stored
K cache is `GGML_TYPE_FP8_E4M3_E8M0_32`; the F32 Q operand is quantized to
temporary FP8 block32 inside the native CUDA FP8 matmul path before cuBLASLt
execution.

### `LLAMA_KCACHE_HYBRID_FP8_E4M3_E8M0_32_LAYERS`

Enables experimental per-layer hybrid K-cache storage where selected K-cache
layers use `GGML_TYPE_FP8_E4M3_E8M0_32` while the remaining layers keep the
configured `--cache-type-k`. Default: unset/off.

Compatibility alias for the newer B switch
`LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8=1`.

Supported values:

```text
high_medium
0,1,4,5,6,8,10,11,12,14,23,35
```

The `high_medium` alias maps to the stable threshold-16 NVFP4 K-cache outlier
high and medium layers observed in the prompt-consistency experiment:
`0,1,4,5,6,8,10,11,12,14,23,35`. When enabled, the selected layer K tensors are
allocated as FP8(E4M3+E8M0 block32), so CUDA `set_rows` quantization and KQ
matmul dispatch use the existing FP8 K-cache paths for those layers. This
hybrid switch only takes effect when `--cache-type-k nvfp4`; other K-cache
types ignore it. This hybrid mode inherits the FP8 K-cache runtime limits:
flash attention is not supported, and KQ/V offload must be enabled.

## NVFP4 CUDA Native Matmul

### `GGML_CUDA_NVFP4_BF16_QUANT`

Parent switch for the experimental BF16 trunc-NN NVFP4 RHS activation quantizer
in the CUDA native NVFP4 matmul path. Default: off.

This switch only changes behavior together with
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1`. When disabled, or when
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN` is disabled, the existing FP32
nearest-neighbor NVFP4 quantizer remains active unless
`GGML_CUDA_NVFP4_TRUNC_BF16_INPUT=1` selects the separate trunc-input FP32
nearest-neighbor path.

Initial scope: `GGML_TYPE_NVFP4 x GGML_TYPE_F32 -> GGML_TYPE_F32` CUDA native
matmul activation quantization, including static input-scale and dynamic
per-row/per-tensor RHS scale modes. This switch does not change stored NVFP4
weights or generic dequantization paths.

### `GGML_CUDA_NVFP4_NATIVE_ROW_SPLIT`

Diagnostic switch for native CUDA NVFP4 matmul. Default: unset/off.

When enabled, native NVFP4 matmul runs each RHS token column separately through
cuBLASLt instead of one batched `N` dimension call when `N > 1`. This is useful
for isolating whether ubatch-dependent GEMM shape changes affect upstream F32
activations before K-cache outlier extraction. It should not be used for
performance measurements.

### `GGML_CUDA_NVFP4_FP4MULMAT`

Enables the experimental fp4_mulmat-derived CUDA NVFP4 matmul model path.
Default: unset/off.

When enabled, the native CUDA NVFP4 matmul path still quantizes F32 RHS
activations through the current NVFP4 activation quantizer, then evaluates the
NVFP4 block dot product with the experimental FP4 accumulator model instead of
cuBLASLt. This is intended for hardware-model comparison and correctness
experiments, not performance measurement.

The path logs once when selected. Combine with
`GGML_CUDA_NVFP4_FP4MULMAT_LOG=1` to log the first several selections during a
run.

### `GGML_CUDA_NVFP4_FP4MULMAT_LOG`

Diagnostic logging switch for `GGML_CUDA_NVFP4_FP4MULMAT`. Default:
unset/off.

When enabled, prints selection logs for the first several fp4_mulmat-derived
NVFP4 matmul calls instead of only the first call. It does not enable the
fp4_mulmat path by itself.

### `GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN`

Enables an experimental hardware-friendly BF16-input NVFP4 RHS activation
quantizer that more closely follows `GGML_CUDA_NVFP4_TRUNC_BF16_INPUT=1`.
Default: off.

When enabled with `GGML_CUDA_NVFP4_BF16_QUANT=1`, the CUDA quantizer truncates
each F32 RHS activation value to the BF16 value range by clearing the lower 16
bits of the FP32 representation, then performs the internal block-scale and
E2M1 magnitude-selection arithmetic with FP32 multiply/add/compare operations.
The E4M3 block scale is generated from FP32 bit fields, and FP4 magnitudes are
selected with nearest-neighbor thresholds whose exact ties choose the lower code,
matching the existing table-search nearest-neighbor behavior.

The implementation avoids runtime division, FP8 conversion intrinsics,
lookup-table nearest-neighbor searches, and special math functions in the BF16
quantization inner path. Dynamic RHS scale discovery uses the same BF16-truncated
values for `amax`.

### `GGML_CUDA_NVFP4_BF16_QUANT_BF16_INTERNAL`

Enables an experimental BF16-precision internal arithmetic variant of the BF16
trunc-NN NVFP4 RHS activation quantizer. Default: off.

This switch only changes behavior when both `GGML_CUDA_NVFP4_BF16_QUANT=1` and
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1` are also enabled. The activation input
is still truncated to BF16 bits, and block maximum search still compares BF16
absolute-value bits. With this switch enabled, the target and nearest-neighbor
threshold multiply/add operations used for E2M1 magnitude selection are
truncated to the BF16 value range before comparison. The default blockscale
calculation still uses the FP32 tensor/global scale arithmetic unless
`GGML_CUDA_NVFP4_BF16_QUANT_BF16_BLOCK_SCALE=1` is also enabled.

### `GGML_CUDA_NVFP4_BF16_QUANT_BF16_BLOCK_SCALE`

Enables an experimental BF16-precision blockscale calculation for the BF16
trunc-NN NVFP4 RHS activation quantizer. Default: off.

This switch only changes behavior when `GGML_CUDA_NVFP4_BF16_QUANT=1`,
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1`, and
`GGML_CUDA_NVFP4_BF16_QUANT_BF16_INTERNAL=1` are also enabled. The activation
input is truncated to BF16 bits and group maximum search compares BF16
absolute-value bits. In addition, the tensor/global scale operand is truncated
to the BF16 value range and blockscale multiply operations are truncated after
each multiply. This models a lower-cost hardware path where both blockscale and
quantized-value selection use BF16-like arithmetic while preserving the
hardware-friendly bit-field E4M3 scale encoder.

### `GGML_CUDA_NVFP4_TRUNC_BF16_INPUT`

Enables an experimental pre-quantization truncation step in the CUDA native NVFP4
FP32 nearest-neighbor RHS activation quantizer. Default: off.

When enabled and the BF16 trunc-NN path is disabled, the FP32 RHS activation
value is first truncated to the BF16 value range by clearing the lower 16 bits
of the FP32 representation. The existing FP32 nearest-neighbor NVFP4 block-scale
and E2M1 code selection then runs on the truncated value. Dynamic RHS scale
discovery uses the same truncated values for its `amax` computation.

Initial scope: `GGML_TYPE_NVFP4 x GGML_TYPE_F32 -> GGML_TYPE_F32` CUDA native
matmul activation quantization. This switch is intended to model callers whose
activation input values are already BF16-truncated while preserving the existing
nearest-neighbor NVFP4 quantizer.

## NVFP4 K-Cache Outlier Sidecar

### `LLAMA_NVFP4_KCACHE_OUTLIER`

Enables the compact NVFP4 K-cache outlier sidecar. Default: off.

When this switch is off, `--cache-type-k nvfp4` uses the normal NVFP4 K-cache
path with per-row K global scales and no outlier extraction/correction.

When this switch is on and hybrid FP8 K-cache is not enabled, each NVFP4 K-cache
layer uses the balanced per-layer threshold and compact capacity profile fixed
in `src/llama-kv-cache-nvfp4-outlier-config.h`. The K-cache residual
quantization uses the layer threshold as tensor amax for the K global scale.

When this switch is on together with
`LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8=1`, the selected high/medium layers are
stored as FP8(E4M3+E8M0 block32), and the remaining NVFP4 K-cache layers use the
balanced threshold profile and current context-specific capacity profile fixed
in `src/llama-kv-cache-nvfp4-outlier-config.h`.

### `LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE`

Selects the full-NVFP4 K-cache outlier threshold/capacity profile when
`LLAMA_NVFP4_KCACHE_OUTLIER=1` and hybrid FP8 K-cache is not enabled. Default:
unset, which uses the original `balanced` profile.

Supported values:

- `new`: use the ratio-1e-4 profile snapshot in
  `docs/development/nvfp4-kcache-outlier-thresholds/profiles/ratio-1e4/`.
  Raw evidence remains in
  `experiments/20260605T072559Z-kcache-outlier-threshold-ratio-sweep/` and
  `experiments/20260605T081206Z-kcache-outlier-ratio1e4-default-ppl/`.
- `bf16`: use the opt-in BF16/new-FP4-quantizer profile for full NVFP4 K/V-cache
  experiments. This profile is intended for runs with
  `GGML_CUDA_NVFP4_BF16_QUANT=1`,
  `GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1`, and
  `GGML_CUDA_TRUNC_ENABLE=1`. It currently starts from the ratio-1e-4 `new`
  profile values until a fresh BF16/new-FP4 sweep is applied.

### `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD`

Overrides the per-layer balanced NVFP4 K-cache outlier threshold with one global
absolute-value threshold. Default: unset/off.

This is intended for threshold sweep diagnostics. When unset,
`LLAMA_NVFP4_KCACHE_OUTLIER=1` uses the selected per-layer profile.

### `LLAMA_NVFP4_KCACHE_OUTLIER_OVERFLOW_LOG`

Enables the Release-mode compact sidecar overflow diagnostic for NVFP4 K-cache
outlier extraction. Default: unset/off.

When enabled, each extraction may copy compact sidecar metadata to the host and
synchronize the stream to print overflow diagnostics if any touched row exceeds
its compact row capacity. When disabled, Release execution skips this diagnostic
host-copy path. Debug builds still print the existing detailed outlier counts
under `#ifndef NDEBUG`.

### `LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8`

Switch B. Enables the fixed high/medium hybrid FP8 K-cache layer set:

```text
0,1,4,5,6,8,10,11,12,14,23,35
```

Default: off. This switch only affects `--cache-type-k nvfp4`. On its own, it
does not enable NVFP4 K-cache outlier sidecar; combine it with switch A
`LLAMA_NVFP4_KCACHE_OUTLIER=1` to run the current hybrid outlier configuration.

Scripts and reusable profile snapshots for deriving a new balanced profile from
threshold sweep artifacts live in:

```text
docs/development/nvfp4-kcache-outlier-thresholds/
docs/development/nvfp4-kcache-outlier-thresholds/scripts/parse-kcache-outlier-threshold-sweep.py
docs/development/nvfp4-kcache-outlier-thresholds/scripts/derive-kcache-outlier-balanced-config.py
docs/development/nvfp4-kcache-outlier-thresholds/scripts/run-kcache-outlier-balanced-experiment.sh
```

### `LLAMA_NVFP4_KCACHE_OUTLIER_DETERMINISTIC_FILL`

Diagnostic switch for the compact NVFP4 K-cache outlier sidecar. Default:
unset/off.

When enabled, CUDA fills each compact outlier row in ascending column order
using a slow deterministic kernel instead of the default parallel atomic fill.
This is intended to isolate whether compact sidecar entry order affects KQ
correction and PPL. It should not be used for performance measurements.

### `LLAMA_NVFP4_KCACHE_OUTLIER_NO_CORRECTION`

Diagnostic switch for the compact NVFP4 K-cache outlier sidecar. Default:
unset/off.

When enabled, CUDA still extracts outliers and quantizes the residual K cache,
but skips applying the outlier correction to KQ. This isolates residual K-cache
quantization from correction accumulation behavior. It is expected to change
model quality and should not be used as a correctness mode.

### `LLAMA_NVFP4_KCACHE_OUTLIER_FINGERPRINT`

Diagnostic switch for the compact NVFP4 K-cache outlier sidecar. Default:
unset/off.

When enabled, CUDA logs host-side hashes of the touched sidecar counts,
offsets, compact indices, compact values, residual amax rows, destination row
ranges, and source F32 K activation aggregates after each extract operation
when stream capture allows host copies. The source aggregates include
commutative sums/xors so multiple microbatch logs can be combined and compared
with a larger-ubatch extract. This is for comparing ubatch-dependent sidecar
contents and upstream K activation values, and is not suitable for performance
measurements.

## NVFP4 V-Cache

### `LLAMA_NVFP4_VCACHE_PER_BLOCK_SCALE`

Enables the old per-block external-scale path for NVFP4 V-cache
quantization. Default: off.

When this switch is off, the default NVFP4 V-cache path uses one per-tensor
global scale for all V-cache layers and streams. The default scalar is derived
from the wiki calibration nearest-rank P90 absmax:

```text
absmax = 80.428
global_scale = 6 * 224 / 80.428 = 16.7106
```

When this switch is on, the V cache stores external scales per row and per
16-token block, matching the older V-cache quantization behavior. This is kept
for A/B runs and debugging.

### `LLAMA_NVFP4_VCACHE_LAYER_GLOBAL_SCALE`

Enables the per-layer JSON global-scale path. Default: off.

This switch has priority over `LLAMA_NVFP4_VCACHE_PER_BLOCK_SCALE`.
Set it to `1` to load the historical default JSON file
`experiments/qwen3-8b-v-layer-absmax.json`, or set it to a JSON path accepted by
`llama_nvfp4_vcache_load_layer_absmax()`.

When enabled, each layer gets one global scale computed as:

```text
global_scale = 6 * 224 / layer_absmax
```

### `LLAMA_NVFP4_VCACHE_FAST_UPDATE`

Controls CUDA NVFP4 V-cache single-token fast update. Default: on.

When enabled, CUDA set_rows may patch single-token updates without requantizing
the whole 16-token V-cache block when the existing block scale remains valid.
Set to `0` to force the full block requantization path.
