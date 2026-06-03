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

Enables the experimental BF16-round NVFP4 RHS activation quantizer in the CUDA
native NVFP4 matmul path. Default: off.

When enabled, CUDA native NVFP4 matmul converts each F32 RHS activation value to
rounded BF16 bits first, then uses the QuantBF16-style fixed-point NVFP4 block
scale and E2M1 magnitude threshold algorithm. When disabled, the existing FP32
nearest-neighbor NVFP4 quantizer remains active.

Initial scope: `GGML_TYPE_NVFP4 x GGML_TYPE_F32 -> GGML_TYPE_F32` CUDA native
matmul activation quantization, including static input-scale and dynamic
per-row/per-tensor RHS scale modes. This switch does not change stored NVFP4
weights or generic dequantization paths.

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
current hybrid threshold/capacity profile fixed in
`src/llama-kv-cache-nvfp4-outlier-config.h`.

### `LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8`

Switch B. Enables the fixed high/medium hybrid FP8 K-cache layer set:

```text
0,1,4,5,6,8,10,11,12,14,23,35
```

Default: off. This switch only affects `--cache-type-k nvfp4`. On its own, it
does not enable NVFP4 K-cache outlier sidecar; combine it with switch A
`LLAMA_NVFP4_KCACHE_OUTLIER=1` to run the current hybrid outlier configuration.

Scripts for deriving a new balanced profile from threshold sweep artifacts live
in:

```text
scripts/parse-kcache-outlier-threshold-sweep.py
scripts/derive-kcache-outlier-balanced-config.py
scripts/run-kcache-outlier-balanced-experiment.sh
```

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

Enables CUDA NVFP4 V-cache single-token fast update. Default: off.

When enabled, CUDA set_rows may patch single-token updates without
requantizing the whole 16-token V-cache block.
