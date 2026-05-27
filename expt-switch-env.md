# Experiment Switch Environment Variables

## FP8 E4M3 E8M0 32 K-Cache

### `--cache-type-k fp8_e4m3_e8m0_32`

Enables the experimental FP8(E4M3+E8M0 block32) K-cache path. Default: off
because the standard K-cache type default remains `f16`.

Initial scope: CUDA non-flash KQ with KQV offload enabled. During KQ, the stored
K cache is `GGML_TYPE_FP8_E4M3_E8M0_32`; the F32 Q operand is quantized to
temporary FP8 block32 inside the native CUDA FP8 matmul path before cuBLASLt
execution.

## NVFP4 K-Cache Outlier Sidecar

### `LLAMA_NVFP4_KCACHE_OUTLIER`

Enables the experimental NVFP4 K-cache outlier sidecar path. Default: off.

When enabled for `K cache = nvfp4`, K values whose absolute value is above the
configured threshold are extracted before cache quantization. The residual K
positions are quantized as zero, and the extracted signed F32 values are added
back into KQ by multiplying them with the corresponding pre-quantization F32 Q
values.

Initial scope: CUDA NVFP4 K-cache, non-flash-attention KQ.

### `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD`

Absolute-value threshold for K-cache outlier extraction. Default: `16`.

The predicate is:

```text
abs(K) > LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD
```

### `LLAMA_NVFP4_KCACHE_OUTLIER_MAX`

Maximum number of outlier values stored per K-cache row. Default: `32`.

The count sidecar records the true number of outliers per row. The value/index
sidecars store only the first `LLAMA_NVFP4_KCACHE_OUTLIER_MAX` entries, and KQ
correction uses only stored entries. Use the log switch below to detect overflow.

### `LLAMA_NVFP4_KCACHE_OUTLIER_LOG`

Enables outlier sidecar logging. Default: off.

When enabled, startup logs print the configured threshold and per-row storage
capacity. Runtime set_rows logs summarize rows processed, total outliers,
maximum outliers in a row, and rows whose true count exceeded storage capacity.

## F16 K-Cache Outlier Sidecar

### `LLAMA_F16_KCACHE_OUTLIER`

Enables the experimental F16 K-cache outlier sidecar path. Default: off.

When enabled for `K cache = f16`, K values whose absolute value is above the
configured threshold are extracted before cache write. The residual K-cache row
stores zero at those positions in F16, and the extracted signed F32 values are
added back into KQ by multiplying them with the corresponding F32 Q values.

Initial scope: CUDA F16 K-cache, non-flash-attention KQ.

### `LLAMA_F16_KCACHE_OUTLIER_THRESHOLD`

Absolute-value threshold for F16 K-cache outlier extraction. Default: `16`.

The predicate is:

```text
abs(K) > LLAMA_F16_KCACHE_OUTLIER_THRESHOLD
```

### `LLAMA_F16_KCACHE_OUTLIER_MAX`

Maximum number of outlier values stored per K-cache row. Default: `32`.

The count sidecar records the true number of outliers per row. The value/index
sidecars store only the first `LLAMA_F16_KCACHE_OUTLIER_MAX` entries, and KQ
correction uses only stored entries. Use the log switch below to detect overflow.

### `LLAMA_F16_KCACHE_OUTLIER_LOG`

Enables F16 K-cache outlier sidecar logging. Default: off.

When enabled, startup logs print the configured threshold and per-row storage
capacity. Runtime set_rows logs summarize rows processed, total outliers,
maximum outliers in a row, and rows whose true count exceeded storage capacity.

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
