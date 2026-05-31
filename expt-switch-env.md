# Experiment Switch Environment Variables

## FP8 E4M3 E8M0 32 K-Cache

### `--cache-type-k fp8_e4m3_e8m0_32`

Enables the experimental FP8(E4M3+E8M0 block32) K-cache path. Default: off
because the standard K-cache type default remains `f16`.

Initial scope: CUDA non-flash KQ with KQV offload enabled. During KQ, the stored
K cache is `GGML_TYPE_FP8_E4M3_E8M0_32`; the F32 Q operand is quantized to
temporary FP8 block32 inside the native CUDA FP8 matmul path before cuBLASLt
execution.

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

Enables the experimental NVFP4 K-cache outlier sidecar path. Default: off.

When enabled for `K cache = nvfp4`, K values whose absolute value is above the
configured threshold are extracted before cache quantization. The residual K
positions are quantized as zero, and the extracted signed F32 values are added
back into KQ by multiplying them with the corresponding pre-quantization F32 Q
values. By default, residual K cache quantization keeps the original per-row
global scale based on residual row amax, and dynamic Q quantization keeps its
original per-row amax behavior. In the NVFP4 K-cache outlier path, outlier
sidecar tensors are bound and residual K cache quantization instead uses the
threshold as the per-tensor amax:

```text
K global_scale = 1344 / LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD
```

During native NVFP4 KQ with dynamic Q quantization, Q computes one runtime
per-tensor amax across the active Q matrix and derives its own global scale from
that value. Q does not use the K outlier threshold for this scale.

Initial scope: CUDA NVFP4 K-cache, non-flash-attention KQ.

### `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD`

Absolute-value threshold for K-cache outlier extraction. Default: `16`.

The predicate is:

```text
abs(K) > LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD
```

### `LLAMA_NVFP4_KCACHE_OUTLIER_LAYER_THRESHOLDS`

Optional comma-separated per-layer absolute-value thresholds for NVFP4 K-cache
outlier extraction. Default: unset.

When set, layer `i` uses entry `i` from this list. Layers beyond the provided
entries fall back to `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD`. This switch only
affects NVFP4 K-cache outlier set_rows extraction and residual K quantization;
it is ignored for F16 K-cache outliers.

### `LLAMA_NVFP4_KCACHE_OUTLIER_MAX`

Maximum number of outlier values stored per K-cache row. Default: `32`.

The count sidecar records the true number of outliers per row. The value/index
sidecars store only the first `LLAMA_NVFP4_KCACHE_OUTLIER_MAX` entries, and KQ
correction uses only stored entries. Use the log switch below to detect overflow.

### `LLAMA_NVFP4_KCACHE_OUTLIER_COMPACT`

Enables compact fixed-capacity sparse-pool storage for NVFP4 K-cache outliers.
Default: off.

When off, the sidecar uses the legacy fixed per-row `count/index/value` layout.
When on, each row stores `count` and `offset`, while outlier `index/value`
entries are appended to a per-layer, per-stream sparse pool. The pool is not
compacted or reclaimed during row rewrites; `clear(true)` resets the backing
buffer and cursor. If the pool capacity is exhausted, later rows keep their
count but receive no stored entries, so KQ correction can only restore stored
entries.

### `LLAMA_NVFP4_KCACHE_OUTLIER_CAPACITY_RATIO`

Fraction of dense K-cache elements reserved as compact outlier pool entries.
Default: `0.004`.

The allocated pool capacity per layer and stream is:

```text
max(LLAMA_NVFP4_KCACHE_OUTLIER_MIN_CAPACITY,
    ceil(kv_size * n_embd_k_gqa * ratio))
```

### `LLAMA_NVFP4_KCACHE_OUTLIER_MIN_CAPACITY`

Minimum compact outlier pool entries allocated per layer and stream. Default:
`kv_size`.

Set this to a smaller value for capacity-ratio experiments that need the actual
pool size to match the ratio-derived capacity. For example, with `n_ctx=512`,
`n_embd_k_gqa=1024`, and
`LLAMA_NVFP4_KCACHE_OUTLIER_CAPACITY_RATIO=0.0003108978271484375`, setting this
to `1` makes the allocated compact capacity `163` entries instead of the
default minimum of `512`.

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

### `LLAMA_F16_KCACHE_OUTLIER_COMPACT`

Enables compact fixed-capacity sparse-pool storage for F16 K-cache outliers.
Default: off. Semantics match `LLAMA_NVFP4_KCACHE_OUTLIER_COMPACT`.

### `LLAMA_F16_KCACHE_OUTLIER_CAPACITY_RATIO`

Fraction of dense K-cache elements reserved as compact F16 outlier pool entries.
Default: `0.004`. Capacity calculation matches
`LLAMA_NVFP4_KCACHE_OUTLIER_CAPACITY_RATIO`.

### `LLAMA_F16_KCACHE_OUTLIER_MIN_CAPACITY`

Minimum compact outlier pool entries allocated per layer and stream for the F16
K-cache outlier sidecar. Default: `kv_size`. Semantics match
`LLAMA_NVFP4_KCACHE_OUTLIER_MIN_CAPACITY`.

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
