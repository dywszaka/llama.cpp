# Experiment Switch Environment Variables

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
