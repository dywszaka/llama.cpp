# Experiment Switch Environment Variables

## CUDA NVFP4 Native Matmul

### `GGML_CUDA_NVFP4_NATIVE_ACT_DUMP`

Exports the first activation tensor quantized inside
`ggml_cuda_mul_mat_nvfp4_native`. Default: off.

Set this switch to a directory path. On the first native NVFP4 activation
quantization, the CUDA path writes:

- `activation-before-f32-hi16.bin`: pre-quantization activation values, exported
  as the high 16 bits of the F32 bit pattern. The dump path does not round; when
  `GGML_CUDA_TRUNC_ENABLE=1` has already rounded tensors to BF16-representable
  F32 values, this is a direct truncation/export of those bits;
- `activation-after-nvfp4.bin`: post-quantization `block_nvfp4` values;
- `metadata.json`: shape, scale mode, global scale, and file metadata used by
  `scripts/compare-nvfp4-activation-dump.py`.

This diagnostic is intended for local baseline validation and prints once. It
synchronizes the active CUDA stream while exporting and should stay disabled for
performance measurements.

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
