# Input Reference

Synthetic F32 tensor fixture for offline evaluator validation.

- Manifest: `manifest.json`
- Raw tensor: `k0.bin`
- Tensor: `name=k-synthetic`, `kind=k`, `dtype=f32`, `ne=[16,1,1,1]`, `nb=[4,64,64,64]`, `byte_size=64`
- Values: 16 little-endian float32 values from `-8.0` through `7.0`
- Baseline algorithm: NVFP4 reference quantize/dequantize via `quantize_row_nvfp4_ref()` and `dequantize_row_nvfp4()` with `--global-scale 1`
