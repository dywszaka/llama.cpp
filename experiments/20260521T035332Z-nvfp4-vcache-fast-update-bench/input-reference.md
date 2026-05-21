# Input And Runtime Reference

Date: 2026-05-21 UTC

This experiment uses the `EXPERI.md` `llama-bench` baseline as the starting
point.

## Baseline Contract

- Binary: `${ROOT_DIR}/build_cuda/bin/llama-bench`
- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- K cache type: `--cache-type-k f16`
- V cache type: `--cache-type-v f16`
- GPU layers: `--n-gpu-layers 40`
- Batch size: `--batch-size 2048`
- UBatch size: `--ubatch-size 512`
- Threads: `-t 32`
- Prompt tokens: `-p 512`
- Generation tokens: `-n 128`

## Experiment Deltas

- `nvfp4-fast-update-off` changes V cache to `--cache-type-v nvfp4`, enables
  `LLAMA_EXPERIMENT_NVFP4_VCACHE=1`, adds `--kv-unified 1`, and sets
  `LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=0`.
- `nvfp4-fast-update-on` changes only
  `LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=1` relative to the off run.

`--kv-unified 1` is required by the NVFP4 V-cache runtime path; the baseline
keeps the `EXPERI.md` f16/f16 cache configuration for reference.
