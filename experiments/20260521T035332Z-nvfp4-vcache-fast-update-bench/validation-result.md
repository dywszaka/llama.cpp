# Validation Result

Date: 2026-05-21 UTC

## Completed

- Built `llama-bench` successfully:

```bash
cmake --build build_cuda --target llama-bench -j2
```

- Validated shell syntax:

```bash
bash -n experiments/20260521T035332Z-nvfp4-vcache-fast-update-bench/run-bench.sh
bash -n experiments/20260521T035332Z-nvfp4-vcache-fast-update-bench/parse-results.sh
```

- Confirmed `llama-bench` accepts the new unified KV option:

```bash
build_cuda/bin/llama-bench --help
```

Relevant help output includes:

```text
-kvu, --kv-unified <0|1>                  (default: 0)
```

- Confirmed `llama-bench` accepts `--cache-type-v nvfp4 --kv-unified 1` with a
  CUDA smoke run.

- Ran the full benchmark:

```bash
experiments/20260521T035332Z-nvfp4-vcache-fast-update-bench/run-bench.sh
```

Generated artifacts:

- `logs/*.raw.log`
- `logs/*.stderr.log`
- `metrics/01-baseline.json`
- `metrics/02-nvfp4-fast-update-off.json`
- `metrics/03-nvfp4-fast-update-on.json`
- `metrics/results.csv`
- `metrics/results.json`
- `summary.md`

## Results

Primary decode metric, `tg128`:

| Run | tok/s | stdev tok/s |
| --- | ---: | ---: |
| baseline `f16/f16` | 86.10 | 6.58 |
| `f16/nvfp4`, fast_update off | 80.13 | 3.08 |
| `f16/nvfp4`, fast_update on | 80.02 | 3.10 |

`fast_update on` vs `off`: `-0.11 tok/s`, `-0.14%`.

## Switch Confirmation

- Off run logged `LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=0 -> disabled`.
- On run logged `LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=1 -> enabled`.
- Both NVFP4 runs logged `kv_unified = true`.
- Both NVFP4 runs logged `CUDA NVFP4 V-cache p*v matmul path=cublasLt-fp4`.

## Earlier Blocker

Before GPU recovery, the benchmark was blocked by:

```text
nvidia-smi: Failed to initialize NVML: Unknown Error
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
```

After GPU recovery, the benchmark completed successfully.
