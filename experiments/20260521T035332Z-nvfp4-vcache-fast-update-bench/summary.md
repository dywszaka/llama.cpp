# NVFP4 V-cache Fast Update llama-bench Summary
Date: 2026-05-21 UTC
## Parameters
- Binary: `build_cuda/bin/llama-bench`
- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- GPU layers: `--n-gpu-layers 40`
- Batch: `--batch-size 2048 --ubatch-size 512`
- Threads: `-t 32`
- Tests: `pp512` and `tg128`; `tg128` is the primary fast_update metric.
- Repetitions: `5` unless `BENCH_REPS` was set when running `run-bench.sh`.

## Results
| Run | Test | K cache | V cache | fast_update | kv_unified | tok/s | stdev tok/s |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| baseline | pp | `f16` | `f16` | - | false | 4181.25 | 20.28 |
| baseline | tg | `f16` | `f16` | - | false | 86.10 | 6.58 |
| nvfp4-fast-update-off | pp | `f16` | `nvfp4` | false | true | 3094.56 | 7.51 |
| nvfp4-fast-update-off | tg | `f16` | `nvfp4` | false | true | 80.13 | 3.08 |
| nvfp4-fast-update-on | pp | `f16` | `nvfp4` | true | true | 3093.27 | 10.04 |
| nvfp4-fast-update-on | tg | `f16` | `nvfp4` | true | true | 80.02 | 3.10 |

## Decode Deltas
| Comparison | tok/s delta | tok/s delta % |
| --- | ---: | ---: |
| nvfp4 fast_update off vs baseline | -5.97 | -6.94% |
| nvfp4 fast_update on vs baseline | -6.09 | -7.07% |
| fast_update on vs off | -0.11 | -0.14% |

## Validation Notes
- `llama-bench` reports speed only; it does not produce a PPL/accuracy metric.
- Use existing PPL experiment results for precision context, or run a separate PPL sanity check if accuracy must be revalidated for this exact build.
- Raw stdout JSON and stderr logs are preserved under `metrics/` and `logs/`.
