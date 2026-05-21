# Task Plan: NVFP4 V-cache FP4 P*V cuBLASLt Path

## Goal
Add an experimental cuBLASLt Tensor Core FP4*FP4 path for NVFP4 V-cache P*V, then validate correctness and run PPL.

## Current Status
- Phase 1: context and design - complete
- Phase 2: focused failing test hook - pending
- Phase 3: implementation - pending
- Phase 4: focused CUDA validation - pending
- Phase 5: PPL experiment - pending

## Design Decisions
- Add new switch `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV_LT`, default off.
- Existing `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV=1` remains required for P dynamic NVFP4 quantization.
- New path stages V/P `block_nvfp4` into cuBLASLt packed FP4 data and UE4M3 scale channels.
- Fold V-cache external per-block `v_scale[row, block]` into A scale channel during V staging.
- Leave P per-row `p_scale[col]` out of B scale channel and multiply output columns after matmul.
- If cuBLASLt path is unavailable or fails, fall back to existing `k_vcache_nvfp4_matmul_fp4_p_4d`.

## Verification Plan
- Build `test-vcache-nvfp4-matmul`.
- Run focused test without LT switch.
- Run focused test with `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV=1`.
- Run focused test with both `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV=1` and `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV_LT=1`.
- If focused validation passes, run PPL using the existing current-code PPL parameters plus the new LT switch.

## Errors Encountered
| Error | Attempt | Resolution |
| --- | --- | --- |
| Lt matmul returned CUBLAS_STATUS_NOT_SUPPORTED and fallback then mismatched | focused test with LT switch | Need fix descriptor or avoid dst pollution on failed Lt before fallback |
| PPL run aborted with CUDA OOM in native NVFP4 ffn_gate repack during warmup | first current-LT PPL run | Investigate GPU memory/processes and rerun with a clean GPU or lower memory pressure |

---

# Task Plan: 2026-05-20 Three-Run PPL Experiment

## Goal
Run three PPL configurations and preserve scripts, logs, parsed metrics, and a concise summary under `experiments/`.

## Runs
- baseline: `EXPERI.md` llama-perplexity baseline, K/V cache `f16/f16`.
- layer-global-scale: same as existing `run-layer-global-scale.sh`, K/V cache `f16/nvfp4`.
- layer-global-scale-k-nvfp4: same as layer-global-scale, with `--cache-type-k nvfp4`.

## Status
- Phase 1: inspect baseline/script context - complete
- Phase 2: create experiment folder/scripts - complete
- Phase 3: run baseline - complete
- Phase 4: run layer-global-scale - complete
- Phase 5: run K-cache NVFP4 variant - complete
- Phase 6: parse and summarize - complete

---

# Task Plan: Wiki P99 Uniform Absmax PPL Experiment

## Goal
Add one PPL experiment to the existing wiki-vs-qwen layer-global-scale experiment family using a uniform absmax equal to the nearest-rank P99 of `wiki-v-layer-absmax.json`.

## Status
- Phase 1: confirm P99 value and existing experiment format - complete
- Phase 2: create P99 JSON and run script - complete
- Phase 3: run PPL and capture raw log - complete
- Phase 4: parse metrics and update summary - complete
- Phase 5: verify artifacts and report result - complete

---

# Task Plan: Wiki P95 Uniform Absmax PPL Experiment

## Goal
Add one PPL experiment to the existing wiki-vs-qwen layer-global-scale experiment family using a uniform absmax equal to the nearest-rank P95 of `wiki-v-layer-absmax.json`.

## Status
- Phase 1: confirm P95 value and existing experiment format - complete
- Phase 2: create P95 JSON and run script - complete
- Phase 3: run PPL and capture raw log - complete
- Phase 4: parse metrics and update summary - complete
- Phase 5: verify artifacts and report result - complete

## Decisions
- Use nearest-rank percentile, consistent with the existing P90/P95/P99/P50 uniform experiments.
- For 36 values, `ceil(0.95 * 36) = 35`; selected P95 absmax is `129.048`.
- Use the same llama-perplexity parameters and environment switches as the existing wiki P90/P99/P50 runs, changing only the layer absmax JSON and raw log path.

## Errors Encountered
| Error | Attempt | Resolution |
| --- | --- | --- |
| CUDA initialization failed (`no CUDA-capable device is detected`), then CPU fallback aborted in NVFP4 path | first P95 PPL run | Discard as non-comparable run; check GPU/NVML availability and rerun only when CUDA is visible |

## Decisions
- Use nearest-rank percentile, consistent with the existing P90/P50 uniform experiments.
- For 36 values, `ceil(0.99 * 36) = 36`; selected P99 absmax is `130.771`.
- Use the same llama-perplexity parameters and environment switches as the existing wiki P90/P50 runs, changing only the layer absmax JSON and raw log path.

---

# Task Plan: 2026-05-21 NVFP4 V-cache Fast Update llama-bench Experiment

## Goal
Compare decode speed for `cache-type-v=nvfp4` with `LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=0` and `=1` using `llama-bench`, with a `f16/f16` baseline for reference.

## Runs
- baseline: `EXPERI.md` llama-bench baseline, K/V cache `f16/f16`.
- nvfp4-fast-update-off: same llama-bench parameters, K/V cache `f16/nvfp4`, NVFP4 V-cache enabled, fast update disabled.
- nvfp4-fast-update-on: same as previous, fast update enabled.

## Status
- Phase 1: inspect llama-bench and NVFP4 V-cache requirements - complete
- Phase 2: add llama-bench `--kv-unified` support for NVFP4 V-cache - complete
- Phase 3: create experiment folder/scripts - complete
- Phase 4: build and validate script syntax/help - complete
- Phase 5: run benchmark and parse metrics - complete
- Phase 6: summarize and commit verified changes - complete

## Decisions
- Use `llama-bench` generation test `tg128` as the primary fast_update signal because `test_gen()` calls `llama_decode()` with one token per iteration.
- Keep the baseline command aligned with `EXPERI.md`; add `--kv-unified 1` only to NVFP4 V-cache runs because that runtime path requires unified KV cache.
- Enable verbose logging for the bench runs so one-shot experiment switch logs are captured in stderr.

## Errors Encountered
| Error | Attempt | Resolution |
| --- | --- | --- |
| `llama-bench` did not support `--kv-unified` | preparing NVFP4 V-cache bench command | Added narrow `--kv-unified/-kvu` option and passed it to `llama_context_params.kv_unified` |
| CUDA unavailable: `nvidia-smi` failed with `Failed to initialize NVML: Unknown Error`; `llama-bench` reported no CUDA-capable device | attempted local smoke benchmark | Recorded as blocked; scripts are ready to run when CUDA/NVML recovers |
| `llama-bench --cache-type-v nvfp4` failed with invalid parameter | first full bench run after GPU recovered | Fixed llama-bench K/V cache type parser to match common cache allow-lists; V cache now accepts `nvfp4` and FP8 V-cache types |

## Results
- Baseline `tg128`: `86.10 tok/s` (`stddev 6.58`).
- NVFP4 V-cache fast_update off `tg128`: `80.13 tok/s` (`stddev 3.08`).
- NVFP4 V-cache fast_update on `tg128`: `80.02 tok/s` (`stddev 3.10`).
- fast_update on vs off: `-0.11 tok/s`, `-0.14%`.
