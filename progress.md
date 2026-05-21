# Progress: NVFP4 V-cache FP4 P*V cuBLASLt Path

## 2026-05-19
- Read current V-cache matmul implementation and existing NVFP4 native matmul path.
- Confirmed user-approved design: fold V external per-block scale into A scale channel; post-scale only P per-row scale.
- Created planning files.

- Added initial LT switch, V/P staging, post-scale helper, and dispatch fallback.

- Root cause for first LT failure: unpadded cols triggered cuBLASLt unsupported shape; changed LT path to pad cols to 16 and write to temporary C before copying valid columns to dst.

- Changed LT path to stage all slice outputs in a large temporary buffer and write to dst only after every Lt call succeeds.

- Corrected LT scale-channel staging: use raw E4M3 block scale magnitude for V/P scale channels; V external per-block scale is folded into A channel.

- Updated focused test tolerance for LT path because folding V F32 per-block scale into UE4M3 Lt scale channel adds extra scale quantization. Added one-shot active log for successful Lt path.

- Test reference now computes LT-equivalent V dequantization by quantizing folded V scale to E4M3 before dot comparison.

- Focused LT validation passed with active cuBLASLt log. Next: build and run PPL.

---

# Progress: 2026-05-20 Three-Run PPL Experiment

- Inspected `EXPERI.md`, existing layer-global-scale script, binary/model/prompt/scale-file availability, and current GPU memory state.
- Planned dedicated experiment directory with three scripts: baseline, layer-global-scale, and layer-global-scale with K cache NVFP4.
- Created `experiments/ppl-three-run-layer-global-scale-20260520T024947Z/` with scripts, input reference, logs, metrics, and summary.
- Completed baseline PPL: `10.4023 +/- 0.08130`.
- Completed layer-global-scale PPL with K `f16`, V `nvfp4`: `10.2732 +/- 0.07932`.
- Completed layer-global-scale PPL with K `nvfp4`, V `nvfp4`: `10.5416 +/- 0.08113`.

---

# Progress: Wiki P99 Uniform Absmax PPL Experiment

- Confirmed nearest-rank P99 over the 36 wiki layer absmax values selects rank 36, `absmax=130.771`, giving `global_scale=1344/130.771=10.277508`.
- Created P99 uniform config and run script under `experiments/ppl-layer-global-scale-wiki-vs-qwen-20260520/`.
- Completed P99 uniform PPL run: `10.2745 +/- 0.07935`, prompt throughput `2674.47 tok/s`.

---

# Progress: Wiki P95 Uniform Absmax PPL Experiment

- Confirmed nearest-rank P95 over the 36 wiki layer absmax values selects rank 35, `absmax=129.048`, giving `global_scale=1344/129.048=10.414729`.
- Created P95 uniform config and run script under `experiments/ppl-layer-global-scale-wiki-vs-qwen-20260520/`.
- First P95 run was invalid: CUDA initialization failed, `nvidia-smi` also failed with `Failed to initialize NVML: Unknown Error`, and the process fell back to CPU then aborted in the NVFP4 CPU path.
- After GPU/NVML recovered, reran P95 on CUDA successfully: `10.2692 +/- 0.07931`, prompt throughput `2683.65 tok/s`.

---

# Progress: 2026-05-21 NVFP4 V-cache Fast Update llama-bench Experiment

- Confirmed `llama-perplexity` is not a good fast_update workload because its baseline path uses batched prompt eval rather than single-token decode.
- Switched experiment design to `llama-bench` after user direction.
- Added `--kv-unified/-kvu` plumbing to `tools/llama-bench/llama-bench.cpp` so NVFP4 V-cache benchmark contexts can be initialized.
- Created `experiments/20260521T035332Z-nvfp4-vcache-fast-update-bench/` with run script, parser, input reference, summary, and validation result.
- Built `llama-bench` successfully and validated script syntax/help output.
- Could not run the requested CUDA benchmark because local CUDA/NVML is unavailable: `nvidia-smi` reports `Failed to initialize NVML: Unknown Error`.
- After GPU recovered, the first full run completed baseline but failed at `--cache-type-v nvfp4`; fixed the local llama-bench K/V cache type parser to allow NVFP4 V-cache consistently with `common/arg.cpp`.
- Rebuilt `llama-bench`, reran all three benchmark cases successfully, and generated metrics/summary artifacts.
- Results: baseline `tg128=86.10 tok/s`; NVFP4 V-cache fast_update off `80.13 tok/s`; fast_update on `80.02 tok/s`; on vs off delta `-0.14%`.
