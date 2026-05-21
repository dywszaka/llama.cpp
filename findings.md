# Findings: NVFP4 V-cache FP4 P*V cuBLASLt Path

## Existing Path
- `ggml/src/ggml-cuda/vcache-nvfp4-matmul.cu` contains `k_vcache_nvfp4_matmul_fp4_p_4d`.
- Current FP4-P path quantizes P to `block_nvfp4`, then decodes V and P inside a custom CUDA kernel and accumulates scalar FP32.
- Existing switch: `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV`.

## Scale Semantics
- V block contribution is `fp4_v * e_v * v_scale[row, block]`.
- P block contribution is `fp4_p * e_p * p_scale[p_row]`.
- `p_scale[p_row]` is constant across the whole reduction for one output column and can be post-multiplied.
- `v_scale[row, block]` varies inside the reduction and must be folded into the cuBLASLt A scale channel during staging.

## Reuse Points
- `ggml/src/ggml-cuda/nvfp4-matmul.cu` has the cuBLASLt FP4 descriptor pattern.
- It uses `CUDA_R_4F_E2M1`, `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`, packed data channels, and separate scale channels.
- Scale-channel attributes are available only when `CUBLAS_VERSION >= 130000` or `CUBLAS_VER_MAJOR >= 13`.

- LT focused test uses wider absolute tolerance (2.0) because V per-block F32 scale is folded into UE4M3 Lt channel, unlike custom kernel F32 multiplication.

- Restricted LT path to kv_size >= 512 because smaller shapes hit cuBLASLt NOT_SUPPORTED and are not relevant to target decode/PPL path.

- Corrected LT test reference for CUDA_R_4F_E2M1 value semantics: ggml kvalues table is doubled, so LT-equivalent reference uses 0.5 * kvalues.

---

# Findings: 2026-05-20 Three-Run PPL Experiment

- Baseline PPL parameters are documented in `EXPERI.md`: `build_cuda/bin/llama-perplexity`, model `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`, prompt `data/wikitext/wikitext-2-raw/wiki.test.raw`, `--cache-type-k f16`, `--cache-type-v f16`, `--n_gpu_layers 40`, `--batch-size 512`, `--ubatch-size 512`, `-t 32`, `-c 512`.
- Existing `run-layer-global-scale.sh` lives at `experiments/ppl-nvfp4-vcache-layer-global-scale-20260519T080910Z/run-layer-global-scale.sh`.
- That script sets `LLAMA_EXPERIMENT_NVFP4_VCACHE=1`, `LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=1`, `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV=1`, `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV_LT=1`, and `LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE=experiments/qwen3-8b-v-layer-absmax.json`.
- Existing layer-global-scale script uses `--cache-type-k f16`, `--cache-type-v nvfp4`, and `--kv-unified`.
- GPU 0 has an existing `llama-server` process using about 10.8GB before these runs; if PPL fails from OOM, record it rather than terminating unrelated work.
- Results from `experiments/ppl-three-run-layer-global-scale-20260520T024947Z`:
  - baseline `f16/f16`: PPL `10.4023 +/- 0.08130`, prompt throughput `3424.30 tok/s`.
  - layer-global-scale `f16/nvfp4`: PPL `10.2732 +/- 0.07932`, prompt throughput `2680.75 tok/s`.
  - layer-global-scale `nvfp4/nvfp4`: PPL `10.5416 +/- 0.08113`, prompt throughput `2165.35 tok/s`.
- Second and third runs both logged `cuBLASLt FP4 P*V path active`.

---

# Findings: Wiki P99 Uniform Absmax PPL Experiment

- Existing experiment family: `experiments/ppl-layer-global-scale-wiki-vs-qwen-20260520/`.
- Existing P90/P50 uniform runs use nearest-rank percentile over the 36 values in `wiki-v-layer-absmax.json`.
- P99 nearest-rank over 36 values uses `ceil(0.99 * 36) = 36`; this selects the maximum wiki layer absmax, `130.771`.
- The V-cache layer-global-scale loader converts absmax to global scale with `global_scale = 1344 / absmax`, so the P99 uniform config should log approximately `global_scale=10.2775`.

---

# Findings: Wiki P95 Uniform Absmax PPL Experiment

- P95 nearest-rank over 36 values uses `ceil(0.95 * 36) = 35`; this selects the second-largest wiki layer absmax, `129.048`.
- The V-cache layer-global-scale loader converts absmax to global scale with `global_scale = 1344 / absmax`, so the P95 uniform config should log approximately `global_scale=10.4147`.

---

# Findings: 2026-05-21 NVFP4 V-cache Fast Update llama-bench Experiment

- `llama-perplexity` calls `llama_decode()`, but with the baseline `-c 512 --batch-size 512` path it evaluates prompt chunks in batches, so it does not exercise the `fast_update && n_tokens == 1` branch reliably.
- `tools/llama-bench/llama-bench.cpp` generation benchmark `test_gen()` calls `llama_decode(ctx, llama_batch_get_one(&token, 1))` for each generated token, which matches the CUDA set_rows fast update condition.
- NVFP4 V-cache runtime rejects initialization unless `LLAMA_EXPERIMENT_NVFP4_VCACHE=1`, `flash_attn=0`, KQV offload is enabled, and `kv_unified=1`.
- Existing `llama-bench` parser did not expose `kv_unified`; it needs a narrow `--kv-unified/-kvu` option to benchmark NVFP4 V-cache directly.
- `llama-bench` can emit JSON via `-o json`; JSON contains `avg_ts`, `stddev_ts`, `samples_ts`, `n_prompt`, and `n_gen`.
