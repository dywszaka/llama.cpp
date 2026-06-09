# Generating NVFP4 K-cache Outlier Balanced Profiles

This document is the reviewable workflow for deriving a new balanced threshold and compact capacity profile for `LLAMA_NVFP4_KCACHE_OUTLIER`.

Use it when the model, context size, prompt/data mix, cache layout, or runtime parameters change enough that the existing profile in `src/llama-kv-cache-nvfp4-outlier-config.h` is no longer trustworthy.

## Output

The workflow produces two per-layer arrays:

```cpp
static constexpr float llama_nvfp4_kcache_outlier_layer_thresholds_balanced[] = { ... };
static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities_balanced[] = { ... };
```

For hybrid FP8 mode, the same process can produce the capacity table used by the remaining NVFP4 layers, but the current script names and generated snippet target the balanced full-NVFP4 profile by default.

## Required Inputs

Record these in the experiment folder before running the sweep:

- Model path.
- Dataset or prompt source.
- Number of model layers.
- K row width used for density conversion, normally `n_embd_k_gqa = n_head_kv * head_dim`.
- Context size.
- Batch and ubatch size.
- GPU layer count.
- CPU thread count.
- CUDA device.
- K and V cache types.
- Whether hybrid FP8 K-cache layers are enabled.
- Build directory and git revision.

Start from `expt-baseline.md` and change only parameters required by the new model or the explicit experiment. If you change a baseline parameter, write the reason into the experiment summary.

## Artifact Layout

Create one timestamped directory:

```text
experiments/YYYYMMDDThhmmssZ-kcache-outlier-balanced-profile-{model-or-run}/
  input-reference.txt
  scripts/
  runs/
  results/
  summary.md
```

Store copied scripts under `scripts/`, raw threshold logs under `runs/`, and parsed or generated outputs under `results/`. After review, copy reusable profile outputs into `docs/development/nvfp4-kcache-outlier-thresholds/profiles/`; keep raw logs in `experiments/`.

Minimum copied scripts:

```bash
THRESHOLD_DIR="${ROOT_DIR}/docs/development/nvfp4-kcache-outlier-thresholds"
cp "${THRESHOLD_DIR}/scripts/parse-kcache-outlier-threshold-sweep.py" "${EXP_DIR}/scripts/"
cp "${THRESHOLD_DIR}/scripts/derive-kcache-outlier-balanced-config.py" "${EXP_DIR}/scripts/"
```

## Step 1: Choose Threshold Sweep Grid

Use a threshold grid wide enough to cover dense and sparse regimes for the new model. The current Qwen3-8B run used:

```text
16 20 24 28 32 40 48 56 64 80 96 112 128 160 192 224 256 320 384
```

For a new model, keep this grid for the first run unless there is clear evidence that K activations live at a very different scale. If all layers produce too many outliers at `384`, extend upward. If most layers produce zero outliers at `16`, extend downward.

## Step 2: Run Threshold Sweep

Each threshold run must enable outlier extraction logging so the parser can recover layer counts:

```bash
CUDA_VISIBLE_DEVICES=0 \
LLAMA_NVFP4_KCACHE_OUTLIER=1 \
LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD="${THRESHOLD}" \
LLAMA_NVFP4_KCACHE_OUTLIER_LOG=1 \
  "${ROOT_DIR}/build_cuda/bin/llama-perplexity" \
    -m "${MODEL}" \
    -f "${DATA}" \
    --cache-type-k nvfp4 \
    --cache-type-v f16 \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --batch-size "${BATCH_SIZE}" \
    --ubatch-size "${UBATCH_SIZE}" \
    -t "${THREADS}" \
    -c "${CTX_SIZE}" \
    --kv-unified \
    --chunks "${CHUNKS}" \
    > "${EXP_DIR}/runs/threshold_${THRESHOLD}.raw.log" 2>&1
```

Notes:

- `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD` is a sweep-time override. Confirm the current build still supports it before launching a long sweep.
- `LLAMA_NVFP4_KCACHE_OUTLIER_LOG=1` is diagnostic and can affect timing. Use these logs to derive profile data, not performance claims.
- Use enough chunks to get stable PPL and representative K-cache writes. The current helper defaults to 200 chunks.
- Keep cache type, context, model, prompt data, and batching fixed across thresholds.

For hybrid profile generation, also set:

```bash
LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8=1
```

and document that selected layers are not NVFP4 sidecar layers.

## Step 3: Parse Sweep Logs

Run:

```bash
python3 docs/development/nvfp4-kcache-outlier-thresholds/scripts/parse-kcache-outlier-threshold-sweep.py \
  --runs-dir "${EXP_DIR}/runs" \
  --output-dir "${EXP_DIR}/results" \
  --layers "${N_LAYERS}" \
  --head-dim "${K_ROW_WIDTH}"
```

Outputs:

```text
results/threshold-summary.csv
results/threshold-layer-density.csv
```

`threshold-summary.csv` contains one row per scanned threshold:

- PPL and error, when present in the raw log.
- Total outliers.
- Overall density.
- Max per-layer density.
- Max call total.
- Overflow rows.

`threshold-layer-density.csv` contains one row per layer, with threshold-specific columns:

- `th{N}_density_pct`
- `th{N}_total_outliers`
- `th{N}_max_call_total`
- `th{N}_max_row_outliers`

Parser assumptions:

- Raw logs are named `threshold_<N>.raw.log`.
- Outlier log lines include `ggml_cuda_nvfp4_kcache_outlier_extract`.
- If target tensor names include `cache_k_l{layer}`, the parser uses that layer id. Otherwise it infers layer id by record order.
- `--head-dim` is used only for density conversion; counts and capacities come from logs.

## Step 4: Derive Balanced Thresholds and Capacities

Run:

```bash
python3 docs/development/nvfp4-kcache-outlier-thresholds/scripts/derive-kcache-outlier-balanced-config.py \
  --layer-density "${EXP_DIR}/results/threshold-layer-density.csv" \
  --threshold-summary "${EXP_DIR}/results/threshold-summary.csv" \
  --output-dir "${EXP_DIR}/results" \
  --target-count 200 \
  --max-ppl-delta 0.35 \
  --capacity-margin 1.25 \
  | tee "${EXP_DIR}/results/derive.stdout"
```

Important knobs:

- `--target-count`: preferred total outliers per layer in the sweep. Larger values preserve more exact K entries but use more sidecar memory.
- `--max-layer-count`: optional hard cap. Default is `4 * target-count`.
- `--max-ppl-delta`: rejects thresholds whose global-threshold PPL is too far above the best scanned PPL.
- `--capacity-margin`: multiplies selected `max_call_total` to size each layer's compact pool.
- `--min-capacity`: lower bound, default `1`.
- `--max-threshold`: ignores threshold columns above this value.
- `--count-weight`, `--ppl-weight`, `--zero-weight`: scoring weights.

Outputs:

```text
results/balanced-config.json
results/balanced-layer-config.csv
results/balanced-config-snippet.h
results/derive.stdout
```

Review `balanced-layer-config.csv` before applying the result. Look for:

- Layers with zero selected outliers.
- Layers at the max allowed threshold.
- Layers with unexpectedly large capacity.
- Large gap between `total_outliers` and `max_call_total`.
- Any threshold selected only because tighter thresholds were rejected by PPL.

## Step 5: Apply the Profile

Copy values from:

```text
results/balanced-config-snippet.h
```

into `src/llama-kv-cache-nvfp4-outlier-config.h`.

For the current full-NVFP4 balanced path, update:

```cpp
llama_nvfp4_kcache_outlier_layer_thresholds_balanced
llama_nvfp4_kcache_outlier_layer_capacities_balanced
```

Also update nearby comments to record:

- Canonical profile snapshot directory.
- Raw evidence experiment directory.
- Model and data source.
- Context size.
- Derivation command or script.
- Objective knobs such as target count, max PPL delta, and capacity margin.

Do not overwrite the hybrid tables unless the experiment explicitly targeted hybrid mode.

## Step 6: Build and Focused Validation

Build:

```bash
cmake --build build_cuda --target llama-perplexity test-nvfp4-kcache-outlier -j8
```

Run focused CUDA test:

```bash
ctest --test-dir build_cuda -R '^test-nvfp4-kcache-outlier$' --output-on-failure
```

Run a startup or smoke check that confirms:

- `NVFP4 K-cache compact outlier sidecar enabled` appears only when expected.
- `threshold_profile` and `layer_capacity_profile` match the intended mode.
- `NVFP4 K-cache compact outlier sidecar size = ...` is present.
- `overflow_rows=0` in normal validation logs, or any overflow is explained.

## Step 7: PPL Validation

After applying the profile and rebuilding, run PPL with the intended profile:

```bash
CUDA_VISIBLE_DEVICES=0 \
LLAMA_NVFP4_KCACHE_OUTLIER=1 \
  "${ROOT_DIR}/build_cuda/bin/llama-perplexity" \
    -m "${MODEL}" \
    -f "${DATA}" \
    --cache-type-k nvfp4 \
    --cache-type-v f16 \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --batch-size "${BATCH_SIZE}" \
    --ubatch-size "${UBATCH_SIZE}" \
    -t "${THREADS}" \
    -c "${CTX_SIZE}" \
    --kv-unified \
    --chunks "${CHUNKS}" \
    > "${EXP_DIR}/runs/balanced_profile.raw.log" 2>&1
```

Compare against:

- F16/F16 baseline from `expt-baseline.md`, adjusted only as required for the new model.
- Plain NVFP4 K-cache without outlier sidecar.
- Any previous profile for the same model and parameters.

Do not compare against a run with different model, prompt, context, batching, CUDA device, or cache layout as a direct A/B result.

## Step 8: Summarize Evidence

Write `summary.md` with:

- Objective and changed parameters.
- Exact model, data, and runtime configuration.
- Threshold grid.
- Parser command.
- Derivation command and scoring knobs.
- Generated threshold and capacity arrays.
- Sidecar memory-size startup log.
- PPL result and comparison table.
- Overflow status.
- Known limitations or confounders.

Include links or relative paths to:

- Raw threshold logs.
- Parsed CSVs.
- Balanced config JSON and CSV.
- Applied code diff.
- Validation logs.

## Common Failure Modes

No outlier lines parsed:

- Confirm `LLAMA_NVFP4_KCACHE_OUTLIER_LOG=1`.
- Confirm the run actually used `--cache-type-k nvfp4`.
- Confirm `LLAMA_NVFP4_KCACHE_OUTLIER=1`.
- Confirm the K cache was allocated on CUDA, not CPU.

Many rows overflow in validation:

- Increase `--capacity-margin`.
- Increase `--target-count` only if the selected thresholds are too high and quality suffers.
- Re-run the sweep with more representative prompts or larger context.

All selected thresholds are very high:

- The target count may be too low.
- The PPL rejection window may be too strict.
- The model's K activation scale may differ from the current threshold grid.

Profile works in the sweep but fails in long-context validation:

- Generate a context-specific capacity profile.
- Use larger `--chunks` or a long-context prompt mix.
- Consider a separate table keyed by `kv_size`, as the current hybrid implementation does for `ctx8192`.

Unexpected PPL regression:

- Check whether Q quantization used per-tensor outlier mode.
- Check startup logs for the intended profile.
- Confirm no fallback path skipped sparse correction.
- Compare against the global-threshold scan PPL used by the derivation script.
