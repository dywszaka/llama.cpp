# Baseline KLD Small Wikitext

This tool directory defines a reusable, small KLD evaluation companion for
future PPL experiments. It does not contain experiment run data by default. It
provides the dataset selection rule, baseline log-prob generation command,
experiment KLD command, bounded diagnostic artifact contract, parser, and tests.

## Purpose

Future PPL experiments can run this KLD smoke with the same parameter group and
compare the group against a fixed f16/f16 KV baseline. The run is intentionally
small: by default it selects 8 complete Wikitext documents and evaluates up to
8 context chunks at `-c 512`.

## Baseline

The baseline starts from `expt-baseline.md` and keeps:

- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- Dataset source: `data/wikitext/wikitext-2-raw/wiki.test.raw`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- GPU layers: `--n_gpu_layers 40`
- Batch size: `--batch-size 512`
- Threads: `-t 32`
- Context size: `-c 512`
- KV mode: `--kv-unified`
- K/V cache baseline: `--cache-type-k f16 --cache-type-v f16`

The KLD baseline uses `--kl-divergence-base baseline-logprobs/ubatch_<N>.kld`
without `--kl-divergence` to generate compact log-prob data.

## Dataset

Run the default command to prepare the small Wikitext sample and manifest:

```bash
tools/kld/scripts/run-kld-small.sh
```

By default, invariant baseline artifacts are written to
`experiments/kld-baseline-data`. Override with `BASELINE_DIR=...` only when
creating an intentionally separate baseline dataset.

Generated artifacts:

- `data/wikitext-small.raw`
- `data/wikitext-small.manifest.json`
- `input-reference.md`

Default selection:

- `SAMPLE_COUNT=8`
- `START_DOCUMENT=0`
- `MIN_CHARS=200`
- `CHUNKS=8`

## Running

Generate baseline log-prob files:

```bash
RUN_BASELINE=1 tools/kld/scripts/run-kld-small.sh
```

Baseline generation uses sparse threshold format by default
(`SPARSE_BASELINE=1`). Use `SPARSE_BASELINE=0` only when testing compatibility
with the original dense `.kld` format.

Run the default experiment matrix against the baseline:

```bash
EXP_DIR=experiments/YYYYMMDDThhmmssZ-kld-comparison \
RUN_KLD=1 tools/kld/scripts/run-kld-small.sh
```

If `EXP_DIR` is omitted while `RUN_KLD=1`, the script creates a timestamped
`experiments/<timestamp>-kld-comparison` folder. Do not write comparison logs or
metrics into `experiments/kld-baseline-data`.

The default matrix is:

```text
nvfp4_outlier:nvfp4:nvfp4:LLAMA_NVFP4_KCACHE_OUTLIER=1
```

Override it for a future PPL parameter group:

```bash
CASE_MATRIX='nvfp4_outlier:nvfp4:nvfp4:LLAMA_NVFP4_KCACHE_OUTLIER=1' \
UBATCH_SIZES='128 512' \
RUN_KLD=1 \
tools/kld/scripts/run-kld-small.sh
```

`CASE_MATRIX` entries are whitespace-separated and use:

```text
case_name:k_cache_type:v_cache_type:ENV_KEY=VALUE
```

Use comma-separated assignments when a case needs multiple switches:

```text
case_name:k_cache_type:v_cache_type:ENV_A=1,ENV_B=high_medium
```

Use `-` for a case with no extra environment switch.

## Diagnostics

Diagnostics are intentionally bounded. Keep histograms and sampled rows only;
do not store full tensor dumps. The detailed schema and sample policy are in
`tools/kld/collection-contract.md`.

Recommended roles:

- K-cache histogram and sampled rows.
- V-cache histogram and sampled rows.
- Attention score histogram and sampled rows.
- Attention probability histogram when available.

## Results

After a KLD run or after adding raw logs manually, parse metrics with:

```bash
PARSE_ONLY=1 tools/kld/scripts/run-kld-small.sh
```

Parsed outputs:

- `results/metrics.json`
- `results/metrics.tsv`
- `summary.md`

## Validation

Lightweight script tests:

```bash
python3 -m unittest tools/kld/tests/test_kld_tools.py
```
