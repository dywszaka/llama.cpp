# Diagnostic Collection Contract

This experiment is a KLD quality smoke template. Diagnostics must explain the
KLD result without turning the run into a full tensor dump.

## Dataset Scope

- Source: `data/wikitext/wikitext-2-raw/wiki.test.raw`.
- Sample unit: complete Wikitext documents, not arbitrary byte slices.
- Default sample count: `8` documents.
- Default context/chunk limit: `-c 512 --chunks 8`.
- The generated dataset is `data/wikitext-small.raw`.
- The generated manifest is `data/wikitext-small.manifest.json`.

The sample is intentionally small so future PPL experiments can run this KLD
smoke alongside the larger PPL run. Increase `SAMPLE_COUNT` or `CHUNKS` only
when the KLD signal is too noisy for the experiment under test.

## Required KLD Data

For each ubatch size and experiment case:

- Baseline log-prob file: `baseline-logprobs/ubatch_<ubatch>.kld`.
- Raw baseline log: `logs/baseline_ubatch_<ubatch>.raw.log`.
- Raw KLD log: `logs/kld_<case>_ubatch_<ubatch>.raw.log`.
- Parsed metrics: `results/metrics.json` and `results/metrics.tsv`.

The baseline log-prob file is produced by `llama-perplexity` with f16/f16 KV and
`--kl-divergence-base`. The experiment reads the matching file with
`--kl-divergence --kl-divergence-base`.

## Diagnostic Artifact Limits

Do not store complete K, V, attention score, probability, or logits tensors.
Each diagnostic artifact must be one of:

- Aggregate histogram JSON.
- Per-layer or per-head histogram JSON.
- Bounded row sample JSONL.
- A manifest describing the sampling configuration and runtime switches.

Baseline data, baseline log-prob files, and invariant KLD configuration live in
`experiments/kld-baseline-data`. Per-experiment diagnostics belong in the dated
comparison experiment folder, one diagnostic folder per KLD case:

```text
diagnostics/kld_<case>_ubatch_<ubatch>/
```

## Histogram Schema

Histogram files use the suffix `.hist.json`:

```json
{
  "schema_version": 1,
  "name": "attention-score",
  "case": "kld_nvfp4_outlier_ubatch_512",
  "tensor_role": "attention_score",
  "sample_count": 4096,
  "axis": "all_sampled_values",
  "bins": [
    {"lo": -16.0, "hi": -8.0, "count": 12},
    {"lo": -8.0, "hi": -4.0, "count": 41}
  ]
}
```

Recommended files:

- `k.hist.json`: sampled K-cache value distribution.
- `v.hist.json`: sampled V-cache value distribution.
- `attention-score.hist.json`: sampled pre-softmax attention score distribution.
- `attention-prob.hist.json`: sampled post-softmax probability distribution when available.

## Row Sample Schema

Row samples use the suffix `.sample-rows.jsonl`. Each line is one bounded sample:

```json
{"schema_version":1,"tensor_role":"k","layer":0,"head":0,"row":0,"cols":[0,1,2,3,4,5,6,7],"values":[0.0,0.125,-0.25,0.5,0.0,0.0,0.25,-0.125]}
```

Default row sampling:

- Rows: `0`, `1`, `127`, `255`, and the last valid row in the sampled chunk.
- Columns: `0..31` only.
- Layers: `0`, `17`, and `35` for Qwen3 8B unless the experiment targets a
  specific layer group.
- Heads: `0`, middle head, and last head when head structure is available.
- Max JSONL records per tensor role per case: `256`.

Recommended files:

- `k.sample-rows.jsonl`
- `v.sample-rows.jsonl`
- `attention-score.sample-rows.jsonl`

## Manifest Schema

Each diagnostic folder may include `manifest.json`:

```json
{
  "schema_version": 1,
  "case": "kld_nvfp4_outlier_ubatch_512",
  "dataset": "../data/wikitext-small.manifest.json",
  "row_sample": {
    "rows": [0, 1, 127, 255, "last"],
    "columns": [0, 31],
    "max_records_per_tensor_role": 256
  },
  "histograms": {
    "binning": "fixed_log_abs_or_linear_by_tensor_role",
    "full_tensor_dump_allowed": false
  }
}
```

## Future Runtime Hook

When adding runtime collection code, gate it behind an off-by-default diagnostic
switch and record that switch in `expt-switch-env.md`. The switch should accept
the case diagnostic directory as its value and write only the bounded artifacts
defined above.
