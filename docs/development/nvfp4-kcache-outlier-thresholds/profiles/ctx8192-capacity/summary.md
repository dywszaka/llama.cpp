# Balanced Threshold ctx8192 Capacity Derivation

Purpose: re-derive `llama_nvfp4_kcache_outlier_layer_capacities_ctx8192` after
removing the hybrid threshold-16 profile and making hybrid FP8 + NVFP4 K-cache
outlier use `llama_nvfp4_kcache_outlier_layer_thresholds_balanced` by default.

Configuration:

- K cache: `--cache-type-k nvfp4`
- V cache: `--cache-type-v nvfp4`
- Outlier: `LLAMA_NVFP4_KCACHE_OUTLIER=1`
- Hybrid layers: `LLAMA_KCACHE_HYBRID_FP8_E4M3_E8M0_32_LAYERS=high_medium`
- Threshold profile observed: `balanced`
- Capacity profile observed during profiling: `ctx8192`
- Context: `-c 8192`
- Batch / ubatch: `512 / 512`
- KV mode: `--kv-unified`

Profile result:

- PPL: `8.4501 +/- 0.06093`
- rows=512 records: `13824`
- total outliers: `592`
- compact overflow: `0`
- overflow rows: `0`

Derived rule:

```text
capacity[layer] = 0 for hybrid FP8 layers
capacity[layer] = max(ctx512_capacity[layer], ceil(observed_peak_compact_used[layer] * 1.5))
```

Derived `llama_nvfp4_kcache_outlier_layer_capacities_ctx8192`:

```text
0,0,418,72,0,0,0,68,0,14,0,0,0,29,0,46,174,31,294,16,129,321,221,0,17,61,26,48,28,29,883,30,8,18,751,0
```

This matches the existing ctx512 compact-min capacity table because balanced
thresholds produce very sparse outliers at `-c 8192`; the largest observed
`peak_used` was `3`.

Artifacts:

- Script: `scripts/run_profile.sh`
- Parser: `scripts/analyze_capacity.py`
- Raw log: `runs/balanced_threshold_ctx8192_profile.raw.log`
- JSON profile: `results/capacity_profile.json`
- Layer TSV: `results/layer_capacity_profile.tsv`
