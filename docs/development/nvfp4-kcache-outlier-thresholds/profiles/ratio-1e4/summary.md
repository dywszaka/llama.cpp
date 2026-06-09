# K-Cache Outlier Threshold Ratio Sweep Final Selection

## Purpose

Select per-layer `LLAMA_NVFP4_KCACHE_OUTLIER` thresholds so each layer max-batch outlier ratio is around `1e-4`.

## Inputs

- Real data: `data/wikitext/wikitext-2-raw/wiki.test.raw`
- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- Runtime delta from baseline: `--cache-type-k nvfp4`, `LLAMA_NVFP4_KCACHE_OUTLIER=1`, `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD=<swept>`, `--no-warmup`
- Batch statistic: 512 rows per batch, denominator `512 * 1024 = 524288` K values per layer per batch
- Selection rule: per layer, choose threshold whose max 512-row batch outlier ratio is closest to `1e-4` across all sweep runs

## Sweep Runs

- Coarse: `experiments/20260605T065001Z-kcache-outlier-threshold-ratio-sweep`
- Fine: `experiments/20260605T070048Z-kcache-outlier-threshold-ratio-sweep`
- Refinement: `experiments/20260605T072559Z-kcache-outlier-threshold-ratio-sweep`

## Result

- Layers selected: `36`
- Ratio range: `6.86645508e-05` to `0.000144958496`
- Zero-outlier selected layers: `0`
- Layers within `[5e-5, 2e-4]`: `36/36`
- Layers within absolute delta `5e-5`: `36/36`

## Recommended Thresholds

```cpp
static constexpr float llama_nvfp4_kcache_outlier_layer_thresholds_ratio_1e4[] = {
     214.00f,   42.00f,   19.00f,   16.25f,   42.00f,   72.00f,   26.00f,   15.25f,   40.00f,
      13.00f,   38.00f,   27.00f,   23.00f,   14.50f,   21.00f,   14.50f,   17.00f,   13.50f,
      17.75f,   12.50f,   13.50f,   18.25f,   17.75f,   23.00f,   13.00f,   16.25f,   14.50f,
      15.50f,   14.50f,   15.50f,   20.25f,   15.00f,   15.75f,   13.00f,   20.25f,   30.00f,
};
```

## Output

- Final CSV: `results/final-selected-thresholds.csv`
- Per-threshold ratios in each sweep directory: `results/threshold-layer-batch-ratio.csv`
