# K-Cache Outlier Threshold Ratio Sweep

## Purpose

Sweep global `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD` values and choose per-layer
thresholds whose max-batch outlier ratio is closest to `0.0001`.

## Inputs

- Real data: `/home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw`
- Chunks: `2`
- Thresholds: `16 24`
- No warmup: `1`

## Outputs

- `results/threshold-layer-batch-ratio.csv`
- `results/selected-thresholds.csv`
