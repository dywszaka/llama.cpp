# K-Cache Outlier Threshold Ratio Sweep

## Purpose

Sweep global `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD` values and choose per-layer
thresholds whose max-batch outlier ratio is closest to `0.0001`.

## Inputs

- Real data: `/home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw`
- Chunks: `50`
- Thresholds: `11 11.5 12.5 13 13.5 14.5 15 15.5 16.5 17 17.5 18.5 19 19.5 20.5 21 21.5 22.5 23 23.5 25 26 27 29 30 31 33 34 36 38 42 44 46 50 56 60 72 80 88 112 144 160 176 208 224 240`
- No warmup: `1`

## Outputs

- `results/threshold-layer-batch-ratio.csv`
- `results/selected-thresholds.csv`
