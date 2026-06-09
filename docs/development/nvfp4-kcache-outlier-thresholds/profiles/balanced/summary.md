# Balanced K-Cache Outlier Threshold Config

## Purpose

Derive a per-layer NVFP4 K-cache outlier configuration from the previous
threshold/capacity experiment so layer outlier counts are closer to each other
while avoiding thresholds whose scanned PPL regresses too much.

## Inputs

- Previous experiment: `/home/allen/host_workspace/develop/llama.cpp/experiments/20260529T080102Z-kcache-outlier-layer-threshold-capacity`
- Layer density CSV: `/home/allen/host_workspace/develop/llama.cpp/experiments/20260529T080102Z-kcache-outlier-layer-threshold-capacity/results/threshold-layer-density.csv`
- Threshold summary CSV: `/home/allen/host_workspace/develop/llama.cpp/experiments/20260529T080102Z-kcache-outlier-layer-threshold-capacity/results/threshold-summary.csv`
- Target layer outliers in sweep: `200`
- Max accepted global-scan PPL delta: `0.35`
- Capacity margin: `1.25`

## Outputs

- `results/balanced-config.json`
- `results/balanced-layer-config.csv`
- `results/balanced-config-snippet.h`
- `results/derive.stdout`

## Validation Status

The derived balanced profile was applied to
`src/llama-kv-cache-nvfp4-outlier-config.h` for the A-only
`LLAMA_NVFP4_KCACHE_OUTLIER=1` mode. The default `--cache-type-k nvfp4` path now
keeps outlier sidecar disabled.

Focused validation:

- `cmake --build build_cuda --target llama-perplexity test-nvfp4-kcache-outlier -j2`
- `bash tests/test-kcache-nvfp4-default-no-outlier-smoke.sh build_cuda/bin/llama-perplexity /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf data/wikitext/wikitext-2-raw/wiki.test.raw`
- `bash tests/test-kcache-hybrid-outlier-layer-capacity-smoke.sh build_cuda/bin/llama-perplexity /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf data/wikitext/wikitext-2-raw/wiki.test.raw`
- `build_cuda/bin/test-nvfp4-kcache-outlier`

## Balanced A-Only PPL

Command:

```bash
docs/development/nvfp4-kcache-outlier-thresholds/scripts/run-kcache-outlier-balanced-experiment.sh --run-ppl
```

Result:

| profile | PPL | +/- | rows=512 outliers | max call total | max row outliers | overflow rows | KV MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced A-only | `10.7696` | `0.08581` | `11706` | `24` | `3` | `0` | `46.35` |

The run used thresholds:

```text
256,48,24,32,48,192,32,24,192,24,192,32,24,24,24,24,24,32,32,24,32,32,24,24,24,32,32,32,32,32,32,32,32,32,32,32
```

and capacities:

```text
1,2,2,1,14,1,2,5,1,2,1,5,24,5,5,3,3,1,1,4,1,1,4,29,3,1,1,1,1,1,1,1,1,1,1,25
```

Compared with the previous `min24-capacity163` result from
`experiments/20260529T080102Z-kcache-outlier-layer-threshold-capacity`, this
profile has higher PPL (`10.7696` vs `10.6811`) but fewer logged rows=512
outliers (`11706` vs `16670`) and lower total compact capacity (`156` vs
uniform `163` per layer).

## Reproduction Scripts

Reusable scripts:

- `docs/development/nvfp4-kcache-outlier-thresholds/scripts/parse-kcache-outlier-threshold-sweep.py`
- `docs/development/nvfp4-kcache-outlier-thresholds/scripts/derive-kcache-outlier-balanced-config.py`
- `docs/development/nvfp4-kcache-outlier-thresholds/scripts/run-kcache-outlier-balanced-experiment.sh`
