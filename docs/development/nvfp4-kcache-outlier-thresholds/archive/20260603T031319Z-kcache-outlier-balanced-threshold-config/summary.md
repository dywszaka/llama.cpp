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

This experiment derived the configuration from existing sweep artifacts and the
generated balanced profile was copied into
`src/llama-kv-cache-nvfp4-outlier-config.h`.

Focused validation run after applying the switch/config behavior:

- `cmake --build build_cuda --target llama-perplexity test-nvfp4-kcache-outlier -j2`
- `bash tests/test-kcache-nvfp4-default-no-outlier-smoke.sh build_cuda/bin/llama-perplexity /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf data/wikitext/wikitext-2-raw/wiki.test.raw`
- `bash tests/test-kcache-hybrid-outlier-layer-capacity-smoke.sh build_cuda/bin/llama-perplexity /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf data/wikitext/wikitext-2-raw/wiki.test.raw`
- `build_cuda/bin/test-nvfp4-kcache-outlier`

Full PPL validation of the balanced A-only profile is still pending because it
requires a full `llama-perplexity` run after the generated config is applied.
