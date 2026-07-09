# llama-tensor-export-eval Usage

`llama-tensor-export-eval` reads a tensor export `manifest.json` and runs offline
evaluation algorithms. It prints one JSON report to stdout, so shell scripts
should redirect stdout to a result file.

## Build

```bash
cmake --build build-default --target llama-tensor-export-eval -j 4
```

For debugging:

```bash
cmake -S . -B build-ci-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-ci-debug --target llama-tensor-export-eval -j 4
```

## Export Input Tensors

Attention replay algorithms need a tensor export that includes the layer 0
attention inputs and outputs:

```bash
EXPORT_DIR=/tmp/layer0-attention-export
MODEL=/path/to/model.gguf

mkdir -p "$EXPORT_DIR"

LLAMA_EXPT_TENSOR_EXPORT_DIR="$EXPORT_DIR" \
LLAMA_EXPT_TENSOR_EXPORT_KINDS="k,q,kq,kq_softmax,kq_mask,k_attn,q_attn" \
./build-default/bin/llama-cli \
  -m "$MODEL" \
  -p "Hello" \
  -n 1 \
  -c 512 \
  --batch-size 32 \
  --ubatch-size 32 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --kv-unified \
  -t 4
```

The export directory must contain:

```text
manifest.json
*.bin
```

## Basic Command Shape

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest /path/to/export/manifest.json \
  --algorithm attention_replay \
  > /path/to/result.json
```

The tool writes diagnostics to stderr and the JSON report to stdout.

## Supported Algorithms

### `attention_replay`

Replays exported layer 0 attention with the exported K/Q/mask tensors and
compares replayed KQ and softmax against the exported runtime tensors.

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest "$EXPORT_DIR/manifest.json" \
  --algorithm attention_replay \
  > attention-replay.json
```

Important output fields:

- `records[].kq_metrics`
- `records[].softmax_metrics`
- `records[].softmax_nmse`
- `records[].max_abs_err_kq`
- `records[].max_abs_err_softmax`

### `attention_replay_nvfp4_outlier`

Applies the NVFP4 outlier K quant-round algorithm and NVFP4 dynamic-row Q
quant-round algorithm before replaying attention.

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest "$EXPORT_DIR/manifest.json" \
  --algorithm attention_replay_nvfp4_outlier \
  > attention-replay-nvfp4-outlier.json
```

Important output fields:

- `quant_round_algorithm`
- `records[].k_quantization`
- `records[].q_quantization`
- `records[].k_quant_metrics`
- `records[].q_quant_metrics`
- `records[].softmax_mse`
- `records[].softmax_nmse`
- `records[].softmax_kld`

### `attention_replay_fp8_e4m3_e8m0`

Applies FP8 E4M3 plus E8M0 block32 quant-round to both K and Q before replaying
attention.

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest "$EXPORT_DIR/manifest.json" \
  --algorithm attention_replay_fp8_e4m3_e8m0 \
  > attention-replay-fp8-e4m3-e8m0.json
```

Important output fields are the same as the NVFP4 outlier replay:

- `records[].softmax_mse`
- `records[].softmax_nmse`
- `records[].softmax_kld`
- `records[].k_quantization`
- `records[].q_quantization`

### `nvfp4_ref`

Runs NVFP4 reference quantize/dequantize on records from a manifest. Use
`--global-scale` to set the NVFP4 global scale.

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest /path/to/manifest.json \
  --algorithm nvfp4_ref \
  --global-scale 1.0 \
  > nvfp4-ref.json
```

### `nvfp4_k_channel_sort`

Sorts K channels by first-row absolute value before NVFP4 roundtrip, restores the
original channel order, and compares error against the baseline roundtrip.

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest /path/to/k-only-manifest.json \
  --algorithm nvfp4_k_channel_sort \
  --global-scale 1.0 \
  > nvfp4-k-channel-sort.json
```

Shortcut:

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest /path/to/k-only-manifest.json \
  --k-channel-sort \
  > nvfp4-k-channel-sort.json
```

### `nvfp4_k_channel_mean_sort`

Sorts K channels by absolute mean across rows before NVFP4 roundtrip.

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest /path/to/k-only-manifest.json \
  --algorithm nvfp4_k_channel_mean_sort \
  --global-scale 1.0 \
  > nvfp4-k-channel-mean-sort.json
```

Shortcut:

```bash
./build-default/bin/llama-tensor-export-eval \
  --manifest /path/to/k-only-manifest.json \
  --k-channel-mean-sort \
  > nvfp4-k-channel-mean-sort.json
```

## Example Batch Script

This matches the common attention replay workflow:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/home/allen/host_workspace/develop/llama.cpp
EXPORT_DIR="$ROOT_DIR/experiments/20260708T080134Z-layer0-attn-softmax-export/export"
OUT_DIR="$ROOT_DIR/experiments/my-attention-eval/logs"
BIN="$ROOT_DIR/build-default/bin/llama-tensor-export-eval"

mkdir -p "$OUT_DIR"

"$BIN" \
  --manifest "$EXPORT_DIR/manifest.json" \
  --algorithm attention_replay \
  > "$OUT_DIR/attention-replay.json"

"$BIN" \
  --manifest "$EXPORT_DIR/manifest.json" \
  --algorithm attention_replay_nvfp4_outlier \
  > "$OUT_DIR/attention-replay-nvfp4-outlier.json"

"$BIN" \
  --manifest "$EXPORT_DIR/manifest.json" \
  --algorithm attention_replay_fp8_e4m3_e8m0 \
  > "$OUT_DIR/attention-replay-fp8-e4m3-e8m0.json"
```

## Notes

- Attention replay currently targets layer 0 exports named like `kq-softmax-0`.
- Attention replay expects the softmax record metadata to contain `src_k`,
  `src_q`, `src_kq`, `src_mask`, `kq_scale`, and `max_bias`.
- FP8 E4M3+E8M0 block32 quant-round requires K/Q row sizes divisible by 32.
- NVFP4 quant-round requires row sizes divisible by the NVFP4 block size.
- Do not compare two reports as an A/B result unless the export manifest,
  model, prompt, context size, batch sizes, cache types, and other runtime
  parameters match.
