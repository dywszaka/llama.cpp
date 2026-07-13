#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/home/allen/host_workspace/develop/llama.cpp
EXP_DIR="$ROOT_DIR/experiments/20260709T000000Z-attention-replay-nvfp4-outlier-eval"
SOURCE_EXP_DIR="$ROOT_DIR/experiments/20260708T080134Z-layer0-attn-softmax-export"
MANIFEST="$SOURCE_EXP_DIR/export/manifest.json"
LOG_DIR="$EXP_DIR/logs"

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/attention-replay.json "$LOG_DIR"/attention-replay-nvfp4-outlier.json

"$ROOT_DIR/build-default/bin/llama-tensor-export-eval" \
  --manifest "$MANIFEST" \
  --algorithm attention_replay \
  > "$LOG_DIR/attention-replay.json"

"$ROOT_DIR/build-default/bin/llama-tensor-export-eval" \
  --manifest "$MANIFEST" \
  --algorithm attention_replay_nvfp4_outlier \
  > "$LOG_DIR/attention-replay-nvfp4-outlier.json"
