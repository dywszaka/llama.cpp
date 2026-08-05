#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${ROOT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
BIN=${BIN:-"$ROOT_DIR/build-default/bin/llama-tensor-export-eval"}
DEFAULT_MANIFEST="$ROOT_DIR/experiments/20260708T080134Z-layer0-attn-softmax-export/export/manifest.json"
MANIFEST=${MANIFEST:-"$DEFAULT_MANIFEST"}
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/experiments/kq-eval"}

usage() {
  cat >&2 <<EOF
usage: $0 <algorithm> [csv_path]

Examples:
  $0 attention_replay
  $0 attention_replay_nvfp4_outlier
  $0 attention_replay_fp8_e4m3_e8m0 /tmp/kq-metrics.csv

Environment overrides:
  ROOT_DIR   repository root, default: auto-detected
  BIN        llama-tensor-export-eval binary, default: \$ROOT_DIR/build-default/bin/llama-tensor-export-eval
  MANIFEST   tensor export manifest, default: \$ROOT_DIR/experiments/20260708T080134Z-layer0-attn-softmax-export/export/manifest.json
  OUT_DIR    JSON/CSV output directory, default: \$ROOT_DIR/experiments/kq-eval
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

ALGORITHM=$1
CSV_PATH=${2:-"$OUT_DIR/metrics.csv"}
JSON_PATH="$OUT_DIR/$ALGORITHM.json"

mkdir -p "$OUT_DIR" "$(dirname "$CSV_PATH")"

"$BIN" \
  --manifest "$MANIFEST" \
  --algorithm "$ALGORITHM" \
  --csv "$CSV_PATH" \
  > "$JSON_PATH"

printf 'wrote json: %s\n' "$JSON_PATH"
printf 'appended csv: %s\n' "$CSV_PATH"
