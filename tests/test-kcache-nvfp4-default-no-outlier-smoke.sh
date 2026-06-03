#!/usr/bin/env bash
set -euo pipefail

BIN="$1"
MODEL="$2"
DATA="$3"
TMP_OUTPUT="$(mktemp)"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "test-kcache-nvfp4-default-no-outlier-smoke: SKIP (CUDA device unavailable)"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${BIN}" \
    -m "${MODEL}" \
    -f "${DATA}" \
    --chunks 1 \
    --cache-type-k nvfp4 \
    --cache-type-v f16 \
    --n_gpu_layers 40 \
    --batch-size 512 \
    --ubatch-size 512 \
    -t 32 \
    -c 512 \
    --kv-unified \
    > "${TMP_OUTPUT}" 2>&1

if grep -q "CPU KV buffer size" "${TMP_OUTPUT}"; then
    echo "test-kcache-nvfp4-default-no-outlier-smoke: SKIP (KV cache was not allocated on CUDA)"
    exit 0
fi

grep -q "K (nvfp4)" "${TMP_OUTPUT}"
if grep -q "NVFP4 K-cache compact outlier sidecar" "${TMP_OUTPUT}"; then
    echo "NVFP4 K-cache outlier sidecar should be off by default" >&2
    exit 1
fi
