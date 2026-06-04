#!/usr/bin/env bash
set -euo pipefail

BIN="$1"
MODEL="$2"
DATA="$3"
TMP_OUTPUT="$(mktemp)"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "test-kcache-hybrid-outlier-layer-capacity-smoke: SKIP (CUDA device unavailable)"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
LLAMA_NVFP4_KCACHE_OUTLIER=1 \
LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8=1 \
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
    echo "test-kcache-hybrid-outlier-layer-capacity-smoke: SKIP (KV cache was not allocated on CUDA)"
    exit 0
fi

grep -q "hybrid FP8(E4M3+E8M0 block32) K-cache layers enabled: 0,1,4,5,6,8,10,11,12,14,23,35" "${TMP_OUTPUT}"
grep -q "NVFP4 K-cache compact outlier sidecar enabled: threshold_profile=balanced layer_capacity_profile=.* layer_capacities=36" "${TMP_OUTPUT}"
grep -q "NVFP4 K-cache compact outlier sidecar size = .* MiB (.* bytes)" "${TMP_OUTPUT}"
grep -q "K (nvfp4+fp8_e4m3_e8m0_32)" "${TMP_OUTPUT}"
