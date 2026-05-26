#!/usr/bin/env bash
set -euo pipefail

BIN="$1"
MODEL="$2"
TMP_OUTPUT="$(mktemp)"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
GGML_CUDA_FP8_E8M0_NATIVE_NO_FALLBACK=1 \
"${BIN}" \
    -m "${MODEL}" \
    -p "The capital of France is" \
    -n 4 \
    -c 512 \
    --batch-size 512 \
    --ubatch-size 512 \
    --n-gpu-layers 40 \
    -t 32 \
    --cache-type-k fp8_e4m3_e8m0_32 \
    --cache-type-v f16 \
    --kv-unified \
    --no-warmup \
    --no-display-prompt \
    > "${TMP_OUTPUT}" 2>&1

grep -q "flash_attn    = 0" "${TMP_OUTPUT}"
grep -q "K (fp8_e4m3_e8m0_32)" "${TMP_OUTPUT}"
grep -q "V (f16)" "${TMP_OUTPUT}"
grep -q "q\\*k src0{name=cache_k_l0.*type=fp8_e4m3_e8m0_32" "${TMP_OUTPUT}"
