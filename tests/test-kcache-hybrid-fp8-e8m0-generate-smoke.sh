#!/usr/bin/env bash
set -euo pipefail

BIN="$1"
MODEL="$2"
TMP_OUTPUT="$(mktemp)"
trap 'rm -f "${TMP_OUTPUT}"' EXIT

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "test-kcache-hybrid-fp8-e8m0-generate-smoke: SKIP (CUDA device unavailable)"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
LLAMA_KCACHE_HYBRID_FP8_E4M3_E8M0_32_LAYERS=high_medium \
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
    --cache-type-k f16 \
    --cache-type-v f16 \
    --kv-unified \
    --no-warmup \
    -no-cnv \
    --no-display-prompt \
    > "${TMP_OUTPUT}" 2>&1

if grep -q "CPU KV buffer size" "${TMP_OUTPUT}"; then
    echo "test-kcache-hybrid-fp8-e8m0-generate-smoke: SKIP (KV cache was not allocated on CUDA)"
    exit 0
fi

grep -q "flash_attn    = 0" "${TMP_OUTPUT}"
grep -q "hybrid FP8(E4M3+E8M0 block32) K-cache layers enabled: 0,1,4,5,6,8,10,11,12,14,23,35" "${TMP_OUTPUT}"
grep -q "CUDA0 KV buffer size" "${TMP_OUTPUT}"
grep -q "K (f16+fp8_e4m3_e8m0_32)" "${TMP_OUTPUT}"
grep -q "ggml_cuda_fp8_log_e4m3_e8m0_32_e4m2_set_rows_once: .*dst=cache_k_l0.*type=fp8_e4m3_e8m0_32" "${TMP_OUTPUT}"
grep -q "q\\*k src0{name=cache_k_l0.*type=fp8_e4m3_e8m0_32" "${TMP_OUTPUT}"
