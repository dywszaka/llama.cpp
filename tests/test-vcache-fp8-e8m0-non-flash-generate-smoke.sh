#!/usr/bin/env bash

set -euo pipefail

CLI_BIN=${1:?missing llama-cli path}
MODEL_PATH=${2:?missing model path}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "test-vcache-fp8-e8m0-non-flash-generate-smoke: SKIP (nvidia-smi not found)"
    exit 0
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "test-vcache-fp8-e8m0-non-flash-generate-smoke: SKIP (no CUDA device)"
    exit 0
fi

if [[ ! -x "${CLI_BIN}" ]]; then
    echo "test-vcache-fp8-e8m0-non-flash-generate-smoke: SKIP (llama-cli missing)"
    exit 0
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "test-vcache-fp8-e8m0-non-flash-generate-smoke: SKIP (model missing: ${MODEL_PATH})"
    exit 0
fi

TMP_OUTPUT=$(mktemp)
trap 'rm -f "${TMP_OUTPUT}"' EXIT

GGML_CUDA_TRUNC_ENABLE=0 \
LLAMA_EXP_VCACHE_V_FP8=e8m0 \
"${CLI_BIN}" \
    -m "${MODEL_PATH}" \
    -ngl 99 \
    -no-cnv \
    -n 12 \
    -c 256 \
    -b 128 \
    -ub 128 \
    --seed 123 \
    --temp 0 \
    -p "The capital of France is" \
    > "${TMP_OUTPUT}" 2>&1

grep -q "experimental V cache type = fp8_e4m3_e8m0_32" "${TMP_OUTPUT}"
grep -q "flash_attn    = 0" "${TMP_OUTPUT}"
grep -q "V (fp8_e4m3_e8m0_32)" "${TMP_OUTPUT}"
grep -q "Paris" "${TMP_OUTPUT}"

echo "test-vcache-fp8-e8m0-non-flash-generate-smoke: ok"
