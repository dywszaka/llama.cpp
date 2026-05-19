#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="${BIN:-${ROOT_DIR}/build_cuda/bin/llama-bench}"
MODEL_PATH="${MODEL_PATH:-/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_NVFP4_NATIVE="${GGML_CUDA_NVFP4_NATIVE:-1}"

if [[ ! -x "${BIN}" ]]; then
  echo "llama-bench binary not found or not executable: ${BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "model not found: ${MODEL_PATH}" >&2
  exit 1
fi

exec "${BIN}" \
  -m "${MODEL_PATH}" \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --n-gpu-layers 40 \
  --batch-size 2048 \
  --ubatch-size 512 \
  -t 32 \
  -p 512 \
  -n 128 \
  "$@"
