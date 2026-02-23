#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf}"
PORT="${PORT:-8080}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"
export GGML_CUDA_NVFP4_NATIVE="${GGML_CUDA_NVFP4_NATIVE:-1}"

exec "${ROOT_DIR}/build_cuda_release/bin/llama-server" \
  -m "${MODEL_PATH}" \
  --n_gpu_layers 40 \
  --host 127.0.0.1 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --port "${PORT}" \
  -t 32 \
  -c 2048 \
  --no-warmup \
  "$@"
