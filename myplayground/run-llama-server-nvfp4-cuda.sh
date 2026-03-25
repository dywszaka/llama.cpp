#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/allen/host_workspace/develop/llama.cpp"
MODEL_PATH="${MODEL_PATH:-/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-40}"
N_CTX="${N_CTX:-8192}"
N_THREADS="${N_THREADS:-32}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-}"
CACHE_TYPE_V="${CACHE_TYPE_V:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"
export GGML_CUDA_NVFP4_NATIVE="${GGML_CUDA_NVFP4_NATIVE:-1}"

EXTRA_ARGS=()
if [[ -n "${CACHE_TYPE_K}" ]]; then
  EXTRA_ARGS+=(--cache-type-k "${CACHE_TYPE_K}")
fi
if [[ -n "${CACHE_TYPE_V}" ]]; then
  EXTRA_ARGS+=(--cache-type-v "${CACHE_TYPE_V}")
fi

exec "${ROOT_DIR}/build_cuda_r/bin/llama-server" \
  -m "${MODEL_PATH}" \
  --n_gpu_layers "${N_GPU_LAYERS}" \
  --host "${HOST}" \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --port "${PORT}" \
  -t "${N_THREADS}" \
  -c "${N_CTX}" \
  --no-warmup \
  "${EXTRA_ARGS[@]}" \
  "$@"
