#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lerong.chen/qemu"
MODEL="/home/lerong.chen/qwen3-8b-q4_0.gguf"
TABLE="/home/lerong.chen/0729-rope-node4/rope-cos-sin-f32.bin"
ARTIFACT_DIR="${ROOT_DIR}/vendor-llama-cpp/experiments/20260804T023900Z-rope-qemu-compare"

ROPE_FP32_LOG_DIR="${ROOT_DIR}/rope-fp32-llama-logs" \
"${ROOT_DIR}/call_rope_fp32/run.sh" -- \
docker run --rm --gpus all --ipc host --network host \
  -u "$(id -u):$(id -g)" \
  -e "HOME=${HOME}" \
  -e GGML_CUDA_ROPE_QEMU_MODE=compare \
  -e GGML_CUDA_ROPE_QEMU_ENDPOINT=tcp://127.0.0.1:15587 \
  -e "GGML_CUDA_ROPE_QEMU_TABLE=${TABLE}" \
  -e "GGML_CUDA_ROPE_QEMU_ARTIFACT=${ARTIFACT_DIR}/compare.jsonl" \
  -e "GGML_CUDA_ROPE_QEMU_MISMATCH_LOG=${ARTIFACT_DIR}/mismatch.jsonl" \
  -v "${HOME}:${HOME}" \
  -w "${ROOT_DIR}" \
  qemu-llama-softmax:local \
  vendor-llama-cpp/build_rope_qemu/bin/llama-cli \
  -m "${MODEL}" -p hello -n 1 -c 8192 -ngl 999 -no-cnv --no-warmup
