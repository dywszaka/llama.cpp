#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lerong.chen/qemu"
EXPERIMENT_DIR="${ROOT_DIR}/vendor-llama-cpp/experiments/20260804T025000Z-rope-qemu-server"

docker run --rm --gpus all --ipc host --network host \
  --name rope-qemu-server-validation \
  -u "$(id -u):$(id -g)" \
  -e "HOME=${HOME}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e "LLAMA_STDOUT_FILE=${EXPERIMENT_DIR}/server.log" \
  -e GGML_CUDA_ROPE_QEMU_MODE=qemu_cuda \
  -e GGML_CUDA_ROPE_QEMU_TABLE=/home/lerong.chen/0729-rope-node4/rope-cos-sin-f32.bin \
  -v "${HOME}:${HOME}" \
  -w "${ROOT_DIR}" \
  qemu-llama-softmax:local \
  vendor-llama-cpp/build_rope_qemu/bin/llama-server \
    -m /home/lerong.chen/qwen3-8b-nvfp4.gguf \
    --n_gpu_layers 40 \
    --host 127.0.0.1 \
    --batch-size 512 \
    --ubatch-size 512 \
    --port 58082 \
    -t 32 \
    -c 8192 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --kv-unified \
    --log-file "${EXPERIMENT_DIR}/server.log"
