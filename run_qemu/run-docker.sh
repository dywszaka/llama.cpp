#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${SOFTMAX_DOCKER_IMAGE:-qemu-llama-softmax:local}"
HOST_PORT="${LLAMA_HOST_PORT:-58080}"
CONTAINER_PORT="${LLAMA_CONTAINER_PORT:-8080}"

usage() {
  cat <<'EOF'
Usage:
  ./run-docker.sh compare   MODEL.gguf [PROMPT]
  ./run-docker.sh qemu      MODEL.gguf
  ./run-docker.sh qemu_cuda MODEL.gguf
  ./run-docker.sh MODE -- <llama command> [args...]

qemu_cuda runs the deterministic CUDA implementation without starting QEMU or
ZMQ and without CUDA D2H/H2D staging. compare runs llama CUDA, QEMU/RVV, and
qemu_cuda, while keeping the llama CUDA result downstream.

The container server listens on 8080 and is exposed as host port 58080 by
default. Override the host port with LLAMA_HOST_PORT.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  compare|qemu|qemu_cuda) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "mode must be compare, qemu, or qemu_cuda" >&2; usage >&2; exit 2 ;;
esac

docker build \
  --build-arg "CUDA_IMAGE=${CUDA_IMAGE:-nvcr.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}" \
  -f "${ROOT_DIR}/Dockerfile.softmax-qemu" \
  -t "${IMAGE}" \
  "${ROOT_DIR}"

docker run --rm --gpus all --ipc host \
  --label "qemu.softmax.stack=1" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -u "$(id -u):$(id -g)" \
  -e "HOME=${HOME}" \
  -e "LLAMA_PORT=${CONTAINER_PORT}" \
  -v "${HOME}:${HOME}" \
  -w "${ROOT_DIR}" \
  "${IMAGE}" \
  "${ROOT_DIR}/run.sh" "$@"
