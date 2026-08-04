#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/vendor-llama-cpp"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-${LLAMA_DIR}/build_softmax_qemu}"
SOFTMAX_LOG_DIR="${SOFTMAX_LOG_DIR:-${PWD}/softmax-logs}"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh compare   -- <llama command> [args...]
  ./run.sh qemu      -- <llama command> [args...]
  ./run.sh qemu_cuda -- <llama command> [args...]
  ./run.sh compare   MODEL.gguf [PROMPT]
  ./run.sh qemu      MODEL.gguf
  ./run.sh qemu_cuda MODEL.gguf

compare runs llama CUDA, FP32 RVV/NI900 Exp, and the mirrored qemu_cuda softmax,
writes numerical and bit-exact comparison artifacts, and keeps the llama CUDA
result. qemu_cuda stays on device and does not start QEMU, ZMQ, or D2H/H2D.

Set BUILD_LLAMA=1 to configure/build build_softmax_qemu before launch.
Set SOFTMAX_LOG_DIR to choose the log/artifact directory.
Per-call softmax timing is enabled by default for qemu/compare and saved in
llama.log. Pure qemu_cuda mode suppresses per-call timing logs.
run-docker.sh maps host port 58080 to container port 8080 by default.
Use ./kill.sh to stop the complete softmax QEMU/llama stack.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  compare) qemu_mode="compare"; default_tool="llama-cli"; start_qemu=1 ;;
  qemu) qemu_mode="qemu"; default_tool="llama-server"; start_qemu=1 ;;
  qemu_cuda) qemu_mode="qemu_cuda"; default_tool="llama-server"; start_qemu=0 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "mode must be compare, qemu, or qemu_cuda" >&2; usage >&2; exit 2 ;;
esac
shift

if [[ "${BUILD_LLAMA:-0}" == "1" || ! -x "${LLAMA_BUILD_DIR}/bin/${default_tool}" || \
      ! -f "${LLAMA_BUILD_DIR}/CMakeCache.txt" || \
      "$(sed -n 's/^GGML_CUDA_SOFTMAX_QEMU:BOOL=//p' "${LLAMA_BUILD_DIR}/CMakeCache.txt" 2>/dev/null)" != "ON" ]]; then
  cmake -S "${LLAMA_DIR}" -B "${LLAMA_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=OFF \
    -DGGML_CUDA_SOFTMAX_QEMU=ON \
    -DLLAMA_CURL=OFF
  cmake --build "${LLAMA_BUILD_DIR}" --target llama-cli llama-server -j "${BUILD_JOBS:-$(nproc)}"
fi

if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    echo "a llama command is required after --" >&2
    exit 2
  fi
  command=("$@")
elif [[ $# -gt 0 ]]; then
  model="$1"
  shift
  if [[ "${qemu_mode}" == "compare" ]]; then
    prompt="${1:-hello}"
    command=("${LLAMA_BUILD_DIR}/bin/llama-cli" -m "${model}" -p "${prompt}" -n "${LLAMA_N_PREDICT:-1}" -ngl 999 -no-cnv)
  else
    command=("${LLAMA_BUILD_DIR}/bin/llama-server" -m "${model}" --host "${LLAMA_HOST:-0.0.0.0}" --port "${LLAMA_PORT:-8080}" -ngl 999)
  fi
else
  echo "provide a command after -- or a model path" >&2
  exit 2
fi

mkdir -p "${SOFTMAX_LOG_DIR}"
export SOFTMAX_LOG_DIR
export GGML_CUDA_SOFT_MAX_QEMU_MODE="${qemu_mode}"
export GGML_CUDA_SOFT_MAX_QEMU_ARTIFACT="${SOFTMAX_LOG_DIR}/softmax-qemu-compare.jsonl"
export GGML_CUDA_SOFT_MAX_QEMU_MISMATCH_LOG="${SOFTMAX_LOG_DIR}/softmax-qemu-cuda-mismatch.jsonl"
if [[ "${qemu_mode}" == "qemu_cuda" ]]; then
  export GGML_CUDA_SOFT_MAX_QEMU_TIMING=0
else
  export GGML_CUDA_SOFT_MAX_QEMU_TIMING="${GGML_CUDA_SOFT_MAX_QEMU_TIMING:-1}"
fi

if [[ "${start_qemu}" == "1" ]]; then
  exec env SOFTMAX_FP32_LOG_DIR="${SOFTMAX_LOG_DIR}" \
    "${ROOT_DIR}/call_softmax_fp32/run.sh" -- "${command[@]}"
fi

: > "${SOFTMAX_LOG_DIR}/llama.log"
exec > >(tee -a "${SOFTMAX_LOG_DIR}/llama.log") 2>&1
exec "${command[@]}"
