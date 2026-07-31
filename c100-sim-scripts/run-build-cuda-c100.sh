#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./build-cuda-c100.sh [debug]

Build the CUDA+C100 Docker variant with all default targets.

Modes:
  default   Build release into build-cuda-c100
  debug     Build debug into build-cuda-c100-debug
EOF
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && cd .. &&  pwd)"
cd "${ROOT_DIR}"

C100_SIM_ROOT="${C100_SIM_ROOT:-${ROOT_DIR}/c100-sim}"
export C100_SIM_ROOT

mode="${1:-release}"
variant=

case "${mode}" in
    release)
        BUILD_TYPE=Release
        CUDA_C100_BUILD_DIR="/workspace/llama.cpp/build-cuda-c100"
        export BUILD_TYPE CUDA_C100_BUILD_DIR
        variant="cuda-c100"
        ;;
    debug)
        BUILD_TYPE=Debug
        CUDA_C100_DEBUG_BUILD_DIR="/workspace/llama.cpp/build-cuda-c100-debug"
        export BUILD_TYPE CUDA_C100_DEBUG_BUILD_DIR
        variant="cuda-c100-debug"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

BUILD_TARGET=all \
CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120" \
"${ROOT_DIR}/.devops/build-local.sh" "${variant}"
