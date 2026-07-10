#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

grep -Fq 'set(C100_SIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/c100-sim" CACHE PATH' \
    "${ROOT_DIR}/CMakeLists.txt"

grep -Fq 'C100_SIM_ROOT="${C100_SIM_ROOT:-${LLAMA_CPP_ROOT}/c100-sim}"' \
    "${ROOT_DIR}/.devops/llamacpp-cuda-variants.sh"

test ! -e "${ROOT_DIR}/c100-sim-scripts/release-build-cuda-c100-perplexity.sh"

grep -Fq 'C100_SIM_ROOT="${C100_SIM_ROOT:-${ROOT_DIR}/c100-sim}"' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'mode="${1:-release}"' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'if [[ $# -gt 1 ]]; then' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'Usage:' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq './build-cuda-c100.sh [debug]' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'case "${mode}" in' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'release)' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'BUILD_TYPE=Release' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'CUDA_C100_BUILD_DIR="/workspace/llama.cpp/build-cuda-c100"' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'debug)' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'BUILD_TYPE=Debug' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'CUDA_C100_DEBUG_BUILD_DIR="/workspace/llama.cpp/build-cuda-c100-debug"' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq 'BUILD_TARGET=all' \
    "${ROOT_DIR}/build-cuda-c100.sh"

grep -Fq '".devops/build-local.sh" "${variant}"' \
    "${ROOT_DIR}/build-cuda-c100.sh"

test ! -e "${ROOT_DIR}/c100-sim-scripts/build-cuda-c100.sh"

grep -Fq 'CUDA_C100_DEVICE_DEBUG="${CUDA_C100_DEVICE_DEBUG:-0}"' \
    "${ROOT_DIR}/.devops/llamacpp-cuda-variants.sh"

grep -Fq '"-DCMAKE_CUDA_FLAGS_DEBUG=-g -lineinfo"' \
    "${ROOT_DIR}/.devops/llamacpp-cuda-variants.sh"

grep -Fq '"-DCMAKE_CUDA_FLAGS_DEBUG=-G -g -lineinfo"' \
    "${ROOT_DIR}/.devops/llamacpp-cuda-variants.sh"

grep -Fq 'if [[ "${CUDA_C100_DEVICE_DEBUG}" == "1" ]]; then' \
    "${ROOT_DIR}/.devops/llamacpp-cuda-variants.sh"

grep -Fq 'configure_file("${CMAKE_CURRENT_LIST_DIR}/c100-spike-configure-input.in"' \
    "${ROOT_DIR}/cmake/c100-runtime.cmake"

grep -Fq 'COMMAND "${CMAKE_COMMAND}" -E remove_directory "${C100_SIM_SPIKE_BUILD_DIR}"' \
    "${ROOT_DIR}/cmake/c100-runtime.cmake"

grep -Fq 'DEPENDS "${C100_SIM_SPIKE_CONFIGURE_INPUT}"' \
    "${ROOT_DIR}/cmake/c100-runtime.cmake"

grep -Fq '@C100_SIM_SPIKE_ROOT@' \
    "${ROOT_DIR}/cmake/c100-spike-configure-input.in"
