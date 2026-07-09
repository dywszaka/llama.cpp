#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

grep -Fq 'set(C100_SIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/c100-sim" CACHE PATH' \
    "${ROOT_DIR}/CMakeLists.txt"

grep -Fq 'C100_SIM_ROOT="${C100_SIM_ROOT:-${LLAMA_CPP_ROOT}/c100-sim}"' \
    "${ROOT_DIR}/.devops/llamacpp-cuda-variants.sh"

grep -Fq 'C100_SIM_ROOT="${C100_SIM_ROOT:-${ROOT_DIR}/c100-sim}"' \
    "${ROOT_DIR}/c100-sim-scripts/release-build-cuda-c100-perplexity.sh"
