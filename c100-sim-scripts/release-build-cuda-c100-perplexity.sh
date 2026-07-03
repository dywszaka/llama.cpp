#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

BUILD_TARGET=llama-perplexity \
BUILD_TYPE=Release \
CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120" \
".devops/build-local.sh" cuda-c100
