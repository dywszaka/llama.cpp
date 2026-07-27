#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for kv_size in 512 2048 8192 32768; do
    GGML_CUDA_NVFP4_VCACHE_BATCHED=1 \
        CUDA_VISIBLE_DEVICES=0 \
        "${ROOT_DIR}/build_cuda/bin/test-vcache-nvfp4-matmul" --benchmark-only "${kv_size}"
done
