#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GGML_CUDA_NVFP4_VCACHE_BATCHED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    "${ROOT_DIR}/build_cuda/bin/llama-bench" \
        -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
        --cache-type-k f16 \
        --cache-type-v nvfp4 \
        --n-gpu-layers 40 \
        --batch-size 2048 \
        --ubatch-size 512 \
        --kv-unified 1 \
        --flash-attn 0 \
        --no-kv-offload 0 \
        -t 32 \
        -p 512 \
        -n 128 \
        -r 3
