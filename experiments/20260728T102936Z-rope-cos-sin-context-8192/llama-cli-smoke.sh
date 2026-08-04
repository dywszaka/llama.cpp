#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_ROPE_QEMU_ENABLED=1 \
    "${ROOT_DIR}/build_cuda/bin/llama-cli" \
        -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
        --n-gpu-layers 40 \
        -t 32 \
        -c 8192 \
        --batch-size 512 \
        --ubatch-size 512 \
        --cache-type-k f16 \
        --cache-type-v f16 \
        --kv-unified \
        --no-warmup \
        --no-display-prompt \
        --simple-io \
        --no-conversation \
        --ignore-eos \
        --seed 1 \
        --temp 0 \
        -n 1 \
        --prompt "RoPE smoke" \
        > "$(dirname -- "${BASH_SOURCE[0]}")/llama-cli-smoke.log" 2>&1
