#!/bin/bash
BIN=build_cuda_release/bin/llama-bench
MODEL_DIR=/home/allen/host_workspace/develop/models/
THREADS=8

MODELS=(
    qwen3-8b-nvfp4.gguf
)

for M in "${MODELS[@]}"; do
  echo "=== Testing $M ==="
  $BIN -m $MODEL_DIR/$M --threads $THREADS --batch-size 512
done