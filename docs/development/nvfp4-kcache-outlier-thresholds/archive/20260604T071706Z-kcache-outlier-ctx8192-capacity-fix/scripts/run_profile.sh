#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments/20260604T071706Z-kcache-outlier-ctx8192-capacity-fix"
BIN="${ROOT_DIR}/build_cuda/bin/llama-perplexity"
MODEL="/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf"
DATA="${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw"
LOG="${EXP_DIR}/runs/fourth_ctx8192_capacity_fix.raw.log"

mkdir -p "${EXP_DIR}/runs" "${EXP_DIR}/results"

{
  echo "case=fourth_ctx8192_capacity_fix"
  echo "source_case=20260604T055253Z case 04_nvfp4k_nvfp4v_outlier_hybrid_fp8"
  echo "cache_type_k=nvfp4"
  echo "cache_type_v=nvfp4"
  echo "LLAMA_NVFP4_KCACHE_OUTLIER=1"
  echo "LLAMA_KCACHE_HYBRID_FP8_E4M3_E8M0_32_LAYERS=high_medium"
  echo "ctx_size=8192"
  echo "binary=${BIN}"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${LOG}"

env \
  -u LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8 \
  -u LLAMA_NVFP4_KCACHE_OUTLIER_LOG \
  -u LLAMA_NVFP4_KCACHE_OUTLIER_COMPACT \
  -u LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD \
  -u LLAMA_NVFP4_KCACHE_OUTLIER_MIN_CAPACITY \
  -u LLAMA_NVFP4_KCACHE_OUTLIER_LAYER_CAPACITIES \
  -u LLAMA_NVFP4_KCACHE_OUTLIER_LAYER_THRESHOLDS \
  -u LLAMA_NVFP4_KCACHE_OUTLIER_CAPACITY_RATIO \
  -u GGML_CUDA_NVFP4_FATTN \
  -u GGML_CUDA_NVFP4_FATTN_NO_FALLBACK \
  -u GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH \
  -u GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH \
  -u GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC \
  -u GGML_CUDA_NVFP4_FATTN_P_DIRECT \
  -u GGML_CUDA_NVFP4_FATTN_DEBUG \
  CUDA_VISIBLE_DEVICES=0 \
  LLAMA_NVFP4_KCACHE_OUTLIER=1 \
  LLAMA_KCACHE_HYBRID_FP8_E4M3_E8M0_32_LAYERS=high_medium \
  "${BIN}" \
    -m "${MODEL}" \
    -f "${DATA}" \
    --cache-type-k nvfp4 \
    --cache-type-v nvfp4 \
    --n_gpu_layers 40 \
    --batch-size 512 \
    --ubatch-size 512 \
    -t 32 \
    -c 8192 \
    --kv-unified \
  >> "${LOG}" 2>&1

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG}"
