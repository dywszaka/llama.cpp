#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAB_DIR="${ROOT_DIR}/mylab/kqv-heatmap"
LOG_DIR="${LAB_DIR}/logs"
RESULTS_DIR="${LAB_DIR}/results"
RAW_DIR="${RESULTS_DIR}/raw_tensors"

mkdir -p \
  "${LOG_DIR}" \
  "${RESULTS_DIR}" \
  "${RAW_DIR}/q_raw_f32" \
  "${RAW_DIR}/kq_raw_f32" \
  "${RAW_DIR}/v_raw_f32" \
  "${RAW_DIR}/vp_raw_f32"

CUDA_VISIBLE_DEVICES=0 \
  "${ROOT_DIR}/build_cuda/bin/llama-kcache-mean" \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f "${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw" \
    --include-prompt \
    -n 0 \
    -o "${RESULTS_DIR}/kcache_mean_one_chunk.jsonl" \
    --tensor-dist-json "${RESULTS_DIR}/tensor_distribution.json" \
    --tensor-raw-dump-dir "${RAW_DIR}" \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --n_gpu_layers 40 \
    --batch-size 512 \
    --ubatch-size 512 \
    -t 32 \
    -c 512 \
    --kv-unified \
    --chunks 1 \
    2>&1 | tee "${LOG_DIR}/kqv_export.raw.log"
