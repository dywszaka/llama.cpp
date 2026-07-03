#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments/20260703T075735Z-cuda-c100-release-ppl-one-chunk"
BIN="${ROOT_DIR}/build-cuda-c100/bin/llama-perplexity"
LOG="${EXP_DIR}/logs/ppl.log"
TIME_LOG="${EXP_DIR}/results/time.txt"

cd "${ROOT_DIR}"

start_epoch="$(date +%s)"
status=0

(
  export CUDA_VISIBLE_DEVICES=0
  export PROJECT_ROOT="${ROOT_DIR}/build-cuda-c100"
  export LD_LIBRARY_PATH="${ROOT_DIR}/build-cuda-c100/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  unset LLAMA_EXPT_C100_SOFT_MAX
  unset GGML_SCHED_DEBUG

  "${BIN}" \
      --device CUDA0 \
      -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
      -f "${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw" \
      --cache-type-k f16 \
      --cache-type-v f16 \
      --n_gpu_layers 40 \
      --batch-size 512 \
      --ubatch-size 512 \
      -t 32 \
      -c 512 \
      --kv-unified \
      --chunks 1
) 2>&1 | tee "${LOG}" || status="${PIPESTATUS[0]}"

end_epoch="$(date +%s)"
{
  printf 'start_epoch=%s\n' "${start_epoch}"
  printf 'end_epoch=%s\n' "${end_epoch}"
  printf 'elapsed_seconds=%s\n' "$((end_epoch - start_epoch))"
  printf 'exit_status=%s\n' "${status}"
} > "${TIME_LOG}"

exit "${status}"
