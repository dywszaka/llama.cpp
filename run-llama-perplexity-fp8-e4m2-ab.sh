#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="${BIN:-${ROOT_DIR}/build_cuda/bin/llama-perplexity}"
MODEL_PATH="${MODEL_PATH:-/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/ppl-fp8-e4m2-results}"

COMMON_ARGS=(
  -m "${MODEL_PATH}"
  -f "${PROMPT_FILE}"
  --cache-type-k nvfp4
  --cache-type-v fp8_e4m3_e8m0_32
  --n_gpu_layers 40
  --batch-size 2048
  --ubatch-size 512
  -t 32
  -c 2048
)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_NVFP4_NATIVE="${GGML_CUDA_NVFP4_NATIVE:-1}"
export GGML_CUDA_TRUNC_ENABLE="${GGML_CUDA_TRUNC_ENABLE:-0}"
export GGML_CUDA_TRUNC_LOG="${GGML_CUDA_TRUNC_LOG:-0}"
export GGML_CUDA_NVFP4_FP4MULMAT="${GGML_CUDA_NVFP4_FP4MULMAT:-1}"
export GGML_CUDA_NVFP4_FP4MULMAT_LOG="${GGML_CUDA_NVFP4_FP4MULMAT_LOG:-1}"

if [[ ! -x "${BIN}" ]]; then
  echo "llama-perplexity binary not found or not executable: ${BIN}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "model not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

baseline_log="${OUT_DIR}/fp8-e4m3.log"
e4m2_log="${OUT_DIR}/fp8-e4m2.log"

echo "Running baseline fp8_e4m3_e8m0_32 PPL..."
GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2=0 \
  "${BIN}" "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "${baseline_log}"

echo "Running fp8_e4m2 experiment PPL..."
GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2=1 \
  "${BIN}" "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "${e4m2_log}"

echo "Logs:"
echo "  baseline: ${baseline_log}"
echo "  e4m2:     ${e4m2_log}"
