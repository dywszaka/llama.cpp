#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="${ROOT_DIR}/experiments/20260521T035332Z-nvfp4-vcache-fast-update-bench"
LOG_DIR="${EXP_DIR}/logs"
METRICS_DIR="${EXP_DIR}/metrics"
mkdir -p "${LOG_DIR}" "${METRICS_DIR}"

BENCH_BIN="${ROOT_DIR}/build_cuda/bin/llama-bench"
MODEL_PATH="/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf"

COMMON_ARGS=(
    -m "${MODEL_PATH}"
    --n-gpu-layers 40
    --batch-size 2048
    --ubatch-size 512
    -t 32
    -p 512
    -n 128
    -r "${BENCH_REPS:-5}"
    -o json
    -v
)

clear_nvfp4_env() {
    unset LLAMA_EXPERIMENT_NVFP4_VCACHE
    unset LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE
    unset LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV
    unset LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV_LT
    unset LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE
    unset GGML_CUDA_NVFP4_FATTN
    unset GGML_CUDA_NVFP4_FATTN_NO_FALLBACK
    unset GGML_CUDA_NVFP4_FATTN_P_DIRECT
    unset GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH
    unset GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH
    unset GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC
}

run_case() {
    local name="$1"
    shift
    local json="${METRICS_DIR}/${name}.json"
    local err="${LOG_DIR}/${name}.stderr.log"
    local raw="${LOG_DIR}/${name}.raw.log"

    echo "== ${name} ==" | tee "${raw}"
    "$@" > "${json}" 2> >(tee "${err}" >&2)
    {
        echo
        echo "## stdout JSON (${json})"
        cat "${json}"
        echo
        echo "## stderr (${err})"
        cat "${err}"
    } >> "${raw}"
}

export CUDA_VISIBLE_DEVICES=0

clear_nvfp4_env
run_case "01-baseline" \
    env GGML_CUDA_NVFP4_NATIVE=1 \
    "${BENCH_BIN}" "${COMMON_ARGS[@]}" \
    --cache-type-k f16 \
    --cache-type-v f16

clear_nvfp4_env
run_case "02-nvfp4-fast-update-off" \
    env GGML_CUDA_NVFP4_NATIVE=1 \
        LLAMA_EXPERIMENT_NVFP4_VCACHE=1 \
        LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=0 \
    "${BENCH_BIN}" "${COMMON_ARGS[@]}" \
    --cache-type-k f16 \
    --cache-type-v nvfp4 \
    --kv-unified 1

clear_nvfp4_env
run_case "03-nvfp4-fast-update-on" \
    env GGML_CUDA_NVFP4_NATIVE=1 \
        LLAMA_EXPERIMENT_NVFP4_VCACHE=1 \
        LLAMA_EXPERIMENT_NVFP4_VCACHE_FAST_UPDATE=1 \
    "${BENCH_BIN}" "${COMMON_ARGS[@]}" \
    --cache-type-k f16 \
    --cache-type-v nvfp4 \
    --kv-unified 1

"${EXP_DIR}/parse-results.sh"
