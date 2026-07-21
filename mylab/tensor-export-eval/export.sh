#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Experiment parameters. Edit them directly before running.
TYPE="decode" # decode or prefill
PROMPT=$(cat wikitext-chunk-512.txt)
echo "PROMPT=${PROMPT}"
OP="GGML_OP_RMS_NORM"
LAYER=0
CUDA_DEVICE="1"
MODEL_PATH="${ROOT_DIR}/models/qwen3-8b-nvfp4.gguf"
GGML_CUDA_RMS_NORM_QEMU_MODE="qemu_cuda"

# Leave empty to create a timestamped directory under experiments/.
RUN_DIR=""

# Set to 1 to rebuild llama-cli before running.
REBUILD=1

# Add intentional llama-cli overrides here. Keep empty for baseline parameters.
EXTRA_ARGS=()

LLAMA_CLI="${ROOT_DIR}/build_cuda/bin/llama-cli"

TYPE="${TYPE,,}"
OP="${OP^^}"
OP="${OP#GGML_OP_}"
OP="${OP//-/_}"

case "${TYPE}" in
    decode)
        # One generated token only evaluates prefill. Requesting two tokens
        # executes exactly one single-token decode graph after prefill.
        N_PREDICT=2
        ;;
    prefill)
        N_PREDICT=1
        ;;
    *)
        echo "invalid TYPE='${TYPE}'; expected decode or prefill" >&2
        exit 1
        ;;
esac

if [[ -z "${OP}" ]]; then
    echo "OP must not be empty" >&2
    exit 1
fi

if [[ ! "${LAYER}" =~ ^[0-9]+$ ]]; then
    echo "invalid LAYER='${LAYER}'; expected a non-negative integer" >&2
    exit 1
fi

if [[ ${REBUILD} -eq 1 ]]; then
    cmake --build "${ROOT_DIR}/build_cuda" --target llama-cli -j
fi

if [[ ! -x "${LLAMA_CLI}" ]]; then
    echo "llama-cli not found or not executable: ${LLAMA_CLI}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "model not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ -z "${RUN_DIR}" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    op_slug="${OP,,}"
    op_slug="${op_slug//_/-}"
    RUN_DIR="${ROOT_DIR}/experiments/${timestamp}-op-tensor-export-${TYPE}-${op_slug}"
fi

if [[ -e "${RUN_DIR}" ]]; then
    echo "run directory already exists: ${RUN_DIR}" >&2
    exit 1
fi

TENSOR_DIR="${RUN_DIR}/tensors"
mkdir -p "${TENSOR_DIR}"

COMMAND=(
    "${LLAMA_CLI}"
    -m "${MODEL_PATH}"
    --n-gpu-layers 40
    -t 32
    -c 8192
    --batch-size 512
    --ubatch-size 512
    --cache-type-k f16
    --cache-type-v f16
    --kv-unified
    --no-warmup
    --no-display-prompt
    --simple-io
    --no-conversation
    --ignore-eos
    --seed 1
    --temp 0
    -n "${N_PREDICT}"
    --prompt "${PROMPT}"
    "${EXTRA_ARGS[@]}"
)

printf '%s' "${PROMPT}" > "${RUN_DIR}/prompt.txt"
{
    printf 'TYPE=%q\n' "${TYPE}"
    printf 'PROMPT=%q\n' "${PROMPT}"
    printf 'OP=%q\n' "${OP}"
    printf 'LAYER=%q\n' "${LAYER}"
    printf 'CUDA_DEVICE=%q\n' "${CUDA_DEVICE}"
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'GGML_CUDA_RMS_NORM_QEMU_MODE=%q\n' "${GGML_CUDA_RMS_NORM_QEMU_MODE}"
    printf 'N_PREDICT=%q\n' "${N_PREDICT}"
    printf 'CODE_REVISION=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "${RUN_DIR}/config.env"

{
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_DEVICE}"
    printf 'GGML_CUDA_RMS_NORM_QEMU_MODE=%q ' "${GGML_CUDA_RMS_NORM_QEMU_MODE}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_DIR=%q ' "${TENSOR_DIR}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_OP=%q ' "${OP}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_TYPE=%q ' "${TYPE}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_LAYER=%q ' "${LAYER}"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "${RUN_DIR}/command.txt"

git -C "${ROOT_DIR}" status --short > "${RUN_DIR}/git-status.txt" 2>/dev/null || true

set +e
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
GGML_CUDA_RMS_NORM_QEMU_MODE="${GGML_CUDA_RMS_NORM_QEMU_MODE}" \
GGML_CUDA_DISABLE_GRAPHS=0 \
GGML_CUDA_TRUNC_ENABLE=1 \
GGML_CUDA_TRUNC_LOG=1 \
LLAMA_EXPT_TENSOR_EXPORT_DIR="${TENSOR_DIR}" \
LLAMA_EXPT_TENSOR_EXPORT_OP="${OP}" \
LLAMA_EXPT_TENSOR_EXPORT_TYPE="${TYPE}" \
LLAMA_EXPT_TENSOR_EXPORT_LAYER="${LAYER}" \
    "${COMMAND[@]}" > "${RUN_DIR}/run.log" 2>&1
exit_code=$?
set -e

MANIFEST_PATH="${TENSOR_DIR}/manifest.json"
if [[ ${exit_code} -ne 0 ]]; then
    {
        echo "# Op tensor export"
        echo
        echo "- Type: \`${TYPE}\`"
        echo "- Op: \`${OP}\`"
        echo "- Layer: ${LAYER}"
        echo "- Validation: \`failed\`"
        echo "- Exit code: ${exit_code}"
        echo "- Error: llama-cli failed; inspect \`run.log\`."
    } > "${RUN_DIR}/summary.md"
    echo "export failed: ${RUN_DIR}" >&2
    exit "${exit_code}"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
    {
        echo "# Op tensor export"
        echo
        echo "- Type: \`${TYPE}\`"
        echo "- Op: \`${OP}\`"
        echo "- Layer: ${LAYER}"
        echo "- Validation: \`failed\`"
        echo "- Error: no manifest was produced."
    } > "${RUN_DIR}/summary.md"
    echo "no manifest was produced: ${RUN_DIR}" >&2
    exit 2
fi

matched_nodes="$(sed -nE 's/^[[:space:]]*"matched_nodes":[[:space:]]*([0-9]+),?$/\1/p' "${MANIFEST_PATH}" | head -n 1)"
manifest_layer="$(sed -nE 's/^[[:space:]]*"layer":[[:space:]]*([0-9]+),?$/\1/p' "${MANIFEST_PATH}" | head -n 1)"
dst_records="$(grep -c '"role": "dst"' "${MANIFEST_PATH}" || true)"
src0_records="$(grep -c '"role": "src0"' "${MANIFEST_PATH}" || true)"
src1_records="$(grep -c '"role": "src1"' "${MANIFEST_PATH}" || true)"

matched_nodes="${matched_nodes:-0}"
valid=false
if [[ ${manifest_layer:-invalid} == "${LAYER}" && ${matched_nodes} -gt 0 && ${dst_records} -eq ${matched_nodes} ]]; then
    valid=true
fi

{
    printf 'EXIT_CODE=%q\n' "${exit_code}"
    printf 'TYPE=%q\n' "${TYPE}"
    printf 'OP=%q\n' "${OP}"
    printf 'LAYER=%q\n' "${LAYER}"
    printf 'MATCHED_NODES=%q\n' "${matched_nodes}"
    printf 'DST_RECORDS=%q\n' "${dst_records}"
    printf 'SRC0_RECORDS=%q\n' "${src0_records}"
    printf 'SRC1_RECORDS=%q\n' "${src1_records}"
    printf 'VALID=%q\n' "${valid}"
} > "${RUN_DIR}/validation.env"

{
    echo "# Op tensor export"
    echo
    echo "- Type: \`${TYPE}\`"
    echo "- Op: \`${OP}\`"
    echo "- Layer: ${LAYER}"
    echo "- Matched nodes: ${matched_nodes}"
    echo "- dst records: ${dst_records}"
    echo "- src0 records: ${src0_records}"
    echo "- src1 records: ${src1_records}"
    echo "- Validation: \`${valid}\`"
} > "${RUN_DIR}/summary.md"

if [[ "${valid}" != true ]]; then
    echo "manifest validation failed: ${RUN_DIR}" >&2
    exit 3
fi

echo "export completed: ${RUN_DIR}"
