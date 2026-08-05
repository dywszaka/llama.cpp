#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Experiment parameters. Edit them directly or override them in the environment.
TYPE="${TYPE:-decode}" # decode or prefill
PROMPT="${PROMPT:-$(cat "${ROOT_DIR}/mylab/tensor-export-eval/wikitext-chunk-512.txt")}"
echo "PROMPT=${PROMPT}"
OP="${OP-GGML_OP_SOFT_MAX}"
NAME="${NAME-}"
LAYER="${LAYER:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen3-8b-nvfp4.gguf}"

# Leave empty to create a timestamped directory under experiments/.
RUN_DIR="${RUN_DIR:-}"

# Set to 1 to rebuild llama-cli before running.
REBUILD="${REBUILD:-1}"

# Set to 1 to export F32 tensors as raw BF16 by truncating the low 16 bits.
BF16_DUMP="${BF16_DUMP:-0}"

# CMake build directory. If unset, prefer an existing configured CUDA build.
BUILD_DIR="${BUILD_DIR:-}"
if [[ -z "${BUILD_DIR}" ]]; then
    for candidate in "${ROOT_DIR}/build_cuda" "${ROOT_DIR}/build-cuda"; do
        if [[ -f "${candidate}/CMakeCache.txt" ]]; then
            BUILD_DIR="${candidate}"
            break
        fi
    done
fi
if [[ -z "${BUILD_DIR}" ]]; then
    if [[ -d "${ROOT_DIR}/build_cuda" ]]; then
        BUILD_DIR="${ROOT_DIR}/build_cuda"
    elif [[ -d "${ROOT_DIR}/build-cuda" ]]; then
        BUILD_DIR="${ROOT_DIR}/build-cuda"
    else
        BUILD_DIR="${ROOT_DIR}/build_cuda"
    fi
fi

# Add intentional llama-cli overrides here. Keep empty for baseline parameters.
EXTRA_ARGS=()

LLAMA_CLI="${LLAMA_CLI:-${BUILD_DIR}/bin/llama-cli}"
RMS_NORM_VALIDATOR_SOURCE="${ROOT_DIR}/mylab/tensor-export-eval/verify-rms-norm.py"
MUL_MAT_VALIDATOR_SOURCE="${ROOT_DIR}/mylab/tensor-export-eval/verify-mul-mat.py"
SOFT_MAX_VALIDATOR_SOURCE="${ROOT_DIR}/mylab/tensor-export-eval/verify-soft-max.py"
ROPE_VALIDATOR_SOURCE="${ROOT_DIR}/mylab/tensor-export-eval/verify-rope.py"
ROPE_COS_SIN_DIR="${ROPE_COS_SIN_DIR:-${ROOT_DIR}/experiments/rope-cos-sin-c-8192}"
ROPE_COS_SIN_MANIFEST_SOURCE="${ROPE_COS_SIN_DIR}/manifest.json"
ROPE_COS_SIN_DATA_SOURCE="${ROPE_COS_SIN_DIR}/rope-cos-sin-f32.bin"

TYPE="${TYPE,,}"
OP="${OP^^}"
OP="${OP#GGML_OP_}"
OP="${OP//-/_}"

VALIDATOR_SOURCE=""
VALIDATOR_NAME=""
case "${OP}" in
    RMS_NORM)
        VALIDATOR_SOURCE="${RMS_NORM_VALIDATOR_SOURCE}"
        VALIDATOR_NAME="verify-rms-norm.py"
        ;;
    MUL_MAT)
        VALIDATOR_SOURCE="${MUL_MAT_VALIDATOR_SOURCE}"
        VALIDATOR_NAME="verify-mul-mat.py"
        ;;
    SOFT_MAX)
        VALIDATOR_SOURCE="${SOFT_MAX_VALIDATOR_SOURCE}"
        VALIDATOR_NAME="verify-soft-max.py"
        ;;
    ROPE)
        VALIDATOR_SOURCE="${ROPE_VALIDATOR_SOURCE}"
        VALIDATOR_NAME="verify-rope.py"
        ;;
esac

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

if [[ -z "${NAME}" && -z "${OP}" ]]; then
    echo "NAME and OP must not both be empty" >&2
    exit 1
fi

if [[ ! "${LAYER}" =~ ^[0-9]+$ ]]; then
    echo "invalid LAYER='${LAYER}'; expected a non-negative integer" >&2
    exit 1
fi

case "${BF16_DUMP}" in
    0|1) ;;
    *)
        echo "invalid BF16_DUMP='${BF16_DUMP}'; expected 0 or 1" >&2
        exit 1
        ;;
esac

if [[ ${REBUILD} -eq 1 ]]; then
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        echo "CMake cache not found in BUILD_DIR='${BUILD_DIR}'." >&2
        echo "Configure the build first, or set BUILD_DIR to an existing CMake build directory." >&2
        echo "Example: cmake -S '${ROOT_DIR}' -B '${BUILD_DIR}' -DGGML_CUDA=ON" >&2
        exit 1
    fi
    cmake --build "${BUILD_DIR}" --target llama-cli -j
fi

if [[ ! -x "${LLAMA_CLI}" ]]; then
    echo "llama-cli not found or not executable: ${LLAMA_CLI}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "model not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ -n "${VALIDATOR_SOURCE}" && ! -f "${VALIDATOR_SOURCE}" ]]; then
    echo "validator not found for OP=${OP}: ${VALIDATOR_SOURCE}" >&2
    exit 1
fi

if [[ "${OP}" == "ROPE" &&
      ( ! -f "${ROPE_COS_SIN_MANIFEST_SOURCE}" || ! -f "${ROPE_COS_SIN_DATA_SOURCE}" ) ]]; then
    echo "ROPE cos/sin table is incomplete: ${ROPE_COS_SIN_DIR}" >&2
    exit 1
fi

if [[ -z "${RUN_DIR}" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    if [[ -n "${NAME}" ]]; then
        name_slug="$(printf '%s' "${NAME,,}" | tr -cs '[:alnum:].-' '-')"
        name_slug="${name_slug#-}"
        name_slug="${name_slug%-}"
        RUN_DIR="${ROOT_DIR}/experiments/${timestamp}-tensor-export-${TYPE}-name-${name_slug}-layer-${LAYER}"
    else
        op_slug="${OP,,}"
        op_slug="${op_slug//_/-}"
        RUN_DIR="${ROOT_DIR}/experiments/${timestamp}-tensor-export-${TYPE}-op-${op_slug}-layer-${LAYER}"
    fi
fi

if [[ -e "${RUN_DIR}" ]]; then
    echo "run directory already exists: ${RUN_DIR}" >&2
    exit 1
fi

TENSOR_DIR="${RUN_DIR}/tensors"
mkdir -p "${TENSOR_DIR}"
if [[ -n "${VALIDATOR_SOURCE}" ]]; then
    install -m 0755 "${VALIDATOR_SOURCE}" "${RUN_DIR}/${VALIDATOR_NAME}"
fi
if [[ "${OP}" == "ROPE" ]]; then
    install -m 0644 "${ROPE_COS_SIN_MANIFEST_SOURCE}" "${RUN_DIR}/rope-cos-sin-manifest.json"
    install -m 0644 "${ROPE_COS_SIN_DATA_SOURCE}" "${RUN_DIR}/rope-cos-sin-f32.bin"
fi

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
    printf 'NAME=%q\n' "${NAME}"
    printf 'LAYER=%q\n' "${LAYER}"
    printf 'CUDA_DEVICE=%q\n' "${CUDA_DEVICE}"
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'N_PREDICT=%q\n' "${N_PREDICT}"
    printf 'BF16_DUMP=%q\n' "${BF16_DUMP}"
    printf 'BUILD_DIR=%q\n' "${BUILD_DIR}"
    printf 'LLAMA_CLI=%q\n' "${LLAMA_CLI}"
    printf 'CODE_REVISION=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "${RUN_DIR}/config.env"

{
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_DEVICE}"
    printf 'GGML_CUDA_RMS_NORM_QEMU_MODE=%q ' "qemu_cuda"
    printf 'GGML_CUDA_DISABLE_GRAPHS=%q ' "0"
    printf 'GGML_CUDA_TRUNC_ENABLE=%q ' "1"
    printf 'GGML_CUDA_TRUNC_LOG=%q ' "1"
    printf 'GGML_CUDA_NVFP4_FP4MULMAT=%q ' "1"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_DIR=%q ' "${TENSOR_DIR}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_OP=%q ' "${OP}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_NAME=%q ' "${NAME}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_TYPE=%q ' "${TYPE}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_LAYER=%q ' "${LAYER}"
    printf 'LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP=%q ' "${BF16_DUMP}"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "${RUN_DIR}/command.txt"

git -C "${ROOT_DIR}" status --short > "${RUN_DIR}/git-status.txt" 2>/dev/null || true

set +e
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
GGML_CUDA_RMS_NORM_QEMU_MODE="qemu_cuda" \
GGML_CUDA_DISABLE_GRAPHS=0 \
GGML_CUDA_TRUNC_ENABLE=1 \
GGML_CUDA_TRUNC_LOG=1 \
GGML_CUDA_NVFP4_FP4MULMAT=1 \
GGML_CUDA_NVFP4_BF16_QUANT=1 \
GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1 \
LLAMA_EXPT_TENSOR_EXPORT_DIR="${TENSOR_DIR}" \
LLAMA_EXPT_TENSOR_EXPORT_OP="${OP}" \
LLAMA_EXPT_TENSOR_EXPORT_NAME="${NAME}" \
LLAMA_EXPT_TENSOR_EXPORT_TYPE="${TYPE}" \
LLAMA_EXPT_TENSOR_EXPORT_LAYER="${LAYER}" \
LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP="${BF16_DUMP}" \
    "${COMMAND[@]}" > "${RUN_DIR}/run.log" 2>&1
exit_code=$?
set -e

MANIFEST_PATH="${TENSOR_DIR}/manifest.json"
if [[ ${exit_code} -ne 0 ]]; then
    {
        echo "# Tensor export"
        echo
        echo "- Type: \`${TYPE}\`"
        echo "- Op: \`${OP}\`"
        echo "- Name: \`${NAME}\`"
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
        echo "# Tensor export"
        echo
        echo "- Type: \`${TYPE}\`"
        echo "- Op: \`${OP}\`"
        echo "- Name: \`${NAME}\`"
        echo "- Layer: ${LAYER}"
        echo "- Validation: \`failed\`"
        echo "- Error: no manifest was produced."
    } > "${RUN_DIR}/summary.md"
    echo "no manifest was produced: ${RUN_DIR}" >&2
    exit 2
fi

matched_nodes="$(sed -nE 's/^[[:space:]]*"matched_nodes":[[:space:]]*([0-9]+),?$/\1/p' "${MANIFEST_PATH}" | head -n 1)"
manifest_layer="$(sed -nE 's/^[[:space:]]*"layer":[[:space:]]*([0-9]+),?$/\1/p' "${MANIFEST_PATH}" | head -n 1)"
manifest_format="$(sed -nE 's/^[[:space:]]*"format":[[:space:]]*"([^"]*)",?$/\1/p' "${MANIFEST_PATH}" | head -n 1)"
selection_priority="$(sed -nE 's/^[[:space:]]*"priority":[[:space:]]*"([^"]*)",?$/\1/p' "${MANIFEST_PATH}" | head -n 1)"
requested_name="$(sed -nE 's/^[[:space:]]*"requested_name":[[:space:]]*"([^"]*)",?$/\1/p' "${MANIFEST_PATH}" | head -n 1)"
dst_records="$(grep -c '"role": "dst"' "${MANIFEST_PATH}" || true)"
src0_records="$(grep -c '"role": "src0"' "${MANIFEST_PATH}" || true)"
src1_records="$(grep -c '"role": "src1"' "${MANIFEST_PATH}" || true)"
src2_records="$(grep -c '"role": "src2"' "${MANIFEST_PATH}" || true)"
first_dst_path="$(awk '
    /"role": "dst"/ { selected = 1; next }
    selected && /"path":/ {
        path = $0
        sub(/^[[:space:]]*"path":[[:space:]]*"/, "", path)
        sub(/",?[[:space:]]*$/, "", path)
        print path
        exit
    }
' "${MANIFEST_PATH}")"
validator_exit_code=""
validator_valid=""
if [[ "${BF16_DUMP}" == "1" && -n "${VALIDATOR_NAME}" && "${OP}" != "SOFT_MAX" ]]; then
    validator_valid="skipped_bf16_dump"
elif [[ -n "${VALIDATOR_NAME}" && -n "${first_dst_path}" ]]; then
    set +e
    "${RUN_DIR}/${VALIDATOR_NAME}" "${TENSOR_DIR}/${first_dst_path}" \
        > "${RUN_DIR}/validator.log" 2>&1
    validator_exit_code=$?
    set -e
    if [[ ${validator_exit_code} -eq 0 ]]; then
        validator_valid=true
    else
        validator_valid=false
    fi
fi
resolved_name=""
dst_name_mismatches=0
if [[ -n "${NAME}" ]]; then
    resolved_name="${NAME}"
    if [[ ! "${resolved_name}" =~ -[0-9]+$ ]]; then
        resolved_name="${resolved_name}-${LAYER}"
    fi
    dst_name_mismatches="$(awk -v expected="${resolved_name}" '
        /"role": "dst"/ { in_dst = 1; next }
        in_dst && /"name":/ {
            name = $0
            sub(/^[[:space:]]*"name":[[:space:]]*"/, "", name)
            sub(/",?[[:space:]]*$/, "", name)
            if (name != expected) {
                mismatches++
            }
            in_dst = 0
        }
        END { print mismatches + 0 }
    ' "${MANIFEST_PATH}")"
fi

matched_nodes="${matched_nodes:-0}"
valid=false
if [[ "${manifest_format}" == "llama_expt_op_tensor_export_v2" &&
      ${manifest_layer:-invalid} == "${LAYER}" &&
      ${matched_nodes} -gt 0 &&
      ${dst_records} -eq ${matched_nodes} ]]; then
    if [[ -n "${NAME}" ]]; then
        if [[ "${selection_priority}" == "tensor_name" &&
              "${requested_name}" == "${NAME}" &&
              ${dst_name_mismatches} -eq 0 ]]; then
            valid=true
        fi
    elif [[ "${selection_priority}" == "op" && -z "${requested_name}" ]]; then
        valid=true
    fi
fi

{
    printf 'EXIT_CODE=%q\n' "${exit_code}"
    printf 'TYPE=%q\n' "${TYPE}"
    printf 'OP=%q\n' "${OP}"
    printf 'NAME=%q\n' "${NAME}"
    printf 'LAYER=%q\n' "${LAYER}"
    printf 'BF16_DUMP=%q\n' "${BF16_DUMP}"
    printf 'MANIFEST_FORMAT=%q\n' "${manifest_format}"
    printf 'SELECTION_PRIORITY=%q\n' "${selection_priority}"
    printf 'REQUESTED_NAME=%q\n' "${requested_name}"
    printf 'RESOLVED_NAME=%q\n' "${resolved_name}"
    printf 'DST_NAME_MISMATCHES=%q\n' "${dst_name_mismatches}"
    printf 'MATCHED_NODES=%q\n' "${matched_nodes}"
    printf 'DST_RECORDS=%q\n' "${dst_records}"
    printf 'SRC0_RECORDS=%q\n' "${src0_records}"
    printf 'SRC1_RECORDS=%q\n' "${src1_records}"
    printf 'SRC2_RECORDS=%q\n' "${src2_records}"
    printf 'VALIDATOR=%q\n' "${VALIDATOR_NAME}"
    printf 'VALIDATOR_EXIT_CODE=%q\n' "${validator_exit_code}"
    printf 'VALIDATOR_VALID=%q\n' "${validator_valid}"
    printf 'VALID=%q\n' "${valid}"
} > "${RUN_DIR}/validation.env"

{
    echo "# Tensor export"
    echo
    echo "- Type: \`${TYPE}\`"
    echo "- Op: \`${OP}\`"
    echo "- Name: \`${NAME}\`"
    echo "- Layer: ${LAYER}"
    echo "- BF16 dump: \`${BF16_DUMP}\`"
    echo "- Selection priority: \`${selection_priority}\`"
    if [[ -n "${NAME}" ]]; then
        echo "- Resolved dst name: \`${resolved_name}\`"
        echo "- dst name mismatches: ${dst_name_mismatches}"
    fi
    echo "- Matched nodes: ${matched_nodes}"
    echo "- dst records: ${dst_records}"
    echo "- src0 records: ${src0_records}"
    echo "- src1 records: ${src1_records}"
    echo "- src2 records: ${src2_records}"
    if [[ -n "${VALIDATOR_NAME}" ]]; then
        echo "- Bundled validator: \`${VALIDATOR_NAME}\`"
        if [[ "${validator_valid}" == "skipped_bf16_dump" ]]; then
            echo "- Validator result: \`skipped\` (BF16_DUMP=1 changes F32 export dtype)"
        else
            echo "- Validator result: \`${validator_valid}\` (exit ${validator_exit_code})"
        fi
    fi
    echo "- Validation: \`${valid}\`"
    if [[ "${BF16_DUMP}" == "1" && "${OP}" != "SOFT_MAX" ]]; then
        :
    elif [[ "${OP}" == "RMS_NORM" && -n "${first_dst_path}" ]]; then
        echo
        echo "## RMSNorm data validation"
        echo
        echo "Pass one result file; its input is resolved from manifest.json:"
        echo
        echo '```bash'
        printf './%s tensors/%s\n' "${VALIDATOR_NAME}" "${first_dst_path}"
        echo '```'
    elif [[ "${OP}" == "MUL_MAT" && -n "${first_dst_path}" ]]; then
        echo
        echo "## MUL_MAT data validation"
        echo
        echo "Pass one result file; its inputs are resolved from manifest.json:"
        echo
        echo '```bash'
        printf './%s tensors/%s\n' "${VALIDATOR_NAME}" "${first_dst_path}"
        echo '```'
    elif [[ "${OP}" == "SOFT_MAX" && -n "${first_dst_path}" ]]; then
        echo
        echo "## SOFT_MAX data validation"
        echo
        echo "Pass one result file; its inputs and op parameters are resolved from manifest.json:"
        echo
        echo '```bash'
        printf './%s tensors/%s\n' "${VALIDATOR_NAME}" "${first_dst_path}"
        echo '```'
    elif [[ "${OP}" == "ROPE" && -n "${first_dst_path}" ]]; then
        echo
        echo "## ROPE data validation"
        echo
        echo "The validator resolves src0, positions, and op parameters from the manifest and looks up cos/sin from the bundled static table:"
        echo
        echo '```bash'
        printf './%s tensors/%s\n' "${VALIDATOR_NAME}" "${first_dst_path}"
        echo '```'
    fi
} > "${RUN_DIR}/summary.md"

if [[ "${valid}" != true ]]; then
    echo "manifest validation failed: ${RUN_DIR}" >&2
    exit 3
fi

if [[ -n "${VALIDATOR_NAME}" && "${validator_valid}" != true && "${validator_valid}" != "skipped_bf16_dump" ]]; then
    echo "tensor value validation failed: ${RUN_DIR}" >&2
    exit 4
fi

echo "export completed: ${RUN_DIR}"
