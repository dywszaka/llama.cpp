#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${ROOT_DIR}/experiments}"
COMPARE_PY="${SCRIPT_DIR}/compare.py"

usage() {
    cat <<'EOF'
Usage:
  compare.sh EXPERIMENT_A EXPERIMENT_B TENSOR_NAME [compare.py options]

The tensor files are resolved as:
  <project>/experiments/EXPERIMENT_A/tensors/TENSOR_NAME
  <project>/experiments/EXPERIMENT_B/tensors/TENSOR_NAME

Example:
  compare.sh \
    20260720T070840Z-op-tensor-export-decode-rms-norm \
    20260720T071045Z-op-tensor-export-decode-rms-norm \
    0-node1-dst-norm-0.bin

Set EXPERIMENTS_DIR to override the default experiments directory. Additional
arguments, such as --atol, --rtol, and --max-mismatches, are passed to compare.py.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    usage
    exit 0
fi

if [[ $# -lt 3 ]]; then
    usage >&2
    exit 2
fi

EXPERIMENT_A="$1"
EXPERIMENT_B="$2"
TENSOR_NAME="$3"
shift 3

BIN_A="${EXPERIMENTS_DIR}/${EXPERIMENT_A}/tensors/${TENSOR_NAME}"
BIN_B="${EXPERIMENTS_DIR}/${EXPERIMENT_B}/tensors/${TENSOR_NAME}"

if [[ ! -x "${COMPARE_PY}" ]]; then
    echo "compare.py not found or not executable: ${COMPARE_PY}" >&2
    exit 2
fi

if [[ ! -f "${BIN_A}" ]]; then
    echo "tensor file not found: ${BIN_A}" >&2
    exit 2
fi

if [[ ! -f "${BIN_B}" ]]; then
    echo "tensor file not found: ${BIN_B}" >&2
    exit 2
fi

exec "${COMPARE_PY}" "${BIN_A}" "${BIN_B}" "$@"
