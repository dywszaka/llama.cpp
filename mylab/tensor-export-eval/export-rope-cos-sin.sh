#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

CONTEXT_SIZE="${1:-${CONTEXT_SIZE:-}}"
if [[ -z "${CONTEXT_SIZE}" ]]; then
    echo "usage: $0 CONTEXT_SIZE [OUTPUT_DIR]" >&2
    exit 2
fi
if [[ ! "${CONTEXT_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid CONTEXT_SIZE='${CONTEXT_SIZE}'; expected a positive integer" >&2
    exit 2
fi

OUTPUT_DIR="${2:-${OUTPUT_DIR:-}}"
if [[ -z "${OUTPUT_DIR}" ]]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT_DIR="${ROOT_DIR}/experiments/${timestamp}-rope-cos-sin-context-${CONTEXT_SIZE}"
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "output directory already exists: ${OUTPUT_DIR}" >&2
    exit 1
fi

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_cuda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
HELPER_DEVICE=0
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/qwen3-8b-nvfp4.gguf}"
N_DIMS="${N_DIMS:-128}"
ROPE_MODE="${ROPE_MODE:-2}"
N_CTX_ORIG="${N_CTX_ORIG:-40960}"
FREQ_BASE="${FREQ_BASE:-1000000}"
FREQ_SCALE="${FREQ_SCALE:-1}"
EXT_FACTOR="${EXT_FACTOR:-0}"
ATTN_FACTOR="${ATTN_FACTOR:-1}"
BETA_FAST="${BETA_FAST:-32}"
BETA_SLOW="${BETA_SLOW:-1}"

HELPER_SOURCE="${ROOT_DIR}/mylab/tensor-export-eval/generate-rope-cos-sin.cpp"
HELPER_BIN="${BUILD_DIR}/bin/generate-rope-cos-sin"

if [[ ! -d "${BUILD_DIR}/bin" ]]; then
    echo "build_cuda bin directory not found: ${BUILD_DIR}/bin" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

g++ -std=c++17 -O2 \
    -I"${ROOT_DIR}/ggml/include" \
    -I"${ROOT_DIR}/ggml/src" \
    "${HELPER_SOURCE}" \
    -L"${BUILD_DIR}/bin" \
    -Wl,-rpath,"${BUILD_DIR}/bin" \
    -lggml-cuda -lggml -lggml-base \
    -o "${HELPER_BIN}"

{
    printf 'CONTEXT_SIZE=%q\n' "${CONTEXT_SIZE}"
    printf 'OUTPUT_DIR=%q\n' "${OUTPUT_DIR}"
    printf 'CUDA_DEVICE=%q\n' "${CUDA_DEVICE}"
    printf 'HELPER_DEVICE=%q\n' "${HELPER_DEVICE}"
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'N_DIMS=%q\n' "${N_DIMS}"
    printf 'ROPE_MODE=%q\n' "${ROPE_MODE}"
    printf 'N_CTX_ORIG=%q\n' "${N_CTX_ORIG}"
    printf 'FREQ_BASE=%q\n' "${FREQ_BASE}"
    printf 'FREQ_SCALE=%q\n' "${FREQ_SCALE}"
    printf 'EXT_FACTOR=%q\n' "${EXT_FACTOR}"
    printf 'ATTN_FACTOR=%q\n' "${ATTN_FACTOR}"
    printf 'BETA_FAST=%q\n' "${BETA_FAST}"
    printf 'BETA_SLOW=%q\n' "${BETA_SLOW}"
    printf 'CODE_REVISION=%q\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "${OUTPUT_DIR}/config.env"

COMMAND=(
    "${HELPER_BIN}"
    "${OUTPUT_DIR}"
    "${CONTEXT_SIZE}"
    --device "${HELPER_DEVICE}"
    --model "${MODEL_PATH}"
    --n-dims "${N_DIMS}"
    --mode "${ROPE_MODE}"
    --n-ctx-orig "${N_CTX_ORIG}"
    --freq-base "${FREQ_BASE}"
    --freq-scale "${FREQ_SCALE}"
    --ext-factor "${EXT_FACTOR}"
    --attn-factor "${ATTN_FACTOR}"
    --beta-fast "${BETA_FAST}"
    --beta-slow "${BETA_SLOW}"
)

{
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${CUDA_DEVICE}"
    printf 'GGML_CUDA_ROPE_QEMU_ENABLED=%q ' "1"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "${OUTPUT_DIR}/command.txt"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
GGML_CUDA_ROPE_QEMU_ENABLED=1 \
    "${COMMAND[@]}" > "${OUTPUT_DIR}/run.log" 2>&1

sha256sum "${OUTPUT_DIR}/rope-cos-sin-f32.bin" > "${OUTPUT_DIR}/sha256.txt"

python3 - "${OUTPUT_DIR}/manifest.json" > "${OUTPUT_DIR}/validation.log" <<'PY'
import json
import struct
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
context_size, channels, components = (int(value) for value in manifest["shape"])
if components != 2:
    raise SystemExit(f"invalid component count: {components}")

data_path = manifest_path.parent / manifest["data_file"]
raw = data_path.read_bytes()
expected_bytes = context_size * channels * components * 4
if len(raw) != expected_bytes:
    raise SystemExit(f"byte-size mismatch: actual={len(raw)} expected={expected_bytes}")

values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
max_unit_error = 0.0
max_abs = 0.0
for index in range(0, len(values), 2):
    cos_value, sin_value = values[index:index + 2]
    max_unit_error = max(max_unit_error, abs(cos_value*cos_value + sin_value*sin_value - 1.0))
    max_abs = max(max_abs, abs(cos_value), abs(sin_value))

if context_size > 0:
    first = values[:min(8, len(values))]
    expected_first = (1.0, 0.0) * (len(first) // 2)
    if first != expected_first:
        raise SystemExit(f"unexpected position-0 prefix: {first}")
if max_abs > 1.000001:
    raise SystemExit(f"max_abs too large: {max_abs}")
if max_unit_error >= 2.0e-6:
    raise SystemExit(f"unit circle error too large: {max_unit_error}")

print(f"context_size={context_size}")
print(f"channels={channels}")
print(f"elements={len(values)}")
print(f"bytes={len(raw)}")
print(f"max_unit_circle_error={max_unit_error:.9g}")
print(f"max_abs={max_abs:.9g}")
print("validation=passed")
PY

{
    echo "# RoPE cos/sin export"
    echo
    echo "- Context size: ${CONTEXT_SIZE}"
    echo "- Channels: $((N_DIMS / 2))"
    echo "- Data: \`rope-cos-sin-f32.bin\`"
    echo "- Manifest: \`manifest.json\`"
    echo "- Validation: \`passed\`"
} > "${OUTPUT_DIR}/summary.md"

echo "exported RoPE cos/sin table: ${OUTPUT_DIR}"
