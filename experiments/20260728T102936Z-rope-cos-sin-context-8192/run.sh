#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build_cuda"
GENERATOR="${RUN_DIR}/generate-rope-cos-sin"

g++ -std=c++17 -O2 \
    -I"${ROOT_DIR}/ggml/include" \
    "${RUN_DIR}/generate-rope-cos-sin.cpp" \
    -L"${BUILD_DIR}/bin" \
    -Wl,-rpath,"${BUILD_DIR}/bin" \
    -lggml-cuda -lggml -lggml-base \
    -o "${GENERATOR}"

CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_ROPE_QEMU_ENABLED=1 \
    "${GENERATOR}" "${RUN_DIR}" > "${RUN_DIR}/run.log" 2>&1

sha256sum "${RUN_DIR}/rope-cos-sin-f32.bin" > "${RUN_DIR}/sha256.txt"
"${RUN_DIR}/query-rope-cos-sin.py" \
    --position 0:2 \
    --channel-idx 0:4 \
    > "${RUN_DIR}/query-sample.csv"

python3 - "${RUN_DIR}/rope-cos-sin-f32.bin" > "${RUN_DIR}/validation.log" <<'PY'
import struct
import sys

path = sys.argv[1]
raw = open(path, "rb").read()
values = struct.unpack("<" + "f" * (len(raw) // 4), raw)
max_unit_error = 0.0
max_abs = 0.0
for index in range(0, len(values), 2):
    cos_value, sin_value = values[index:index + 2]
    max_unit_error = max(max_unit_error, abs(cos_value*cos_value + sin_value*sin_value - 1.0))
    max_abs = max(max_abs, abs(cos_value), abs(sin_value))

assert len(values) == 8192 * 64 * 2
assert values[:8] == (1.0, 0.0) * 4
assert max_abs <= 1.000001
assert max_unit_error < 2.0e-6
print(f"elements={len(values)}")
print(f"bytes={len(raw)}")
print(f"max_unit_circle_error={max_unit_error:.9g}")
print(f"max_abs={max_abs:.9g}")
PY

check_dir="$(mktemp -d)"
trap 'rm -r -- "${check_dir}"' EXIT
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_ROPE_QEMU_ENABLED=0 \
    "${GENERATOR}" "${check_dir}" > "${check_dir}/run.log" 2>&1
cmp "${RUN_DIR}/rope-cos-sin-f32.bin" "${check_dir}/rope-cos-sin-f32.bin"
printf 'qemu_enabled_vs_disabled=bit_identical\n' >> "${RUN_DIR}/validation.log"
