#!/usr/bin/env python3
"""Validate an exported RMS_NORM result against its input tensor.

Usage:
    verify-rms-norm.py RESULT.bin [INPUT.bin]

Pass the exported result (dst). The matching input (src0) is resolved from the
adjacent manifest.json by node index. An explicit input remains supported for
exports without a manifest. Only F32 RMS_NORM tensor exports are supported.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def bf16_to_f32(value: int) -> float:
    return bits_f32(value << 16)


def bf16_from_f32_rz(value: float) -> int:
    return f32_bits(value) >> 16


def bf16_from_f32_rne(value: float) -> int:
    bits = f32_bits(value)
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent == 0x7F800000 and mantissa != 0:
        return (bits >> 16) | 0x0040

    upper = bits >> 16
    lower = bits & 0xFFFF
    return (upper + int(lower > 0x8000 or (lower == 0x8000 and (upper & 1) != 0))) & 0xFFFF


def bf16_add(left: int, right: int) -> int:
    return bf16_from_f32_rne(f32(bf16_to_f32(left) + bf16_to_f32(right)))


def bf16_mul(left: int, right: int) -> int:
    return bf16_from_f32_rne(f32(bf16_to_f32(left) * bf16_to_f32(right)))


def bf16_fma(left: int, right: int, accumulator: int) -> int:
    value = bf16_to_f32(left) * bf16_to_f32(right) + bf16_to_f32(accumulator)
    return bf16_from_f32_rne(f32(value))


def bf16_sqrt(value: int) -> int:
    return bf16_from_f32_rne(f32(math.sqrt(bf16_to_f32(value))))


def bf16_div(numerator: int, denominator: int) -> int:
    return bf16_from_f32_rne(f32(bf16_to_f32(numerator) / bf16_to_f32(denominator)))


def read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) == 0 or len(data) % 4 != 0:
        raise ValueError(f"{path}: expected a non-empty F32 binary, got {len(data)} bytes")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def rms_norm_f32(values: list[float], row_width: int, epsilon: float) -> list[float]:
    output: list[float] = []
    for offset in range(0, len(values), row_width):
        row = values[offset : offset + row_width]
        mean_square = math.fsum(value * value for value in row) / row_width
        scale = 1.0 / math.sqrt(mean_square + epsilon)
        output.extend(f32(value * scale) for value in row)
    return output


def rms_norm_qemu_bf16(values: list[float], row_width: int, epsilon: float) -> list[float]:
    output: list[float] = []
    inverse_cols = bf16_from_f32_rne(f32(1.0 / f32(float(row_width))))
    epsilon_bf16 = bf16_from_f32_rne(f32(epsilon))

    for offset in range(0, len(values), row_width):
        row = [bf16_from_f32_rz(value) for value in values[offset : offset + row_width]]
        lane_sums = [0] * 32
        for lane in range(32):
            lane_sum = 0
            for column in range(lane, row_width, 32):
                lane_sum = bf16_fma(row[column], row[column], lane_sum)
            lane_sums[lane] = lane_sum

        sum_squares = 0
        for lane_sum in lane_sums:
            sum_squares = bf16_add(sum_squares, lane_sum)
        mean = bf16_mul(sum_squares, inverse_cols)
        mean_with_epsilon = bf16_add(mean, epsilon_bf16)
        root = bf16_sqrt(mean_with_epsilon)
        scale = bf16_div(0x3F80, root)
        output.extend(bf16_to_f32(bf16_mul(value, scale)) for value in row)
    return output


def rms_norm_qemu_fp32(values: list[float], row_width: int, epsilon: float) -> list[float]:
    output: list[float] = []
    inverse_cols = f32(1.0 / f32(float(row_width)))
    epsilon_f32 = f32(epsilon)

    for offset in range(0, len(values), row_width):
        row = [bf16_to_f32(bf16_from_f32_rz(value))
               for value in values[offset : offset + row_width]]
        lane_sums: list[float] = []
        for lane in range(32):
            lane_sum = 0.0
            for column in range(lane, row_width, 32):
                # A BF16 square is exact in F32, so rounding the sum reproduces
                # the FP32 FMA used by the CUDA model without requiring math.fma.
                lane_sum = f32(row[column] * row[column] + lane_sum)
            lane_sums.append(lane_sum)

        sum_squares = 0.0
        for lane_sum in lane_sums:
            sum_squares = f32(sum_squares + lane_sum)
        mean = f32(sum_squares * inverse_cols)
        mean_with_epsilon = f32(mean + epsilon_f32)
        root = f32(math.sqrt(mean_with_epsilon))
        scale = f32(1.0 / root)
        output.extend(
            bf16_to_f32(bf16_from_f32_rne(f32(value * scale)))
            for value in row
        )
    return output


def find_manifest(
        result_path: Path,
        input_path: Path | None,
        requested_manifest: Path | None) -> tuple[Path | None, dict | None]:
    if requested_manifest is not None:
        candidate = requested_manifest.resolve()
        if not candidate.is_file():
            raise ValueError(f"manifest not found: {candidate}")
        return candidate, json.loads(candidate.read_text(encoding="utf-8"))

    candidates = [result_path.parent / "manifest.json"]
    if input_path is not None:
        candidates.append(input_path.parent / "manifest.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
    return None, None


def find_record(manifest_path: Path, manifest: dict, path: Path) -> dict:
    matches = [
        record for record in manifest.get("records", [])
        if (manifest_path.parent / str(record.get("path", ""))).resolve() == path
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest record for {path}, found {len(matches)}")
    return matches[0]


def validate_record(record: dict, expected_role: str) -> None:
    if record.get("op") != "RMS_NORM" or record.get("role") != expected_role:
        raise ValueError(
            f"expected role={expected_role} and op=RMS_NORM, got "
            f"role={record.get('role')} and op={record.get('op')}"
        )
    if record.get("dtype") != "f32":
        raise ValueError("only F32 RMS_NORM exports are supported")


def resolve_input(
        result_path: Path,
        requested_input: Path | None,
        requested_manifest: Path | None) -> tuple[Path, Path | None, dict | None, dict | None]:
    manifest_path, manifest = find_manifest(result_path, requested_input, requested_manifest)
    if manifest is None or manifest_path is None:
        if requested_input is None:
            raise ValueError(
                f"manifest not found next to {result_path}; pass INPUT.bin explicitly"
            )
        return requested_input, None, None, None

    result_record = find_record(manifest_path, manifest, result_path)
    validate_record(result_record, "dst")

    if requested_input is None:
        node_index = result_record.get("node_index")
        matches = [
            record for record in manifest.get("records", [])
            if record.get("node_index") == node_index
            and record.get("op") == "RMS_NORM"
            and record.get("role") == "src0"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one RMS_NORM src0 record for node {node_index}, found {len(matches)}"
            )
        input_record = matches[0]
        input_path = (manifest_path.parent / str(input_record.get("path", ""))).resolve()
    else:
        input_path = requested_input
        input_record = find_record(manifest_path, manifest, input_path)

    validate_record(input_record, "src0")
    if result_record.get("node_index") != input_record.get("node_index"):
        raise ValueError(
            f"node mismatch: result={result_record.get('node_index')} "
            f"input={input_record.get('node_index')}"
        )
    if result_record.get("ne") != input_record.get("ne"):
        raise ValueError(
            f"shape mismatch: result={result_record.get('ne')} input={input_record.get('ne')}"
        )
    return input_path, manifest_path, result_record, input_record


def infer_row_width(
        element_count: int,
        requested_width: int | None,
        input_record: dict | None) -> int:
    if requested_width is not None:
        return requested_width
    if input_record is None:
        return element_count
    return int(input_record["ne"][0])


def detect_mode(result_path: Path, actual: list[float], requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    for parent in [result_path.parent, result_path.parent.parent]:
        command_path = parent / "command.txt"
        if command_path.is_file():
            command = command_path.read_text(encoding="utf-8", errors="replace")
            if "GGML_CUDA_RMS_NORM_QEMU_MODE=qemu_cuda" in command or \
                    "GGML_CUDA_RMS_NORM_QEMU_MODE=qemu " in command:
                return "qemu-fp32"

    if all((f32_bits(value) & 0xFFFF) == 0 for value in actual):
        return "qemu-fp32"
    return "f32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="RMS_NORM result/dst F32 .bin file")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="optional RMS_NORM input/src0; defaults to the same node's src0 in manifest.json",
    )
    parser.add_argument("--manifest", type=Path, help="defaults to RESULT.bin's sibling manifest.json")
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--row-width", type=int, help="override ne[0] when no manifest is available")
    parser.add_argument(
        "--mode",
        choices=["auto", "f32", "qemu-fp32", "qemu-bf16"],
        default="auto",
    )
    parser.add_argument("--atol", type=float, default=5.0e-5)
    parser.add_argument("--rtol", type=float, default=5.0e-5)
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    result_path = args.result.resolve()
    requested_input = args.input.resolve() if args.input is not None else None
    input_path, manifest_path, _, input_record = resolve_input(
        result_path, requested_input, args.manifest
    )
    actual = read_f32(result_path)
    input_values = read_f32(input_path)
    if len(actual) != len(input_values):
        raise ValueError(f"element count mismatch: result={len(actual)} input={len(input_values)}")

    row_width = infer_row_width(len(input_values), args.row_width, input_record)
    if row_width <= 0 or len(input_values) % row_width != 0:
        raise ValueError(
            f"element count {len(input_values)} is not divisible by row width {row_width}"
        )

    mode = detect_mode(result_path, actual, args.mode)
    if mode == "qemu-fp32":
        expected = rms_norm_qemu_fp32(input_values, row_width, args.epsilon)
    elif mode == "qemu-bf16":
        expected = rms_norm_qemu_bf16(input_values, row_width, args.epsilon)
    else:
        expected = rms_norm_f32(input_values, row_width, args.epsilon)

    deltas = [got - want for got, want in zip(actual, expected)]
    abs_deltas = [abs(delta) for delta in deltas]
    mae = math.fsum(abs_deltas) / len(abs_deltas)
    rmse = math.sqrt(math.fsum(delta * delta for delta in deltas) / len(deltas))
    max_abs = max(abs_deltas)
    exact = sum(got == want for got, want in zip(actual, expected))
    within_tolerance = sum(
        abs(got - want) <= args.atol + args.rtol * abs(want)
        for got, want in zip(actual, expected)
    )
    sign_mismatches = sum(
        source != 0.0 and got != 0.0 and math.copysign(1.0, source) != math.copysign(1.0, got)
        for source, got in zip(input_values, actual)
    )
    finite = all(math.isfinite(value) for value in input_values + actual)
    passed = finite and within_tolerance == len(actual)

    print("RMSNorm validation")
    print(f"  result: {result_path}")
    print(f"  input: {input_path}")
    print(f"  manifest: {manifest_path or 'not found'}")
    print(f"  mode: {mode}")
    print(f"  elements: {len(actual)} (row_width={row_width})")
    print(f"  epsilon: {args.epsilon:.9g}")
    print(f"  MAE: {mae:.9g}")
    print(f"  RMSE: {rmse:.9g}")
    print(f"  max abs error: {max_abs:.9g}")
    print(f"  exact matches: {exact}/{len(actual)}")
    print(
        f"  within tolerance (atol={args.atol:.3g}, rtol={args.rtol:.3g}): "
        f"{within_tolerance}/{len(actual)}"
    )
    print(f"  sign mismatches: {sign_mismatches}")
    print(f"  result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main() -> int:
    try:
        return run()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
