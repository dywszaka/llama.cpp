#!/usr/bin/env python3
"""Validate an exported SOFT_MAX result using its manifest inputs and op params.

Usage:
    verify-soft-max.py RESULT.bin

The matching src0, optional mask (src1), optional attention sinks (src2), scale,
and max_bias are resolved from the adjacent manifest.json by node index.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


Q31_ONE = 0x80000000
EXP_STEP_Q31 = 0x7FE00400
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bf16_to_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def f32_to_bf16_rne(value: float) -> int:
    bits = f32_bits(value)
    absolute = bits & 0x7FFFFFFF
    if absolute > 0x7F800000:
        result = bits >> 16
        return result if result & 0x007F else result | 1
    bits = (bits + 0x00007FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return bits >> 16


def normalize_op(value: object) -> str:
    return str(value or "").upper().removeprefix("GGML_OP_").replace("-", "_")


@dataclass
class TensorData:
    path: Path
    record: dict
    raw: bytes

    @property
    def dtype(self) -> str:
        return str(self.record["dtype"]).lower()

    @property
    def ne(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.record["ne"])

    @property
    def nb(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.record["nb"])

    def scalar(self, i0: int, i1: int, i2: int, i3: int) -> float:
        offset = i0 * self.nb[0] + i1 * self.nb[1] + i2 * self.nb[2] + i3 * self.nb[3]
        if self.dtype == "f32":
            return struct.unpack_from("<f", self.raw, offset)[0]
        if self.dtype == "f16":
            return struct.unpack_from("<e", self.raw, offset)[0]
        if self.dtype == "bf16":
            return bf16_to_f32(struct.unpack_from("<H", self.raw, offset)[0])
        raise ValueError(f"unsupported scalar dtype '{self.dtype}' for {self.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="exported SOFT_MAX dst/result F32 .bin file")
    parser.add_argument("--manifest", type=Path, help="defaults to RESULT.bin's sibling manifest.json")
    parser.add_argument("--scale", type=float, help="override scale; required for an older manifest without op_params")
    parser.add_argument(
        "--max-bias",
        type=float,
        help="override max_bias; required for an older manifest without op_params",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "f32", "qemu-bf16"],
        default="auto",
        help="auto reads GGML_CUDA_SOFT_MAX_QEMU_MODE from command.txt",
    )
    parser.add_argument(
        "--result-rounding",
        choices=["auto", "f32", "bf16-rne"],
        default="auto",
        help="normal F32 mode only; auto reads GGML_CUDA_TRUNC_ENABLE from command.txt",
    )
    parser.add_argument("--atol", type=float, default=5.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-4)
    parser.add_argument("--max-mismatches", type=int, default=10)
    return parser.parse_args()


def load_document(result_path: Path, manifest_arg: Path | None) -> tuple[Path, dict]:
    manifest_path = manifest_arg.resolve() if manifest_arg else result_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"manifest not found: {manifest_path}")
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def find_result_record(result_path: Path, manifest_path: Path, document: dict) -> dict:
    matches = [
        record for record in document.get("records", [])
        if (manifest_path.parent / str(record.get("path", ""))).resolve() == result_path
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest record for result, found {len(matches)}")
    record = matches[0]
    if record.get("role") != "dst" or normalize_op(record.get("op")) != "SOFT_MAX":
        raise ValueError("result record must have role=dst and op=SOFT_MAX")
    return record


def records_for_node(document: dict, node_index: int) -> dict[str, dict]:
    return {
        str(record.get("role")): record
        for record in document.get("records", [])
        if record.get("node_index") == node_index
    }


def load_tensor(manifest_path: Path, record: dict) -> TensorData:
    path = (manifest_path.parent / str(record["path"])).resolve()
    raw = path.read_bytes()
    expected_size = int(record["byte_size"])
    if len(raw) != expected_size:
        raise ValueError(f"byte-size mismatch for {path}: file={len(raw)} manifest={expected_size}")
    return TensorData(path=path, record=record, raw=raw)


def check_bf16_representable_f32(
    tensor: TensorData,
    max_failures: int,
) -> tuple[int, int, list[str]]:
    if tensor.dtype != "f32":
        raise ValueError(f"BF16-rounded input check requires F32 storage, got {tensor.dtype}")

    matching = 0
    failures: list[str] = []
    for i3 in range(tensor.ne[3]):
        for i2 in range(tensor.ne[2]):
            for i1 in range(tensor.ne[1]):
                for i0 in range(tensor.ne[0]):
                    value = tensor.scalar(i0, i1, i2, i3)
                    bits = f32_bits(value)
                    if bits & 0xFFFF == 0:
                        matching += 1
                    elif len(failures) < max_failures:
                        failures.append(
                            f"    [{i0},{i1},{i2},{i3}] value={value:.9g} bits=0x{bits:08x}"
                        )
    return matching, math.prod(tensor.ne), failures


def resolve_op_param(result_record: dict, name: str, override: float | None) -> float:
    if override is not None:
        return override
    op_params = result_record.get("op_params")
    if not isinstance(op_params, dict) or name not in op_params:
        raise ValueError(f"manifest result record has no op_params.{name}; pass --{name.replace('_', '-')}")
    return float(op_params[name])


def validate_shapes(
    result: TensorData,
    src0: TensorData,
    mask: TensorData | None,
    sinks: TensorData | None,
) -> None:
    if result.dtype != "f32" or src0.dtype != "f32":
        raise ValueError(f"SOFT_MAX requires F32 dst/src0, got dst={result.dtype} src0={src0.dtype}")
    if result.ne != src0.ne:
        raise ValueError(f"dst/src0 shape mismatch: dst={result.ne} src0={src0.ne}")
    if mask is not None:
        if mask.dtype not in ("f16", "f32"):
            raise ValueError(f"SOFT_MAX mask must be F16 or F32, got {mask.dtype}")
        if mask.ne[0] != src0.ne[0] or mask.ne[1] < src0.ne[1]:
            raise ValueError(f"mask shape is incompatible with src0: mask={mask.ne} src0={src0.ne}")
        if src0.ne[2] % mask.ne[2] != 0 or src0.ne[3] % mask.ne[3] != 0:
            raise ValueError(f"mask batch dimensions do not broadcast: mask={mask.ne} src0={src0.ne}")
    if sinks is not None:
        if sinks.dtype != "f32" or sinks.ne[0] != src0.ne[2]:
            raise ValueError(f"sinks must be F32 with ne[0]={src0.ne[2]}, got dtype={sinks.dtype} shape={sinks.ne}")


def command_text(manifest_path: Path) -> str:
    for candidate in (manifest_path.parent / "command.txt", manifest_path.parent.parent / "command.txt"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    return ""


def detect_mode(manifest_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    command = command_text(manifest_path)
    if "GGML_CUDA_SOFT_MAX_QEMU_MODE=qemu_cuda" in command or \
            "GGML_CUDA_SOFT_MAX_QEMU_MODE=qemu " in command:
        return "qemu-bf16"
    return "f32"


def detect_result_rounding(manifest_path: Path, requested: str, mode: str) -> str:
    if mode == "qemu-bf16":
        return "bf16-rne"
    if requested != "auto":
        return requested
    return "bf16-rne" if "GGML_CUDA_TRUNC_ENABLE=1" in command_text(manifest_path) else "f32"


def alibi_slope(max_bias: float, head: int, nheads: int) -> float:
    if max_bias <= 0.0:
        return 1.0
    n_head_log2 = 1 << int(math.floor(math.log2(nheads)))
    m0 = f32(math.pow(2.0, -max_bias / n_head_log2))
    m1 = f32(math.pow(2.0, -(max_bias / 2.0) / n_head_log2))
    return f32(math.pow(m0, head + 1) if head < n_head_log2 else math.pow(m1, 2 * (head - n_head_log2) + 1))


def prepared_row(
    src0: TensorData,
    mask: TensorData | None,
    scale: float,
    max_bias: float,
    i1: int,
    i2: int,
    i3: int,
) -> list[float]:
    slope = alibi_slope(max_bias, i2, src0.ne[2])
    values: list[float] = []
    for i0 in range(src0.ne[0]):
        value = f32(src0.scalar(i0, i1, i2, i3) * scale)
        if mask is not None:
            mask_value = mask.scalar(i0, i1, i2 % mask.ne[2], i3 % mask.ne[3])
            value = f32(value + slope * mask_value)
        values.append(value)
    return values


def softmax_f32_row(values: list[float], sink: float | None, result_rounding: str) -> list[float]:
    maximum = max(values) if sink is None else max(max(values), sink)
    exponents = [f32(math.exp(value - maximum)) for value in values]
    total = math.fsum(exponents)
    if sink is not None:
        total += f32(math.exp(sink - maximum))
    output = [f32(value / total) for value in exponents]
    if result_rounding == "bf16-rne":
        output = [bf16_to_f32(f32_to_bf16_rne(value)) for value in output]
    return output


def round_shift_u64(value: int, shift: int) -> int:
    if shift == 0:
        return value
    if shift >= 64:
        return 0
    quotient = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return quotient + int(remainder > halfway or (remainder == halfway and quotient & 1))


def bf16_to_q16(bits: int) -> int:
    sign = bits >> 15
    exponent = (bits >> 7) & 0xFF
    fraction = bits & 0x7F
    if exponent == 0:
        return 0
    if exponent == 0xFF:
        return INT32_MIN if sign else INT32_MAX
    significand = 128 + fraction
    shift = exponent - 118
    if shift >= 0:
        magnitude = 0x80000000 if shift >= 56 or significand > (0x80000000 >> shift) else significand << shift
    else:
        magnitude = round_shift_u64(significand, -shift)
    if sign:
        return INT32_MIN if magnitude >= 0x80000000 else -magnitude
    return INT32_MAX if magnitude > INT32_MAX else magnitude


def mul_q31(left: int, right: int) -> int:
    product = left * right
    quotient = product >> 31
    remainder = product & 0x7FFFFFFF
    halfway = 0x40000000
    return quotient + int(remainder > halfway or (remainder == halfway and quotient & 1))


def exp_neg_q31(delta_q16: int) -> int:
    steps = (delta_q16 + 32) >> 6
    if steps >= 32768:
        return 0
    result = Q31_ONE
    factor = EXP_STEP_Q31
    power = steps
    while power:
        if power & 1:
            result = mul_q31(result, factor)
        power >>= 1
        if power:
            factor = mul_q31(factor, factor)
    return result


def divide_rne_u64(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(
        remainder > denominator - remainder
        or (remainder == denominator - remainder and quotient & 1)
    )


def q31_to_bf16_bits(probability_q31: int) -> int:
    if probability_q31 == 0:
        return 0
    msb = probability_q31.bit_length() - 1
    exponent = msb + 96
    significand = round_shift_u64(probability_q31, msb - 7) if msb >= 7 else probability_q31 << (7 - msb)
    if significand == 256:
        significand = 128
        exponent += 1
    return (exponent << 7) | (significand & 0x7F)


def probability_bf16_bits(exponent_q31: int, sum_q31: int) -> int:
    if exponent_q31 == 0 or sum_q31 == 0:
        return 0
    probability_q31 = min(divide_rne_u64(exponent_q31 << 31, sum_q31), Q31_ONE)
    return q31_to_bf16_bits(probability_q31)


def softmax_qemu_bf16_row(values: list[float], sink: float | None) -> list[float]:
    input_bits = [f32_to_bf16_rne(value) for value in values]
    sink_bits = f32_to_bf16_rne(sink) if sink is not None else None
    input_q16 = [bf16_to_q16(value) for value in input_bits]
    maximum = max(input_q16) if sink_bits is None else max(max(input_q16), bf16_to_q16(sink_bits))
    exponents = [exp_neg_q31(maximum - value) for value in input_q16]
    total = sum(exponents)
    if sink_bits is not None:
        total += exp_neg_q31(maximum - bf16_to_q16(sink_bits))
    return [bf16_to_f32(probability_bf16_bits(value, total)) for value in exponents]


def run() -> int:
    args = parse_args()
    if args.atol < 0 or args.rtol < 0 or args.max_mismatches < 0:
        raise ValueError("tolerances and output limits must be non-negative")

    result_path = args.result.resolve()
    manifest_path, document = load_document(result_path, args.manifest)
    result_record = find_result_record(result_path, manifest_path, document)
    node_index = int(result_record["node_index"])
    role_records = records_for_node(document, node_index)
    if "src0" not in role_records:
        raise ValueError(f"SOFT_MAX node {node_index} has no src0 record")

    result = load_tensor(manifest_path, result_record)
    src0 = load_tensor(manifest_path, role_records["src0"])
    mask = load_tensor(manifest_path, role_records["src1"]) if "src1" in role_records else None
    sinks = load_tensor(manifest_path, role_records["src2"]) if "src2" in role_records else None
    validate_shapes(result, src0, mask, sinks)
    scale = resolve_op_param(result_record, "scale", args.scale)
    max_bias = resolve_op_param(result_record, "max_bias", args.max_bias)
    mode = detect_mode(manifest_path, args.mode)
    result_rounding = detect_result_rounding(manifest_path, args.result_rounding, mode)
    input_bf16_matching, input_count, input_bf16_failures = check_bf16_representable_f32(
        src0, args.max_mismatches
    )
    input_bf16_ok = input_bf16_matching == input_count

    count = math.prod(result.ne)
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    exact = 0
    within_tolerance = 0
    mismatches: list[str] = []

    for i3 in range(result.ne[3]):
        for i2 in range(result.ne[2]):
            sink = sinks.scalar(i2, 0, 0, 0) if sinks is not None else None
            for i1 in range(result.ne[1]):
                values = prepared_row(src0, mask, scale, max_bias, i1, i2, i3)
                expected_row = (
                    softmax_qemu_bf16_row(values, sink)
                    if mode == "qemu-bf16"
                    else softmax_f32_row(values, sink, result_rounding)
                )
                for i0, expected in enumerate(expected_row):
                    actual = result.scalar(i0, i1, i2, i3)
                    delta = actual - expected
                    abs_delta = abs(delta)
                    tolerance = args.atol + args.rtol * abs(expected)
                    sum_abs += abs_delta
                    sum_sq += delta * delta
                    max_abs = max(max_abs, abs_delta)
                    exact += actual == expected
                    if math.isfinite(actual) and math.isfinite(expected) and abs_delta <= tolerance:
                        within_tolerance += 1
                    elif len(mismatches) < args.max_mismatches:
                        mismatches.append(
                            f"    [{i0},{i1},{i2},{i3}] actual={actual:.9g} "
                            f"expected={expected:.9g} abs_error={abs_delta:.9g}"
                        )

    mae = sum_abs / count
    rmse = math.sqrt(sum_sq / count)
    passed = input_bf16_ok and within_tolerance == count
    print("SOFT_MAX validation")
    print(f"  result: {result.path}")
    print(f"  manifest: {manifest_path}")
    print(f"  node_index: {node_index}")
    print(f"  mode: {mode}")
    print(f"  result rounding: {result_rounding}")
    print(f"  src0: {src0.path} dtype={src0.dtype} shape={src0.ne}")
    print(
        "  src0 BF16-representable F32: "
        f"{'yes' if input_bf16_ok else 'no'} "
        f"(F32 low 16 bits zero: {input_bf16_matching}/{input_count})"
    )
    if input_bf16_failures:
        print("  first non-BF16-representable src0 values:")
        print("\n".join(input_bf16_failures))
    print(f"  mask: {mask.path if mask else 'none'}" + (f" dtype={mask.dtype} shape={mask.ne}" if mask else ""))
    print(f"  sinks: {sinks.path if sinks else 'none'}" + (f" shape={sinks.ne}" if sinks else ""))
    print(f"  scale: {scale:.9g}")
    print(f"  max_bias: {max_bias:.9g}")
    print(f"  elements: {count}")
    print(f"  MAE: {mae:.9g}")
    print(f"  RMSE: {rmse:.9g}")
    print(f"  max abs error: {max_abs:.9g}")
    print(f"  exact matches: {exact}/{count}")
    print(
        f"  within tolerance (atol={args.atol:.3g}, rtol={args.rtol:.3g}): "
        f"{within_tolerance}/{count}"
    )
    if mismatches:
        print("  first mismatches:")
        print("\n".join(mismatches))
    print(f"  result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main() -> int:
    try:
        return run()
    except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
