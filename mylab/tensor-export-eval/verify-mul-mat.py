#!/usr/bin/env python3
"""Validate an exported MUL_MAT result using inputs resolved from manifest.json.

Usage:
    verify-mul-mat.py RESULT.bin

ggml MUL_MAT computes dst = src0^T * src1, including batch broadcasting over
dimensions 2 and 3. The result record's node_index is used to locate inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


NVFP4_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bf16_to_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def e4m3_to_f32_half(value: int) -> float:
    sign = -1.0 if value & 0x80 else 1.0
    exponent = (value >> 3) & 0x0F
    mantissa = value & 0x07
    if exponent == 0:
        decoded = 0.0 if mantissa == 0 else (2.0**-6) * (mantissa / 8.0)
    elif exponent == 0x0F and mantissa == 0x07:
        decoded = math.nan
    else:
        decoded = (2.0 ** (exponent - 7)) * (1.0 + mantissa / 8.0)
    return sign * decoded * 0.5


def round_f32_to_bf16_rne(value: float) -> float:
    bits = f32_bits(value)
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent == 0x7F800000 and mantissa != 0:
        upper = (bits >> 16) | 0x0040
    else:
        upper = bits >> 16
        lower = bits & 0xFFFF
        upper = (upper + int(lower > 0x8000 or (lower == 0x8000 and (upper & 1) != 0))) & 0xFFFF
    return bf16_to_f32(upper)


def normalize_op(value: object) -> str:
    return str(value or "").upper().removeprefix("GGML_OP_").replace("-", "_")


@dataclass
class TensorData:
    path: Path
    record: dict
    raw: bytes
    scale: "TensorData | None" = None
    implicit_unit_global_scale: bool = False

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

    def row_scale(self, i1: int, i2: int, i3: int) -> float:
        if self.scale is None:
            if self.implicit_unit_global_scale:
                return 1.0
            raise ValueError(f"NVFP4 tensor has no global-scale record: {self.path}")
        sne = self.scale.ne
        if math.prod(sne) == 1:
            return self.scale.scalar(0, 0, 0, 0)
        if sne[0] == self.ne[1] and sne[1] == self.ne[2] and sne[2] == self.ne[3]:
            return self.scale.scalar(i1, i2, i3, 0)
        if sne[0] == self.ne[1] and sne[1:] == (1, 1, 1):
            return self.scale.scalar(i1, 0, 0, 0)
        raise ValueError(
            f"unsupported global-scale shape {sne} for NVFP4 tensor shape {self.ne}"
        )

    def row(self, i1: int, i2: int, i3: int) -> list[float]:
        if self.dtype != "nvfp4":
            return [self.scalar(i0, i1, i2, i3) for i0 in range(self.ne[0])]

        if self.ne[0] % 16 != 0:
            raise ValueError(f"NVFP4 row width must be divisible by 16: {self.ne[0]}")
        global_scale = self.row_scale(i1, i2, i3)
        row_base = i1 * self.nb[1] + i2 * self.nb[2] + i3 * self.nb[3]
        output: list[float] = []
        for block in range(self.ne[0] // 16):
            offset = row_base + block * self.nb[0]
            block_scale = e4m3_to_f32_half(self.raw[offset])
            scale = block_scale / global_scale if global_scale != 0.0 else 0.0
            for packed in self.raw[offset + 1 : offset + 9]:
                output.append(NVFP4_VALUES[packed & 0x0F] * scale)
                output.append(NVFP4_VALUES[packed >> 4] * scale)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="exported MUL_MAT dst/result .bin file")
    parser.add_argument("--manifest", type=Path, help="defaults to RESULT.bin's sibling manifest.json")
    parser.add_argument("--atol", type=float, default=2.0e-3)
    parser.add_argument("--rtol", type=float, default=2.0e-3)
    parser.add_argument(
        "--result-rounding",
        choices=["auto", "f32", "bf16-rne"],
        default="auto",
        help="auto reads GGML_CUDA_TRUNC_ENABLE from the export command",
    )
    parser.add_argument("--max-mismatches", type=int, default=10)
    parser.add_argument(
        "--max-scale-values",
        type=int,
        default=16,
        help="maximum number of values printed for each scale tensor",
    )
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
    if record.get("role") != "dst" or normalize_op(record.get("op")) != "MUL_MAT":
        raise ValueError("result record must have role=dst and op=MUL_MAT")
    return record


def records_for_node(document: dict, node_index: int) -> tuple[dict[str, dict], dict | None]:
    role_records = {
        str(record.get("role")): record
        for record in document.get("records", [])
        if record.get("node_index") == node_index
    }
    capture = next(
        (item for item in document.get("captures", []) if item.get("node_index") == node_index),
        None,
    )
    return role_records, capture


def load_tensor(
    manifest_path: Path,
    record: dict,
    scale_record: dict | None = None,
    *,
    implicit_unit_global_scale: bool = False,
) -> TensorData:
    path = (manifest_path.parent / record["path"]).resolve()
    raw = path.read_bytes()
    expected_size = int(record["byte_size"])
    if len(raw) != expected_size:
        raise ValueError(f"byte-size mismatch for {path}: file={len(raw)} manifest={expected_size}")
    scale = load_tensor(manifest_path, scale_record) if scale_record is not None else None
    return TensorData(
        path=path,
        record=record,
        raw=raw,
        scale=scale,
        implicit_unit_global_scale=implicit_unit_global_scale,
    )


def scale_value_for_rhs_row(
    scale: TensorData,
    src1: TensorData,
    i1: int,
    i2: int,
    i3: int,
) -> float:
    sne = scale.ne
    if math.prod(sne) == 1:
        return scale.scalar(0, 0, 0, 0)
    if sne[0] == src1.ne[1] and sne[1] == src1.ne[2] and sne[2] == src1.ne[3]:
        return scale.scalar(i1, i2, i3, 0)
    if sne[0] == src1.ne[1] and sne[1:] == (1, 1, 1):
        return scale.scalar(i1, 0, 0, 0)
    raise ValueError(f"unsupported final scale shape {sne} for RHS shape {src1.ne}")


def tensor_scalar_values(tensor: TensorData) -> list[float]:
    return [
        tensor.scalar(i0, i1, i2, i3)
        for i3 in range(tensor.ne[3])
        for i2 in range(tensor.ne[2])
        for i1 in range(tensor.ne[1])
        for i0 in range(tensor.ne[0])
    ]


def bf16_storage_summary(tensor: TensorData) -> tuple[str, int, int, list[str]]:
    count = math.prod(tensor.ne)
    if tensor.dtype == "bf16":
        return "yes (native BF16 storage)", count, count, []
    if tensor.dtype != "f32":
        return f"not applicable for dtype={tensor.dtype}", 0, count, []

    values = tensor_scalar_values(tensor)
    matching = 0
    failures: list[str] = []
    for index, value in enumerate(values):
        bits = f32_bits(value)
        if bits & 0xFFFF == 0:
            matching += 1
        elif len(failures) < 4:
            failures.append(f"index={index} value={value:.9g} bits=0x{bits:08x}")
    status = "yes" if matching == count else "no"
    return status, matching, count, failures


def print_scale_details(label: str, tensor: TensorData, max_values: int) -> None:
    values = tensor_scalar_values(tensor)
    shown = values[:max_values]
    suffix = "" if len(shown) == len(values) else f" ... ({len(values) - len(shown)} more)"
    encoding = tensor.record.get("scale_encoding", "unspecified")
    semantics = tensor.record.get("scale_semantics", "unspecified")
    role = tensor.record.get("role", "unknown")
    bf16_status, bf16_matching, bf16_count, bf16_failures = bf16_storage_summary(tensor)

    print(f"  {label}:")
    print(f"    role: {role}")
    print(f"    file: {tensor.path}")
    print(f"    dtype: {tensor.dtype}")
    print(f"    shape: {tensor.ne}")
    print(f"    scale encoding: {encoding}")
    print(f"    scale semantics: {semantics}")
    print(f"    values: {shown}{suffix}")
    if tensor.dtype == "f32":
        shown_bits = [f"0x{f32_bits(value):08x}" for value in shown]
        print(f"    F32 bits: {shown_bits}{suffix}")
    if values:
        print(f"    value range: min={min(values):.9g} max={max(values):.9g}")
    print(
        "    BF16 value check: "
        f"{bf16_status} (F32 low 16 bits zero: {bf16_matching}/{bf16_count})"
    )
    for failure in bf16_failures:
        print(f"      non-BF16 value: {failure}")


def resolve_inputs(
    manifest_path: Path,
    role_records: dict[str, dict],
    capture: dict | None,
) -> tuple[TensorData, TensorData, TensorData | None, str]:
    if capture and capture.get("status") == "native_nvfp4_valid":
        effective = capture.get("effective_srcs", {})
        a_desc = effective.get("a", {})
        b_desc = effective.get("b", {})
        a_role = a_desc.get("tensor_role")
        b_role = b_desc.get("tensor_role")
        final_scale_role = effective.get("matmul_scale_role")
        if final_scale_role:
            required = [a_role, b_role, final_scale_role]
            if any(not role or role not in role_records for role in required):
                raise ValueError("FP4MULMAT capture is missing an effective input or final scale record")
            return (
                load_tensor(
                    manifest_path,
                    role_records[a_role],
                    implicit_unit_global_scale=True,
                ),
                load_tensor(
                    manifest_path,
                    role_records[b_role],
                    implicit_unit_global_scale=True,
                ),
                load_tensor(manifest_path, role_records[final_scale_role]),
                "native_nvfp4_fp4mulmat_final_scale",
            )

        a_scale_role = a_desc.get("global_scale_role")
        b_scale_role = b_desc.get("global_scale_role")
        required = [a_role, b_role, a_scale_role, b_scale_role]
        if any(not role or role not in role_records for role in required):
            raise ValueError("native NVFP4 capture is missing effective input or scale records")
        return (
            load_tensor(manifest_path, role_records[a_role], role_records[a_scale_role]),
            load_tensor(manifest_path, role_records[b_role], role_records[b_scale_role]),
            None,
            "native_nvfp4_effective_inputs",
        )

    if "src0" not in role_records or "src1" not in role_records:
        raise ValueError("manifest node is missing src0 or src1")
    src0_record = role_records["src0"]
    src1_record = role_records["src1"]
    src0_scale = role_records.get("src0_global_scale") if str(src0_record["dtype"]).lower() == "nvfp4" else None
    src1_scale = role_records.get("src1_global_scale") if str(src1_record["dtype"]).lower() == "nvfp4" else None
    return (
        load_tensor(manifest_path, src0_record, src0_scale),
        load_tensor(manifest_path, src1_record, src1_scale),
        None,
        "manifest_src0_src1",
    )


def validate_shapes(src0: TensorData, src1: TensorData, result: TensorData) -> None:
    if result.dtype != "f32":
        raise ValueError(f"only F32 MUL_MAT results are supported, got {result.dtype}")
    if src0.ne[0] != src1.ne[0]:
        raise ValueError(f"inner dimension mismatch: src0.ne0={src0.ne[0]} src1.ne0={src1.ne[0]}")
    expected = (src0.ne[1], src1.ne[1], src1.ne[2], src1.ne[3])
    if result.ne != expected:
        raise ValueError(f"result shape mismatch: result={result.ne} expected={expected}")
    if src1.ne[2] % src0.ne[2] != 0 or src1.ne[3] % src0.ne[3] != 0:
        raise ValueError(f"invalid batch broadcast: src0={src0.ne} src1={src1.ne}")


def detect_result_rounding(manifest_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested.replace("-", "_")
    command_path = manifest_path.parent.parent / "command.txt"
    if command_path.is_file():
        command = command_path.read_text(encoding="utf-8", errors="replace")
        if "GGML_CUDA_TRUNC_ENABLE=1" in command:
            return "bf16_rne"
    return "f32"


def run() -> int:
    args = parse_args()
    if args.atol < 0 or args.rtol < 0 or args.max_mismatches < 0 or args.max_scale_values < 0:
        raise ValueError("tolerances and output limits must be non-negative")

    result_path = args.result.resolve()
    manifest_path, document = load_document(result_path, args.manifest)
    result_record = find_result_record(result_path, manifest_path, document)
    node_index = int(result_record["node_index"])
    role_records, capture = records_for_node(document, node_index)
    src0, src1, final_scale, input_mode = resolve_inputs(manifest_path, role_records, capture)
    result = load_tensor(manifest_path, result_record)
    validate_shapes(src0, src1, result)
    result_rounding = detect_result_rounding(manifest_path, args.result_rounding)

    r2 = src1.ne[2] // src0.ne[2]
    r3 = src1.ne[3] // src0.ne[3]
    count = math.prod(result.ne)
    sum_abs = 0.0
    sum_sq = 0.0
    max_abs = 0.0
    within_tolerance = 0
    exact = 0
    mismatches: list[str] = []

    for i3 in range(result.ne[3]):
        for i2 in range(result.ne[2]):
            a2 = i2 // r2
            a3 = i3 // r3
            b_rows = [src1.row(i1, i2, i3) for i1 in range(src1.ne[1])]
            for i0 in range(result.ne[0]):
                a_row = src0.row(i0, a2, a3)
                for i1, b_row in enumerate(b_rows):
                    expected = f32(math.fsum(a * b for a, b in zip(a_row, b_row)))
                    if final_scale is not None:
                        scale = scale_value_for_rhs_row(final_scale, src1, i1, i2, i3)
                        expected = f32(expected * scale)
                    if result_rounding == "bf16_rne":
                        expected = round_f32_to_bf16_rne(expected)
                    actual = result.scalar(i0, i1, i2, i3)
                    delta = actual - expected
                    abs_delta = abs(delta)
                    tolerance = args.atol + args.rtol * abs(expected)
                    sum_abs += abs_delta
                    sum_sq += delta * delta
                    max_abs = max(max_abs, abs_delta)
                    exact += actual == expected
                    if abs_delta <= tolerance:
                        within_tolerance += 1
                    elif len(mismatches) < args.max_mismatches:
                        mismatches.append(
                            f"    [{i0},{i1},{i2},{i3}] actual={actual:.9g} "
                            f"expected={expected:.9g} abs_error={abs_delta:.9g}"
                        )

    mae = sum_abs / count
    rmse = math.sqrt(sum_sq / count)
    passed = within_tolerance == count
    print("MUL_MAT validation")
    print(f"  result: {result.path}")
    print(f"  manifest: {manifest_path}")
    print(f"  node_index: {node_index}")
    print(f"  input mode: {input_mode}")
    print(f"  result rounding: {result_rounding}")
    print(f"  src0: {src0.path} dtype={src0.dtype} shape={src0.ne}")
    print(f"  src1: {src1.path} dtype={src1.dtype} shape={src1.ne}")
    scale_tensors: list[tuple[str, TensorData]] = []
    if final_scale is not None:
        scale_tensors.append(("final matmul scale", final_scale))
    else:
        if src0.scale is not None:
            scale_tensors.append(("src0 global scale", src0.scale))
        if src1.scale is not None:
            scale_tensors.append(("src1 global scale", src1.scale))
    if scale_tensors:
        print("  scales used:")
        for label, tensor in scale_tensors:
            print_scale_details(label, tensor, args.max_scale_values)
    else:
        print("  scales used: none")
    print(f"  result shape: {result.ne}")
    formula = "dst = (src0^T * src1) * matmul_scale" if final_scale is not None else "dst = src0^T * src1"
    print(f"  formula: {formula}")
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
