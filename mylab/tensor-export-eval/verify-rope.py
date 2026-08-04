#!/usr/bin/env python3
"""Validate an exported ROPE dst from src0 and a precomputed cos/sin table.

The matching src0 values, position ids, and RoPE parameters are resolved from
the tensor export manifest. Cosine and sine are only loaded from the static
table; this validator never recomputes them.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


ROPE_TYPE_NEOX = 2
ROPE_TYPE_MROPE = 8
ROPE_TYPE_VISION = 24


def f32_to_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def bf16_to_f32(value: int) -> float:
    return bits_to_f32(value << 16)


def f32_to_bf16_rne(value: float) -> int:
    fp32_bits = f32_to_bits(value)
    sign = (fp32_bits >> 31) & 0x1
    exp = (fp32_bits >> 23) & 0xFF
    mant = fp32_bits & 0x007FFFFF

    if exp == 0xFF:
        if mant != 0:
            bf16_bits = (sign << 15) | 0x7F80 | ((mant >> 16) & 0x3F)
            return bf16_bits | 0x0040 if (bf16_bits & 0x7F) == 0 else bf16_bits
        return (sign << 15) | 0x7F80

    if exp == 0 and mant == 0:
        return sign << 15

    if exp == 0:
        shift = 0
        while (mant & (1 << 22)) == 0 and shift < 22:
            mant <<= 1
            shift += 1
        exp = 1 - shift
        mant &= 0x007FFFFF
    else:
        mant |= 0x00800000

    guard_bit = (mant >> 15) & 0x1
    round_bit = (mant >> 14) & 0x1
    sticky = 1 if (mant & 0x3FFF) != 0 else 0
    bf16_mant = (mant >> 16) & 0x7F

    if guard_bit == 1 and (round_bit == 1 or sticky == 1 or (bf16_mant & 0x1) == 1):
        bf16_mant += 1

    if bf16_mant > 0x7F:
        bf16_mant = 0
        exp += 1
        if exp > 0xFE:
            return (sign << 15) | 0x7F80

    if exp < 1:
        return sign << 15

    return (sign << 15) | ((exp & 0xFF) << 7) | (bf16_mant & 0x7F)


def apply_result_rounding(value: float, mode: str) -> float:
    if mode == "f32":
        return value
    if mode == "bf16-rne":
        return bf16_to_f32(f32_to_bf16_rne(value))
    raise ValueError(f"unsupported result rounding mode {mode!r}")


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

    def scalar(self, i0: int, i1: int, i2: int, i3: int) -> float | int:
        offset = i0 * self.nb[0] + i1 * self.nb[1] + i2 * self.nb[2] + i3 * self.nb[3]
        if self.dtype == "f32":
            return struct.unpack_from("<f", self.raw, offset)[0]
        if self.dtype == "f16":
            return struct.unpack_from("<e", self.raw, offset)[0]
        if self.dtype == "i32":
            return struct.unpack_from("<i", self.raw, offset)[0]
        raise ValueError(f"unsupported dtype {self.dtype!r} for {self.path}")


@dataclass
class CosSinTable:
    path: Path
    manifest: dict
    raw: bytes

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.manifest["shape"])

    def pair(self, position: int, channel_idx: int) -> tuple[float, float]:
        context_size, channels, components = self.shape
        if components != 2:
            raise ValueError(f"cos/sin table must have two components, got {components}")
        if not 0 <= position < context_size:
            raise ValueError(f"position {position} is outside table range [0, {context_size})")
        if not 0 <= channel_idx < channels:
            raise ValueError(f"channel_idx {channel_idx} is outside table range [0, {channels})")
        offset = ((position * channels + channel_idx) * 2) * 4
        return struct.unpack_from("<ff", self.raw, offset)


def normalize_op(value: object) -> str:
    return str(value or "").upper().removeprefix("GGML_OP_").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="exported ROPE dst/result binary")
    parser.add_argument("--manifest", type=Path, help="defaults to RESULT's sibling manifest.json")
    parser.add_argument(
        "--cos-sin-manifest",
        type=Path,
        help="defaults to rope-cos-sin-manifest.json next to this script",
    )
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-5)
    parser.add_argument(
        "--result-rounding",
        choices=["auto", "f32", "bf16-rne"],
        default="auto",
        help="defaults to bf16-rne when command.txt enabled GGML_CUDA_TRUNC_ENABLE",
    )
    parser.add_argument("--max-mismatches", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def records_for_result(result_path: Path, manifest_path: Path, manifest: dict) -> dict[str, dict]:
    result_matches = [
        record for record in manifest.get("records", [])
        if (manifest_path.parent / str(record.get("path", ""))).resolve() == result_path
    ]
    if len(result_matches) != 1:
        raise ValueError(f"expected one manifest record for {result_path}, found {len(result_matches)}")
    result_record = result_matches[0]
    if result_record.get("role") != "dst" or normalize_op(result_record.get("op")) != "ROPE":
        raise ValueError("result record must have role=dst and op=ROPE")
    node_index = result_record.get("node_index")
    records = {
        str(record.get("role")): record
        for record in manifest.get("records", [])
        if record.get("node_index") == node_index and normalize_op(record.get("op")) == "ROPE"
    }
    for role in ("dst", "src0", "src1"):
        if role not in records:
            raise ValueError(f"ROPE node {node_index} has no {role} record")
    return records


def command_text(manifest_path: Path) -> str:
    for candidate in (manifest_path.parent / "command.txt", manifest_path.parent.parent / "command.txt"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    return ""


def detect_result_rounding(manifest_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "bf16-rne" if "GGML_CUDA_TRUNC_ENABLE=1" in command_text(manifest_path) else "f32"


def load_tensor(manifest_path: Path, record: dict) -> TensorData:
    path = (manifest_path.parent / str(record["path"])).resolve()
    raw = path.read_bytes()
    expected_size = int(record["byte_size"])
    if len(raw) != expected_size:
        raise ValueError(f"byte-size mismatch for {path}: file={len(raw)} manifest={expected_size}")
    return TensorData(path=path, record=record, raw=raw)


def load_cos_sin_table(manifest_path: Path) -> CosSinTable:
    manifest = load_json(manifest_path, "cos/sin manifest")
    if manifest.get("format") != "llama_cuda_rope_cos_sin_v1":
        raise ValueError(f"unsupported cos/sin table format: {manifest.get('format')!r}")
    if manifest.get("dtype") != "f32_le":
        raise ValueError(f"unsupported cos/sin table dtype: {manifest.get('dtype')!r}")
    if manifest.get("component_order") != ["cos", "sin"]:
        raise ValueError("cos/sin table component order must be [cos, sin]")
    path = (manifest_path.parent / str(manifest["data_file"])).resolve()
    raw = path.read_bytes()
    shape = tuple(int(value) for value in manifest["shape"])
    expected_size = 4
    for dim in shape:
        expected_size *= dim
    if len(raw) != expected_size:
        raise ValueError(f"cos/sin table size mismatch: file={len(raw)} expected={expected_size}")
    return CosSinTable(path=path, manifest=manifest, raw=raw)


def require_compatible_table(op_params: dict, records: dict[str, dict], table: CosSinTable) -> None:
    mode = int(op_params["mode"])
    if mode & ROPE_TYPE_MROPE or mode == ROPE_TYPE_VISION:
        raise ValueError(f"multi-dimensional RoPE mode {mode} is not supported by this 1D position table")
    if not mode & ROPE_TYPE_NEOX:
        raise ValueError(f"expected GPT-NeoX RoPE mode, got mode={mode}")
    if "src2" in records:
        raise ValueError("frequency-factor src2 is present; the static table was exported without frequency factors")

    table_params = table.manifest["rope_params"]
    exact_fields = ("n_dims", "n_ctx_orig")
    float_fields = (
        "freq_base", "freq_scale", "ext_factor", "attn_factor", "beta_fast", "beta_slow"
    )
    for field in exact_fields:
        if int(op_params[field]) != int(table_params[field]):
            raise ValueError(
                f"RoPE/table {field} mismatch: op={op_params[field]} table={table_params[field]}"
            )
    for field in float_fields:
        if float(op_params[field]) != float(table_params[field]):
            raise ValueError(
                f"RoPE/table {field} mismatch: op={op_params[field]} table={table_params[field]}"
            )

    channels = table.shape[1]
    if channels != int(op_params["n_dims"]) // 2:
        raise ValueError(f"table channels={channels} do not match n_dims={op_params['n_dims']}")


def validate_shapes(dst: TensorData, src0: TensorData, positions: TensorData, n_dims: int) -> None:
    if dst.dtype not in ("f16", "f32") or src0.dtype != dst.dtype:
        raise ValueError(f"ROPE requires matching F16/F32 dst/src0, got {dst.dtype}/{src0.dtype}")
    if positions.dtype != "i32":
        raise ValueError(f"ROPE positions must be I32, got {positions.dtype}")
    if dst.ne != src0.ne:
        raise ValueError(f"dst/src0 shape mismatch: dst={dst.ne} src0={src0.ne}")
    if n_dims <= 0 or n_dims > src0.ne[0] or n_dims % 2 != 0:
        raise ValueError(f"invalid n_dims={n_dims} for src0 shape {src0.ne}")
    expected_positions = src0.ne[2] * src0.ne[3]
    if positions.ne[0] != expected_positions:
        raise ValueError(
            f"position count mismatch: positions={positions.ne[0]} tokens={expected_positions}"
        )


def validate_rope(
    dst: TensorData,
    src0: TensorData,
    positions: TensorData,
    table: CosSinTable,
    n_dims: int,
    result_rounding: str,
    atol: float,
    rtol: float,
    max_mismatches: int,
) -> tuple[int, float, float, list[str], set[int]]:
    mismatches = 0
    max_abs = 0.0
    max_rel = 0.0
    details: list[str] = []
    used_positions: set[int] = set()

    for i3 in range(src0.ne[3]):
        for i2 in range(src0.ne[2]):
            token_index = i3 * src0.ne[2] + i2
            position = int(positions.scalar(token_index, 0, 0, 0))
            used_positions.add(position)
            for i1 in range(src0.ne[1]):
                for channel_idx in range(n_dims // 2):
                    cos_value, sin_value = table.pair(position, channel_idx)
                    x0 = float(src0.scalar(channel_idx, i1, i2, i3))
                    x1 = float(src0.scalar(channel_idx + n_dims // 2, i1, i2, i3))
                    expected0 = apply_result_rounding(x0 * cos_value - x1 * sin_value, result_rounding)
                    expected1 = apply_result_rounding(x0 * sin_value + x1 * cos_value, result_rounding)
                    for i0, expected in (
                        (channel_idx, expected0),
                        (channel_idx + n_dims // 2, expected1),
                    ):
                        actual = float(dst.scalar(i0, i1, i2, i3))
                        absolute = abs(actual - expected)
                        relative = absolute / max(abs(actual), abs(expected), 1.0e-30)
                        max_abs = max(max_abs, absolute)
                        max_rel = max(max_rel, relative)
                        if absolute > atol + rtol * abs(expected):
                            mismatches += 1
                            if len(details) < max_mismatches:
                                details.append(
                                    f"[{i0},{i1},{i2},{i3}] position={position} "
                                    f"channel_idx={channel_idx} expected={expected:.9g} "
                                    f"actual={actual:.9g} abs={absolute:.9g} rel={relative:.9g}"
                                )

                for i0 in range(n_dims, src0.ne[0]):
                    expected = apply_result_rounding(float(src0.scalar(i0, i1, i2, i3)), result_rounding)
                    actual = float(dst.scalar(i0, i1, i2, i3))
                    absolute = abs(actual - expected)
                    relative = absolute / max(abs(actual), abs(expected), 1.0e-30)
                    max_abs = max(max_abs, absolute)
                    max_rel = max(max_rel, relative)
                    if absolute > atol + rtol * abs(expected):
                        mismatches += 1
                        if len(details) < max_mismatches:
                            details.append(
                                f"[{i0},{i1},{i2},{i3}] unrotated expected={expected:.9g} "
                                f"actual={actual:.9g} abs={absolute:.9g} rel={relative:.9g}"
                            )

    return mismatches, max_abs, max_rel, details, used_positions


def main() -> int:
    args = parse_args()
    result_path = args.result.resolve()
    manifest_path = (
        args.manifest.resolve() if args.manifest else result_path.parent / "manifest.json"
    )
    cos_sin_manifest_path = (
        args.cos_sin_manifest.resolve()
        if args.cos_sin_manifest
        else Path(__file__).resolve().with_name("rope-cos-sin-manifest.json")
    )

    manifest = load_json(manifest_path, "tensor export manifest")
    records = records_for_result(result_path, manifest_path, manifest)
    dst = load_tensor(manifest_path, records["dst"])
    src0 = load_tensor(manifest_path, records["src0"])
    positions = load_tensor(manifest_path, records["src1"])
    op_params = records["dst"].get("op_params")
    if not isinstance(op_params, dict):
        raise ValueError("ROPE dst manifest record has no op_params")
    table = load_cos_sin_table(cos_sin_manifest_path)
    require_compatible_table(op_params, records, table)
    n_dims = int(op_params["n_dims"])
    validate_shapes(dst, src0, positions, n_dims)
    result_rounding = detect_result_rounding(manifest_path, args.result_rounding)

    mismatches, max_abs, max_rel, details, used_positions = validate_rope(
        dst, src0, positions, table, n_dims, result_rounding, args.atol, args.rtol, args.max_mismatches
    )
    elements = 1
    for dim in dst.ne:
        elements *= dim
    print(f"result={dst.path}")
    print(f"input={src0.path}")
    print(f"positions={positions.path}")
    print(f"cos_sin_table={table.path}")
    print(f"shape={dst.ne} n_dims={n_dims} mode={op_params['mode']}")
    print(f"result_rounding={result_rounding}")
    print(f"position_range={min(used_positions)}:{max(used_positions) + 1}")
    print(f"elements={elements} mismatches={mismatches}")
    print(f"max_abs_error={max_abs:.9g} max_rel_error={max_rel:.9g}")
    for detail in details:
        print(f"mismatch: {detail}")

    if mismatches:
        print("validation=failed")
        return 1
    print("validation=passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
