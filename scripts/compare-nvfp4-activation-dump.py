#!/usr/bin/env python3
"""Read and optionally compare a native NVFP4 activation quantization dump."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import tempfile
from pathlib import Path


NVFP4_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
QK_NVFP4 = 16
BLOCK_NVFP4_SIZE = 9


def bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def f32_to_bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x007FFFFF

    if exp == 0xFF:
        if mant != 0:
            out = (sign << 15) | 0x7F80 | ((mant >> 16) & 0x3F)
            return out | 0x0040 if (out & 0x7F) == 0 else out
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

    guard = (mant >> 15) & 1
    round_bit = (mant >> 14) & 1
    sticky = 1 if (mant & 0x3FFF) != 0 else 0
    bf16_mant = (mant >> 16) & 0x7F
    if guard == 1 and (round_bit == 1 or sticky == 1 or (bf16_mant & 1) == 1):
        bf16_mant += 1

    if bf16_mant > 0x7F:
        bf16_mant = 0
        exp += 1
        if exp > 0xFE:
            return (sign << 15) | 0x7F80

    if exp < 1:
        return sign << 15

    return (sign << 15) | ((exp & 0xFF) << 7) | (bf16_mant & 0x7F)


def f32_to_hi16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return bits >> 16


def e4m3_to_f32(byte: int) -> float:
    sign = -1.0 if (byte & 0x80) else 1.0
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    if exponent == 0:
        if mantissa == 0:
            return math.copysign(0.0, sign)
        return sign * (2.0 ** -6) * (mantissa / 8.0)
    if exponent == 0x0F:
        return math.nan if mantissa == 0x07 else sign * 448.0
    return sign * (2.0 ** (exponent - 7)) * (1.0 + mantissa / 8.0)


def e4m3_to_f32_half(byte: int) -> float:
    return e4m3_to_f32(byte) * 0.5


def best_index_e4m3(value: float) -> int:
    if not math.isfinite(value):
        return 0
    best_i = 0
    best_err = math.inf
    for i in range(256):
        candidate = e4m3_to_f32(i)
        if not math.isfinite(candidate):
            continue
        err = abs(candidate - value)
        if err < best_err:
            best_i = i
            best_err = err
    return best_i


def best_index_nvfp4(value: float) -> int:
    return min(range(16), key=lambda i: abs(NVFP4_VALUES[i] - value))


def quantize_row_nvfp4(values: list[float], global_scale: float) -> bytes:
    assert len(values) == QK_NVFP4
    vmax = max(abs(v) for v in values)
    scale_q = best_index_e4m3(global_scale * (vmax / 6.0))
    scale_f = e4m3_to_f32_half(scale_q)
    inv_scale = global_scale / scale_f if global_scale != 0.0 and scale_f != 0.0 else 0.0
    packed = bytearray([scale_q])
    for i in range(0, QK_NVFP4, 2):
        q0 = best_index_nvfp4(values[i] * inv_scale)
        q1 = best_index_nvfp4(values[i + 1] * inv_scale)
        packed.append(q0 | (q1 << 4))
    return bytes(packed)


def row_global_scale(meta: dict, row: int) -> float:
    if meta.get("scale_mode") == "dynamic_per_row":
        amax = float(meta["dynamic_amax_rows"][row])
        return (6.0 * 224.0) / amax if amax > 0.0 and math.isfinite(amax) else 0.0
    return float(meta["global_scale"])


def load_dump(path: Path) -> tuple[dict, list[float], bytes]:
    meta = json.loads((path / "metadata.json").read_text())
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    before_path = path / meta.get("before_file", "activation-before-f32-hi16.bin")
    if not before_path.exists() and "before_file" not in meta:
        before_path = path / "activation-before-bf16.bin"
    after_path = path / meta.get("after_file", "activation-after-nvfp4.bin")
    before_raw = before_path.read_bytes()
    after_raw = after_path.read_bytes()

    expected_before = rows * cols * 2
    expected_after = rows * (cols // QK_NVFP4) * BLOCK_NVFP4_SIZE
    if len(before_raw) != expected_before:
        raise ValueError(f"{before_path} has {len(before_raw)} bytes, expected {expected_before}")
    if len(after_raw) != expected_after:
        raise ValueError(f"{after_path} has {len(after_raw)} bytes, expected {expected_after}")

    before_bits = struct.unpack("<" + "H" * (rows * cols), before_raw)
    before = [bf16_to_f32(x) for x in before_bits]
    return meta, before, after_raw


def dequant_value(after_raw: bytes, rows: int, cols: int, row: int, col: int, global_scale: float) -> tuple[float, int, int]:
    blocks_per_row = cols // QK_NVFP4
    block = col // QK_NVFP4
    in_block = col % QK_NVFP4
    off = (row * blocks_per_row + block) * BLOCK_NVFP4_SIZE
    scale_byte = after_raw[off]
    packed = after_raw[off + 1 + in_block // 2]
    q = (packed & 0x0F) if (in_block % 2) == 0 else (packed >> 4)
    scale = e4m3_to_f32_half(scale_byte)
    out_scale = scale / global_scale if global_scale != 0.0 else 0.0
    return NVFP4_VALUES[q] * out_scale, q, scale_byte


def compare_dump(path: Path, samples: int, csv_path: Path | None) -> int:
    meta, before, after_raw = load_dump(path)
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    if cols % QK_NVFP4 != 0:
        raise ValueError(f"cols must be divisible by {QK_NVFP4}, got {cols}")

    count = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    max_item = None
    csv_file = None
    if csv_path is not None:
        csv_file = csv_path.open("w", encoding="utf-8")
        csv_file.write("row,col,before_bf16,after_nvfp4_dequant,abs_diff,q,scale_byte,global_scale\n")

    try:
        for row in range(rows):
            gs = row_global_scale(meta, row)
            for col in range(cols):
                before_value = before[row * cols + col]
                after_value, q, scale_byte = dequant_value(after_raw, rows, cols, row, col, gs)
                diff = abs(before_value - after_value)
                count += 1
                sum_abs += diff
                sum_sq += diff * diff
                if diff > max_abs:
                    max_abs = diff
                    max_item = (row, col, before_value, after_value, q, scale_byte, gs)
                if csv_file is not None:
                    csv_file.write(
                        f"{row},{col},{before_value:.9g},{after_value:.9g},{diff:.9g},{q},{scale_byte},{gs:.9g}\n"
                    )
                elif count <= samples:
                    print(
                        f"pair row={row} col={col} before_bf16={before_value:.9g} "
                        f"after_nvfp4_dequant={after_value:.9g} abs_diff={diff:.9g} "
                        f"q={q} scale_byte={scale_byte} global_scale={gs:.9g}"
                    )
    finally:
        if csv_file is not None:
            csv_file.close()

    mean_abs = sum_abs / count if count else 0.0
    rmse = math.sqrt(sum_sq / count) if count else 0.0
    print(
        f"summary tensor={meta.get('tensor', '')} dst={meta.get('dst', '')} "
        f"rows={rows} cols={cols} pairs={count} mean_abs={mean_abs:.9g} "
        f"rmse={rmse:.9g} max_abs={max_abs:.9g}"
    )
    if max_item is not None:
        row, col, before_value, after_value, q, scale_byte, gs = max_item
        print(
            f"max row={row} col={col} before_bf16={before_value:.9g} "
            f"after_nvfp4_dequant={after_value:.9g} q={q} scale_byte={scale_byte} "
            f"global_scale={gs:.9g}"
        )
    return 0


def read_dump(path: Path, samples: int) -> int:
    meta, before, after_raw = load_dump(path)
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    print(
        f"dump tensor={meta.get('tensor', '')} dst={meta.get('dst', '')} "
        f"rows={rows} cols={cols} before_dtype={meta.get('before_dtype', '')} "
        f"after_dtype={meta.get('after_dtype', '')}"
    )
    for row in range(rows):
        gs = row_global_scale(meta, row)
        for col in range(cols):
            if row * cols + col >= samples:
                return 0
            after_value, q, scale_byte = dequant_value(after_raw, rows, cols, row, col, gs)
            print(
                f"value row={row} col={col} before={before[row * cols + col]:.9g} "
                f"after_nvfp4_dequant={after_value:.9g} q={q} scale_byte={scale_byte} "
                f"global_scale={gs:.9g}"
            )
    return 0


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        dump_dir = Path(tmp)
        rows = 2
        cols = 16
        global_scale = 16.0
        values = [
            -0.75, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5,
            0.75, 1.0, 1.25, 1.5, -1.0, -1.25, -1.5, 1.75,
            0.03125, -0.03125, 0.0625, -0.0625, 0.09375, -0.09375, 0.1875, -0.1875,
            0.375, -0.375, 0.625, -0.625, 0.875, -0.875, 1.125, -1.125,
        ]
        before_bits = [f32_to_hi16_bits(v) for v in values]
        before_values = [bf16_to_f32(v) for v in before_bits]
        after = bytearray()
        for row in range(rows):
            after.extend(quantize_row_nvfp4(before_values[row * cols:(row + 1) * cols], global_scale))
        (dump_dir / "activation-before-f32-hi16.bin").write_bytes(struct.pack("<" + "H" * len(before_bits), *before_bits))
        (dump_dir / "activation-after-nvfp4.bin").write_bytes(bytes(after))
        (dump_dir / "metadata.json").write_text(json.dumps({
            "format": "self-test",
            "tensor": "self-test-src1",
            "dst": "self-test-dst",
            "before_file": "activation-before-f32-hi16.bin",
            "before_dtype": "bf16_trunc_bits_le",
            "after_file": "activation-after-nvfp4.bin",
            "after_dtype": "block_nvfp4",
            "rows": rows,
            "cols": cols,
            "qk_nvfp4": QK_NVFP4,
            "block_nvfp4_size": BLOCK_NVFP4_SIZE,
            "global_scale": global_scale,
            "scale_mode": "bound_static",
        }))
        return compare_dump(dump_dir, samples=4, csv_path=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", nargs="?", type=Path, help="Directory containing metadata.json and dump binaries")
    parser.add_argument("--read-only", action="store_true", help="Read both files and print sampled decoded values without computing diff summary")
    parser.add_argument("--samples", type=int, default=32, help="Number of paired values to print before the summary")
    parser.add_argument("--csv", type=Path, help="Write all paired values to CSV instead of printing samples")
    parser.add_argument("--self-test", action="store_true", help="Run a small synthetic parser/dequantization self-test")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.dump_dir is None:
        parser.error("dump_dir is required unless --self-test is used")
    if args.read_only:
        return read_dump(args.dump_dir, args.samples)
    return compare_dump(args.dump_dir, args.samples, args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
