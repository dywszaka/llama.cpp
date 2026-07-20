#!/usr/bin/env python3

"""Compare two raw tensor-export .bin files using their manifest metadata."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class TensorInput:
    path: Path
    manifest_path: Path
    record: dict

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.record["ne"])

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(self.record["nb"])

    @property
    def dtype(self) -> str:
        return self.record["dtype"]

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)


class BFloat16:
    size = 2

    @staticmethod
    def unpack_from(data: mmap.mmap, offset: int) -> float:
        bits = struct.unpack_from("<H", data, offset)[0] << 16
        return struct.unpack("<f", struct.pack("<I", bits))[0]


SCALAR_DTYPES: dict[str, tuple[int, str | None]] = {
    "f16": (2, "<e"),
    "f32": (4, "<f"),
    "f64": (8, "<d"),
    "bf16": (BFloat16.size, None),
    "i8": (1, "<b"),
    "i16": (2, "<h"),
    "i32": (4, "<i"),
    "i64": (8, "<q"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the shape and values of two headerless tensor-export bin files. "
            "By default, manifest.json is loaded from each bin file's directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bin_a", type=Path, help="first exported .bin file")
    parser.add_argument("bin_b", type=Path, help="second exported .bin file")
    parser.add_argument(
        "--manifest-a",
        type=Path,
        help="manifest for bin_a; defaults to bin_a's sibling manifest.json",
    )
    parser.add_argument(
        "--manifest-b",
        type=Path,
        help="manifest for bin_b; defaults to bin_b's sibling manifest.json",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="absolute tolerance for decoded numeric values",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="relative tolerance: abs(a-b) <= atol + rtol * abs(b)",
    )
    parser.add_argument(
        "--equal-nan",
        action="store_true",
        help="treat NaNs at the same logical position as equal",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=10,
        help="maximum number of mismatched elements to print",
    )
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if args.max_mismatches < 0:
        parser.error("--max-mismatches must be non-negative")
    return args


def load_tensor(bin_path: Path, manifest_path: Path | None) -> TensorInput:
    path = bin_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"bin file does not exist: {path}")

    manifest = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else path.parent / "manifest.json"
    )
    if not manifest.is_file():
        raise ValueError(
            f"manifest does not exist: {manifest}; shape is not stored in the raw bin, "
            "so pass --manifest-a/--manifest-b when it is elsewhere"
        )

    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read manifest {manifest}: {exc}") from exc

    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError(f"manifest has no records array: {manifest}")

    matches = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            continue
        record_path = (manifest.parent / record["path"]).resolve()
        if record_path == path:
            matches.append(record)

    if not matches:
        raise ValueError(f"bin file is not referenced by manifest: {path} ({manifest})")
    if len(matches) > 1:
        raise ValueError(f"bin file is referenced by multiple manifest records: {path}")

    record = matches[0]
    validate_record(path, manifest, record)
    return TensorInput(path=path, manifest_path=manifest, record=record)


def validate_record(path: Path, manifest_path: Path, record: dict) -> None:
    for field in ("name", "dtype", "ne", "nb", "path", "byte_size"):
        if field not in record:
            raise ValueError(f"record for {path} is missing '{field}' in {manifest_path}")

    if not isinstance(record["dtype"], str):
        raise ValueError(f"record dtype must be a string: {path}")
    if (
        not isinstance(record["ne"], list)
        or not isinstance(record["nb"], list)
        or len(record["ne"]) != 4
        or len(record["nb"]) != 4
    ):
        raise ValueError(f"record ne and nb must be four-element arrays: {path}")
    if any(not isinstance(value, int) or value <= 0 for value in record["ne"]):
        raise ValueError(f"record ne must contain positive integers: {path}")
    if any(not isinstance(value, int) or value < 0 for value in record["nb"]):
        raise ValueError(f"record nb must contain non-negative integers: {path}")
    if not isinstance(record["byte_size"], int) or record["byte_size"] < 0:
        raise ValueError(f"record byte_size must be a non-negative integer: {path}")

    actual_size = path.stat().st_size
    if actual_size != record["byte_size"]:
        raise ValueError(
            f"raw byte size does not match manifest for {path}: "
            f"file={actual_size}, manifest={record['byte_size']}"
        )

    dtype = record["dtype"]
    if dtype in SCALAR_DTYPES:
        element_size = SCALAR_DTYPES[dtype][0]
        max_offset = sum(
            (extent - 1) * stride
            for extent, stride in zip(record["ne"], record["nb"])
        )
        if max_offset + element_size > actual_size:
            raise ValueError(
                f"shape/stride metadata addresses beyond the raw file for {path}: "
                f"last_end={max_offset + element_size}, file={actual_size}"
            )


def files_equal(path_a: Path, path_b: Path) -> bool:
    if path_a.stat().st_size != path_b.stat().st_size:
        return False
    with path_a.open("rb") as file_a, path_b.open("rb") as file_b:
        while True:
            chunk_a = file_a.read(CHUNK_SIZE)
            chunk_b = file_b.read(CHUNK_SIZE)
            if chunk_a != chunk_b:
                return False
            if not chunk_a:
                return True


def logical_indices(shape: Sequence[int]) -> Iterator[tuple[int, tuple[int, ...]]]:
    for flat_index in range(math.prod(shape)):
        remainder = flat_index
        coordinates = []
        for extent in shape:
            coordinates.append(remainder % extent)
            remainder //= extent
        yield flat_index, tuple(coordinates)


def byte_offset(coordinates: Sequence[int], strides: Sequence[int]) -> int:
    return sum(index * stride for index, stride in zip(coordinates, strides))


def read_scalar(data: mmap.mmap, offset: int, dtype: str) -> int | float:
    _, format_string = SCALAR_DTYPES[dtype]
    if dtype == "bf16":
        return BFloat16.unpack_from(data, offset)
    assert format_string is not None
    return struct.unpack_from(format_string, data, offset)[0]


def values_equal(
    a: int | float,
    b: int | float,
    atol: float,
    rtol: float,
    equal_nan: bool,
) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        a_float = float(a)
        b_float = float(b)
        if math.isnan(a_float) or math.isnan(b_float):
            return equal_nan and math.isnan(a_float) and math.isnan(b_float)
        if a_float == b_float:
            return True
        if math.isinf(a_float) or math.isinf(b_float):
            return False
        return abs(a_float - b_float) <= atol + rtol * abs(b_float)
    return a == b


def compare_decoded_values(
    tensor_a: TensorInput,
    tensor_b: TensorInput,
    atol: float,
    rtol: float,
    equal_nan: bool,
    max_mismatches: int,
) -> tuple[int, float, float, list[str]]:
    mismatch_count = 0
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mismatch_lines: list[str] = []

    with tensor_a.path.open("rb") as file_a, tensor_b.path.open("rb") as file_b:
        data_a = mmap.mmap(file_a.fileno(), 0, access=mmap.ACCESS_READ)
        data_b = mmap.mmap(file_b.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for flat_index, coordinates in logical_indices(tensor_a.shape):
                value_a = read_scalar(
                    data_a,
                    byte_offset(coordinates, tensor_a.strides),
                    tensor_a.dtype,
                )
                value_b = read_scalar(
                    data_b,
                    byte_offset(coordinates, tensor_b.strides),
                    tensor_b.dtype,
                )

                if isinstance(value_a, float) or isinstance(value_b, float):
                    a_float = float(value_a)
                    b_float = float(value_b)
                    if math.isnan(a_float) or math.isnan(b_float):
                        abs_diff = math.nan
                        rel_diff = math.nan
                    else:
                        abs_diff = abs(a_float - b_float)
                        rel_diff = (
                            abs_diff / abs(b_float)
                            if b_float != 0
                            else (0.0 if abs_diff == 0 else math.inf)
                        )
                        max_abs_diff = max(max_abs_diff, abs_diff)
                        max_rel_diff = max(max_rel_diff, rel_diff)
                else:
                    abs_diff = float(abs(value_a - value_b))
                    rel_diff = (
                        abs_diff / abs(value_b)
                        if value_b != 0
                        else (0.0 if abs_diff == 0 else math.inf)
                    )
                    max_abs_diff = max(max_abs_diff, abs_diff)
                    max_rel_diff = max(max_rel_diff, rel_diff)

                if values_equal(value_a, value_b, atol, rtol, equal_nan):
                    continue

                mismatch_count += 1
                if len(mismatch_lines) < max_mismatches:
                    mismatch_lines.append(
                        f"  index={flat_index} coords={list(coordinates)} "
                        f"a={value_a!r} b={value_b!r} abs_diff={abs_diff!r}"
                    )
        finally:
            data_a.close()
            data_b.close()

    return mismatch_count, max_abs_diff, max_rel_diff, mismatch_lines


def print_tensor(label: str, tensor: TensorInput) -> None:
    record = tensor.record
    extras = []
    for field in ("kind", "op", "role", "node_index"):
        if field in record:
            extras.append(f"{field}={record[field]}")
    print(f"Tensor {label}:")
    print(f"  bin: {tensor.path}")
    print(f"  manifest: {tensor.manifest_path}")
    print(f"  name: {record['name']}")
    if extras:
        print(f"  metadata: {', '.join(extras)}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  shape (ggml ne): {list(tensor.shape)}")
    print(f"  strides (ggml nb): {list(tensor.strides)}")
    print(f"  elements: {tensor.element_count}")
    print(f"  bytes: {record['byte_size']}")


def main() -> int:
    args = parse_args()
    try:
        tensor_a = load_tensor(args.bin_a, args.manifest_a)
        tensor_b = load_tensor(args.bin_b, args.manifest_b)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_tensor("A", tensor_a)
    print_tensor("B", tensor_b)
    print("Comparison:")

    shape_matches = tensor_a.shape == tensor_b.shape
    dtype_matches = tensor_a.dtype == tensor_b.dtype
    strides_match = tensor_a.strides == tensor_b.strides
    names_match = tensor_a.record["name"] == tensor_b.record["name"]
    print(f"  shape: {'MATCH' if shape_matches else 'DIFFERENT'}")
    print(f"  dtype: {'MATCH' if dtype_matches else 'DIFFERENT'}")
    print(f"  strides: {'MATCH' if strides_match else 'DIFFERENT'}")
    print(f"  name: {'MATCH' if names_match else 'DIFFERENT'}")

    if not shape_matches or not dtype_matches:
        print("  values: NOT COMPARED (shape or dtype differs)")
        print("Result: DIFFERENT")
        return 1

    bytewise_equal = files_equal(tensor_a.path, tensor_b.path)
    print(f"  raw bytes: {'MATCH' if bytewise_equal else 'DIFFERENT'}")
    if bytewise_equal and strides_match:
        print(f"  values: MATCH ({tensor_a.element_count} logical elements)")
        print("Result: MATCH")
        return 0

    if tensor_a.dtype not in SCALAR_DTYPES:
        if bytewise_equal:
            print(
                f"  values: NOT COMPARED (dtype '{tensor_a.dtype}' does not have a "
                "scalar decoder and strides differ)"
            )
            print("Result: DIFFERENT")
            return 1
        print(
            f"  values: DIFFERENT (raw bytes differ and dtype '{tensor_a.dtype}' "
            "does not have a scalar decoder)"
        )
        print("Result: DIFFERENT")
        return 1

    mismatch_count, max_abs_diff, max_rel_diff, mismatch_lines = compare_decoded_values(
        tensor_a,
        tensor_b,
        args.atol,
        args.rtol,
        args.equal_nan,
        args.max_mismatches,
    )
    if mismatch_count == 0:
        print(
            f"  values: MATCH ({tensor_a.element_count} logical elements, "
            f"atol={args.atol:g}, rtol={args.rtol:g})"
        )
        print(f"  max_abs_diff: {max_abs_diff!r}")
        print(f"  max_rel_diff: {max_rel_diff!r}")
        print("Result: MATCH")
        return 0

    print(
        f"  values: DIFFERENT ({mismatch_count}/{tensor_a.element_count} logical elements, "
        f"atol={args.atol:g}, rtol={args.rtol:g})"
    )
    print(f"  max_abs_diff: {max_abs_diff!r}")
    print(f"  max_rel_diff: {max_rel_diff!r}")
    if mismatch_lines:
        print("  first mismatches:")
        print("\n".join(mismatch_lines))
    print("Result: DIFFERENT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
