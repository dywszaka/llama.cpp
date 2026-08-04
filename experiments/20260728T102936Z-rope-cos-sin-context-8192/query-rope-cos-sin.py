#!/usr/bin/env python3
"""Query an exported RoPE cos/sin table by position and channel_idx ranges."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def parse_range(text: str, limit: int, label: str) -> tuple[int, int]:
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else limit
    else:
        start = int(text)
        end = start + 1
    if start < 0 or end < start or end > limit:
        raise ValueError(f"invalid {label} range {text!r}; expected 0 <= start <= end <= {limit}")
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--position", default=":", help="single index or start:end (end exclusive)")
    parser.add_argument("--channel-idx", default=":", help="single index or start:end (end exclusive)")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context_size, channels, components = (int(value) for value in manifest["shape"])
    if components != 2 or manifest["component_order"] != ["cos", "sin"]:
        raise ValueError("unsupported component layout")
    position_start, position_end = parse_range(args.position, context_size, "position")
    channel_start, channel_end = parse_range(args.channel_idx, channels, "channel_idx")

    data_path = manifest_path.parent / manifest["data_file"]
    row_stride = channels * 2 * 4
    records: list[dict[str, int | float]] = []
    with data_path.open("rb") as data:
        for position in range(position_start, position_end):
            for channel_idx in range(channel_start, channel_end):
                offset = position * row_stride + channel_idx * 2 * 4
                data.seek(offset)
                raw = data.read(8)
                if len(raw) != 8:
                    raise ValueError(f"short read at byte offset {offset}")
                cos_value, sin_value = struct.unpack("<ff", raw)
                records.append({
                    "position": position,
                    "channel_idx": channel_idx,
                    "cos": cos_value,
                    "sin": sin_value,
                })

    if args.format == "json":
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        print("position,channel_idx,cos,sin")
        for record in records:
            print(
                f"{record['position']},{record['channel_idx']},"
                f"{record['cos']:.9g},{record['sin']:.9g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
