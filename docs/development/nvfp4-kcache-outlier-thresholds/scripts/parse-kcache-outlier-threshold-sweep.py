#!/usr/bin/env python3
"""Parse NVFP4 K-cache outlier threshold sweep raw logs.

The parser accepts logs named `threshold_<N>.raw.log`. Older logs do not include
layer IDs in each outlier line, so layer IDs are inferred from record order:
layer 0..N-1 repeat for warmup and each PPL chunk.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


OUTLIER_RE = re.compile(
    r"ggml_cuda_nvfp4_kcache_outlier_extract: "
    r"(?:target=(?P<target>.+?) )?"
    r"rows=(?P<rows>\d+) threshold=(?P<threshold>[0-9.]+) "
    r"stored_max=(?P<stored_max>\d+) compact_capacity=(?P<compact_capacity>\d+) "
    r"(?:compact_used=(?P<compact_used>\d+) )?"
    r"(?:compact_overflow=(?P<compact_overflow>\d+) )?"
    r"total_outliers=(?P<total_outliers>\d+) "
    r"max_row_outliers=(?P<max_row_outliers>\d+) "
    r"overflow_rows=(?P<overflow_rows>\d+)"
)
FINAL_RE = re.compile(r"Final estimate: PPL = ([0-9.eE+-]+) \+/- ([0-9.eE+-]+)")
THRESHOLD_NAME_RE = re.compile(r"threshold_(\d+(?:\.\d+)?)\.raw\.log$")
TARGET_LAYER_RE = re.compile(r"cache_k_l(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--head-dim", type=int, default=1024,
                        help="K row width used to convert counts to density")
    return parser.parse_args()


def threshold_from_path(path: Path) -> int:
    match = THRESHOLD_NAME_RE.search(path.name)
    if not match:
        raise SystemExit(f"cannot infer threshold from {path}")
    return int(float(match.group(1)))


def parse_log(path: Path, layers: int, head_dim: int) -> tuple[int, dict, list[dict]]:
    text = path.read_text(errors="replace")
    threshold = threshold_from_path(path)
    final = FINAL_RE.search(text)

    layer_stats = {
        layer: {
            "layer": layer,
            "rows": 0,
            "records": 0,
            "total_outliers": 0,
            "max_call_total": 0,
            "max_row_outliers": 0,
            "overflow_rows": 0,
        }
        for layer in range(layers)
    }

    records = 0
    for match in OUTLIER_RE.finditer(text):
        target = match.group("target")
        target_match = TARGET_LAYER_RE.search(target or "")
        layer = int(target_match.group(1)) if target_match else records % layers
        if layer >= layers:
            raise SystemExit(f"layer {layer} outside configured layer count {layers} in {path}")

        rows = int(match.group("rows"))
        outliers = int(match.group("total_outliers"))
        max_row = int(match.group("max_row_outliers"))
        overflow = int(match.group("overflow_rows"))

        stat = layer_stats[layer]
        stat["rows"] += rows
        stat["records"] += 1
        stat["total_outliers"] += outliers
        stat["max_call_total"] = max(stat["max_call_total"], outliers)
        stat["max_row_outliers"] = max(stat["max_row_outliers"], max_row)
        stat["overflow_rows"] += overflow
        records += 1

    layer_rows = []
    for layer in range(layers):
        stat = layer_stats[layer]
        denominator = stat["rows"] * head_dim
        density = 100.0 * stat["total_outliers"] / denominator if denominator else 0.0
        layer_rows.append({
            "layer": layer,
            "density_pct": density,
            "total_outliers": stat["total_outliers"],
            "max_call_total": stat["max_call_total"],
            "max_row_outliers": stat["max_row_outliers"],
            "overflow_rows": stat["overflow_rows"],
            "records": stat["records"],
            "rows": stat["rows"],
        })

    total_rows = sum(row["rows"] for row in layer_rows)
    total_outliers = sum(row["total_outliers"] for row in layer_rows)
    summary = {
        "threshold": threshold,
        "ppl": float(final.group(1)) if final else "",
        "ppl_err": float(final.group(2)) if final else "",
        "records": records,
        "total_outliers": total_outliers,
        "rows": total_rows,
        "density_pct": 100.0 * total_outliers / (total_rows * head_dim) if total_rows else 0.0,
        "max_layer_density_pct": max(row["density_pct"] for row in layer_rows),
        "max_call_total": max(row["max_call_total"] for row in layer_rows),
        "max_row_outliers": max(row["max_row_outliers"] for row in layer_rows),
        "overflow_rows": sum(row["overflow_rows"] for row in layer_rows),
        "layers_over_0_01pct": sum(1 for row in layer_rows if row["density_pct"] >= 0.01),
    }
    return threshold, summary, layer_rows


def write_outputs(results: list[tuple[int, dict, list[dict]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda item: item[0])

    summary_fields = [
        "threshold",
        "ppl",
        "ppl_err",
        "records",
        "total_outliers",
        "rows",
        "density_pct",
        "max_layer_density_pct",
        "max_call_total",
        "max_row_outliers",
        "overflow_rows",
        "layers_over_0_01pct",
    ]
    with (output_dir / "threshold-summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for _, summary, _ in results:
            writer.writerow(summary)

    layer_count = len(results[0][2]) if results else 0
    thresholds = [threshold for threshold, _, _ in results]
    fields = ["layer"]
    for threshold in thresholds:
        fields.extend([
            f"th{threshold}_density_pct",
            f"th{threshold}_total_outliers",
            f"th{threshold}_max_call_total",
            f"th{threshold}_max_row_outliers",
        ])

    by_threshold = {threshold: rows for threshold, _, rows in results}
    with (output_dir / "threshold-layer-density.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for layer in range(layer_count):
            row = {"layer": layer}
            for threshold in thresholds:
                stat = by_threshold[threshold][layer]
                row[f"th{threshold}_density_pct"] = stat["density_pct"]
                row[f"th{threshold}_total_outliers"] = stat["total_outliers"]
                row[f"th{threshold}_max_call_total"] = stat["max_call_total"]
                row[f"th{threshold}_max_row_outliers"] = stat["max_row_outliers"]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    paths = sorted(args.runs_dir.glob("threshold_*.raw.log"), key=threshold_from_path)
    if not paths:
        raise SystemExit(f"no threshold_*.raw.log files found in {args.runs_dir}")
    results = [parse_log(path, args.layers, args.head_dim) for path in paths]
    write_outputs(results, args.output_dir)
    print(args.output_dir / "threshold-summary.csv")
    print(args.output_dir / "threshold-layer-density.csv")


if __name__ == "__main__":
    main()
