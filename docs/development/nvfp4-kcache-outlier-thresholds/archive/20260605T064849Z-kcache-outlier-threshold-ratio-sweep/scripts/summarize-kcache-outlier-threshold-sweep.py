#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


OUTLIER_RE = re.compile(
    r"ggml_cuda_nvfp4_kcache_outlier_extract: "
    r"target=cache_k_l(?P<layer>\d+).*? "
    r"rows=(?P<rows>\d+) threshold=(?P<threshold>[0-9.]+) "
    r".*?total_outliers=(?P<total_outliers>\d+) "
    r"max_row_outliers=(?P<max_row_outliers>\d+) "
    r"overflow_rows=(?P<overflow_rows>\d+)"
)
FINAL_RE = re.compile(r"Final estimate: PPL = ([0-9.eE+-]+) \+/- ([0-9.eE+-]+)")
THRESHOLD_RE = re.compile(r"threshold_(\d+(?:\.\d+)?)\.raw\.log$")


def parse_log(path, values_per_row, batch_rows):
    threshold_match = THRESHOLD_RE.search(path.name)
    if not threshold_match:
        raise ValueError(f"cannot parse threshold from {path}")
    threshold = float(threshold_match.group(1))
    text = path.read_text(errors="replace")
    final = FINAL_RE.search(text)

    records_by_layer = {}
    for m in OUTLIER_RE.finditer(text):
        layer = int(m.group("layer"))
        rows = int(m.group("rows"))
        outliers = int(m.group("total_outliers"))
        if rows != batch_rows:
            continue
        records_by_layer.setdefault(layer, []).append(outliers)

    rows = []
    for layer, records in sorted(records_by_layer.items()):
        max_outliers = max(records) if records else 0
        total_values = batch_rows * values_per_row
        ratio = max_outliers / total_values if total_values else 0.0
        rows.append({
            "threshold": threshold,
            "layer": layer,
            "records": len(records),
            "max_batch_outliers": max_outliers,
            "target_values": total_values,
            "max_batch_outlier_ratio": ratio,
            "total_outliers": sum(records),
            "nonzero_batches": sum(1 for x in records if x > 0),
            "ppl": float(final.group(1)) if final else "",
            "ppl_err": float(final.group(2)) if final else "",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-ratio", type=float, default=1e-4)
    parser.add_argument("--values-per-row", type=int, default=1024)
    parser.add_argument("--batch-rows", type=int, default=512)
    args = parser.parse_args()

    all_rows = []
    for path in sorted(args.runs_dir.glob("threshold_*.raw.log")):
        all_rows.extend(parse_log(path, args.values_per_row, args.batch_rows))
    if not all_rows:
        raise SystemExit(f"no threshold logs parsed under {args.runs_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "threshold", "layer", "records", "max_batch_outliers", "target_values",
        "max_batch_outlier_ratio", "total_outliers", "nonzero_batches", "ppl", "ppl_err",
    ]
    per_layer_path = args.output_dir / "threshold-layer-batch-ratio.csv"
    with per_layer_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(all_rows)

    by_layer = {}
    for row in all_rows:
        by_layer.setdefault(int(row["layer"]), []).append(row)

    selected = []
    for layer, rows in sorted(by_layer.items()):
        def score(row):
            ratio = float(row["max_batch_outlier_ratio"])
            return (abs(math.log((ratio + 1e-12) / args.target_ratio)), abs(ratio - args.target_ratio))
        best = min(rows, key=score)
        selected.append({
            **best,
            "target_ratio": args.target_ratio,
            "ratio_delta": float(best["max_batch_outlier_ratio"]) - args.target_ratio,
        })

    selected_fields = fields + ["target_ratio", "ratio_delta"]
    selected_path = args.output_dir / "selected-thresholds.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, selected_fields)
        writer.writeheader()
        writer.writerows(selected)


if __name__ == "__main__":
    main()
