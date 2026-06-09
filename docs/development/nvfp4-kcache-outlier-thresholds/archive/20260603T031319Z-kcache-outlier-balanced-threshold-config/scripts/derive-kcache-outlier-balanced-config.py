#!/usr/bin/env python3
"""Derive balanced NVFP4 K-cache outlier layer threshold/capacity profiles.

The input files are produced by the K-cache outlier threshold sweep experiments:

  threshold-layer-density.csv
  threshold-summary.csv

The objective is deliberately simple and reproducible: prefer profiles whose
per-layer outlier counts are close to a target count, while rejecting profiles
whose nearest global-threshold PPL is too far above the reference PPL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-density", required=True, type=Path)
    parser.add_argument("--threshold-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-count", type=float, default=200.0,
                        help="preferred total outliers per layer in the sweep")
    parser.add_argument("--max-ppl-delta", type=float, default=0.35,
                        help="allowed PPL increase versus the best scanned threshold")
    parser.add_argument("--capacity-margin", type=float, default=1.25,
                        help="capacity multiplier over selected max_call_total")
    parser.add_argument("--min-capacity", type=int, default=1)
    parser.add_argument("--max-threshold", type=int, default=384)
    parser.add_argument("--count-weight", type=float, default=1.0)
    parser.add_argument("--ppl-weight", type=float, default=8.0)
    parser.add_argument("--zero-weight", type=float, default=0.35,
                        help="extra penalty when a layer has zero selected outliers")
    return parser.parse_args()


def load_threshold_summary(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            threshold = int(float(row["threshold"]))
            result[threshold] = {
                "ppl": float(row["ppl"]),
                "ppl_err": float(row.get("ppl_err") or 0.0),
                "total_outliers": float(row.get("total_outliers") or 0.0),
                "density_pct": float(row.get("density_pct") or 0.0),
                "max_layer_density_pct": float(row.get("max_layer_density_pct") or 0.0),
            }
    if not result:
        raise SystemExit(f"no threshold summary rows found in {path}")
    return result


def available_thresholds(row: dict[str, str], max_threshold: int) -> list[int]:
    thresholds: list[int] = []
    for key in row:
        if not key.startswith("th") or not key.endswith("_total_outliers"):
            continue
        threshold = int(key[2:key.index("_")])
        if threshold <= max_threshold:
            thresholds.append(threshold)
    return sorted(set(thresholds))


def load_layers(path: Path, max_threshold: int) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            thresholds = available_thresholds(row, max_threshold)
            options = []
            for threshold in thresholds:
                prefix = f"th{threshold}_"
                options.append({
                    "threshold": threshold,
                    "density_pct": float(row[f"{prefix}density_pct"]),
                    "total_outliers": int(float(row[f"{prefix}total_outliers"])),
                    "max_call_total": int(float(row[f"{prefix}max_call_total"])),
                    "max_row_outliers": int(float(row[f"{prefix}max_row_outliers"])),
                })
            layers.append({"layer": int(row["layer"]), "options": options})
    if not layers:
        raise SystemExit(f"no layer rows found in {path}")
    return layers


def option_score(
        option: dict[str, Any],
        target_count: float,
        summary: dict[int, dict[str, float]],
        best_ppl: float,
        max_ppl_delta: float,
        count_weight: float,
        ppl_weight: float,
        zero_weight: float) -> float:
    count = float(option["total_outliers"])
    threshold = int(option["threshold"])
    ppl = summary.get(threshold, {"ppl": best_ppl + max_ppl_delta})["ppl"]
    ppl_delta = max(0.0, ppl - best_ppl)
    if ppl_delta > max_ppl_delta:
        return math.inf

    # Log distance treats 20 vs 200 similarly to 200 vs 2000.
    count_term = abs(math.log1p(count) - math.log1p(target_count))
    ppl_term = ppl_delta / max(max_ppl_delta, 1e-9)
    zero_term = zero_weight if count == 0 and target_count > 0 else 0.0
    return count_weight * count_term + ppl_weight * ppl_term + zero_term


def choose_profile(args: argparse.Namespace) -> dict[str, Any]:
    summary = load_threshold_summary(args.threshold_summary)
    layers = load_layers(args.layer_density, args.max_threshold)
    best_ppl = min(item["ppl"] for item in summary.values())

    selected = []
    for layer in layers:
        scored = []
        for option in layer["options"]:
            score = option_score(
                option,
                args.target_count,
                summary,
                best_ppl,
                args.max_ppl_delta,
                args.count_weight,
                args.ppl_weight,
                args.zero_weight,
            )
            scored.append((score, option))
        scored.sort(key=lambda item: (item[0], item[1]["threshold"]))
        if math.isinf(scored[0][0]):
            raise SystemExit(
                f"no acceptable threshold for layer {layer['layer']} with max PPL delta {args.max_ppl_delta}")
        chosen = dict(scored[0][1])
        chosen["layer"] = layer["layer"]
        chosen["score"] = scored[0][0]
        selected.append(chosen)

    counts = [item["total_outliers"] for item in selected]
    positive_counts = [c for c in counts if c > 0]
    max_call_totals = [item["max_call_total"] for item in selected]
    capacities = [
        max(args.min_capacity, int(math.ceil(item["max_call_total"] * args.capacity_margin)))
        for item in selected
    ]

    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    positive_mean = sum(positive_counts) / len(positive_counts) if positive_counts else 0.0
    positive_variance = (
        sum((c - positive_mean) ** 2 for c in positive_counts) / len(positive_counts)
        if positive_counts else 0.0
    )

    return {
        "source": {
            "layer_density": str(args.layer_density),
            "threshold_summary": str(args.threshold_summary),
        },
        "objective": {
            "target_count": args.target_count,
            "max_ppl_delta": args.max_ppl_delta,
            "capacity_margin": args.capacity_margin,
            "min_capacity": args.min_capacity,
            "max_threshold": args.max_threshold,
            "count_weight": args.count_weight,
            "ppl_weight": args.ppl_weight,
            "zero_weight": args.zero_weight,
            "best_scanned_ppl": best_ppl,
        },
        "layer_threshold_csv": ",".join(str(item["threshold"]) for item in selected),
        "layer_capacity_csv": ",".join(str(capacity) for capacity in capacities),
        "threshold_array": [item["threshold"] for item in selected],
        "capacity_array": capacities,
        "selected": selected,
        "summary": {
            "layers": len(selected),
            "total_outliers": sum(counts),
            "min_layer_outliers": min(counts),
            "max_layer_outliers": max(counts),
            "mean_layer_outliers": mean,
            "stddev_layer_outliers": math.sqrt(variance),
            "positive_layers": len(positive_counts),
            "positive_mean_layer_outliers": positive_mean,
            "positive_stddev_layer_outliers": math.sqrt(positive_variance),
            "max_call_total": max(max_call_totals),
            "max_capacity": max(capacities),
            "total_capacity": sum(capacities),
        },
    }


def write_outputs(profile: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "balanced-config.json").write_text(json.dumps(profile, indent=2) + "\n")

    with (output_dir / "balanced-layer-config.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "threshold",
                "capacity",
                "total_outliers",
                "density_pct",
                "max_call_total",
                "max_row_outliers",
                "score",
            ],
        )
        writer.writeheader()
        for item, capacity in zip(profile["selected"], profile["capacity_array"]):
            row = dict(item)
            row["capacity"] = capacity
            writer.writerow(row)

    header = []
    header.append("// Generated by scripts/derive-kcache-outlier-balanced-config.py")
    header.append("static constexpr float llama_nvfp4_kcache_outlier_layer_thresholds[] = {")
    header.append("        " + ", ".join(f"{float(v):.1f}f" for v in profile["threshold_array"]) + ",")
    header.append("};")
    header.append("")
    header.append("static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities_balanced[] = {")
    header.append("        " + ", ".join(str(v) for v in profile["capacity_array"]) + ",")
    header.append("};")
    header.append("")
    (output_dir / "balanced-config-snippet.h").write_text("\n".join(header))


def main() -> None:
    args = parse_args()
    profile = choose_profile(args)
    write_outputs(profile, args.output_dir)
    print(json.dumps(profile["summary"], indent=2))
    print(f"thresholds={profile['layer_threshold_csv']}")
    print(f"capacities={profile['layer_capacity_csv']}")


if __name__ == "__main__":
    main()
