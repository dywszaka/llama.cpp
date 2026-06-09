#!/usr/bin/env python3
import json
import math
import re
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
LOG = EXP / "runs" / "fourth_ctx8192_capacity_fix.raw.log"
RESULTS = EXP / "results"

EXTRACT_RE = re.compile(
    r"target=cache_k_l(\d+).*?rows=(\d+).*?threshold=([0-9.]+).*?"
    r"stored_max=(\d+).*?compact_capacity=(\d+).*?compact_used=(\d+).*?"
    r"(?:compact_overflow=(\d+).*?)?"
    r"total_outliers=(\d+).*?max_row_outliers=(\d+).*?overflow_rows=(\d+)"
)
FINAL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")
KV_PROFILE_RE = re.compile(
    r"NVFP4 K-cache compact outlier sidecar enabled: threshold_profile=(\S+) layer_capacity_profile=(\S+) layer_capacities=(\d+)"
)
KV_SIZE_RE = re.compile(r"K \(([^)]+)\):\s+([0-9.]+) MiB, V \(([^)]+)\):\s+([0-9.]+) MiB")
KV_BUFFER_RE = re.compile(r"CUDA\d+ KV buffer size =\s+([0-9.]+) MiB")

CTX8192_CAPACITY = {
    0: 0, 1: 0, 2: 768, 3: 72, 4: 0, 5: 0, 6: 0, 7: 105, 8: 0,
    9: 14, 10: 0, 11: 0, 12: 0, 13: 29, 14: 0, 15: 59, 16: 314, 17: 67,
    18: 532, 19: 49, 20: 129, 21: 609, 22: 875, 23: 0, 24: 17, 25: 207,
    26: 26, 27: 124, 28: 65, 29: 47, 30: 1885, 31: 42, 32: 85, 33: 18,
    34: 2012, 35: 0,
}
HYBRID_FP8_LAYERS = {0, 1, 4, 5, 6, 8, 10, 11, 12, 14, 23, 35}


def percentile(values, q):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def parse_log():
    text = LOG.read_text(errors="replace")
    records = []
    for m in EXTRACT_RE.finditer(text):
        layer = int(m.group(1))
        rows = int(m.group(2))
        rec = {
            "layer": layer,
            "rows": rows,
            "threshold": float(m.group(3)),
            "stored_max": int(m.group(4)),
            "compact_capacity": int(m.group(5)),
            "compact_used": int(m.group(6)),
            "compact_overflow": int(m.group(7) or 0),
            "total_outliers": int(m.group(8)),
            "max_row_outliers": int(m.group(9)),
            "overflow_rows": int(m.group(10)),
        }
        rec["capacity_util"] = rec["compact_used"] / rec["compact_capacity"] if rec["compact_capacity"] else None
        records.append(rec)

    final = FINAL_RE.search(text)
    profile = KV_PROFILE_RE.search(text)
    kv_size = KV_SIZE_RE.search(text)
    kv_buffer = KV_BUFFER_RE.search(text)
    return text, records, final, profile, kv_size, kv_buffer


def summarize(records, rows_filter):
    by_layer = defaultdict(list)
    for rec in records:
        if rows_filter(rec):
            by_layer[rec["layer"]].append(rec)

    rows = []
    for layer in sorted(set(CTX8192_CAPACITY) - HYBRID_FP8_LAYERS):
        recs = by_layer.get(layer, [])
        cap = CTX8192_CAPACITY[layer]
        used = [r["compact_used"] for r in recs]
        total = [r["total_outliers"] for r in recs]
        utils = [r["capacity_util"] for r in recs if r["capacity_util"] is not None]
        row = {
            "layer": layer,
            "configured_capacity": cap,
            "records": len(recs),
            "thresholds": sorted({r["threshold"] for r in recs}),
            "peak_compact_used": max(used) if used else 0,
            "p50_compact_used": percentile(used, 0.50),
            "p95_compact_used": percentile(used, 0.95),
            "p99_compact_used": percentile(used, 0.99),
            "peak_util": max(utils) if utils else 0.0,
            "p95_util": percentile(utils, 0.95) if utils else 0.0,
            "p99_util": percentile(utils, 0.99) if utils else 0.0,
            "near_full_records_90pct": sum(1 for u in utils if u >= 0.90),
            "near_full_records_75pct": sum(1 for u in utils if u >= 0.75),
            "total_outliers_sum": sum(total),
            "peak_total_outliers": max(total) if total else 0,
            "peak_max_row_outliers": max((r["max_row_outliers"] for r in recs), default=0),
            "overflow_rows_sum": sum(r["overflow_rows"] for r in recs),
            "compact_overflow_sum": sum(r["compact_overflow"] for r in recs),
        }
        row["headroom_at_peak"] = cap - row["peak_compact_used"]
        rows.append(row)
    return rows


def write_outputs():
    RESULTS.mkdir(parents=True, exist_ok=True)
    text, records, final, profile, kv_size, kv_buffer = parse_log()
    ppl_records = [r for r in records if r["rows"] == 512]
    warmup_records = [r for r in records if r["rows"] != 512]
    layer_rows = summarize(records, lambda r: r["rows"] == 512)

    overall = {
        "log": str(LOG.relative_to(EXP)),
        "ppl": float(final.group(1)) if final else None,
        "ppl_err": float(final.group(2)) if final else None,
        "threshold_profile": profile.group(1) if profile else None,
        "capacity_profile": profile.group(2) if profile else None,
        "layer_capacity_count": int(profile.group(3)) if profile else None,
        "kv_buffer_mib": float(kv_buffer.group(1)) if kv_buffer else None,
        "k_cache_summary": kv_size.group(1) if kv_size else None,
        "k_cache_mib": float(kv_size.group(2)) if kv_size else None,
        "v_cache_summary": kv_size.group(3) if kv_size else None,
        "v_cache_mib": float(kv_size.group(4)) if kv_size else None,
        "records_total": len(records),
        "records_ppl_rows512": len(ppl_records),
        "records_warmup_other_rows": len(warmup_records),
        "total_outliers_rows512": sum(r["total_outliers"] for r in ppl_records),
        "overflow_rows_rows512": sum(r["overflow_rows"] for r in ppl_records),
        "compact_overflow_rows512": sum(r["compact_overflow"] for r in ppl_records),
        "max_row_outliers_rows512": max((r["max_row_outliers"] for r in ppl_records), default=0),
        "max_compact_used_rows512": max((r["compact_used"] for r in ppl_records), default=0),
    }

    risky = [
        r for r in layer_rows
        if r["records"] and (r["peak_util"] >= 0.90 or r["overflow_rows_sum"] or r["compact_overflow_sum"])
    ]
    top_util = sorted([r for r in layer_rows if r["records"]], key=lambda r: r["peak_util"], reverse=True)[:10]
    top_outliers = sorted([r for r in layer_rows if r["records"]], key=lambda r: r["total_outliers_sum"], reverse=True)[:10]

    output = {
        "overall": overall,
        "layer_rows": layer_rows,
        "risky_layers": risky,
        "top_peak_util_layers": top_util,
        "top_total_outlier_layers": top_outliers,
    }
    (RESULTS / "outlier_profile.json").write_text(json.dumps(output, indent=2) + "\n")

    cols = [
        "layer", "configured_capacity", "records", "peak_compact_used", "headroom_at_peak",
        "peak_util", "p95_util", "p99_util", "near_full_records_90pct", "near_full_records_75pct",
        "total_outliers_sum", "peak_total_outliers", "peak_max_row_outliers", "overflow_rows_sum",
        "compact_overflow_sum",
    ]
    lines = ["\t".join(cols)]
    for row in layer_rows:
        lines.append("\t".join(str(row.get(c, "")) for c in cols))
    (RESULTS / "layer_capacity_profile.tsv").write_text("\n".join(lines) + "\n")

    summary = ["# K-cache Outlier ctx8192 Capacity Fix Validation", ""]
    summary.append("Profile target: current fourth-case configuration at `-c 8192` after increasing ctx8192 compact capacities.")
    summary.append("")
    summary.append("Configuration observed in log:")
    summary.append("")
    summary.append(f"- threshold profile: `{overall['threshold_profile']}`")
    summary.append(f"- capacity profile: `{overall['capacity_profile']}`")
    summary.append(f"- K cache: `{overall['k_cache_summary']}` `{overall['k_cache_mib']} MiB`")
    summary.append(f"- V cache: `{overall['v_cache_summary']}` `{overall['v_cache_mib']} MiB`")
    summary.append(f"- KV buffer: `{overall['kv_buffer_mib']} MiB`")
    summary.append(f"- PPL: `{overall['ppl']} +/- {overall['ppl_err']}`")
    summary.append("")
    summary.append("Overall outlier evidence, excluding warmup rows:")
    summary.append("")
    summary.append(f"- records with `rows=512`: `{overall['records_ppl_rows512']}`")
    summary.append(f"- total outliers: `{overall['total_outliers_rows512']}`")
    summary.append(f"- max row outliers: `{overall['max_row_outliers_rows512']}`")
    summary.append(f"- overflow rows: `{overall['overflow_rows_rows512']}`")
    summary.append(f"- compact overflow: `{overall['compact_overflow_rows512']}`")
    summary.append("")
    summary.append("## Capacity Pressure")
    summary.append("")
    summary.append("| layer | cap | peak used | headroom | peak util | p99 util | records >=90% | total outliers | max row | overflow |")
    summary.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in top_util:
        summary.append(
            f"| {row['layer']} | {row['configured_capacity']} | {row['peak_compact_used']} | "
            f"{row['headroom_at_peak']} | {row['peak_util']:.3f} | {row['p99_util']:.3f} | "
            f"{row['near_full_records_90pct']} | {row['total_outliers_sum']} | "
            f"{row['peak_max_row_outliers']} | {row['overflow_rows_sum']} |"
        )
    summary.append("")
    summary.append("## Outlier Volume")
    summary.append("")
    summary.append("| layer | cap | total outliers | peak used | peak util | p95 util | max row | overflow |")
    summary.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in top_outliers:
        summary.append(
            f"| {row['layer']} | {row['configured_capacity']} | {row['total_outliers_sum']} | "
            f"{row['peak_compact_used']} | {row['peak_util']:.3f} | {row['p95_util']:.3f} | "
            f"{row['peak_max_row_outliers']} | {row['overflow_rows_sum']} |"
        )
    summary.append("")
    summary.append("## Assessment")
    summary.append("")
    if overall["overflow_rows_rows512"] == 0 and overall["compact_overflow_rows512"] == 0:
        summary.append("- Capacity is not clipping in this run: both `overflow_rows` and `compact_overflow` are zero for all `rows=512` records.")
    else:
        summary.append("- Capacity clipping is present: inspect layers with non-zero overflow before trusting PPL.")
    if risky:
        summary.append("- Some layers run close to capacity. This is a robustness concern, but not an observed correctness failure because no overflow occurred.")
    else:
        summary.append("- No layer reaches 90% of configured capacity in the parsed PPL records.")
    summary.append("- Threshold is fixed at 16 for hybrid mode. Capacity evidence alone cannot prove threshold quality; a too-high threshold can still miss sub-threshold K components without showing overflow.")
    summary.append("- If PPL remains poor, the next likely causes are threshold/profile quality or the FP8 hybrid K layers, not sidecar capacity shortage.")
    summary.append("")
    summary.append("Artifacts:")
    summary.append("")
    summary.append("- Raw log: `runs/fourth_ctx8192_capacity_fix.raw.log`")
    summary.append("- JSON profile: `results/outlier_profile.json`")
    summary.append("- Layer TSV: `results/layer_capacity_profile.tsv`")
    (EXP / "summary.md").write_text("\n".join(summary) + "\n")

    print(EXP / "summary.md")
    print(RESULTS / "layer_capacity_profile.tsv")


if __name__ == "__main__":
    write_outputs()
