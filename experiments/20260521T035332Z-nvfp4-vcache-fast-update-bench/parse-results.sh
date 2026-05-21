#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_DIR="${EXP_DIR}/metrics"
SUMMARY="${EXP_DIR}/summary.md"

python3 - "${EXP_DIR}" <<'PY'
import csv
import json
import pathlib
import sys

exp_dir = pathlib.Path(sys.argv[1])
metrics_dir = exp_dir / "metrics"

cases = [
    ("baseline", "01-baseline", "f16", "f16", None),
    ("nvfp4-fast-update-off", "02-nvfp4-fast-update-off", "f16", "nvfp4", False),
    ("nvfp4-fast-update-on", "03-nvfp4-fast-update-on", "f16", "nvfp4", True),
]

rows = []
for run, stem, cache_k, cache_v, fast_update in cases:
    path = metrics_dir / f"{stem}.json"
    data = json.loads(path.read_text())
    for item in data:
        test = "tg" if int(item["n_gen"]) > 0 and int(item["n_prompt"]) == 0 else "pp"
        rows.append({
            "run": run,
            "script_output": path.name,
            "test": test,
            "cache_k": cache_k,
            "cache_v": cache_v,
            "fast_update": "" if fast_update is None else str(fast_update).lower(),
            "n_prompt": int(item["n_prompt"]),
            "n_gen": int(item["n_gen"]),
            "avg_ts": float(item["avg_ts"]),
            "stddev_ts": float(item["stddev_ts"]),
            "avg_ns": int(item["avg_ns"]),
            "stddev_ns": int(item["stddev_ns"]),
            "samples_ts": item.get("samples_ts", []),
            "kv_unified": bool(item.get("kv_unified", False)),
            "build_commit": item["build_commit"],
            "build_number": int(item["build_number"]),
            "gpu_info": item["gpu_info"],
        })

csv_path = metrics_dir / "results.csv"
fieldnames = [
    "run", "test", "cache_k", "cache_v", "fast_update", "n_prompt", "n_gen",
    "avg_ts", "stddev_ts", "avg_ns", "stddev_ns", "kv_unified",
    "script_output", "build_commit", "build_number", "gpu_info",
]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fieldnames})

json_path = metrics_dir / "results.json"
json_path.write_text(json.dumps(rows, indent=2) + "\n")

by_key = {(row["run"], row["test"]): row for row in rows}

def pct(new, old):
    return (new - old) / old * 100.0 if old else 0.0

baseline_tg = by_key.get(("baseline", "tg"))
off_tg = by_key.get(("nvfp4-fast-update-off", "tg"))
on_tg = by_key.get(("nvfp4-fast-update-on", "tg"))

summary = []
summary.append("# NVFP4 V-cache Fast Update llama-bench Summary\n")
summary.append("Date: 2026-05-21 UTC\n")
summary.append("## Parameters\n")
summary.append("- Binary: `build_cuda/bin/llama-bench`\n")
summary.append("- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`\n")
summary.append("- CUDA device: `CUDA_VISIBLE_DEVICES=0`\n")
summary.append("- GPU layers: `--n-gpu-layers 40`\n")
summary.append("- Batch: `--batch-size 2048 --ubatch-size 512`\n")
summary.append("- Threads: `-t 32`\n")
summary.append("- Tests: `pp512` and `tg128`; `tg128` is the primary fast_update metric.\n")
summary.append("- Repetitions: `5` unless `BENCH_REPS` was set when running `run-bench.sh`.\n")
summary.append("\n## Results\n")
summary.append("| Run | Test | K cache | V cache | fast_update | kv_unified | tok/s | stdev tok/s |\n")
summary.append("| --- | --- | --- | --- | --- | --- | ---: | ---: |\n")
for row in rows:
    summary.append(
        f"| {row['run']} | {row['test']} | `{row['cache_k']}` | `{row['cache_v']}` | "
        f"{row['fast_update'] or '-'} | {str(row['kv_unified']).lower()} | "
        f"{row['avg_ts']:.2f} | {row['stddev_ts']:.2f} |\n"
    )

summary.append("\n## Decode Deltas\n")
summary.append("| Comparison | tok/s delta | tok/s delta % |\n")
summary.append("| --- | ---: | ---: |\n")
if baseline_tg and off_tg:
    delta = off_tg["avg_ts"] - baseline_tg["avg_ts"]
    summary.append(f"| nvfp4 fast_update off vs baseline | {delta:.2f} | {pct(off_tg['avg_ts'], baseline_tg['avg_ts']):+.2f}% |\n")
if baseline_tg and on_tg:
    delta = on_tg["avg_ts"] - baseline_tg["avg_ts"]
    summary.append(f"| nvfp4 fast_update on vs baseline | {delta:.2f} | {pct(on_tg['avg_ts'], baseline_tg['avg_ts']):+.2f}% |\n")
if off_tg and on_tg:
    delta = on_tg["avg_ts"] - off_tg["avg_ts"]
    summary.append(f"| fast_update on vs off | {delta:.2f} | {pct(on_tg['avg_ts'], off_tg['avg_ts']):+.2f}% |\n")

summary.append("\n## Validation Notes\n")
summary.append("- `llama-bench` reports speed only; it does not produce a PPL/accuracy metric.\n")
summary.append("- Use existing PPL experiment results for precision context, or run a separate PPL sanity check if accuracy must be revalidated for this exact build.\n")
summary.append("- Raw stdout JSON and stderr logs are preserved under `metrics/` and `logs/`.\n")

(exp_dir / "summary.md").write_text("".join(summary))
PY
