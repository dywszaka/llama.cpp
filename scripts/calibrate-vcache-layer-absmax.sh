#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/calibrate-vcache-layer-absmax.sh [options] [-- extra llama-vcache-calibration args]

Run an offline f16/f16 baseline decode over calibration text, collect Vcur layer
activation absmax values, and write a compact JSON file for
LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE.

Defaults follow EXPERI.md's local baseline unless the option is explicitly
overridden.

Options:
  --out FILE                 Compact output JSON, default: experiments/qwen3-8b-v-layer-absmax.json
  --raw-out FILE             Raw calibration report, default: <out>.raw.json
  --bin FILE                 Calibration binary, default: build_cuda/bin/llama-vcache-calibration
  --model FILE               GGUF model, default: Qwen3 8B NVFP4 model from EXPERI.md
  --calib-text FILE          One prompt per line, default: tools/vcache-calibration/calibration-prompts.txt
  --ctx-size N               Context size, default: 512
  --batch-size N             Logical batch size, default: 512
  --ubatch-size N            Physical ubatch size, default: 512
  --threads N                CPU threads, default: 32
  --n-gpu-layers N           GPU layers, default: 40
  --cache-type-k TYPE        K cache type, default: f16
  --cache-type-v TYPE        V cache type, default: f16
  --dry-run                  Print commands without running
  -h, --help                 Show this help

Environment:
  CUDA_VISIBLE_DEVICES       Default: 0
  GGML_CUDA_NVFP4_NATIVE     Default: 1
EOF
}

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v arg "%q" "$arg"
    out+="${arg} "
  done
  printf '%s\n' "${out% }"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

out="${ROOT_DIR}/experiments/qwen3-8b-v-layer-absmax.json"
raw_out=""
bin="${ROOT_DIR}/build_cuda/bin/llama-vcache-calibration"
model="/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf"
calib_text="${ROOT_DIR}/tools/vcache-calibration/calibration-prompts.txt"
ctx_size="512"
batch_size="512"
ubatch_size="512"
threads="32"
n_gpu_layers="40"
cache_type_k="f16"
cache_type_v="f16"
dry_run=0
extra_args=()

while (($# > 0)); do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --raw-out) raw_out="$2"; shift 2 ;;
    --bin) bin="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --calib-text) calib_text="$2"; shift 2 ;;
    --ctx-size|-c) ctx_size="$2"; shift 2 ;;
    --batch-size|-b) batch_size="$2"; shift 2 ;;
    --ubatch-size|-ub) ubatch_size="$2"; shift 2 ;;
    --threads|-t) threads="$2"; shift 2 ;;
    --n-gpu-layers|--n_gpu_layers|-ngl) n_gpu_layers="$2"; shift 2 ;;
    --cache-type-k|-ctk) cache_type_k="$2"; shift 2 ;;
    --cache-type-v|-ctv) cache_type_v="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; extra_args=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${raw_out}" ]]; then
  raw_out="${out%.json}.raw.json"
fi

mkdir -p "$(dirname "${out}")" "$(dirname "${raw_out}")"

cmd=(
  "${bin}"
  -m "${model}"
  -f "${calib_text}"
  -o "${raw_out}"
  --cache-type-k "${cache_type_k}"
  --cache-type-v "${cache_type_v}"
  --n-gpu-layers "${n_gpu_layers}"
  --batch-size "${batch_size}"
  --ubatch-size "${ubatch_size}"
  -t "${threads}"
  -c "${ctx_size}"
)

if ((${#extra_args[@]} > 0)); then
  cmd+=("${extra_args[@]}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GGML_CUDA_NVFP4_NATIVE="${GGML_CUDA_NVFP4_NATIVE:-1}"

echo "calibration command:"
quote_cmd "${cmd[@]}"

if ((dry_run)); then
  echo "dry-run: compact output would be written to ${out}"
  echo "dry-run: raw report would be written to ${raw_out}"
  exit 0
fi

"${cmd[@]}"

python3 - "${raw_out}" "${out}" <<'PY'
import json
import math
import sys

raw_path, out_path = sys.argv[1:3]

with open(raw_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

layers = raw.get("layers")
if not isinstance(layers, list) or not layers:
    raise SystemExit(f"{raw_path}: missing non-empty layers array")

compact_layers = []
seen = set()
for entry in layers:
    layer = entry.get("layer")
    absmax = entry.get("abs_max")
    if not isinstance(layer, int):
        raise SystemExit(f"{raw_path}: invalid layer entry {entry!r}")
    if not isinstance(absmax, (int, float)) or not math.isfinite(absmax) or absmax <= 0:
        raise SystemExit(f"{raw_path}: invalid abs_max for layer {layer}: {absmax!r}")
    if layer in seen:
        raise SystemExit(f"{raw_path}: duplicate layer {layer}")
    seen.add(layer)
    compact_layers.append({"layer": layer, "absmax": float(absmax)})

compact_layers.sort(key=lambda x: x["layer"])
expected = list(range(compact_layers[-1]["layer"] + 1))
actual = [x["layer"] for x in compact_layers]
if actual != expected:
    raise SystemExit(f"{raw_path}: layer ids are not contiguous from 0: {actual}")

compact = {"layer_absmax": compact_layers}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(compact, f, indent=2)
    f.write("\n")

print(f"wrote compact V layer absmax JSON: {out_path}")
PY
