#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXP_DIR="${ROOT_DIR}/experiments/${STAMP}-kcache-outlier-balanced-threshold-config"
PREVIOUS_EXP="${ROOT_DIR}/experiments/20260529T080102Z-kcache-outlier-layer-threshold-capacity"
THRESHOLDS=(16 20 24 28 32 40 48 56 64 80 96 112 128 160 192 224 256 320 384)
RUN_SWEEP=0
RUN_PPL=0
TARGET_COUNT=200
MAX_PPL_DELTA=0.35
CAPACITY_MARGIN=1.25

usage() {
  cat >&2 <<EOF
usage: $0 [options]

Options:
  --run-sweep              run fresh threshold sweep before deriving config.
                          Use only on builds that support
                          LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD.
  --run-ppl               print the compact PPL validation command after deriving
                          config. Apply the generated snippet and rebuild first.
  --target-count N         preferred total outliers per layer in sweep (${TARGET_COUNT})
  --max-ppl-delta X        reject thresholds whose global-scan PPL delta exceeds X (${MAX_PPL_DELTA})
  --capacity-margin X      compact capacity margin over max_call_total (${CAPACITY_MARGIN})
  --exp-dir DIR            write artifacts to DIR (default: timestamped experiments dir)
  --previous-exp DIR       reuse sweep artifacts from DIR when --run-sweep is absent
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-sweep)
      RUN_SWEEP=1
      shift
      ;;
    --run-ppl)
      RUN_PPL=1
      shift
      ;;
    --target-count)
      TARGET_COUNT="$2"
      shift 2
      ;;
    --max-ppl-delta)
      MAX_PPL_DELTA="$2"
      shift 2
      ;;
    --capacity-margin)
      CAPACITY_MARGIN="$2"
      shift 2
      ;;
    --exp-dir)
      EXP_DIR="$2"
      shift 2
      ;;
    --previous-exp)
      PREVIOUS_EXP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${EXP_DIR}/scripts" "${EXP_DIR}/runs" "${EXP_DIR}/results"

cp "${SCRIPT_DIR}/derive-kcache-outlier-balanced-config.py" "${EXP_DIR}/scripts/"
cp "${SCRIPT_DIR}/parse-kcache-outlier-threshold-sweep.py" "${EXP_DIR}/scripts/"

cat > "${EXP_DIR}/input-reference.txt" <<EOF
model=/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf
data=${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw
baseline=${ROOT_DIR}/expt-baseline.md
previous_experiment=${PREVIOUS_EXP}
target_count=${TARGET_COUNT}
max_ppl_delta=${MAX_PPL_DELTA}
capacity_margin=${CAPACITY_MARGIN}
EOF

run_threshold_case() {
  local threshold="$1"
  local log_file="${EXP_DIR}/runs/threshold_${threshold}.raw.log"
  {
    echo "case=threshold_${threshold}"
    echo "threshold=${threshold}"
    echo "chunks=200"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    LLAMA_NVFP4_KCACHE_OUTLIER=1 \
    LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD="${threshold}" \
    LLAMA_NVFP4_KCACHE_OUTLIER_LOG=1 \
      "${ROOT_DIR}/build_cuda/bin/llama-perplexity" \
        -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
        -f "${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw" \
        --cache-type-k nvfp4 \
        --cache-type-v f16 \
        --n_gpu_layers 40 \
        --batch-size 512 \
        --ubatch-size 512 \
        -t 32 \
        -c 512 \
        --kv-unified \
        --chunks 200
  } > "${log_file}" 2>&1
}

if [[ "${RUN_SWEEP}" -eq 1 ]]; then
  for threshold in "${THRESHOLDS[@]}"; do
    run_threshold_case "${threshold}"
  done
  python3 "${SCRIPT_DIR}/parse-kcache-outlier-threshold-sweep.py" \
    --runs-dir "${EXP_DIR}/runs" \
    --output-dir "${EXP_DIR}/results"
  PREVIOUS_EXP="${EXP_DIR}"
fi

LAYER_DENSITY="${PREVIOUS_EXP}/results/threshold-layer-density.csv"
THRESHOLD_SUMMARY="${PREVIOUS_EXP}/results/threshold-summary.csv"
if [[ ! -f "${LAYER_DENSITY}" || ! -f "${THRESHOLD_SUMMARY}" ]]; then
  echo "missing sweep artifacts under ${PREVIOUS_EXP}" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/derive-kcache-outlier-balanced-config.py" \
  --layer-density "${LAYER_DENSITY}" \
  --threshold-summary "${THRESHOLD_SUMMARY}" \
  --output-dir "${EXP_DIR}/results" \
  --target-count "${TARGET_COUNT}" \
  --max-ppl-delta "${MAX_PPL_DELTA}" \
  --capacity-margin "${CAPACITY_MARGIN}" \
  | tee "${EXP_DIR}/results/derive.stdout"

cat > "${EXP_DIR}/summary.md" <<EOF
# Balanced K-Cache Outlier Threshold Config

## Purpose

Derive a per-layer NVFP4 K-cache outlier configuration from the previous
threshold/capacity experiment so layer outlier counts are closer to each other
while avoiding thresholds whose scanned PPL regresses too much.

## Inputs

- Previous experiment: \`${PREVIOUS_EXP}\`
- Layer density CSV: \`${LAYER_DENSITY}\`
- Threshold summary CSV: \`${THRESHOLD_SUMMARY}\`
- Target layer outliers in sweep: \`${TARGET_COUNT}\`
- Max accepted global-scan PPL delta: \`${MAX_PPL_DELTA}\`
- Capacity margin: \`${CAPACITY_MARGIN}\`

## Outputs

- \`results/balanced-config.json\`
- \`results/balanced-layer-config.csv\`
- \`results/balanced-config-snippet.h\`
- \`results/derive.stdout\`

## Validation Status

This script derived the configuration from existing sweep artifacts. Full PPL
validation requires applying the generated per-layer threshold/capacity snippet
to the current NVFP4 K-cache outlier config path or running a build that supports
external per-layer config injection.
EOF

if [[ "${RUN_PPL}" -eq 1 ]]; then
  cat > "${EXP_DIR}/scripts/run_balanced_profile_ppl.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="\$(cd "\$(dirname "\$0")/../../.." && pwd)"
EXP_DIR="\$(cd "\$(dirname "\$0")/.." && pwd)"
CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}" \\
LLAMA_NVFP4_KCACHE_OUTLIER=1 \\
  "\${ROOT_DIR}/build_cuda/bin/llama-perplexity" \\
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \\
    -f "\${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw" \\
    --cache-type-k nvfp4 \\
    --cache-type-v f16 \\
    --n_gpu_layers 40 \\
    --batch-size 512 \\
    --ubatch-size 512 \\
    -t 32 \\
    -c 512 \\
    --kv-unified \\
    > "\${EXP_DIR}/runs/balanced_profile.raw.log" 2>&1
EOF
  chmod +x "${EXP_DIR}/scripts/run_balanced_profile_ppl.sh"
  echo "Wrote ${EXP_DIR}/scripts/run_balanced_profile_ppl.sh"
  echo "Apply ${EXP_DIR}/results/balanced-config-snippet.h, rebuild, then run that script."
fi

echo "${EXP_DIR}"
