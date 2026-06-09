#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXP_DIR="${ROOT_DIR}/experiments/${STAMP}-kcache-outlier-threshold-ratio-sweep"
CHUNKS="${CHUNKS:-200}"
TARGET_RATIO="${TARGET_RATIO:-0.0001}"
NO_WARMUP="${NO_WARMUP:-1}"

if [[ "$#" -eq 0 ]]; then
  THRESHOLDS=(8 10 12 14 16 18 20 22 24 28 32 40 48 64 96 128 192 256 384)
else
  THRESHOLDS=("$@")
fi

mkdir -p "${EXP_DIR}/runs" "${EXP_DIR}/results" "${EXP_DIR}/scripts"
cp "$0" "${EXP_DIR}/scripts/"
cp "${ROOT_DIR}/scripts/summarize-kcache-outlier-threshold-sweep.py" "${EXP_DIR}/scripts/"

cat > "${EXP_DIR}/input-reference.txt" <<EOF
baseline=${ROOT_DIR}/expt-baseline.md
model=/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf
data=${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw
binary=${ROOT_DIR}/build_cuda/bin/llama-perplexity
chunks=${CHUNKS}
target_ratio=${TARGET_RATIO}
thresholds=${THRESHOLDS[*]}
changed_from_baseline=--cache-type-k nvfp4, LLAMA_NVFP4_KCACHE_OUTLIER=1, LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD
diagnostic_runtime=NO_WARMUP=${NO_WARMUP}; skips startup warmup when enabled so ratio logs only cover real PPL input chunks
EOF

EXTRA_ARGS=()
if [[ "${NO_WARMUP}" != "0" ]]; then
  EXTRA_ARGS+=(--no-warmup)
fi

for threshold in "${THRESHOLDS[@]}"; do
  log_file="${EXP_DIR}/runs/threshold_${threshold}.raw.log"
  echo "running threshold=${threshold}" | tee "${EXP_DIR}/runs/threshold_${threshold}.status"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  LLAMA_NVFP4_KCACHE_OUTLIER=1 \
  LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD="${threshold}" \
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
      "${EXTRA_ARGS[@]}" \
      --chunks "${CHUNKS}" \
      > "${log_file}" 2>&1
done

python3 "${ROOT_DIR}/scripts/summarize-kcache-outlier-threshold-sweep.py" \
  --runs-dir "${EXP_DIR}/runs" \
  --output-dir "${EXP_DIR}/results" \
  --target-ratio "${TARGET_RATIO}"

cat > "${EXP_DIR}/summary.md" <<EOF
# K-Cache Outlier Threshold Ratio Sweep

## Purpose

Sweep global \`LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD\` values and choose per-layer
thresholds whose max-batch outlier ratio is closest to \`${TARGET_RATIO}\`.

## Inputs

- Real data: \`${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw\`
- Chunks: \`${CHUNKS}\`
- Thresholds: \`${THRESHOLDS[*]}\`
- No warmup: \`${NO_WARMUP}\`

## Outputs

- \`results/threshold-layer-batch-ratio.csv\`
- \`results/selected-thresholds.csv\`
EOF

echo "${EXP_DIR}"
