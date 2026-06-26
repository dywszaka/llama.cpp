#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAB_DIR="${ROOT_DIR}/mylab/kqv-heatmap"
RESULTS_DIR="${LAB_DIR}/results"
VIZ_DIR="${RESULTS_DIR}/visualization"

python3 "${LAB_DIR}/scripts/generate_v_vp_heatmaps.py" \
  --raw-dir "${RESULTS_DIR}/raw_tensors" \
  --output-dir "${VIZ_DIR}"

find "${VIZ_DIR}/v_vp_heatmaps" -type f | sort > "${VIZ_DIR}/v_vp_heatmaps.files.txt"
sha256sum "${VIZ_DIR}/v_vp_heatmaps.html" "${VIZ_DIR}/v_vp_heatmaps_manifest.json" \
  $(cat "${VIZ_DIR}/v_vp_heatmaps.files.txt") > "${VIZ_DIR}/v_vp_heatmaps.sha256"
