# NVFP4 K-cache Outlier Thresholds

This directory owns reusable material for offline NVFP4 K-cache outlier
threshold/profile derivation. Timestamped `experiments/` directories should keep
raw run evidence such as logs and status files; generated profile snapshots,
derivation scripts, and workflow documentation live here.

## Layout

- `balanced-profile.md`: workflow for deriving the balanced per-layer threshold
  and compact sidecar capacity profile.
- `scripts/parse-kcache-outlier-threshold-sweep.py`: parses
  `threshold_*.raw.log` PPL/outlier sweep logs into layer-density CSVs.
- `scripts/derive-kcache-outlier-balanced-config.py`: selects per-layer
  balanced thresholds and capacities from parsed sweep outputs.
- `scripts/run-kcache-outlier-balanced-experiment.sh`: helper that creates a new
  timestamped experiment folder for raw evidence while using the scripts in this
  directory.
- `scripts/summarize-kcache-outlier-threshold-sweep.py`: summarizes
  max-batch outlier ratios and selects thresholds near a target ratio.
- `profiles/balanced/`: current balanced profile snapshot and derivation
  summary.
- `profiles/ratio-1e4/`: current `LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE=new`
  threshold snapshot and capacity calibration inputs.
- `profiles/bf16/`: target location for the opt-in
  `LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE=bf16` profile derived with the BF16/new
  FP4 quantizer switches enabled.
- `profiles/ctx8192-capacity/`: context-8192 capacity snapshot for the balanced
  threshold profile.
- `archive/`: older intermediate threshold/profile derivation scripts,
  summaries, and generated tables moved out of `experiments/`. These are kept
  for provenance but are not the current canonical profile.

## Evidence

The profile snapshots were derived from raw logs that remain under:

- `experiments/20260603T032033Z-kcache-outlier-balanced-threshold-config/`
- `experiments/20260604T085500Z-balanced-threshold-ctx8192-capacity-derive/`
- `experiments/20260605T072559Z-kcache-outlier-threshold-ratio-sweep/`
- `experiments/20260605T081206Z-kcache-outlier-ratio1e4-default-ppl/`

Do not use those experiment directories as the canonical location for scripts or
generated profile tables. If a new reusable profile is derived, copy the reviewed
profile artifacts here and update `src/llama-kv-cache-nvfp4-outlier-config.h`.
