# K-cache Outlier ctx8192 Capacity Fix Validation

Profile target: current fourth-case configuration at `-c 8192` after increasing ctx8192 compact capacities.

Configuration observed in log:

- threshold profile: `hybrid_threshold16`
- capacity profile: `ctx8192`
- K cache: `nvfp4+fp8_e4m3_e8m0_32` `207.0 MiB`
- V cache: `nvfp4` `162.0 MiB`
- KV buffer: `371.32 MiB`
- PPL: `10.1907 +/- 0.07278`

Overall outlier evidence, excluding warmup rows:

- records with `rows=512`: `13824`
- total outliers: `2064753`
- max row outliers: `8`
- overflow rows: `0`
- compact overflow: `2`

## Capacity Pressure

| layer | cap | peak used | headroom | peak util | p99 util | records >=90% | total outliers | max row | overflow |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 28 | 65 | 76 | -11 | 1.169 | 0.408 | 1 | 6223 | 2 | 0 |
| 31 | 42 | 44 | -2 | 1.048 | 0.506 | 1 | 2043 | 3 | 0 |
| 15 | 59 | 52 | 7 | 0.881 | 0.492 | 0 | 4284 | 8 | 0 |
| 20 | 129 | 112 | 17 | 0.868 | 0.374 | 0 | 3154 | 6 | 0 |
| 29 | 47 | 40 | 7 | 0.851 | 0.436 | 0 | 3388 | 2 | 0 |
| 21 | 609 | 509 | 100 | 0.836 | 0.784 | 0 | 195436 | 4 | 0 |
| 25 | 207 | 172 | 35 | 0.831 | 0.671 | 0 | 22450 | 4 | 0 |
| 27 | 124 | 103 | 21 | 0.831 | 0.542 | 0 | 18293 | 2 | 0 |
| 30 | 1885 | 1551 | 334 | 0.823 | 0.764 | 0 | 668375 | 6 | 0 |
| 2 | 768 | 614 | 154 | 0.799 | 0.769 | 0 | 233827 | 6 | 0 |

## Outlier Volume

| layer | cap | total outliers | peak used | peak util | p95 util | max row | overflow |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1885 | 668375 | 1551 | 0.823 | 0.737 | 6 | 0 |
| 34 | 2012 | 618006 | 1470 | 0.731 | 0.649 | 4 | 0 |
| 2 | 768 | 233827 | 614 | 0.799 | 0.669 | 6 | 0 |
| 21 | 609 | 195436 | 509 | 0.836 | 0.745 | 4 | 0 |
| 22 | 875 | 142173 | 578 | 0.661 | 0.493 | 5 | 0 |
| 18 | 532 | 66825 | 334 | 0.628 | 0.421 | 6 | 0 |
| 16 | 314 | 46876 | 200 | 0.637 | 0.440 | 5 | 0 |
| 25 | 207 | 22450 | 172 | 0.831 | 0.425 | 4 | 0 |
| 27 | 124 | 18293 | 103 | 0.831 | 0.460 | 2 | 0 |
| 3 | 72 | 17419 | 56 | 0.778 | 0.611 | 3 | 0 |

## Assessment

- Capacity clipping is present: inspect layers with non-zero overflow before trusting PPL.
- Some layers run close to capacity. This is a robustness concern, but not an observed correctness failure because no overflow occurred.
- Threshold is fixed at 16 for hybrid mode. Capacity evidence alone cannot prove threshold quality; a too-high threshold can still miss sub-threshold K components without showing overflow.
- If PPL remains poor, the next likely causes are threshold/profile quality or the FP8 hybrid K layers, not sidecar capacity shortage.

Artifacts:

- Raw log: `runs/fourth_ctx8192_capacity_fix.raw.log`
- JSON profile: `results/outlier_profile.json`
- Layer TSV: `results/layer_capacity_profile.tsv`
