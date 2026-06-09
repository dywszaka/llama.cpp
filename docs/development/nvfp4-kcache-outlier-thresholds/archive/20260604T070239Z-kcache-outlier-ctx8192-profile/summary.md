# K-cache Outlier ctx8192 Profile

Profile target: current fourth-case configuration at `-c 8192`.

Configuration observed in log:

- threshold profile: `hybrid_threshold16`
- capacity profile: `ctx8192`
- K cache: `nvfp4+fp8_e4m3_e8m0_32` `207.0 MiB`
- V cache: `nvfp4` `162.0 MiB`
- KV buffer: `371.31 MiB`
- PPL: `10.4403 +/- 0.0749`

Overall outlier evidence, excluding warmup rows:

- records with `rows=512`: `13824`
- total outliers: `2069391`
- max row outliers: `8`
- overflow rows: `0`
- compact overflow: `52`

## Capacity Pressure

| layer | cap | peak used | headroom | peak util | p99 util | records >=90% | total outliers | max row | overflow |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 8 | 68 | -60 | 8.500 | 1.656 | 23 | 781 | 2 | 0 |
| 19 | 16 | 39 | -23 | 2.438 | 0.625 | 4 | 536 | 3 | 0 |
| 28 | 28 | 52 | -24 | 1.857 | 0.902 | 6 | 6194 | 2 | 0 |
| 17 | 31 | 53 | -22 | 1.710 | 0.613 | 3 | 1400 | 4 | 0 |
| 16 | 174 | 251 | -77 | 1.443 | 0.948 | 9 | 45333 | 5 | 0 |
| 25 | 128 | 165 | -37 | 1.289 | 1.057 | 13 | 22537 | 4 | 0 |
| 27 | 77 | 99 | -22 | 1.286 | 0.860 | 4 | 18113 | 3 | 0 |
| 29 | 29 | 37 | -8 | 1.276 | 0.793 | 3 | 3311 | 2 | 0 |
| 7 | 68 | 84 | -16 | 1.235 | 0.651 | 1 | 9227 | 8 | 0 |
| 22 | 580 | 700 | -120 | 1.207 | 0.889 | 3 | 143753 | 5 | 0 |

## Outlier Volume

| layer | cap | total outliers | peak used | peak util | p95 util | max row | overflow |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1664 | 674800 | 1508 | 0.906 | 0.838 | 6 | 0 |
| 34 | 1667 | 616858 | 1609 | 0.965 | 0.781 | 4 | 0 |
| 2 | 695 | 233827 | 614 | 0.883 | 0.739 | 6 | 0 |
| 21 | 564 | 197572 | 487 | 0.863 | 0.794 | 4 | 0 |
| 22 | 580 | 143753 | 700 | 1.207 | 0.706 | 5 | 0 |
| 18 | 409 | 64505 | 425 | 1.039 | 0.537 | 5 | 0 |
| 16 | 174 | 45333 | 251 | 1.443 | 0.773 | 5 | 0 |
| 25 | 128 | 22537 | 165 | 1.289 | 0.766 | 4 | 0 |
| 27 | 77 | 18113 | 99 | 1.286 | 0.727 | 3 | 0 |
| 3 | 72 | 17449 | 60 | 0.833 | 0.597 | 3 | 0 |

## Assessment

- Capacity clipping is present: inspect layers with non-zero overflow before trusting PPL.
- Some layers run close to capacity. This is a robustness concern, but not an observed correctness failure because no overflow occurred.
- Threshold is fixed at 16 for hybrid mode. Capacity evidence alone cannot prove threshold quality; a too-high threshold can still miss sub-threshold K components without showing overflow.
- If PPL remains poor, the next likely causes are threshold/profile quality or the FP8 hybrid K layers, not sidecar capacity shortage.

Artifacts:

- Raw log: `runs/fourth_ctx8192_profile.raw.log`
- JSON profile: `results/outlier_profile.json`
- Layer TSV: `results/layer_capacity_profile.tsv`
