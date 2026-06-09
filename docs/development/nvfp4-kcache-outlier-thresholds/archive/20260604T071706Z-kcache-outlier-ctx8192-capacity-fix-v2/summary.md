# K-cache Outlier ctx8192 Capacity Fix v2 Validation

Profile target: current fourth-case configuration at `-c 8192` after increasing ctx8192 compact capacities.

Configuration observed in log:

- threshold profile: `hybrid_threshold16`
- capacity profile: `ctx8192`
- K cache: `nvfp4+fp8_e4m3_e8m0_32` `207.0 MiB`
- V cache: `nvfp4` `162.0 MiB`
- KV buffer: `371.32 MiB`
- PPL: `10.1924 +/- 0.07293`

Overall outlier evidence, excluding warmup rows:

- records with `rows=512`: `13824`
- total outliers: `2060915`
- max row outliers: `8`
- overflow rows: `0`
- compact overflow: `0`

## Capacity Pressure

| layer | cap | peak used | headroom | peak util | p99 util | records >=90% | total outliers | max row | overflow |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 67 | 61 | 6 | 0.910 | 0.201 | 1 | 1220 | 4 | 0 |
| 3 | 72 | 61 | 11 | 0.847 | 0.674 | 0 | 17387 | 3 | 0 |
| 21 | 609 | 512 | 97 | 0.841 | 0.772 | 0 | 197851 | 3 | 0 |
| 25 | 207 | 171 | 36 | 0.826 | 0.571 | 0 | 21605 | 3 | 0 |
| 2 | 768 | 614 | 154 | 0.799 | 0.769 | 0 | 233827 | 6 | 0 |
| 33 | 18 | 14 | 4 | 0.778 | 0.347 | 0 | 543 | 3 | 0 |
| 30 | 1885 | 1466 | 419 | 0.778 | 0.753 | 0 | 667491 | 6 | 0 |
| 34 | 2012 | 1549 | 463 | 0.770 | 0.684 | 0 | 613989 | 4 | 0 |
| 16 | 314 | 227 | 87 | 0.723 | 0.561 | 0 | 47467 | 5 | 0 |
| 18 | 532 | 337 | 195 | 0.633 | 0.531 | 0 | 65832 | 5 | 0 |

## Outlier Volume

| layer | cap | total outliers | peak used | peak util | p95 util | max row | overflow |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1885 | 667491 | 1466 | 0.778 | 0.723 | 6 | 0 |
| 34 | 2012 | 613989 | 1549 | 0.770 | 0.645 | 4 | 0 |
| 2 | 768 | 233827 | 614 | 0.799 | 0.669 | 6 | 0 |
| 21 | 609 | 197851 | 512 | 0.841 | 0.739 | 3 | 0 |
| 22 | 875 | 143903 | 546 | 0.624 | 0.477 | 5 | 0 |
| 18 | 532 | 65832 | 337 | 0.633 | 0.401 | 5 | 0 |
| 16 | 314 | 47467 | 227 | 0.723 | 0.463 | 5 | 0 |
| 25 | 207 | 21605 | 171 | 0.826 | 0.431 | 3 | 0 |
| 27 | 124 | 17817 | 78 | 0.629 | 0.444 | 3 | 0 |
| 3 | 72 | 17387 | 61 | 0.847 | 0.597 | 3 | 0 |

## Assessment

- Capacity is not clipping in this run: both `overflow_rows` and `compact_overflow` are zero for all `rows=512` records.
- Some layers run close to capacity. This is a robustness concern, but not an observed correctness failure because no overflow occurred.
- Threshold is fixed at 16 for hybrid mode. Capacity evidence alone cannot prove threshold quality; a too-high threshold can still miss sub-threshold K components without showing overflow.
- If PPL remains poor, the next likely causes are threshold/profile quality or the FP8 hybrid K layers, not sidecar capacity shortage.

Artifacts:

- Raw log: `runs/fourth_ctx8192_capacity_fix_v2.raw.log`
- JSON profile: `results/outlier_profile.json`
- Layer TSV: `results/layer_capacity_profile.tsv`
