# KQV Heatmap Lab

This lab exports and visualizes Q, KQ, V, and VP tensors for the one-chunk
baseline run used by `experiments/20260624T090752Z-baseline-onechunk-k-export/`.

The export path uses `tools/kcache-mean` graph callbacks:

- `Qcur-*` -> Q
- `kq-*` -> KQ, before softmax
- `Vcur-*` -> V
- `kqv-*` -> VP, the non-flash attention `V @ P` result before output projection

## Run

From the repository root:

```bash
mylab/kqv-heatmap/scripts/run_kqv_export.sh
mylab/kqv-heatmap/scripts/generate_q_kq_heatmaps.sh
mylab/kqv-heatmap/scripts/generate_v_vp_heatmaps.sh
```

Outputs are written under `mylab/kqv-heatmap/results/`, which is git-ignored
because it contains large raw tensors and PNGs.

## Shapes

- Q: `512 tokens x 4096 channels`
- KQ: `512 query tokens x 16384 flattened key/head channels`
- V: `512 tokens x 1024 channels`
- VP: `512 tokens x 4096 channels`

Heatmap x-axis is token position. Heatmap y-axis is channel. Colors use
`abs(x)` with per-layer `p99(abs(x))` clipping.

## Baseline Parameters

- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- Prompt file: `data/wikitext/wikitext-2-raw/wiki.test.raw`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- K/V cache: `--cache-type-k f16 --cache-type-v f16`
- GPU layers: `--n_gpu_layers 40`
- Batch/ubatch: `--batch-size 512 --ubatch-size 512`
- Threads: `-t 32`
- Context: `-c 512`
- KV mode: `--kv-unified`
- Chunk limit: `--chunks 1`
