# NVFP4 K-Cache Outlier Sidecar Design

## Goal

Add an experimental NVFP4 K-cache quantization path that extracts signed
`abs(K) > threshold` outliers before cache quantization, stores their original
F32 values and positions in sidecar tensors, quantizes the residual K values
with the existing per-token path, and adds the exact outlier dot-product
contribution back into KQ.

## Scope

The first implementation is intentionally narrow:

- CUDA NVFP4 K-cache only.
- Non-flash-attention KQ only.
- Default behavior unchanged.
- Outlier counting logs are controlled by a separate log switch.
- Fixed per-cache-row sidecar capacity, with real outlier counts still recorded
  even when the stored value/index arrays overflow.

Flash attention remains out of scope because it has a separate NVFP4 attention
implementation. The switch should not silently claim coverage there.

## Switches

Add these entries to `expt-switch-env.md`:

- `LLAMA_NVFP4_KCACHE_OUTLIER`: enables the outlier sidecar path. Default: off.
- `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD`: absolute-value threshold. Default:
  `16`.
- `LLAMA_NVFP4_KCACHE_OUTLIER_MAX`: maximum stored outliers per K-cache row.
  Default: `32`.
- `LLAMA_NVFP4_KCACHE_OUTLIER_LOG`: enables once-only summary logging and
  runtime count logging. Default: off.

The predicate is `abs(K) > threshold`. Stored values preserve sign.

## Data Layout

For every NVFP4 K-cache layer, when the feature is enabled:

- `k_outlier_count`: `I32`, shape `[kv_size * n_stream]`
- `k_outlier_index`: `I32`, shape `[max_outliers, kv_size * n_stream]`
- `k_outlier_value`: `F32`, shape `[max_outliers, kv_size * n_stream]`

The count tensor stores the true count for a row. The index/value tensors store
only the first `max_outliers` entries. KQ correction clamps
`min(count, max_outliers)`.

Outlier index is the flattened K dimension in the original K-cache row:

```text
global_dim = kv_head * head_dim + dim
```

## Write Path

`ggml_set_rows()` for NVFP4 K-cache receives the sidecar tensors through narrow
NVFP4 K-cache outlier metadata helpers.

When sidecars are present:

1. Reset `k_outlier_count[dst_row]` for each written row.
2. Compute per-row residual `amax`, ignoring values where `abs(K) > threshold`.
3. Extract outliers into sidecars with atomic row-local slots.
4. Quantize residual K using the current per-token NVFP4 quantization path, but
   treat outlier positions as zero.
5. Write the existing K `input_scale` from the residual `amax`.

When sidecars are absent, the current NVFP4 K-cache write path is unchanged.

## KQ Path

The KQ graph still builds:

```text
kq = ggml_mul_mat(k, q)
kq = kq * k_scale
```

The correction is applied inside the CUDA NVFP4 native matmul path after the
normal output has reached final KQ scale. For each output logit:

```text
kq[token, query] += sum(outlier_value[token, dim] * q_f32[query, dim])
```

For GQA, correction filters sidecar entries by the current KV head:

```text
head_begin = kv_head * head_dim
head_end   = head_begin + head_dim
```

Only entries in that range contribute to the current KQ slice.

## Metadata

The existing `ggml_tensor_set_nvfp4_scale()` helper uses the final `src[]`
metadata slot. The outlier implementation adds narrow helpers for three more
tail slots:

- count tensor
- index tensor
- value tensor

This is the only ggml-core-facing change and keeps the algorithm-specific
logic out of generic ops.

## Logging

With `LLAMA_NVFP4_KCACHE_OUTLIER_LOG=1`, log once during K-cache construction:

- enabled state
- threshold
- max stored outliers per row

Runtime logging should report aggregate counts from set_rows, including
overflow if any row count exceeds capacity. Logging must remain disabled by
default.

## Validation

Focused validation:

- CUDA unit test for extraction:
  - outliers are counted using `abs(K) > threshold`;
  - signed values and flattened positions are stored;
  - residual quantization sees outlier positions as zero.
- CUDA unit test for correction:
  - correction adds `outlier_value * Q_f32[dim]` to KQ;
  - GQA head filtering ignores outliers from other KV heads.

Build validation:

```bash
cmake --build build_cuda_release --target test-nvfp4-kcache-outlier -j 16
cmake --build build_cuda_release --target llama-cli -j 16
```

Runtime validation when CUDA is available:

```bash
CUDA_VISIBLE_DEVICES=0 build_cuda_release/bin/test-nvfp4-kcache-outlier
```

Optional smoke:

```bash
LLAMA_NVFP4_KCACHE_OUTLIER=1 \
LLAMA_NVFP4_KCACHE_OUTLIER_LOG=1 \
LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD=16 \
CUDA_VISIBLE_DEVICES=0 build_cuda_release/bin/llama-cli \
  -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
  -p "Write one short sentence about CUDA." \
  -n 16 -c 2048 -ngl 40 -ctk nvfp4 -ctv f16 --kv-unified --flash-attn 0
```
