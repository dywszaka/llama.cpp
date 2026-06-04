# NVFP4 K-cache Outlier Sidecar Review

This document describes the current `LLAMA_NVFP4_KCACHE_OUTLIER` implementation for code review. It focuses on the runtime data structures, graph plumbing, CUDA write path, KQ correction algorithm, and known limitations.

## Scope

The feature is an experimental CUDA path for `--cache-type-k nvfp4`. It stores most K-cache values in NVFP4, extracts large K values into a compact F32 sidecar, and adds their exact sparse contribution back into KQ.

Runtime enablement:

```text
LLAMA_NVFP4_KCACHE_OUTLIER=1
```

Optional hybrid mode:

```text
LLAMA_NVFP4_KCACHE_OUTLIER=1
LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8=1
```

When hybrid mode is enabled, selected high/medium layers use `GGML_TYPE_FP8_E4M3_E8M0_32` K-cache instead of NVFP4, so those layers do not allocate or use NVFP4 outlier sidecars.

Main files:

- `src/llama-kv-cache-nvfp4-outlier-config.h`
- `src/llama-kv-cache-unified.cpp`
- `src/llama-kv-cache-unified.h`
- `src/llama-graph.cpp`
- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-cuda/expt/nvfp4/kcache-outlier.cu`
- `ggml/src/ggml-cuda/expt/nvfp4/kcache-outlier.cuh`
- `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-set-rows.cu`
- `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-matmul.cu`

## High-level Algorithm

For each K row written to the KV cache:

```text
K_original = K_residual + K_outlier_sparse
```

The main K-cache tensor stores `K_residual` in NVFP4. Any element whose absolute value exceeds the configured per-layer threshold is treated as an outlier:

```text
if abs(K[row][dim]) > threshold:
    K_outlier_sparse[row][dim] = K[row][dim]
    K_residual[row][dim] = 0
else:
    K_outlier_sparse[row][dim] = 0
    K_residual[row][dim] = K[row][dim]
```

During KQ:

```text
K_original * Q
  = K_residual_nvfp4 * Q_nvfp4
  + K_outlier_sparse * Q_f32
```

The dense residual term uses the existing native NVFP4 cuBLASLt FP4 matmul path. The sparse outlier term is applied by a CUDA correction kernel after the dense matmul writes F32 KQ.

## Switches and Profiles

Switch definitions live in `src/llama-kv-cache-nvfp4-outlier-config.h` and are documented in `expt-switch-env.md`.

`LLAMA_NVFP4_KCACHE_OUTLIER`:

- Default: off.
- Enables compact NVFP4 K-cache outlier sidecars only when `type_k == GGML_TYPE_NVFP4`.
- Uses the balanced per-layer threshold and capacity profile when hybrid FP8 is not active.

`LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8`:

- Default: off.
- Maps to the fixed layer set `0,1,4,5,6,8,10,11,12,14,23,35`.
- On its own, it does not enable the sidecar. It only selects hybrid K-cache storage when the outlier switch is also enabled or when the compatibility layer switch is explicitly used.

Profiles:

- Balanced full-NVFP4 thresholds: `llama_nvfp4_kcache_outlier_layer_thresholds_balanced`.
- Balanced full-NVFP4 capacities: `llama_nvfp4_kcache_outlier_layer_capacities_balanced`.
- Hybrid thresholds: `llama_nvfp4_kcache_outlier_layer_thresholds_hybrid`, currently all `16`.
- Hybrid capacities: `llama_nvfp4_kcache_outlier_layer_capacities` for smaller contexts and `llama_nvfp4_kcache_outlier_layer_capacities_ctx8192` for `kv_size >= 8192`.

Threshold selection:

```text
threshold(layer) =
    hybrid ? hybrid_thresholds[layer] : balanced_thresholds[layer]
```

Capacity selection:

```text
capacity(layer, kv_size) =
    hybrid
        ? (kv_size >= 8192 ? ctx8192_capacity[layer] : ctx512_capacity[layer])
        : balanced_capacity[layer]
```

If a layer index is outside the configured table, the implementation falls back to a threshold of `16` and capacity `1`.

## KV-cache Data Structures

The main K-cache tensor remains:

```text
cache_k_l{layer}: type_k[n_embd_k_gqa, kv_size, n_stream]
```

For NVFP4 K-cache, the existing K scale sidecar remains:

```text
cache_k_gscale_l{layer}: F32[kv_size * n_stream]
```

Despite the name, the stored value is:

```text
k_scale[dst_row] = input_scale = 1 / global_scale
```

When outliers are enabled for a layer, the compact sidecar adds five tensors:

```text
cache_k_outlier_count_l{layer}:  I32[kv_size * n_stream]
cache_k_outlier_offset_l{layer}: I32[kv_size * n_stream]
cache_k_outlier_cursor_l{layer}: I32[n_stream]
cache_k_outlier_index_l{layer}:  I32[capacity, n_stream]
cache_k_outlier_value_l{layer}:  F32[capacity, n_stream]
```

Logical meaning:

```text
count[kv_pos]       = number of outliers detected for this K-cache row
offset[kv_pos]      = start offset in the compact pool, or -1 if not stored
cursor[stream]      = next append position in the stream-local compact pool
index[offset + i]   = outlier dimension inside the full K row
value[offset + i]   = original signed F32 K value
```

`index` is a row-local K dimension in `[0, n_embd_k_gqa)`, not a byte offset and not a flattened tensor index. It includes all KV heads for the layer:

```text
n_embd_k_gqa = n_head_kv * head_dim
```

The compact pools are stream-local through the second dimension of `index` and `value`.

## Allocation and Startup Logs

Allocation happens in `llama_kv_cache_unified::llama_kv_cache_unified()`.

The feature is considered active when:

```text
type_k == GGML_TYPE_NVFP4 && llama_nvfp4_kcache_outlier_enabled()
```

Per-layer sidecars are allocated only for layers whose actual K-cache type remains `GGML_TYPE_NVFP4`. In hybrid mode, layers mapped to FP8 do not allocate sidecars.

Startup logs currently include:

```text
NVFP4 K-cache compact outlier sidecar enabled:
  threshold_profile=...
  layer_capacity_profile=...
  layer_capacities=...
```

and:

```text
NVFP4 K-cache compact outlier sidecar size = X.XX MiB (N bytes)
```

The size log is the sum of `ggml_nbytes()` for `count`, `offset`, `cursor`, `index`, and `value` tensors actually created across included cache layers.

## Metadata Plumbing

The sidecars are attached to tensors through ggml metadata helpers that use reserved `tensor->src[]` slots:

```c
ggml_tensor_set_nvfp4_kcache_outliers_compact(tensor, counts, offsets, indices, values);
ggml_tensor_set_nvfp4_kcache_outlier_cursor(tensor, cursor);
```

The getters mirror this:

```c
ggml_tensor_get_nvfp4_kcache_outlier_counts(tensor)
ggml_tensor_get_nvfp4_kcache_outlier_offsets(tensor)
ggml_tensor_get_nvfp4_kcache_outlier_cursor(tensor)
ggml_tensor_get_nvfp4_kcache_outlier_indices(tensor)
ggml_tensor_get_nvfp4_kcache_outlier_values(tensor)
```

The metadata is attached in two places:

1. `cpy_k()` attaches full sidecar tensors to the `ggml_set_rows()` result, so CUDA set_rows can extract outliers while writing K.
2. `get_k()` attaches sidecar views to the K-cache read view, so KQ matmul can apply sparse correction.

`llm_graph_context::build_attn_mha()` permutes K before KQ. After permuting, it reattaches the sidecar metadata to the permuted K tensor so the CUDA matmul path can still see it.

## K Write Path

Graph entry:

```text
llama_kv_cache_unified::cpy_k()
  -> ggml_set_rows(ctx, k, k_cur, k_idxs)
```

For outlier-enabled layers, `cpy_k()` also stores the per-layer threshold in `res->op_params[0]`.

CUDA entry:

```text
ggml_cuda_set_rows_nvfp4_common()
```

If all outlier sidecar metadata is present, the write path calls:

```text
ggml_cuda_nvfp4_kcache_outlier_extract()
```

before quantizing the residual row into NVFP4.

### Extract Pass

`ggml_cuda_nvfp4_kcache_outlier_extract()` runs several kernels:

1. Reset touched destination rows:

```text
count[dst_row] = 0
offset[dst_row] = -1
```

2. Count outliers and compute residual amax:

```text
for dim in row:
    if abs(v) > threshold:
        count[dst_row] += 1
    else:
        residual_amax[src_row] = max(residual_amax[src_row], abs(v))
```

If the row has only outliers and no residual nonzero values, residual amax is set to `1.0` when the outlier count is positive. If the row has no values above threshold and no residual max, it is `0.0`.

3. Reserve compact pool space:

```text
offset = atomicAdd(cursor, count[dst_row])
offset[dst_row] =
    offset + count[dst_row] <= compact_capacity ? offset : -1
```

The cursor counts all requested entries, even if a row later fails to fit. Rows are stored atomically and only if the entire row fits.

4. Reset counts for fill:

```text
count[dst_row] = 0
```

5. Fill compact entries:

```text
if abs(v) > threshold && offset[dst_row] >= 0:
    slot = atomicAdd(count + dst_row, 1)
    entry = offset[dst_row] + slot
    index[entry] = dim
    value[entry] = v
```

After fill, `count[dst_row]` is the number of outliers actually stored for that row. If the row did not fit, `offset[dst_row] == -1` and `count[dst_row]` remains `0`.

### Residual NVFP4 Quantization

After extraction, `k_set_rows_nvfp4` or `k_set_rows_nvfp4_8` writes the main K-cache row. With outliers enabled:

```text
zero_outliers = true
use_threshold_global_scale = true
```

Each element is transformed before quantization:

```text
raw_xi = K_f32[row][dim]
xi = abs(raw_xi) > threshold ? 0.0f : raw_xi
```

The block-local `vmax` is computed on the residual values after zeroing outliers.

The global scale helper is:

```text
global_scale = 1344 / amax
```

where `1344 = 6 * 224`. In outlier mode for K, the amax is the layer threshold:

```text
global_scale_k = 1344 / threshold
k_scale[dst_row] = input_scale_k = threshold / 1344
```

Without outliers, the amax is the per-row residual amax:

```text
global_scale_k = 1344 / residual_amax[row]
k_scale[dst_row] = residual_amax[row] / 1344
```

## Q Quantization During KQ

KQ enters the native NVFP4 path when:

```text
src0->type == GGML_TYPE_NVFP4
src1->type == GGML_TYPE_F32
dst->type  == GGML_TYPE_F32
```

The CUDA dispatcher first tries V-cache-specific NVFP4 matmul, then the native NVFP4 matmul.

For K-cache outlier KQ, Q is still stored as F32 at graph level and is quantized dynamically inside `ggml_cuda_mul_mat_nvfp4_native_impl()`.

Normal dynamic Q scale mode uses per-row Q amax. When the K operand has outlier sidecar metadata and no static input scale is bound, the native path switches Q quantization to one dynamic per-tensor amax for the current Q matrix:

```text
amax_q_tensor = max(abs(Q))
global_scale_q = 1344 / amax_q_tensor
input_scale_q = out_scale / global_scale_q
```

`out_scale` is usually `1.0` for KQ. The outlier K threshold is not reused for Q.

This per-tensor Q scale is selected by `use_outlier_q_tensor_scale`.

## Dense FP4 Matmul

The dense residual term uses the existing cuBLASLt FP4 path:

```text
K_residual_nvfp4 * Q_nvfp4 -> F32 KQ
```

ggml `block_nvfp4` stores packed FP4 data and in-band E4M3 scale bytes together. cuBLASLt expects separate `CUDA_R_4F_E2M1` data and scale channels. The native path therefore splits both K and Q into:

```text
packed data channel
scale channel
```

K repacking can be cached because the K-cache is reused. Q is quantized and repacked at runtime.

After dynamic Q scale mode, the native path applies Q-side output scale compensation to KQ before outlier correction.

## Sparse Correction

After the dense matmul succeeds, native NVFP4 matmul calls:

```text
ggml_cuda_nvfp4_kcache_outlier_apply_correction()
```

if `apply_outlier_correction` is true and sidecar metadata exists.

For every `(kv_pos, q_pos)` output element, the kernel does:

```text
kv_head = q_head / (q_heads / kv_heads)
head_begin = kv_head * head_dim
head_end = head_begin + head_dim

corr = 0
for i in 0 .. count[kv_pos]-1:
    entry = offset[kv_pos] + i
    global_dim = index[entry]

    if head_begin <= global_dim < head_end:
        local_dim = global_dim - head_begin
        corr += value[entry] * Q_f32[q_pos][local_dim]

if k_scale exists and is finite and positive:
    corr /= k_scale[kv_pos]

KQ[kv_pos][q_pos] += corr
```

The GQA filter is important. The compact row stores outliers over the full K row across all KV heads, but each Q head only attends to one KV head. Correction therefore ignores outliers outside the current KV head's `[head_begin, head_end)` range.

The division by `k_scale[kv_pos]` compensates for the downstream graph multiply. The graph later multiplies the full KQ result by `k_scale`, so the sparse term is pre-divided to ensure the final output contains the exact F32 outlier contribution once:

```text
(dense + outlier / k_scale) * k_scale
  = dense * k_scale + outlier
```

## Batched and Stream Handling

The native NVFP4 implementation has a batched path for tensors with nontrivial `ne[2]` or `ne[3]`. It slices K, Q, and dst matrices and recursively runs dense native matmul with `apply_outlier_correction=false`; then it applies correction per slice using sidecar pointers offset by stream id.

For `n_stream > 1`, sidecar tensors are stream-separated:

- `count` and `offset` are sliced by stream through 4D views in `get_k()`.
- `cursor`, `index`, and `value` are sliced by stream through 1D/2D views.
- Stream copy copies K, V, K scale, and all outlier sidecar stream views.

## KV Movement, State, and Lifecycle

`clear(true)` clears backend buffers, including sidecar tensors and cursors.

Stream copy:

- Copies all sidecar tensors for the source stream to the destination stream.
- This preserves compact pool entries because the stream-local `index/value` pool and `cursor` are copied together.

Defrag:

- Copies K/V and K scale for moved cell ranges.
- Copies `count` and `offset` for moved K rows.
- Does not compact or rewrite `index/value/cursor` pools.
- This can preserve correctness as long as copied `offset` values still point to the original pool entries, but it does not reclaim pool space and does not make moved rows own new compact segments.

State export/import:

- Disabled when `nvfp4_kcache_outlier` is active.
- `state_write()` and `state_read()` throw runtime errors for this mode.

## Overflow Behavior

The compact pool has fixed per-layer capacity. Allocation is row-atomic:

- If an entire row's outliers fit, the row gets an offset and all entries are stored.
- If the row does not fit, `offset[row] = -1` and no entries are stored for that row.

After fill:

```text
offset[row] < 0 => correction skips row
count[row] == stored outlier count
```

In debug builds, count logging can print detailed statistics. In release builds, overflow logging is limited to an overflow warning path when stream copying for logs is allowed.

## Correctness Invariants

Sidecar metadata is only valid when all compact tensors are present:

```text
counts != null
offsets != null
cursor != null
indices != null
values != null
```

Type invariants:

```text
counts: I32
offsets: I32
cursor: I32
indices: I32
values: F32
```

Shape invariants:

```text
counts.ne[0] == kv_size * n_stream, or a stream/n_kv view of it
offsets.ne[0] == kv_size * n_stream, or a stream/n_kv view of it
cursor.ne[0] == n_stream, or a stream view
indices.ne[0] == compact_capacity
values.ne[0] == compact_capacity
```

K write invariants:

- `ggml_set_rows` destination rows must be valid cache positions.
- The outlier threshold must be positive; otherwise CUDA falls back to the default `16`.
- Residual NVFP4 quantization must zero the same elements that extraction classified as outliers.

KQ invariants:

- Correction must run after dense matmul and after dynamic Q scale compensation.
- Correction must use original F32 Q, not quantized Q.
- Correction must filter entries by GQA KV head.
- Correction must pre-divide by K graph scale when K scale is present.

## Tests

Focused CUDA coverage lives in `tests/test-nvfp4-kcache-outlier.cu`.

It checks:

- compact offset assignment and pool fill;
- full-row capacity behavior when the pool is too small;
- K scale helper behavior for row amax vs threshold amax;
- Q tensor amax helper behavior;
- correction filtering by head;
- correction filtering by GQA KV head;
- K scale compensation in correction.

Smoke coverage:

- `tests/test-kcache-nvfp4-default-no-outlier-smoke.sh` checks the sidecar is off by default.
- `tests/test-kcache-outlier-hybrid-b-switch-smoke.sh` checks hybrid switch B alone does not enable the sidecar.
- `tests/test-kcache-hybrid-outlier-layer-capacity-smoke.sh` checks hybrid sidecar startup logs, K-cache type string, and the sidecar memory-size log.

## Known Review Points

1. Compact pool capacity is static and profile-derived. It is not adaptive to prompt distribution, model changes, or long-running row rewrites.
2. Cursor is monotonic within a stream until the KV buffer is cleared. Row rewrites do not reclaim old compact segments.
3. Defrag copies `count/offset` but does not compact or copy sidecar pool entries for individual rows. This avoids rewriting sparse pools but leaves pool fragmentation and should be reviewed for all defrag scenarios.
4. State import/export is intentionally unsupported while this feature is active.
5. The sidecar uses `tensor->src[]` metadata slots near `GGML_MAX_SRC`. This is narrow but implicit; future metadata users must avoid slot collisions.
6. Correction currently belongs to the native NVFP4 CUDA matmul path. If execution falls back to another KQ path, review whether sparse correction is still applied or whether the path must be disallowed for correctness-sensitive runs.
7. Hybrid FP8 layers bypass the NVFP4 sidecar entirely. Any review of aggregate memory or quality needs to account for which layers are NVFP4 residual+sidecar and which are FP8.

