# Experiment Switch Environment Variables

## CUDA ADD QEMU Offload

### `GGML_CUDA_ADD_QEMU_MODE`

Selects `cuda|qemu|qemu_cuda|compare` for `GGML_OP_ADD`. The build option is
`-DGGML_CUDA_ADD_QEMU=ON`; unset defaults to `cuda`, preserving the original
llama.cpp CUDA ADD implementation. CUDA preprocessing expands ggml's complete
four-dimensional `src1` repeat/broadcast mapping and converts both F16/F32
operands to dense canonical BF16 using RZ/high-16-bit truncation.

The replacement numerical model is the `call_add_fp32` contract: both BF16
inputs are widened exactly to FP32, added with FP32 round-to-nearest semantics,
and narrowed once to BF16 with RNE. `qemu_cuda` is device-only. `compare` runs
the original CUDA ADD plus QEMU/RVV and qemu_cuda, compares QEMU and qemu_cuda
as raw BF16 bits, and keeps the original CUDA result downstream. All non-CUDA
modes disable CUDA graph capture for graphs containing ADD.

### `GGML_CUDA_ADD_QEMU_ENDPOINT`

Daemon endpoint, default `tcp://127.0.0.1:15586`.

### `GGML_CUDA_ADD_QEMU_TIMEOUT_MS`

ZMQ timeout in milliseconds, default `300000`.

### `GGML_CUDA_ADD_QEMU_ARTIFACT`

Compare JSONL path, default `experiments/add-qemu-compare.jsonl`. It records
llama-vs-QEMU MSE/RMSE/max error and QEMU-vs-qemu_cuda BF16 mismatch counts.

### `GGML_CUDA_ADD_QEMU_MISMATCH_LOG`

Mismatch-only JSONL path, default `experiments/add-qemu-cuda-mismatch.jsonl`.
It records both complete canonical inputs and both complete BF16 outputs.

### `GGML_CUDA_ADD_QEMU_TIMING`

Enables diagnostic ADD timing logs. Default: unset/off. Timing synchronizes the
measured call and therefore changes performance behavior.

## CUDA MUL QEMU Offload

### `GGML_CUDA_MUL_QEMU_MODE`

Selects `cuda|qemu|qemu_cuda|compare` for `GGML_OP_MUL`. The build option is
`-DGGML_CUDA_MUL_QEMU=ON`; unset defaults to `cuda`. CUDA preprocessing expands
the complete ggml four-dimensional broadcast mapping and converts both F16/F32
operands to dense BF16 with RZ/high-16-bit truncation. `qemu_cuda` is
device-only. `compare` executes the original CUDA result plus both BF16 models
and keeps the original result downstream. Non-CUDA modes disable graph capture
and RMS_NORM/MUL fusion.

### `GGML_CUDA_MUL_QEMU_ENDPOINT`

Daemon endpoint, default `tcp://127.0.0.1:15582`.

### `GGML_CUDA_MUL_QEMU_TIMEOUT_MS`

ZMQ timeout in milliseconds, default `300000`.

### `GGML_CUDA_MUL_QEMU_ARTIFACT`

Compare JSONL path, default `experiments/mul-qemu-compare.jsonl`. It records
llama-vs-QEMU MSE/RMSE/max error and QEMU-vs-qemu_cuda BF16 mismatch details.

### `GGML_CUDA_MUL_QEMU_MISMATCH_LOG`

Mismatch-only JSONL path, default `experiments/mul-qemu-cuda-mismatch.jsonl`.
Records both complete canonical inputs and both complete BF16 outputs.

### `GGML_CUDA_MUL_QEMU_TIMING`

Enables diagnostic `MUL_QEMU_TIMING` per-call total timing. It synchronizes the
calling stream and therefore changes performance behavior.

## CUDA SOFT_MAX QEMU Offload

### `GGML_CUDA_SOFT_MAX_QEMU_MODE`

Selects the experimental CUDA/QEMU SOFT_MAX runtime mode. The build must use
`-DGGML_CUDA_SOFTMAX_QEMU=ON`. Default: unset or
`cuda`, preserving the existing CUDA-only path and producing no comparison
artifact.

Supported values:

- `cuda`: run only the existing CUDA SOFT_MAX implementation and use the CUDA
  result.
- `qemu`: apply scale, mask, and ALiBi on CUDA, truncate the effective logits to
  canonical BF16, then run `call_softmax_fp32`: max/subtract/reduction and
  normalization use FP32 RVV while exp uses `ni900_exp_f16m8`. The BF16 result
  is converted to F32 on CUDA and used downstream.
- `qemu_cuda`: run the same FP32-compute/NI900-Exp numerical model entirely on
  the existing CUDA device tensors and use its F32-converted result. This mode does not
  create a ZMQ socket and performs no D2H or H2D transfer.
- `compare`: run the original llama.cpp CUDA softmax, QEMU/RVV BF16 softmax,
  and qemu_cuda concurrently. The original llama.cpp CUDA result is used
  downstream. The artifact records llama-vs-QEMU MSE/RMSE/max error and the
  QEMU-vs-qemu_cuda BF16 bit mismatch count.

`compare_cuda` and `compare_qemu` are accepted as compatibility aliases for
`compare`; both now keep the original llama.cpp CUDA result downstream.

Unknown values fall back to `cuda`. Before the FP32/NI900 softmax, the CUDA
preprocess supports the complete forward protocol described in
`docs/development/cuda-softmax-io-protocol.md`, including F16/F32 masks, mask
strides, ALiBi, scale, and attention sinks. QEMU and qemu_cuda receive identical
BF16 effective logits and BF16 sinks.

Dual-run modes disable CUDA graph capture for graphs containing SOFT_MAX because
the experiment performs host IO, synchronization, and artifact writes. Single
run modes do not generate comparison artifacts.

### `GGML_CUDA_SOFT_MAX_QEMU_ENDPOINT`

ZMQ endpoint for the `call_softmax_fp32` daemon. Default:
`tcp://127.0.0.1:15584`.

### `GGML_CUDA_SOFT_MAX_QEMU_TIMEOUT_MS`

ZMQ send/receive timeout in milliseconds. Default: `300000`.

### `GGML_CUDA_SOFT_MAX_QEMU_ARTIFACT`

Overrides the append-only JSONL artifact path used by `compare`. Default:
`experiments/softmax-qemu-compare.jsonl` relative to
the process working directory. Parent directories are created when possible.
Each line records llama-vs-QEMU `mse`, `rmse`, and `max_abs`, the
QEMU-vs-qemu_cuda BF16 mismatch count, all three softmax timings, element count,
request id, destination tensor name, and ggml operator descriptor.

### `GGML_CUDA_SOFT_MAX_QEMU_MISMATCH_LOG`

Overrides the separate append-only JSONL diagnostic used only when QEMU/RVV and
qemu_cuda BF16 outputs are not bit-exact. Default:
`experiments/softmax-qemu-cuda-mismatch.jsonl`. Each mismatch record contains
the full effective BF16 input, optional BF16 sinks, both full BF16 outputs,
shape, mismatch count, and first mismatch index. BF16 values are written as
four-digit hexadecimal bit patterns.

### `GGML_CUDA_SOFT_MAX_QEMU_TIMING`

Enables per-call llama.cpp timing logs. Default: unset/off. The workspace
`run.sh` enables it by default. In pure `qemu_cuda`, CUDA timing events are
created only when this switch is enabled; enabling it synchronizes the timed
call and therefore changes performance behavior.

Each `RVV_SOFTMAX_TIMING` line records the request id, runtime mode, destination
tensor, element count, CUDA D2H time, ZMQ round-trip time, daemon request time,
result-copy time, and total offload time. Set it to `0`, `false`, or `off` to
disable timing logs. `QEMU_CUDA_SOFTMAX_TIMING` records the device-only BF16
preprocess, deterministic softmax, and BF16-to-F32 total duration in
`qemu_cuda` and `compare`, together with cumulative call count and average
duration.
In `compare`, `LLAMA_CUDA_SOFTMAX_TIMING` also records the original llama.cpp
CUDA kernel time.

## CUDA RMS_NORM QEMU Offload

### `GGML_CUDA_RMS_NORM_QEMU_MODE`

Selects the experimental CUDA/QEMU RMS_NORM runtime mode. The build must use
`-DGGML_CUDA_RMS_NORM_QEMU=ON` for modes that contact QEMU. Default: unset or
`cuda`, preserving the existing CUDA implementation.

Supported values:

- `cuda`: use the original F32 CUDA RMS_NORM.
- `qemu`: pack the strided F32 input into dense BF16 on CUDA, send BF16 to the
  QEMU/RVV daemon, convert returned BF16 to F32 on CUDA, and use that result.
- `qemu_cuda`: perform F32-to-BF16 packing, the FP32-compute/BF16-I/O RMS_NORM
  model, and BF16-to-F32 conversion entirely on the current CUDA device. This
  mode creates no ZMQ socket and performs no D2H/H2D transfer. Its 32-lane FP32
  FMA, lane-ordered FP32 reduction, sqrt, reciprocal, and scaling model is
  bit-exact with `call_rms_norm_fp32` on the NI900 RVV implementation.
- `compare`: run original CUDA, QEMU/RVV, and qemu_cuda concurrently and keep
  the original CUDA result downstream. Comparison artifacts include
  llama-vs-QEMU errors and QEMU-vs-qemu_cuda numerical and BF16-bit metrics;
  the expected QEMU-vs-qemu_cuda bit mismatch count is zero.

RMS_NORM follows the project-wide canonical input rule required for all new
QEMU/RVV operators: F32-to-BF16 packing uses RZ/truncation by taking the raw
high 16 bits of each F32 value (`bf16_bits = f32_bits >> 16`). This input-boundary
rule is followed by an exact BF16-to-FP32 expansion. `eps`, `1/ncols`, the
reduction, sqrt, reciprocal, and scaling remain FP32. Only the canonical output
conversion rounds FP32 to BF16 using RNE.

`compare_cuda` and `compare_qemu` are accepted as aliases for `compare`.
Unknown values fall back to `cuda`. All non-`cuda` modes disable RMS_NORM/MUL
fusion and CUDA graph capture so the experimental hook cannot be bypassed.

### `GGML_CUDA_RMS_NORM_QEMU_ENDPOINT`

ZMQ endpoint for the FP32-compute RMS_NORM daemon. Default:
`tcp://127.0.0.1:15583`.

### `GGML_CUDA_RMS_NORM_QEMU_TIMEOUT_MS`

ZMQ send/receive timeout in milliseconds. Default: `300000`.

### `GGML_CUDA_RMS_NORM_QEMU_ARTIFACT`

Overrides the append-only JSONL artifact used by `compare`. Default:
`experiments/rms-norm-qemu-compare.jsonl`.

### `GGML_CUDA_RMS_NORM_QEMU_MISMATCH_LOG`

Overrides the QEMU/qemu_cuda BF16 mismatch JSONL. Default:
`experiments/rms-norm-qemu-cuda-mismatch.jsonl`. Full BF16 input and both BF16
outputs are written only when raw output bits differ.

### `GGML_CUDA_RMS_NORM_QEMU_TIMING`

Enables per-call timing logs. Default: unset/off. `RVV_RMS_NORM_TIMING` records
D2H, RPC, daemon, return-copy, and total offload time. In `qemu_cuda`,
`QEMU_CUDA_RMS_NORM_TIMING` records F32-to-BF16 preprocess, FP32-compute
RMS_NORM with BF16 I/O, BF16-to-F32 conversion, total duration, cumulative call
count, and average duration. Timing events are created only when enabled and
synchronize the timed call. `LLAMA_CUDA_RMS_NORM_TIMING` records original CUDA
time in `compare`.

## CUDA ROPE QEMU Offload

### `GGML_CUDA_ROPE_QEMU_MODE`

Selects the experimental static-table RoPE path. Default: unset or `cuda`, so
the upstream CUDA implementation is unchanged. QEMU-contacting modes require a
build configured with `-DGGML_CUDA_ROPE_QEMU=ON`.

Supported values:

- `cuda`: original llama.cpp CUDA RoPE.
- `qemu`: truncate the native source to canonical BF16 on CUDA, send BF16 plus
  I32 positions to the resident RVV service, convert returned BF16 to the native
  destination type on CUDA, and use that result downstream.
- `qemu_cuda`: use the same BF16-RZ input, F32 lookup-table multiply/FMA model,
  BF16-RNE output and native output conversion on CUDA. The immutable 4 MiB F32
  table is loaded to each CUDA device once; subsequent calls do not use ZMQ,
  D2H, or H2D.
- `compare`: run original CUDA, QEMU/RVV, and qemu_cuda; record original-CUDA
  error and QEMU/qemu_cuda BF16 mismatch counts; keep original CUDA downstream.

`compare_cuda` and `compare_qemu` are accepted as aliases for `compare`.
Unknown values fall back to `cuda`. The experiment only intercepts forward
GPT-NeoX RoPE nodes that exactly match the table parameters documented in
`docs/development/cuda-rope-qemu-rpc.md`; every other RoPE node safely falls
back to the original CUDA path. Non-`cuda` modes disable CUDA graph capture for
graphs containing RoPE.

### `GGML_CUDA_ROPE_QEMU_ENDPOINT`

ZMQ endpoint for the ROPE daemon. Default: `tcp://127.0.0.1:15587`.

### `GGML_CUDA_ROPE_QEMU_TIMEOUT_MS`

ZMQ send/receive timeout in milliseconds. Default: `300000`.

### `GGML_CUDA_ROPE_QEMU_TABLE`

Path to the exact 4 MiB F32 table described by
`rope-cos-sin-manifest.json`. The daemon loads it into globalram at startup;
qemu_cuda loads it once per CUDA device. Default:
`/home/lerong.chen/0729-rope-node4/rope-cos-sin-f32.bin`.

### `GGML_CUDA_ROPE_QEMU_ARTIFACT`

Append-only compare JSONL. Default: `experiments/rope-qemu-compare.jsonl`.

### `GGML_CUDA_ROPE_QEMU_MISMATCH_LOG`

QEMU/qemu_cuda mismatch JSONL. Full canonical input, positions, and both BF16
outputs are written only when bits differ. Default:
`experiments/rope-qemu-cuda-mismatch.jsonl`.

### `GGML_CUDA_ROPE_QEMU_TIMING`

Enables diagnostic per-call timing. Default: unset/off. `RVV_ROPE_TIMING`
records D2H, RPC, daemon, and return-copy time. `QEMU_CUDA_ROPE_TIMING` records
preprocess, table operator, output conversion, and total CUDA time. Timing uses
CUDA event synchronization and changes performance behavior.

## CUDA SWIGLU QEMU Offload

### `GGML_CUDA_GLU_QEMU_MODE`

Selects `cuda|qemu|qemu_cuda|compare` for `GGML_GLU_OP_SWIGLU`. The build
option is `-DGGML_CUDA_GLU_QEMU=ON`; unset defaults to `cuda`. Other GLU
variants keep their original CUDA implementation. Both split-input and
two-tensor SWIGLU layouts are packed on CUDA into identical dense BF16 x/gate
inputs using RZ truncation.

The `call_glu_fp32` numerical model widens x and gate to FP32. Negated x is
narrowed once to the BF16 input boundary of `ni900_exp_f16m8`; the exp result is
widened immediately, and FP32 RVV performs `x / (1 + exp(-x)) * gate` before a
single BF16 RNE output conversion. `qemu_cuda` mirrors the NI900 exp model on
device. `compare` retains the original llama.cpp CUDA output downstream and
records llama-vs-QEMU error plus QEMU-vs-qemu_cuda BF16 mismatches. Non-CUDA
modes disable CUDA graph capture for intercepted SWIGLU nodes.

### `GGML_CUDA_GLU_QEMU_ENDPOINT`

Daemon endpoint, default `tcp://127.0.0.1:15588`.

### `GGML_CUDA_GLU_QEMU_TIMEOUT_MS`

ZMQ timeout in milliseconds, default `300000`.

### `GGML_CUDA_GLU_QEMU_ARTIFACT`

Compare JSONL path, default `experiments/glu-qemu-compare.jsonl`.

### `GGML_CUDA_GLU_QEMU_MISMATCH_LOG`

Mismatch-only JSONL path, default `experiments/glu-qemu-cuda-mismatch.jsonl`.
It records both complete canonical inputs and both BF16 outputs.

### `GGML_CUDA_GLU_QEMU_TIMING`

Enables diagnostic SWIGLU timing logs. Default: unset/off. Timing synchronizes
the measured call and therefore changes performance behavior.

## Tensor Export and Offline Quantization Evaluation

### `LLAMA_EXPT_NVFP4_K_OFFLINE_CHANNEL_ORDER`

Enables an experimental NVFP4 K-cache channel-order runtime path from an
offline-generated JSON order file. Default: unset/off.

When set to a `kcur-mean-sort.raw.json`-style artifact, the runtime validates
that the file contains 36 `Kcur-<layer>` records and that each `channel_order`
is a 128-element permutation. Initial scope: Qwen3 8B, unified KV cache,
non-flash attention, and `--cache-type-k nvfp4`.

For semantic correctness, the path stores K cache rows in the per-layer offline
channel order and applies the same per-layer channel order to Q before the KQ
dot product. This changes NVFP4 K-cache quantization/block grouping while
preserving matched K/Q coordinates for attention. The switch logs once when
enabled and aborts clearly for unsupported graph paths instead of reporting an
invalid PPL.

### `LLAMA_EXPT_TENSOR_EXPORT_DIR`

Enables experimental runtime export of selected computed F32 graph tensors for
offline quantization evaluation. Default: unset/off.

When set to a non-empty output directory, the decode graph export pass scans
completed graph nodes and writes supported F32 tensors whose names map to K, Q,
V, KQ, or KQV records. It creates the directory when needed, writes one
contiguous raw `.bin` file per tensor, and writes `manifest.json` containing
`name`, `kind`, `dtype`, `ne`, `nb`, `path`, and `byte_size`. Unsupported dtypes
are skipped with a warning so normal inference does not fail solely because
export is enabled. Before graph allocation, matching tensors are marked as graph
outputs so their backend storage remains valid until the post-compute export and
cannot be reused by later nodes. The export hook is narrow and does not change
inference math; when this switch is unset or empty, no tensors are retained or
written.

### `LLAMA_EXPT_TENSOR_EXPORT_KINDS`

Comma-separated tensor kinds to export. Default: `k,q,v,kq,kqv`.

Supported values are `k`, `q`, `v`, `kq`, and `kqv`. This switch only filters
which recognized graph tensor names are exported; it does not enable export
without `LLAMA_EXPT_TENSOR_EXPORT_DIR`.

### `LLAMA_EXPT_TENSOR_EXPORT_OP`

Selects op-oriented tensor export mode. Default: unset/off.

When set together with `LLAMA_EXPT_TENSOR_EXPORT_DIR`, the export pass matches
all graph nodes whose `ggml_op_name()` equals this value, ignoring case and an
optional `GGML_OP_` prefix. For the first graph selected by
`LLAMA_EXPT_TENSOR_EXPORT_TYPE`, it writes each matching node's `dst` and
populated `dst->src[0..2]` tensors as raw binary spans and records their role,
dtype, shape, strides, contiguity, and view offset in `manifest.json`. For
`SOFT_MAX`, the dst record also stores `op_params.scale` and
`op_params.max_bias`, while `src2` captures optional attention sinks. For
`ROPE`, the dst record stores the complete RoPE parameters needed to interpret
the inputs and validate the output, including `n_dims`, `mode`, `n_ctx_orig`,
the frequency/scaling values, and multi-RoPE sections. This mode
marks the matching `dst` and populated `src0` through `src2` storage as graph
outputs before allocation. During the selected execution, the scheduler also
stops immediately after each matching node, synchronizes its backend, and
copies those tensors into host snapshots before later nodes can overwrite
aliased storage.
The v2 manifest records
`snapshot_timing: source_producer_and_node_completion`: graph-produced inputs
are copied when their producer finishes so in-place selected ops cannot
overwrite them, while the selected dst and any remaining inputs are copied when
the selected node finishes. Existing user eval callbacks are chained through
the export observer. Retention is applied when the graph is built even if a
compatible prefill graph is later reused for the selected decode execution.
This mode does not use
`LLAMA_EXPT_TENSOR_EXPORT_KINDS`. If
`LLAMA_EXPT_TENSOR_EXPORT_NAME` is also set, tensor-name selection takes
priority and this op value is recorded but ignored for node selection.

### `LLAMA_EXPT_TENSOR_EXPORT_NAME`

Selects a graph tensor by name for op-oriented export. Default: unset/off.

Tensor-name selection has higher priority than `LLAMA_EXPT_TENSOR_EXPORT_OP`.
When a layer is selected, a base name is resolved with the layer suffix; for
example, `LLAMA_EXPT_TENSOR_EXPORT_NAME=kq` and
`LLAMA_EXPT_TENSOR_EXPORT_LAYER=0` select exactly `kq-0`. A name that already
contains a trailing layer suffix must agree with `LLAMA_EXPT_TENSOR_EXPORT_LAYER`.
Without a layer, a base name such as `kq` matches `kq-<layer>` tensors, while an
explicit name such as `kq-0` remains an exact match.

For a selected `NVFP4 x F32 -> F32` `MUL_MAT`, graph construction allocates
graph-owned output sidecars for the effective NVFP4 RHS and its associated
scale. The native CUDA path writes directly into these sidecars. For the
cuBLASLt variant, the v2 manifest exports the original NVFP4 A tensor, A's raw
inverse-global scale and canonical global scale, the original F32 B tensor, the
effective NVFP4 B tensor, and B's canonical global scale. For the FP4MULMAT
variant, it instead exports only one scale record named `matmul_scale`: the
original FP32 final multiplier supplied to the accumulator output multiply.
The manifest records the BF16-RNE operand rounding performed by FP4MULMAT, but
the exported scale itself retains its FP32 low bits.
The separate A and B global-scale records are omitted in that variant. If the
native path is not used, the manifest records `native_nvfp4_not_used` and does
not claim an effective B source. The capture sidecars are created only for the
selected name/type/layer, so a precise name such as `kq-0` avoids retaining
every MUL_MAT input in a layer.

### `LLAMA_EXPT_TENSOR_EXPORT_TYPE`

Selects which first graph is captured by op-oriented export. Default: `decode`.

Supported values are `decode` and `prefill`. `decode` captures the first
single-token-per-sequence graph after the initial prompt. `prefill` captures the
first prompt graph, including a one-token prompt at position zero. This switch
has an effect only when
`LLAMA_EXPT_TENSOR_EXPORT_OP` and `LLAMA_EXPT_TENSOR_EXPORT_DIR` are both set.

### `LLAMA_EXPT_TENSOR_EXPORT_LAYER`

Restricts op-oriented export to a zero-based model layer. Default: unset, which
keeps all matching op nodes.

The exporter matches the layer suffixes used by graph tensor names, including
forms such as `norm-0`, `blk.0.*`, and `cache_k_l0`. This switch has an effect
only in op-oriented export mode. The selected layer is recorded in the op
manifest.

### `LLAMA_EXPT_TENSOR_EXPORT_BF16_DUMP`

Exports F32 tensor values as raw BF16 storage for tensor-export experiments.
Default: unset/off.

When enabled, every F32 tensor record selected by tensor export is written as
BF16 by truncating the low 16 bits of each F32 bit pattern (`bf16_bits =
f32_bits >> 16`). The written file uses compact contiguous BF16 layout, and the
manifest records `dtype: "bf16"`, updated strides and byte size, plus original
F32 dtype/stride metadata for op-oriented exports. Non-F32 tensors are exported
unchanged. The helper script `mylab/tensor-export-eval/export.sh` exposes this
as `BF16_DUMP=1`.

## FP8 E4M3 E8M0 32 K-Cache

### `--cache-type-k fp8_e4m3_e8m0_32`

Enables the experimental FP8(E4M3+E8M0 block32) K-cache path. Default: off
because the standard K-cache type default remains `f16`.

Initial scope: CUDA non-flash KQ with KQV offload enabled. During KQ, the stored
K cache is `GGML_TYPE_FP8_E4M3_E8M0_32`; the F32 Q operand is quantized to
temporary FP8 block32 inside the native CUDA FP8 matmul path before cuBLASLt
execution.

### `LLAMA_KCACHE_HYBRID_FP8_E4M3_E8M0_32_LAYERS`

Enables experimental per-layer hybrid K-cache storage where selected K-cache
layers use `GGML_TYPE_FP8_E4M3_E8M0_32` while the remaining layers keep the
configured `--cache-type-k`. Default: unset/off.

Compatibility alias for the newer B switch
`LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8=1`.

Supported values:

```text
high_medium
0,1,4,5,6,8,10,11,12,14,23,35
```

The `high_medium` alias maps to the stable threshold-16 NVFP4 K-cache outlier
high and medium layers observed in the prompt-consistency experiment:
`0,1,4,5,6,8,10,11,12,14,23,35`. When enabled, the selected layer K tensors are
allocated as FP8(E4M3+E8M0 block32), so CUDA `set_rows` quantization and KQ
matmul dispatch use the existing FP8 K-cache paths for those layers. This
hybrid switch only takes effect when `--cache-type-k nvfp4`; other K-cache
types ignore it. This hybrid mode inherits the FP8 K-cache runtime limits:
flash attention is not supported, and KQ/V offload must be enabled.

## NVFP4 CUDA Native Matmul

### `GGML_CUDA_NVFP4_BF16_QUANT`

Parent switch for the experimental BF16 trunc-NN NVFP4 RHS activation quantizer
in the CUDA native NVFP4 matmul path. Default: off.

This switch only changes behavior together with
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1`. When disabled, or when
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN` is disabled, the existing FP32
nearest-neighbor NVFP4 quantizer remains active unless
`GGML_CUDA_NVFP4_TRUNC_BF16_INPUT=1` selects the separate trunc-input FP32
nearest-neighbor path.

Initial scope: `GGML_TYPE_NVFP4 x GGML_TYPE_F32 -> GGML_TYPE_F32` CUDA native
matmul activation quantization, including static input-scale and dynamic
per-row/per-tensor RHS scale modes. This switch does not change stored NVFP4
weights or generic dequantization paths.

### `GGML_CUDA_NVFP4_NATIVE_ROW_SPLIT`

Diagnostic switch for native CUDA NVFP4 matmul. Default: unset/off.

When enabled, native NVFP4 matmul runs each RHS token column separately through
cuBLASLt instead of one batched `N` dimension call when `N > 1`. This is useful
for isolating whether ubatch-dependent GEMM shape changes affect upstream F32
activations before K-cache outlier extraction. It should not be used for
performance measurements.

### `GGML_CUDA_NVFP4_FP4MULMAT`

Enables the experimental fp4_mulmat-derived CUDA NVFP4 matmul model path.
Default: unset/off.

When enabled, the native CUDA NVFP4 matmul path still quantizes F32 RHS
activations through the current NVFP4 activation quantizer, then evaluates the
NVFP4 block dot product with the experimental FP4 accumulator model instead of
cuBLASLt. This is intended for hardware-model comparison and correctness
experiments, not performance measurement.

The accumulator writeback follows the `call_mul_fp32` model: the accumulator
and column-scale operands are rounded to BF16 with RNE and exactly widened to
FP32, multiplied in FP32, then rounded to BF16 with RNE before being stored in
the F32 destination.

For static input scales this multiplier is
`weight_scale_2 / global_scale`; for dynamic input scales it is computed per RHS
row. NVFP4 tensor export captures the original FP32 final multiplier as the single
exported scale for the FP4MULMAT variant, rather than exporting the separate
weight/input global-scale components.

The path logs once when selected. Combine with
`GGML_CUDA_NVFP4_FP4MULMAT_LOG=1` to log the first several selections during a
run.

### `GGML_CUDA_NVFP4_FP4MULMAT_LOG`

Diagnostic logging switch for `GGML_CUDA_NVFP4_FP4MULMAT`. Default:
unset/off.

When enabled, prints selection logs for the first several fp4_mulmat-derived
NVFP4 matmul calls instead of only the first call. It does not enable the
fp4_mulmat path by itself.

### `GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN`

Enables an experimental hardware-friendly BF16-input NVFP4 RHS activation
quantizer that more closely follows `GGML_CUDA_NVFP4_TRUNC_BF16_INPUT=1`.
Default: off.

When enabled with `GGML_CUDA_NVFP4_BF16_QUANT=1`, the CUDA quantizer truncates
each F32 RHS activation value to the BF16 value range by clearing the lower 16
bits of the FP32 representation, then performs the internal block-scale and
E2M1 magnitude-selection arithmetic with FP32 multiply/add/compare operations.
The E4M3 block scale is generated from FP32 bit fields, and FP4 magnitudes are
selected with nearest-neighbor thresholds whose exact ties choose the lower code,
matching the existing table-search nearest-neighbor behavior.

The implementation avoids runtime division, FP8 conversion intrinsics,
lookup-table nearest-neighbor searches, and special math functions in the BF16
quantization inner path. Dynamic RHS scale discovery uses the same BF16-truncated
values for `amax`.

### `GGML_CUDA_NVFP4_BF16_QUANT_BF16_INTERNAL`

Enables an experimental BF16-precision internal arithmetic variant of the BF16
trunc-NN NVFP4 RHS activation quantizer. Default: off.

This switch only changes behavior when both `GGML_CUDA_NVFP4_BF16_QUANT=1` and
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1` are also enabled. The activation input
is still truncated to BF16 bits, and block maximum search still compares BF16
absolute-value bits. With this switch enabled, the target and nearest-neighbor
threshold multiply/add operations used for E2M1 magnitude selection are
truncated to the BF16 value range before comparison. The default blockscale
calculation still uses the FP32 tensor/global scale arithmetic unless
`GGML_CUDA_NVFP4_BF16_QUANT_BF16_BLOCK_SCALE=1` is also enabled.

### `GGML_CUDA_NVFP4_BF16_QUANT_BF16_BLOCK_SCALE`

Enables an experimental BF16-precision blockscale calculation for the BF16
trunc-NN NVFP4 RHS activation quantizer. Default: off.

This switch only changes behavior when `GGML_CUDA_NVFP4_BF16_QUANT=1`,
`GGML_CUDA_NVFP4_BF16_QUANT_TRUNC_NN=1`, and
`GGML_CUDA_NVFP4_BF16_QUANT_BF16_INTERNAL=1` are also enabled. The activation
input is truncated to BF16 bits and group maximum search compares BF16
absolute-value bits. In addition, the tensor/global scale operand is truncated
to the BF16 value range and blockscale multiply operations are truncated after
each multiply. This models a lower-cost hardware path where both blockscale and
quantized-value selection use BF16-like arithmetic while preserving the
hardware-friendly bit-field E4M3 scale encoder.

### `GGML_CUDA_NVFP4_TRUNC_BF16_INPUT`

Enables an experimental pre-quantization truncation step in the CUDA native NVFP4
FP32 nearest-neighbor RHS activation quantizer. Default: off.

When enabled and the BF16 trunc-NN path is disabled, the FP32 RHS activation
value is first truncated to the BF16 value range by clearing the lower 16 bits
of the FP32 representation. The existing FP32 nearest-neighbor NVFP4 block-scale
and E2M1 code selection then runs on the truncated value. Dynamic RHS scale
discovery uses the same truncated values for its `amax` computation.

Initial scope: `GGML_TYPE_NVFP4 x GGML_TYPE_F32 -> GGML_TYPE_F32` CUDA native
matmul activation quantization. This switch is intended to model callers whose
activation input values are already BF16-truncated while preserving the existing
nearest-neighbor NVFP4 quantizer.

## NVFP4 K-Cache Outlier Sidecar

### `LLAMA_NVFP4_KCACHE_OUTLIER`

Enables the compact NVFP4 K-cache outlier sidecar. Default: off.

When this switch is off, `--cache-type-k nvfp4` uses the normal NVFP4 K-cache
path with per-row K global scales and no outlier extraction/correction.

When this switch is on and hybrid FP8 K-cache is not enabled, each NVFP4 K-cache
layer uses the balanced per-layer threshold and compact capacity profile fixed
in `src/llama-kv-cache-nvfp4-outlier-config.h`. The K-cache residual
quantization uses the layer threshold as tensor amax for the K global scale.

When this switch is on together with
`LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8=1`, the selected high/medium layers are
stored as FP8(E4M3+E8M0 block32), and the remaining NVFP4 K-cache layers use the
balanced threshold profile and current context-specific capacity profile fixed
in `src/llama-kv-cache-nvfp4-outlier-config.h`.

### `LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE`

Selects the full-NVFP4 K-cache outlier threshold/capacity profile when
`LLAMA_NVFP4_KCACHE_OUTLIER=1` and hybrid FP8 K-cache is not enabled. Default:
unset, which uses the original `balanced` profile.

Supported values:

- `new`: use the ratio-1e-4 profile snapshot in
  `docs/development/nvfp4-kcache-outlier-thresholds/profiles/ratio-1e4/`.
  Raw evidence remains in
  `experiments/20260605T072559Z-kcache-outlier-threshold-ratio-sweep/` and
  `experiments/20260605T081206Z-kcache-outlier-ratio1e4-default-ppl/`.

### `LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD`

Overrides the per-layer balanced NVFP4 K-cache outlier threshold with one global
absolute-value threshold. Default: unset/off.

This is intended for threshold sweep diagnostics. When unset,
`LLAMA_NVFP4_KCACHE_OUTLIER=1` uses the selected per-layer profile.

### `LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8`

Switch B. Enables the fixed high/medium hybrid FP8 K-cache layer set:

```text
0,1,4,5,6,8,10,11,12,14,23,35
```

Default: off. This switch only affects `--cache-type-k nvfp4`. On its own, it
does not enable NVFP4 K-cache outlier sidecar; combine it with switch A
`LLAMA_NVFP4_KCACHE_OUTLIER=1` to run the current hybrid outlier configuration.

Scripts and reusable profile snapshots for deriving a new balanced profile from
threshold sweep artifacts live in:

```text
docs/development/nvfp4-kcache-outlier-thresholds/
docs/development/nvfp4-kcache-outlier-thresholds/scripts/parse-kcache-outlier-threshold-sweep.py
docs/development/nvfp4-kcache-outlier-thresholds/scripts/derive-kcache-outlier-balanced-config.py
docs/development/nvfp4-kcache-outlier-thresholds/scripts/run-kcache-outlier-balanced-experiment.sh
```

### `LLAMA_NVFP4_KCACHE_OUTLIER_DETERMINISTIC_FILL`

Diagnostic switch for the compact NVFP4 K-cache outlier sidecar. Default:
unset/off.

When enabled, CUDA fills each compact outlier row in ascending column order
using a slow deterministic kernel instead of the default parallel atomic fill.
This is intended to isolate whether compact sidecar entry order affects KQ
correction and PPL. It should not be used for performance measurements.

### `LLAMA_NVFP4_KCACHE_OUTLIER_NO_CORRECTION`

Diagnostic switch for the compact NVFP4 K-cache outlier sidecar. Default:
unset/off.

When enabled, CUDA still extracts outliers and quantizes the residual K cache,
but skips applying the outlier correction to KQ. This isolates residual K-cache
quantization from correction accumulation behavior. It is expected to change
model quality and should not be used as a correctness mode.

### `LLAMA_NVFP4_KCACHE_OUTLIER_FINGERPRINT`

Diagnostic switch for the compact NVFP4 K-cache outlier sidecar. Default:
unset/off.

When enabled, CUDA logs host-side hashes of the touched sidecar counts,
offsets, compact indices, compact values, residual amax rows, destination row
ranges, and source F32 K activation aggregates after each extract operation
when stream capture allows host copies. The source aggregates include
commutative sums/xors so multiple microbatch logs can be combined and compared
with a larger-ubatch extract. This is for comparing ubatch-dependent sidecar
contents and upstream K activation values, and is not suitable for performance
measurements.

## NVFP4 V-Cache

### `LLAMA_NVFP4_VCACHE_PER_BLOCK_SCALE`

Enables the old per-block external-scale path for NVFP4 V-cache
quantization. Default: off.

When this switch is off, the default NVFP4 V-cache path uses one per-tensor
global scale for all V-cache layers and streams. The default scalar is derived
from the wiki calibration nearest-rank P90 absmax:

```text
absmax = 80.428
global_scale = 6 * 224 / 80.428 = 16.7106
```

When this switch is on, the V cache stores external scales per row and per
16-token block, matching the older V-cache quantization behavior. This is kept
for A/B runs and debugging.

### `LLAMA_NVFP4_VCACHE_LAYER_GLOBAL_SCALE`

Enables the per-layer JSON global-scale path. Default: off.

This switch has priority over `LLAMA_NVFP4_VCACHE_PER_BLOCK_SCALE`.
Set it to `1` to load the historical default JSON file
`experiments/qwen3-8b-v-layer-absmax.json`, or set it to a JSON path accepted by
`llama_nvfp4_vcache_load_layer_absmax()`.

When enabled, each layer gets one global scale computed as:

```text
global_scale = 6 * 224 / layer_absmax
```

### `LLAMA_NVFP4_VCACHE_FAST_UPDATE`

Enables CUDA NVFP4 V-cache single-token fast update. Default: off.

When enabled, CUDA set_rows may patch single-token updates without
requantizing the whole 16-token V-cache block.

## CUDA RoPE QEMU Dispatch

### `GGML_CUDA_ROPE_QEMU_ENABLED`

Enables the experimental CUDA RoPE QEMU dispatch hook. Default: unset/off.

Accepted true values are `1`, `true`, and `on`; accepted false values are an
unset or empty value, `0`, `false`, and `off`. The switch is read once on first
use. When enabled, `GGML_OP_ROPE` receives `qemu_enabled=true`, CUDA graph
capture is disabled for graphs containing RoPE, and the experimental QEMU entry
point is attempted. The current interface-only entry point returns control to
the existing CUDA RoPE kernel, preserving output while the external QEMU
operator is not yet connected. A once-only log confirms both the switch state
and the fallback path.
