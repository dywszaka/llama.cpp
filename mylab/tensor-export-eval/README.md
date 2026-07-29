# Tensor Export Lab

This directory holds tensor-export-eval documentation and lightweight example
metadata for LLAMA_CPP-12.

- `docs/` contains the tensor export and offline quantization evaluation spec.
- `examples/` contains reproducibility notes, manifests, commands, and captured
  metrics from small example runs. Large raw tensor dumps are intentionally not
  kept here by default.

Build-integrated source, CLI, and tests remain in their normal repository
locations:

- `src/expt/tensor-export-eval.{h,cpp}`
- `tools/tensor-export-eval/`
- `tests/test-expt-tensor-export-eval.cpp`

The runtime exporter also supports an op-oriented mode for capturing every
matching node's `dst` and populated `dst->src[0..2]` tensors from the first
selected prefill or decode graph. Edit and run `export.sh` in this directory to
launch the export and record its artifacts under `experiments/`.

For `OP=RMS_NORM`, each export directory contains an executable
`verify-rms-norm.py`. Pass the result/dst binary; the script locates the same
node's input/src0 in the adjacent manifest:

```bash
./verify-rms-norm.py tensors/0-node1-dst-norm-0.bin
```

The script reads shape information from the adjacent `tensors/manifest.json`.
An explicit input/src0 path remains supported for exports without a manifest.
It automatically selects the normal F32 algorithm or the deterministic QEMU
FP32-compute/BF16-I/O algorithm used by `export.sh`; `--mode` can override
detection. The older all-BF16 reconstruction remains available explicitly as
`--mode qemu-bf16` for legacy captures.

For `OP=MUL_MAT`, the export directory instead contains
`verify-mul-mat.py`. Pass only the result/dst binary; the script locates the
same node's inputs and any NVFP4 effective-input scale records in the adjacent
manifest:

```bash
./verify-mul-mat.py tensors/0-node26-dst-kq-0.bin
```

The MUL_MAT validator supports strided F16/F32/BF16 inputs and native NVFP4
effective inputs. It follows ggml's `dst = src0^T * src1` batch-broadcast
semantics and recognizes the BF16 result rounding enabled by `export.sh`.
For native NVFP4 captures, the report lists every scale file actually used by
the reconstruction. FP4MULMAT exports report the single `matmul_scale` file,
its manifest encoding and semantics, scalar values/range, and whether each F32
value happens to have zero low 16 bits. The scale file preserves the original
FP32 values; the manifest tells the validator to apply BF16-RNE operand rounding
when reconstructing the FP4MULMAT output multiply. Use `--max-scale-values` to
control how many scale values are printed.

For `OP=SOFT_MAX`, the export directory contains `verify-soft-max.py`. The
attention softmax dst is named `kq_soft_max-<layer>`, so it can be selected with
`NAME=kq_soft_max`. The dst manifest record includes `op_params.scale` and
`op_params.max_bias`; the same node's logits (`src0`), optional mask (`src1`),
and optional attention sinks (`src2`) are resolved automatically:

```bash
./verify-soft-max.py tensors/0-node42-dst-kq_soft_max-0.bin
```

The validator reproduces mask broadcasting, ALiBi slopes, attention sinks, and
the configured scale. It supports the normal F32 CUDA calculation and the
deterministic QEMU BF16 algorithm selected by
`GGML_CUDA_SOFT_MAX_QEMU_MODE=qemu` or `qemu_cuda`; `--mode` can override
automatic detection. It also requires every exported `src0`/KQ F32 value to be
BF16-representable (`f32_bits & 0xffff == 0`) and fails validation if any input
value retains non-zero low 16 bits.

For `OP=ROPE`, each export directory contains `verify-rope.py` together with
the static 8192-position CUDA cos/sin table and its manifest. The default
`export.sh` configuration captures the first decode graph's `Qcur-0` RoPE node:

```bash
./verify-rope.py tensors/0-node7-dst-Qcur-0.bin
```

The validator resolves `src0`, the I32 position tensor (`src1`), and all RoPE
parameters from `tensors/manifest.json`. For each position and rotary pair it
looks up cos/sin in `rope-cos-sin-f32.bin`; it does not recompute trigonometric
values. The bundled table covers Qwen3-8B GPT-NeoX RoPE with positions
`[0, 8192)`, `n_dims=128`, and `freq_base=1000000`, and the validator rejects
incompatible parameters or frequency-factor inputs.
