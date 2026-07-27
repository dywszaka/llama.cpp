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
matching node's `dst`, `dst->src[0]`, and `dst->src[1]` from the first selected
prefill or decode graph. Edit and run `export.sh` in this directory to launch
the export and record its artifacts under `experiments/`.

For `OP=RMS_NORM`, each export directory contains an executable
`verify-rms-norm.py`. Pass the result/dst binary first and the input/src0
binary second:

```bash
./verify-rms-norm.py tensors/0-node1-dst-norm-0.bin tensors/1-node1-src0-CUDA0_inp_embd_0.bin
```

The script reads shape information from the adjacent `tensors/manifest.json`.
It automatically selects the normal F32 algorithm or the deterministic QEMU
BF16 algorithm used by `export.sh`; `--mode` can override detection.

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
