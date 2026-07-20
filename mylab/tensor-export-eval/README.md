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
