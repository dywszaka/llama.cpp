# Merge Conflict Report: gitlab/cuda-bf16

Date: 2026-08-05 08:18:30 UTC

Current branch: `integration_c100_sim`

Merged branch: `remotes/gitlab/cuda-bf16` at `e5d9e9c84`

## Initial Conflicts

- `expt-switch-env.md`
- `ggml/src/ggml-cuda/CMakeLists.txt`
- `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-matmul.cu`
- `ggml/src/ggml-cuda/softmax.cu`
- `src/expt/tensor-export-eval.cpp`
- `src/expt/tensor-export-eval.h`
- `src/llama-graph.cpp`
- `tests/test-cuda-bf16-expp.cu`
- `tests/test-expt-tensor-export-eval.cpp`

## Resolution Summary

- Kept the expanded tensor export documentation and added the new attention replay export kinds, per-record `meta`, and CUDA softmax BF16 switch details.
- Kept the CUDA CMake `expt/*.cuh` glob and the new `expt/bf16-expp/*.cuh` headers.
- Kept both CUDA softmax includes: BF16 expp and local softmax-qemu integration.
- Set the graph softmax callback name to `kq-softmax` for attention replay export, while preserving `llama_expt_pin_soft_max_to_c100`.
- Combined tensor export APIs:
  - retained op/name/layer graph capture, observer, NVFP4 RHS capture, and BF16 dump support;
  - added attention replay manifest metadata, quant-round evaluation, and `tensor_export_pin_named_tensor`;
  - added a two-argument `tensor_export_graph()` compatibility wrapper.
- Resolved NVFP4 matmul around the refactored FP4MULMAT implementation:
  - kept target branch split files `nvfp4-fp4mulmat.*`;
  - retained RHS/scale capture sidecars and capture flags;
  - preserved parallel batched slice scratch/stream handling;
  - copied dynamic final scales into capture sidecars after BF16 truncation for FP4MULMAT.
- Combined BF16 expp tests by keeping exhaustive host/CUDA coverage and adding fixed host boundary cases.
- Combined tensor export eval tests by keeping op/name capture tests and adding attention export pinning.
- Updated `test-nvfp4-matmul` capture expectations to match FP4MULMAT final-scale BF16 truncation.

## Validation

Build:

```sh
cmake --build build_cuda --target test-expt-tensor-export-eval test-cuda-bf16-expp test-nvfp4-matmul -j 8
```

Result: passed.

Tests:

```sh
./build_cuda/bin/test-expt-tensor-export-eval
./build_cuda/bin/test-cuda-bf16-expp
./build_cuda/bin/test-nvfp4-matmul
```

Result: all passed on CUDA device `NVIDIA GeForce RTX 5090`.

Notes:

- `build-cuda` and `build-cuda-c100` caches point to old `/workspace/llama.cpp` paths and were not used for validation.
- Conflict marker scan on resolved files found no remaining line-start conflict markers.
- `git diff --check` passed.
