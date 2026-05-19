# NVFP4 V-cache Layer Global Scale Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an experimental NVFP4 V-cache mode that uses one fixed per-layer global scale from a layer absmax JSON instead of per-block external F32 scales.

**Architecture:** Keep the existing per-block scale path as the default. When `LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE` points to a JSON file, KV cache initialization allocates one F32 scale value per layer/stream and stores `global_scale = GGML_CUDA_VCACHE_NVFP4_GLOBAL_SCALE_MAX / layer_absmax[layer]`. CUDA set_rows and P*V kernels detect scalar scale layout and interpret it as `global_scale`; legacy multi-element scale tensors continue to mean per-block `input_scale`.

**Tech Stack:** llama.cpp KV cache graph code, ggml CUDA set_rows kernels, NVFP4 V-cache custom matmul, cuBLASLt FP4 P*V path, CUDA focused tests.

---

### Task 1: Failing Store Test

**Files:**
- Modify: `tests/test-vcache-nvfp4-store.cu`

**Step 1: Write the failing test**

Add a test that attaches a scalar F32 scale tensor to an experimental transposed NVFP4 V-cache set_rows node. Store a known 16-value block with `global_scale = GGML_CUDA_NVFP4_GLOBAL_SCALE_MAX / 64.0f`. Read back the quantized block and verify dequantization uses `value = q * e / global_scale`.

**Step 2: Run test to verify it fails**

Run: `cmake --build build_cuda --target test-vcache-nvfp4-store -j 4 && LLAMA_EXPERIMENT_NVFP4_VCACHE=1 build_cuda/bin/test-vcache-nvfp4-store`

Expected: failure because set_rows rejects or misinterprets scalar scale layout.

### Task 2: Implement Scalar Store Path

**Files:**
- Modify: `ggml/src/ggml-cuda/set-rows.cu`

**Step 1: Update layout detection**

Allow NVFP4 V-cache set_rows when scale tensor is scalar per stream/layer (`ne[0] == 1`) as well as legacy per-block.

**Step 2: Update kernel semantics**

Pass `scale_is_global` to `k_set_rows_nvfp4_vcache`. In scalar mode, read `global_scale = scale[0]`, do not write `scale[row_global * n_blocks + block_idx]`, and compute block scale `e` from the fixed global scale and current block amax.

**Step 3: Run store test**

Run the same focused store test and expect pass.

### Task 3: Failing Matmul Test

**Files:**
- Modify: `tests/test-vcache-nvfp4-matmul.cu`

**Step 1: Write the failing test**

Add scalar-scale coverage for real V-cache view cases. The reference dequantizes V with `e / global_scale` and validates both custom FP4-P and cuBLASLt FP4-P paths.

**Step 2: Run test to verify it fails**

Run: `LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV=1 LLAMA_EXPERIMENT_NVFP4_VCACHE_FP4_PV_LT=1 build_cuda/bin/test-vcache-nvfp4-matmul`

Expected: failure because matmul currently requires per-block scale layout.

### Task 4: Implement Scalar Matmul Path

**Files:**
- Modify: `ggml/src/ggml-cuda/vcache-nvfp4-matmul.cu`

**Step 1: Accept scalar scale layout**

Teach scale layout matching to return a `scale_is_global` flag when `scale->ne[0] == 1`.

**Step 2: Update kernels**

In custom kernels, use `v_d = e / global_scale` for scalar mode. In cuBLASLt staging, write A scale channel as `e / global_scale`, eliminating per-block F32 scale loads in scalar mode.

**Step 3: Run matmul tests**

Run default, FP4-P, and FP4-P-LT focused tests.

### Task 5: KV Cache Integration

**Files:**
- Modify: `src/llama-kv-cache-unified.cpp`
- Modify: `src/llama-kv-cache-unified.h`

**Step 1: Add experiment config helper**

Parse `LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE` once. Treat unset as disabled. Treat `1` as the default `experiments/qwen3-8b-v-layer-absmax.json`; otherwise treat env value as a JSON path.

**Step 2: Allocate scalar V scale**

When enabled and experimental NVFP4 V-cache layout is active, allocate `v_scale` as `ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_stream)`.

**Step 3: Initialize scale values**

After buffer allocation/clear, set each layer stream scalar to `GGML_CUDA_VCACHE_NVFP4_GLOBAL_SCALE_MAX / absmax[layer]`.

**Step 4: Preserve graph attachments**

Ensure `get_v()` and `cpy_v()` attach scalar scale views to V tensors and set_rows nodes.

### Task 6: Final Verification and PPL

**Files:**
- Create/Update: `experiments/<run-id>/`

**Step 1: Build**

Run: `cmake --build build_cuda --target test-vcache-nvfp4-store test-vcache-nvfp4-matmul llama-perplexity -j 4`

**Step 2: Focused tests**

Run store and matmul tests with scalar env enabled.

**Step 3: PPL**

Run the existing Qwen3 8B Wikitext PPL command with `LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE=experiments/qwen3-8b-v-layer-absmax.json`, `cache-type-k=f16`, `cache-type-v=nvfp4`, `--n_gpu_layers 40`, `--batch-size 512`, `--ubatch-size 512`, `-t 32`, `-c 512`.

**Step 4: Commit**

Commit verified source/test changes with a focused message.
