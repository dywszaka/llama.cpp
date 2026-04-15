# Non-Flash-Attn Decode-Only FP8 P/V Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a CUDA-only experimental non-`flash_attn` decode path for `LLAMA_EXP_VCACHE_V_FP8=e8m0` that quantizes both `V` and `P` to `FP8(E4M3)+E8M0` and computes `P x V` with direct FP8 tensor operations, while removing the current auto-`flash_attn` behavior.

**Architecture:** Keep the existing non-`flash_attn` `KQ -> softmax -> KQV` graph structure, but replace the decode-only `kqv = mul_mat(v, kq)` stage with a dedicated CUDA path when `V` and `P` are in the experimental FP8 format. Gate this tightly by shape and backend, and leave all unsupported cases as explicit errors.

**Tech Stack:** llama.cpp graph builder, CUDA backend, custom CUDA kernels, CTest smoke scripts, Qwen3-8B-NVFP4 model.

---

### Task 1: Lock the intended behavior with failing smoke tests

**Files:**
- Modify: `tests/CMakeLists.txt`
- Create: `tests/test-vcache-fp8-e8m0-non-flash-generate-smoke.sh`
- Modify: `tests/test-vcache-fp8-e8m0-auto-flash-attn.sh`

**Step 1: Write the failing test**

Create a smoke test that:

- runs `llama-cli`
- sets `LLAMA_EXP_VCACHE_V_FP8=e8m0`
- does not pass `-fa`
- expects the model to generate a short answer
- expects logs to show experimental `V` type
- expects logs not to show forced `flash_attn = 1`

Adjust the old auto-`flash_attn` test so it now encodes the opposite expectation or remove it if it no longer matches the accepted design.

**Step 2: Run test to verify it fails**

Run: `ctest --test-dir build_cuda -R 'test-vcache-fp8-e8m0-(auto-flash-attn|non-flash-generate-smoke)' --output-on-failure`

Expected: failure because the current code auto-enables `flash_attn` or rejects non-`flash_attn`.

**Step 3: Commit**

Do not commit yet. Continue to implementation after the failing behavior is captured.

### Task 2: Remove the current auto-flash-attn default

**Files:**
- Modify: `src/llama-context.cpp`

**Step 1: Write the failing test**

Reuse the smoke test from Task 1 as the red test.

**Step 2: Run test to verify it fails**

Run the same focused CTest command and confirm the failure is due to forced or required `flash_attn`.

**Step 3: Write minimal implementation**

Update `llama_init_from_model()` so that:

- `LLAMA_EXP_VCACHE_V_FP8` no longer auto-enables `flash_attn`
- unsupported combinations are rejected with targeted errors
- the new non-`flash_attn` experimental mode can proceed to graph building

**Step 4: Run test to verify it still fails for the missing kernel path**

Run the same focused CTest command.

Expected: the old `flash_attn` behavior is gone, but the new non-`flash_attn` path still fails because compute support is not implemented yet.

### Task 3: Add a dedicated decode-only packing path for P

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: `ggml/src/ggml-cuda/common.cuh`
- Create or modify: `ggml/src/ggml-cuda/convert.cu` or a new CUDA helper file for FP8 decode packing

**Step 1: Write the failing test**

Add a focused CUDA unit test for packing a decode `P` vector into `FP8(E4M3)+E8M0` block format.

**Step 2: Run test to verify it fails**

Run: `ctest --test-dir build_cuda -R test-vcache-fp8-e4m3-e8m0 --output-on-failure`

Expected: failure because the new non-`flash_attn` `P` packing helper is not wired in.

**Step 3: Write minimal implementation**

Implement a CUDA helper that packs the decode-time `P` tensor into the experimental block FP8 format without materializing a dequantized compute copy.

**Step 4: Run test to verify it passes**

Run the focused CUDA test again and verify the packing logic is correct.

### Task 4: Add the decode-only FP8 x FP8 non-flash-attn kernel

**Files:**
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: `ggml/src/ggml-cuda/common.cuh`
- Create: a dedicated CUDA source/header for non-`flash_attn` decode FP8 `mul_mat`

**Step 1: Write the failing test**

Add a numeric regression test comparing:

- baseline non-`flash_attn` decode attention output
- experimental non-`flash_attn` FP8 `V + P`

**Step 2: Run test to verify it fails**

Run the focused test executable directly and confirm the output mismatch or unsupported-path failure.

**Step 3: Write minimal implementation**

Implement a decode-only CUDA kernel that:

- reads non-`flash_attn` transposed `V`
- reads packed `P`
- performs direct FP8-input tensor operations
- accumulates into `FP32`

Wire this into the relevant `mul_mat` dispatch only for the supported experimental shape.

**Step 4: Run test to verify it passes**

Run the numeric regression test and confirm it is green.

### Task 5: Hook the graph builder to the new path

**Files:**
- Modify: `src/llama-graph.cpp`
- Modify: `src/llama-context.cpp`

**Step 1: Write the failing test**

Use the non-`flash_attn` generation smoke test as the red test.

**Step 2: Run test to verify it fails**

Run: `ctest --test-dir build_cuda -R 'test-vcache-fp8-e8m0-non-flash-generate-smoke|test-vcache-fp8-e4m3-e8m0' --output-on-failure`

Expected: failure because the graph still routes through the old generic path.

**Step 3: Write minimal implementation**

In the non-`flash_attn` branch of attention graph construction:

- detect decode-only experimental shapes
- insert the new `P` pack and FP8 `mul_mat` route
- explicitly reject unsupported shapes

**Step 4: Run test to verify it passes**

Run the same focused CTest command and confirm both tests pass.

### Task 6: Verify server and release behavior

**Files:**
- No new production files required unless a dedicated server smoke script is needed

**Step 1: Run debug verification**

Run a real debug server startup without `-fa`:

`GGML_CUDA_TRUNC_ENABLE=0 LLAMA_EXP_VCACHE_V_FP8=e8m0 ./build_cuda/bin/llama-server -m ../models/qwen3-8b-nvfp4.gguf --n_gpu_layers 40 --host 127.0.0.1 --batch-size 2048 --ubatch-size 512 --port <port> -t 32 -c 2048 --no-warmup`

Expected:

- no forced `flash_attn`
- experimental `V` type active
- server reaches listening state

**Step 2: Run release verification**

Run:

- `ctest --test-dir build_cuda_release -R 'test-vcache-fp8-e8m0-non-flash-generate-smoke|test-vcache-fp8-e4m3-e8m0-generate-smoke|test-vcache-fp8-e4m3-e8m0' --output-on-failure`

Expected: all focused release checks pass.

**Step 3: Summarize residual risks**

Call out remaining unsupported cases:

- prefill
- `bs > 1`
- non-Qwen head dimensions
- non-CUDA backends
