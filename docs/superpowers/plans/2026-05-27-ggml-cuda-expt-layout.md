# ggml-cuda expt Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CUDA FP8 and NVFP4 experimental implementations under `ggml/src/ggml-cuda/expt/` and document that future ggml-cuda experimental implementation code belongs there.

**Architecture:** Keep `ggml/src/ggml-cuda` as the upstream-sync surface. Experimental implementation files live below `expt/fp8` and `expt/nvfp4`; existing top-level CUDA files retain only thin dispatch/control references.

**Tech Stack:** CMake, CUDA/C++, llama.cpp ggml CUDA backend.

---

### Task 1: Move source directories

**Files:**
- Move: `ggml/src/ggml-cuda/fp8` to `ggml/src/ggml-cuda/expt/fp8`
- Move: `ggml/src/ggml-cuda/nvfp4` to `ggml/src/ggml-cuda/expt/nvfp4`

- [ ] **Step 1: Create the target directory**

Run: `mkdir -p ggml/src/ggml-cuda/expt`

- [ ] **Step 2: Move the two implementation directories**

Run:
```bash
mv ggml/src/ggml-cuda/fp8 ggml/src/ggml-cuda/expt/fp8
mv ggml/src/ggml-cuda/nvfp4 ggml/src/ggml-cuda/expt/nvfp4
```

- [ ] **Step 3: Inspect the new layout**

Run: `find ggml/src/ggml-cuda/expt -maxdepth 2 -type f | sort`

Expected: FP8 and NVFP4 `.cu` / `.cuh` files are listed below `expt/fp8` and `expt/nvfp4`.

### Task 2: Update build and include references

**Files:**
- Modify: `ggml/src/ggml-cuda/CMakeLists.txt`
- Modify: `ggml/src/ggml-cuda/ggml-cuda.cu`
- Modify: `ggml/src/ggml-cuda/fattn.cu`
- Modify: `ggml/src/ggml-cuda/set-rows.cu`
- Modify: `tests/test-nvfp4-kcache-outlier.cu`

- [ ] **Step 1: Update CMake glob paths**

Change `fp8/*.cuh`, `nvfp4/*.cuh`, `fp8/*.cu`, and `nvfp4/*.cu` to the matching `expt/...` paths.

- [ ] **Step 2: Update include paths**

Replace direct includes of `fp8/...` and `nvfp4/...` with `expt/fp8/...` and `expt/nvfp4/...`.

- [ ] **Step 3: Search for stale path references**

Run:
```bash
rg -n '"(fp8|nvfp4)/|ggml-cuda/(fp8|nvfp4)/|fp8/\*|nvfp4/\*' ggml/src/ggml-cuda tests --glob '!**/build*/**'
```

Expected: no stale references to the old top-level `fp8` or `nvfp4` directories.

### Task 3: Update repository-local guidance

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add the `expt` layout contract**

Document that CUDA FP8 and NVFP4 experiment implementations live under `ggml/src/ggml-cuda/expt/`, and top-level `ggml-cuda` files should only carry narrow dispatch/control code.

- [ ] **Step 2: Update NVFP4 runtime map paths**

Change NVFP4 CUDA implementation paths in the runtime map from `ggml/src/ggml-cuda/nvfp4/...` to `ggml/src/ggml-cuda/expt/nvfp4/...`.

- [ ] **Step 3: Search for stale documentation paths**

Run:
```bash
rg -n 'ggml/src/ggml-cuda/(fp8|nvfp4)|ggml-cuda/(fp8|nvfp4)' AGENTS.md docs
```

Expected: no stale documentation references to top-level CUDA experiment directories.

### Task 4: Verify build integration

**Files:**
- Read-only verification across changed files.

- [ ] **Step 1: Check git diff**

Run: `git diff --stat && git diff -- ggml/src/ggml-cuda/CMakeLists.txt ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/fattn.cu ggml/src/ggml-cuda/set-rows.cu tests/test-nvfp4-kcache-outlier.cu AGENTS.md`

- [ ] **Step 2: Configure the CUDA build if needed**

Run: `cmake -S . -B build_cuda_release -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release`

- [ ] **Step 3: Build a focused CUDA target**

Run: `cmake --build build_cuda_release --target ggml-cuda -j 2`

Expected: the CUDA backend target compiles with the new paths.
