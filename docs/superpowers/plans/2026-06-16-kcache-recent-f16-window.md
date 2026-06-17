# K-Cache Recent F16 Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an off-by-default NVFP4 K-cache experiment that preserves the most recent configurable window of K rows in F16 for the regular non-flash-attention KQ path.

**Architecture:** Keep the existing NVFP4 K-cache and K Outlier sidecar as the default storage/computation path. When the new switch is enabled, allocate a full-size F16 K shadow tensor plus I32 per-cell metadata, write each incoming K row to both NVFP4 and F16, mark only recent-window rows active, and have CUDA overwrite the corresponding KQ logits with an F16 dot result after the NVFP4 matmul. This preserves original KV cell ordering and avoids physical cache repartitioning.

**Tech Stack:** C++ llama KV cache and graph code, ggml tensor sidecar metadata, CUDA set_rows/mul_mat experiment helpers, CMake tests.

---

### Task 1: Switch And Window Helper

**Files:**
- Modify: `src/llama-kv-cache-nvfp4-outlier-config.h`
- Test: `tests/test-nvfp4-kcache-outlier-profile.cpp`

- [ ] Write failing tests for default-off switch, default window 100, env window parsing, and active-row predicate.
- [ ] Run `cmake --build build -t test-nvfp4-kcache-outlier-profile && ./build/bin/test-nvfp4-kcache-outlier-profile`; expect failure because helper symbols do not exist.
- [ ] Add helpers:
  - `llama_nvfp4_kcache_recent_f16_enabled()`
  - `llama_nvfp4_kcache_recent_f16_window()`
  - `llama_nvfp4_kcache_recent_f16_is_active(cell_pos, query_pos, window)`
- [ ] Re-run the test and expect pass.

### Task 2: ggml Sidecar Metadata

**Files:**
- Modify: `ggml/include/ggml.h`
- Modify: `ggml/src/ggml.c`
- Test: `tests/test-vcache-nvfp4-layout.cpp`

- [ ] Write a failing metadata round-trip test for binding recent-F16 shadow, active flags, and positions to a tensor.
- [ ] Run the focused test and expect failure because APIs do not exist.
- [ ] Add ggml setter/getter APIs using existing `tensor->src[]` metadata slots.
- [ ] Re-run the focused test and expect pass.

### Task 3: KV Cache Allocation And Write

**Files:**
- Modify: `src/llama-kv-cache-unified.h`
- Modify: `src/llama-kv-cache-unified.cpp`
- Modify: `ggml/src/ggml-cuda/set-rows.cuh`
- Modify: `ggml/src/ggml-cuda/set-rows.cu`
- Modify: `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-set-rows.cu`
- Test: `tests/test-nvfp4-kcache-set-rows.cu`

- [ ] Write failing CUDA set_rows test that enables the switch, writes rows with positions, and checks F16 shadow plus active flags.
- [ ] Run the focused test and expect failure because the recent-F16 write path does not exist.
- [ ] Allocate F16 shadow K and I32 metadata only when type K is NVFP4, K Outlier is enabled, and recent-F16 switch is enabled.
- [ ] Add a K-position input tensor and attach recent-F16 sidecars to `ggml_set_rows`.
- [ ] Extend CUDA NVFP4 set_rows to copy raw K into F16 shadow and mark active flags from positions/window.
- [ ] Re-run focused test and expect pass.

### Task 4: KQ F16 Override

**Files:**
- Modify: `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-matmul.cu`
- Modify: `tests/test-nvfp4-matmul.cu`

- [ ] Write failing backend KQ test that binds recent-F16 sidecars and expects active rows to use exact F16-shadow dot while inactive rows keep NVFP4/outlier behavior.
- [ ] Run `test-nvfp4-matmul`; expect failure because override is not implemented.
- [ ] After NVFP4 native KQ and outlier correction, launch a CUDA kernel over KQ output rows that recomputes active row logits from the F16 shadow and F32 Q.
- [ ] Re-run `test-nvfp4-matmul`; expect pass.

### Task 5: Docs And Validation Artifacts

**Files:**
- Modify: `expt-switch-env.md`
- Create: `experiments/<timestamp>-kcache-recent-f16-window/summary.md`

- [ ] Document the new switches and defaults.
- [ ] Record validation commands and results.
- [ ] Run focused build/tests available locally and report skipped CUDA/PPL validation explicitly if unavailable.
