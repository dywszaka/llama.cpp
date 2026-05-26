# NVFP4 K-Cache Outlier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a switch-gated NVFP4 K-cache outlier sidecar path that stores signed `abs(K) > threshold` values and adds their exact F32 Q contribution back into KQ.

**Architecture:** Keep the feature local to the NVFP4 K-cache path. Allocate sidecar tensors only when the experiment switch is enabled, bind them through narrow ggml metadata helpers, implement extraction/correction in `ggml/src/ggml-cuda/nvfp4/`, and document all switches in `expt-switch-env.md`.

**Tech Stack:** C++, CUDA, ggml tensor metadata, llama.cpp KV-cache graph, CMake CUDA tests.

---

## File Map

- Modify `ggml/include/ggml.h`: declare narrow metadata helpers for NVFP4 K-cache outlier sidecars.
- Modify `ggml/src/ggml.c`: implement metadata helpers using tail `src[]` slots.
- Create `ggml/src/ggml-cuda/nvfp4/kcache-outlier.cuh`: public CUDA helper declarations and environment helpers.
- Create `ggml/src/ggml-cuda/nvfp4/kcache-outlier.cu`: extraction, residual quantization support, correction kernel, and optional logging.
- Modify `ggml/src/ggml-cuda/nvfp4/nvfp4-set-rows.cu`: route NVFP4 set_rows through outlier-aware kernels when metadata is present.
- Modify `ggml/src/ggml-cuda/nvfp4/nvfp4-matmul.cu`: apply KQ correction in the batched KQ path after K-side scale compensation.
- Modify `src/llama-kv-cache-unified.h`: add K outlier sidecar tensors to `kv_layer`.
- Modify `src/llama-kv-cache-unified.cpp`: parse switches, allocate sidecars, bind sidecars on K write/read views, and log enabled state once.
- Modify `src/llama-graph.cpp`: preserve outlier metadata after K permute, next to existing K scale metadata preservation.
- Modify `tests/CMakeLists.txt`: register the new CUDA focused test target.
- Create `tests/test-nvfp4-kcache-outlier.cu`: test extraction and correction helper behavior without a full model.
- Modify `expt-switch-env.md`: document all new environment switches.
- Modify `task_plan.md`, `findings.md`, `progress.md`: record implementation progress.

## Task 1: Metadata Helpers

- [ ] Add declarations to `ggml/include/ggml.h` near the existing NVFP4 scale helpers:

```c
GGML_API void ggml_tensor_set_nvfp4_kcache_outliers(
        struct ggml_tensor       * tensor,
        const struct ggml_tensor * counts,
        const struct ggml_tensor * indices,
        const struct ggml_tensor * values);

GGML_API const struct ggml_tensor * ggml_tensor_get_nvfp4_kcache_outlier_counts(
        const struct ggml_tensor * tensor);

GGML_API const struct ggml_tensor * ggml_tensor_get_nvfp4_kcache_outlier_indices(
        const struct ggml_tensor * tensor);

GGML_API const struct ggml_tensor * ggml_tensor_get_nvfp4_kcache_outlier_values(
        const struct ggml_tensor * tensor);
```

- [ ] Implement in `ggml/src/ggml.c` near `ggml_tensor_set_nvfp4_scale()`:

```c
void ggml_tensor_set_nvfp4_kcache_outliers(
        struct ggml_tensor       * tensor,
        const struct ggml_tensor * counts,
        const struct ggml_tensor * indices,
        const struct ggml_tensor * values) {
    tensor->src[GGML_MAX_SRC - 4] = (struct ggml_tensor *) counts;
    tensor->src[GGML_MAX_SRC - 3] = (struct ggml_tensor *) indices;
    tensor->src[GGML_MAX_SRC - 2] = (struct ggml_tensor *) values;
}
```

- [ ] Build check:

```bash
cmake --build build_cuda_release --target test-nvfp4-matmul -j 16
```

Expected: compile reaches existing target without metadata declaration errors.

## Task 2: Failing Focused Test

- [ ] Create `tests/test-nvfp4-kcache-outlier.cu` with two tests:
  - extraction stores signed values and flattened positions for `abs(K) > threshold`;
  - correction adds only current-head outliers to a KQ matrix.

- [ ] Register it in `tests/CMakeLists.txt` inside `if (GGML_CUDA)`:

```cmake
set(LLAMA_TEST_NAME test-nvfp4-kcache-outlier)
llama_build(test-nvfp4-kcache-outlier.cu)
if (LLAMA_TEST_CUDA_ARCH AND NOT LLAMA_TEST_CUDA_ARCH STREQUAL "LLAMA_TEST_CUDA_ARCH-NOTFOUND")
    set_property(TARGET ${LLAMA_TEST_NAME} PROPERTY CUDA_ARCHITECTURES "${LLAMA_TEST_CUDA_ARCH}")
endif()
llama_test(test-nvfp4-kcache-outlier LABEL "cuda")
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE CUDA::cudart)
unset(LLAMA_TEST_NAME)
```

- [ ] Run:

```bash
cmake --build build_cuda_release --target test-nvfp4-kcache-outlier -j 16
```

Expected before implementation: build fails because `kcache-outlier.cuh` helpers do not exist.

## Task 3: CUDA Outlier Helper

- [ ] Create `ggml/src/ggml-cuda/nvfp4/kcache-outlier.cuh` with helper declarations:
  - switch parsers;
  - `ggml_cuda_nvfp4_kcache_outlier_extract`;
  - `ggml_cuda_nvfp4_kcache_outlier_apply_correction`.

- [ ] Create `ggml/src/ggml-cuda/nvfp4/kcache-outlier.cu`:
  - reset counts for destination rows;
  - compute residual row amax ignoring outliers;
  - extract outliers into fixed sidecars;
  - apply KQ correction using F32 Q and head filtering.

- [ ] Run:

```bash
cmake --build build_cuda_release --target test-nvfp4-kcache-outlier -j 16
CUDA_VISIBLE_DEVICES=0 build_cuda_release/bin/test-nvfp4-kcache-outlier
```

Expected: focused test passes when CUDA is available; if CUDA is unavailable, test reports skip.

## Task 4: Wire K-Cache Allocation and Binding

- [ ] Add sidecar tensors to `llama_kv_cache_unified::kv_layer`.
- [ ] Parse `LLAMA_NVFP4_KCACHE_OUTLIER`, threshold, max, and log switches in `src/llama-kv-cache-unified.cpp`.
- [ ] Allocate sidecars only for `type_k == GGML_TYPE_NVFP4` when enabled.
- [ ] Bind sidecars to `ggml_set_rows()` result in `cpy_k()`.
- [ ] Bind sidecar views to `get_k()` K views.
- [ ] Preserve sidecar metadata after K permute in `src/llama-graph.cpp`.
- [ ] Run:

```bash
cmake --build build_cuda_release --target llama-cli test-nvfp4-kcache-outlier -j 16
```

Expected: build succeeds.

## Task 5: Set-Rows and KQ Integration

- [ ] Include `kcache-outlier.cuh` in `nvfp4-set-rows.cu`.
- [ ] When sidecar metadata is present, run outlier-aware residual amax and quantization.
- [ ] Include `kcache-outlier.cuh` in `nvfp4-matmul.cu`.
- [ ] In the batched KQ path, after each recursive 2D matmul and after existing K-side scale has been applied by graph, launch correction with the original K sidecars, current Q slice, and current KQ slice.
- [ ] Run:

```bash
cmake --build build_cuda_release --target test-nvfp4-kcache-outlier test-nvfp4-matmul llama-cli -j 16
CUDA_VISIBLE_DEVICES=0 build_cuda_release/bin/test-nvfp4-kcache-outlier
CUDA_VISIBLE_DEVICES=0 build_cuda_release/bin/test-nvfp4-matmul
```

Expected: focused outlier test and existing NVFP4 matmul test pass.

## Task 6: Switch Documentation and Logging Validation

- [ ] Update `expt-switch-env.md` with all four new switches and defaults.
- [ ] Add once-only enabled-state logging in K-cache construction when outlier switch is enabled.
- [ ] Add optional runtime count logging in CUDA helper when `LLAMA_NVFP4_KCACHE_OUTLIER_LOG=1`.
- [ ] Run optional smoke if model and GPU are available:

```bash
LLAMA_NVFP4_KCACHE_OUTLIER=1 \
LLAMA_NVFP4_KCACHE_OUTLIER_LOG=1 \
CUDA_VISIBLE_DEVICES=0 build_cuda_release/bin/llama-cli \
  -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
  -p "Write one short sentence about CUDA." \
  -n 16 -c 2048 -ngl 40 -ctk nvfp4 -ctv f16 --kv-unified --flash-attn 0
```

Expected: log confirms switch state and reports outlier counts; generation reaches completion.

## Self-Review

- Spec coverage: the plan covers extraction before quantization, residual zeroing, per-token quantization, KQ correction with F32 Q, default-off switches, log-controlled count printing, and `expt-switch-env.md`.
- Placeholder scan: no `TBD` or unspecified implementation tasks remain.
- Type consistency: sidecars use `I32` counts, `I32` indices, and `F32` values across allocation, metadata, CUDA helpers, and tests.
