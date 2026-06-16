# ggml-cim backend design

## 1. Context and goals

This document defines the first-stage design for adding a new `ggml-cim`
backend, using `ggml-cuda` as the integration reference but keeping only the
minimum files and interfaces needed to register a backend. No CIM kernels,
operator implementations, scheduling policy, graph lowering, or model-specific
behavior are included in this stage.

The goal is to create a narrow, auditable backend shell design that can later be
enabled with `-DGGML_CIM=ON`, appear in the ggml backend registry, expose a
public C API, define the minimal device and buffer interfaces, and fail
unsupported compute paths explicitly.

This document is the only artifact for the current step. It intentionally does
not add `ggml-cim` source files, build files, registry edits, kernels, or runtime
behavior yet.

## 2. Requirements

### 2.1 Functional requirements

- `GGML_CIM` is a CMake option, defaulting to `OFF`.
- The backend builds as target `ggml-cim` through the existing
  `ggml_add_backend()` / `ggml_add_backend_library()` flow.
- Static builds define `GGML_USE_CIM` so `ggml-backend-reg.cpp` can register
  the backend at process startup.
- Dynamic backend builds export `ggml_backend_init()` through
  `GGML_BACKEND_DL_IMPL(ggml_backend_cim_reg)`.
- Public API lives in `ggml/include/ggml-cim.h`.
- Backend source lives under `ggml/src/ggml-cim/`.
- Device enumeration policy is a v1 implementation decision: prefer zero
  devices until a real CIM runtime probe exists; use a stub `CIM0` device only
  behind an explicit development/simulator choice.
- `supports_op()` returns `false` for every operation in v1.
- `offload_op()` returns `false` in v1.
- `graph_compute()` returns `GGML_STATUS_FAILED` in v1.
- Device buffer APIs define the future boundary, but allocation may return
  `NULL` until a real CIM memory API or explicit simulator mode exists.

### 2.2 Quality attributes

- Maintainability: keep CIM integration hooks small and localized.
- Upstream-sync safety: avoid CUDA-scale edits in shared files; add only one
  include/register block and one `ggml_add_backend(CIM)` entry.
- Safety: no operation should silently run on CIM before an implementation is
  present.
- Build portability: the empty shell should compile without a CIM SDK by
  default. SDK/toolchain discovery is deferred to a later implementation phase.

## 3. Constraints and assumptions

- CIM is treated as an experimental backend until the user explicitly changes
  that status.
- No new environment-variable experiment switch is needed for the empty backend;
  CMake option `GGML_CIM` is the only enablement switch.
- If later runtime switches are added, they must be recorded in
  `expt-switch-env.md`.
- The first implementation should not modify llama model placement, KV-cache
  policy, CUDA NVFP4 code, or CPU reference code.
- The initial backend has no accelerated op coverage. Any graph requiring CIM
  compute must fall back through normal scheduler placement because
  `supports_op()` is false.
- The initial device memory numbers can be conservative placeholders. Prefer
  reporting `0, 0` or a documented configured value over pretending to know real
  hardware capacity.

## 4. CUDA reference boundary

`ggml-cuda` is the integration reference for where a backend hooks into ggml,
not a template to copy wholesale.

Keep these CUDA-derived concepts:

- one public backend header in `ggml/include/`;
- one backend source directory under `ggml/src/`;
- CMake enablement through `ggml_add_backend()` and
  `ggml_add_backend_library()`;
- a public `ggml_backend_*_reg()` registry factory;
- explicit backend initialization by device index;
- backend type predicate;
- backend-owned device buffer type accessor;
- device count, description, and memory query helpers.

Do not copy these CUDA-specific mechanisms:

- CUDA Toolkit discovery, `enable_language(CUDA)`, architecture selection, NVCC
  flags, or CUDA link libraries;
- `.cu`, `.cuh`, template-instantiation, cuBLAS, cuBLASLt, CUDA graph, stream,
  VMM, peer-copy, or pinned-host-buffer code;
- split tensor buffers until CIM has a real multi-device memory model;
- CUDA NVFP4, FP8, FlashAttention, MMQ, or other experiment code;
- CUDA environment variables or release-path diagnostics.

## 5. Candidate architectures

### Option A: minimal first-class backend shell

Add `ggml-cim` as a normal ggml backend with public header, CMake option,
registry entry, device interface, backend interface, and buffer type. The source
contains no kernels and advertises no op support.

Strengths:
- Matches existing ggml backend structure.
- Keeps future CIM work behind a stable boundary.
- Allows early build, registry, and packaging validation.

Weaknesses:
- Requires implementing basic buffer semantics even before compute exists.
- Adds public API surface that should remain stable.

### Option B: documentation-only placeholder

Write the design but do not add any backend target or header until kernels are
ready.

Strengths:
- Zero code churn.
- No risk of users enabling a non-functional backend.

Weaknesses:
- Defers build-system and registry risks.
- Does not establish the intended ownership boundary.

### Option C: CUDA fork with stubs

Copy the CUDA backend shape and replace CUDA-specific code with CIM stubs.

Strengths:
- Captures the full backend surface up front.
- Easy to compare with CUDA paths.

Weaknesses:
- Too much unused code and naming debt.
- Higher sync and review burden.
- Easy to accidentally preserve CUDA assumptions that do not apply to CIM.

## 6. Selected architecture

Select Option A.

The CUDA backend is the reference for where a backend integrates, not for how
much implementation to copy. `ggml-cim` should start closer to a small native
backend skeleton: one public header, one `CMakeLists.txt`, and one source file
with reg/device/backend/buffer interfaces. Compute support remains explicitly
unsupported until a later design adds actual CIM execution.

## 7. System decomposition

### Public API: `ggml/include/ggml-cim.h`

Required declarations:

- `#define GGML_CIM_NAME "CIM"`
- `GGML_BACKEND_API ggml_backend_t ggml_backend_cim_init(int device);`
- `GGML_BACKEND_API bool ggml_backend_is_cim(ggml_backend_t backend);`
- `GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cim_buffer_type(int device);`
- `GGML_BACKEND_API int ggml_backend_cim_get_device_count(void);`
- `GGML_BACKEND_API void ggml_backend_cim_get_device_description(int device, char * description, size_t description_size);`
- `GGML_BACKEND_API void ggml_backend_cim_get_device_memory(int device, size_t * free, size_t * total);`
- `GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cim_reg(void);`

Do not add host-pinned buffers, split buffers, stream accessors, graph controls,
or CIM-specific custom proc addresses in v1.

### Backend source: `ggml/src/ggml-cim/ggml-cim.cpp`

Required internal blocks:

- GUID helper: `ggml_backend_cim_guid()`.
- Backend context: stores at least device index and backend name.
- Backend interface:
  - `get_name`
  - `free`
  - `set_tensor_async` as `NULL` in v1
  - `get_tensor_async` as `NULL` in v1
  - `cpy_tensor_async` as `NULL` in v1
  - `synchronize` as `NULL` or no-op depending on whether buffer transfers are synchronous
  - graph plan functions as `NULL`
  - `graph_compute` returning `GGML_STATUS_FAILED`
  - event functions as `NULL`
- Buffer type interface:
  - `get_name`
  - `alloc_buffer`
  - `get_alignment`
  - optional `get_max_size`
  - optional `get_alloc_size`
  - `is_host = false`
- Buffer interface:
  - `free_buffer`
  - `get_base`
  - `init_tensor = NULL` unless CIM tensor metadata is needed later
  - `memset_tensor`
  - `set_tensor`
  - `get_tensor`
  - `cpy_tensor = NULL` in v1 unless same-buffer copies are trivial
  - `clear`
  - `reset = NULL`
- Device interface:
  - `get_name`
  - `get_description`
  - `get_memory`
  - `get_type`
  - `get_props`
  - `init_backend`
  - `get_buffer_type`
  - `get_host_buffer_type = NULL`
  - `buffer_from_host_ptr = NULL`
  - `supports_op = false`
  - `supports_buft`
  - `offload_op = false`
  - event functions as `NULL`
- Registry interface:
  - `get_name`
  - `get_device_count`
  - `get_device`
  - `get_proc_address`, initially only optional
    `ggml_backend_get_features`
- Dynamic backend export:
  - `GGML_BACKEND_DL_IMPL(ggml_backend_cim_reg)`

The first buffer implementation may use host allocation as a stand-in for CIM
device memory only if the buffer type still reports `is_host = false` and is
documented as a staging placeholder. This keeps the memory API testable without
claiming CPU backend compatibility.

### Build file: `ggml/src/ggml-cim/CMakeLists.txt`

Minimal contents:

- `set(TARGET_NAME ggml-cim)`
- `ggml_add_backend_library(${TARGET_NAME} ggml-cim.cpp ../../include/ggml-cim.h)`
- No SDK discovery in v1.
- No extra compile definitions except those required by future SDK selection.

### Registry integration: `ggml/src/ggml-backend-reg.cpp`

Required changes:

- Add:
  - `#ifdef GGML_USE_CIM`
  - `#include "ggml-cim.h"`
  - `#endif`
- Register before CPU and near other accelerator backends:
  - `#ifdef GGML_USE_CIM`
  - `register_backend(ggml_backend_cim_reg());`
  - `#endif`

Place CIM with other accelerator backends rather than after CPU so default
device enumeration remains consistent with existing GPU/backend ordering.

## 8. Build and packaging changes

### `ggml/CMakeLists.txt`

Required changes:

- Add option near backend options:
  - `option(GGML_CIM "ggml: use CIM" OFF)`
- Add public header:
  - `include/ggml-cim.h`

### `ggml/src/CMakeLists.txt`

Required changes:

- Add backend inclusion:
  - `ggml_add_backend(CIM)`

This automatically creates target `ggml-cim`, adds it to
`GGML_AVAILABLE_BACKENDS`, links it statically into `ggml` when
`GGML_BACKEND_DL=OFF`, and defines `GGML_USE_CIM` for static registration.

### `ggml/cmake/ggml-config.cmake.in`

Required changes:

- Add a no-dependency branch for `GGML_CIM` only if future CIM SDK libraries
  need imported package dependencies.
- For the v1 no-SDK shell, no special link-library block is required because the
  generic `GGML_AVAILABLE_BACKENDS` import loop will handle `ggml-cim`.

### Root `CMakeLists.txt`

Required changes:

- No required change unless a deprecated `LLAMA_CIM` alias is introduced.
- Do not add a deprecated alias in v1.

### `Makefile`

Required changes:

- Optional. The repository is CMake-first for new backend work.
- If legacy Makefile support is required later, add `GGML_CIM` object and define
  plumbing separately. Do not block the CMake backend shell on this.

### GitHub labeler and release workflows

Required changes:

- Optional for v1.
- Add `.github/labeler.yml` entries for `ggml/include/ggml-cim.h` and
  `ggml/src/ggml-cim/**` when code lands.
- Do not add release artifacts until the backend has a real SDK/runtime story.

## 9. Runtime behavior

### Device enumeration

V1 should pick one explicit device-enumeration policy during implementation:

- Conservative policy: expose zero devices until a real CIM runtime probe exists.
  This is safest for a no-implementation scaffold because `GGML_CIM=ON` cannot
  appear usable by accident.
- Development-stub policy: expose one placeholder device only behind an explicit
  simulator or development choice.
  - Name: `CIM0`
  - Description: `CIM backend placeholder`
  - Type: `GGML_BACKEND_DEVICE_TYPE_ACCEL`
  - Caps: `async = false`, `host_buffer = false`,
    `buffer_from_host_ptr = false`, `events = false`

Prefer the conservative policy unless the implementation change also adds a
focused registry smoke test that needs a visible placeholder device.

### Memory and buffer behavior

The buffer type exists so tests and future allocation paths have a stable API.
For v1:

- Alignment should use a conservative value such as `GGML_MEM_ALIGN`.
- Allocation failure returns `NULL`.
- `set_tensor`, `get_tensor`, `memset_tensor`, and `clear` are synchronous.
- `supports_buft()` returns true only for the matching CIM device buffer type.
- No split tensor buffer type is provided.
- No pinned host buffer type is provided.

### Compute behavior

No ops are supported:

- `supports_op()` returns false.
- `offload_op()` returns false.
- `graph_compute()` returns `GGML_STATUS_FAILED`.

This prevents accidental placement of graph nodes onto CIM before kernels exist.

## 10. Documentation and experiment records

Required documentation changes when implementation starts:

- Update this document if the file layout or runtime behavior changes.
- No `expt-switch-env.md` change is needed for the CMake-only v1 shell.
- If a runtime environment switch is introduced later, document it in
  `expt-switch-env.md` in the same change.
- Runtime validations that start `llama-server` or run PPL must create an
  `experiments/` folder with commands, logs, inputs, outputs, and result
  summary.
- Simple build and registry smoke tests do not require experiment folders unless
  they become llama-server startup validations.

## 11. Delivery plan

### Milestone 1: backend shell

Implement:

- `ggml/include/ggml-cim.h`
- `ggml/src/ggml-cim/CMakeLists.txt`
- `ggml/src/ggml-cim/ggml-cim.cpp`
- `ggml/CMakeLists.txt` option and public header entry
- `ggml/src/CMakeLists.txt` backend inclusion
- `ggml/src/ggml-backend-reg.cpp` include and registration blocks

Validation:

- Configure with `cmake -B build-cim -DGGML_CIM=ON`.
- Build target `ggml-cim`.
- Build a small target that links `ggml`.
- Run a registry smoke test or existing backend enumeration path to confirm
  backend `CIM` and device `CIM0` appear.
- Confirm no graph op is placed on CIM because `supports_op()` is false.

### Milestone 2: hardware discovery boundary

Add a narrow CIM runtime adapter only after the actual SDK/runtime contract is
known.

Possible additions:

- SDK discovery in `ggml/src/ggml-cim/CMakeLists.txt`.
- Real device count, description, and memory queries.
- Feature reporting through `ggml_backend_get_features`.
- Optional runtime switch documentation if behavior is gated by environment.

### Milestone 3: first compute path

Add the first CIM op behind a focused design.

Required decisions:

- Which op and tensor types are supported first.
- How host-to-CIM and CIM-to-host transfers work.
- Whether tensors need CIM-specific layout or metadata.
- How scheduler placement should prefer or avoid CIM.
- What correctness tests compare against CPU.

## 12. Complete modification checklist

Required for Milestone 1:

- Add `ggml/include/ggml-cim.h`.
- Add `ggml/src/ggml-cim/`.
- Add `ggml/src/ggml-cim/CMakeLists.txt`.
- Add `ggml/src/ggml-cim/ggml-cim.cpp`.
- Add `GGML_CIM` CMake option in `ggml/CMakeLists.txt`.
- Add `include/ggml-cim.h` to `GGML_PUBLIC_HEADERS`.
- Add `ggml_add_backend(CIM)` in `ggml/src/CMakeLists.txt`.
- Include `ggml-cim.h` under `#ifdef GGML_USE_CIM` in
  `ggml/src/ggml-backend-reg.cpp`.
- Register `ggml_backend_cim_reg()` under `#ifdef GGML_USE_CIM` in
  `ggml/src/ggml-backend-reg.cpp`.
- Ensure `GGML_BACKEND_DL_IMPL(ggml_backend_cim_reg)` is present in backend
  source.
- Add `.github/labeler.yml` path labels if repository hygiene requires it in
  the implementation change.

Explicitly not required for Milestone 1:

- CUDA, HIP, MUSA, Metal, SYCL, Vulkan, OpenCL, CANN, or RPC changes.
- llama model loader changes.
- scheduler policy changes.
- tensor type additions.
- quantization changes.
- KV-cache changes.
- environment-variable switches.
- Makefile support.
- release workflow artifacts.
- CIM SDK discovery.
- CIM kernels.

## 13. Risks and open questions

| Item | Status | Notes |
| --- | --- | --- |
| Device type enum | Confirmed | This checkout has `GGML_BACKEND_DEVICE_TYPE_ACCEL`; use it for the CIM placeholder. |
| Buffer placeholder semantics | Assumption | V1 can use host allocation internally while reporting non-host CIM buffer type. This is acceptable only because no compute ops are supported. |
| Dynamic backend loading | Confirmed | Existing `GGML_BACKEND_DL_IMPL` flow supports this shape. |
| Package config dependencies | Confirmed for v1 | No special `ggml-config.cmake.in` dependency block is needed without an SDK. |
| Makefile support | Assumption | CMake is sufficient for v1 unless explicitly requested. |
| Hardware discovery | TBD | Requires CIM SDK/runtime details. |

## Appendix A: decision log

| ID | Decision | Status | Rationale | Date |
| --- | --- | --- | --- | --- |
| ADR-001 | Add `ggml-cim` as a first-class backend shell, not a CUDA fork. | Confirmed | Keeps backend boundary real while avoiding unused CUDA assumptions. | 2026-06-10 |
| ADR-002 | Advertise no op support in v1. | Confirmed | Prevents accidental compute placement before kernels exist. | 2026-06-10 |
| ADR-003 | Use CMake option `GGML_CIM` as the only v1 switch. | Confirmed | No runtime behavior needs an environment switch yet. | 2026-06-10 |
