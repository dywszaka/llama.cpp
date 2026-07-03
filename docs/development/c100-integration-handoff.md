# C100 Integration Handoff

## Goal

Continue optional default-off C100 backend and runtime integration in this
llama.cpp checkout:

```bash
/home/allen/host_workspace/develop/llama.cpp
```

Do not commit changes unless explicitly requested.

## Current State

Current development branch:

```bash
release_cuda_nvfp4
```

Implemented so far:

- Optional default-off C100 build path.
- `GGML_C100` backend option and GGML backend registration.
- `LLAMA_C100_RUNTIME` runtime integration option.
- `C100_SIM_ROOT` simulator root cache path.
- CMake bridge module at `cmake/c100-runtime.cmake`.
- Prefixed simulator/runtime targets to avoid target-name collisions, such as
  `c100_sim_*` and `c100_runtime`.
- Firmware build/copy wiring for `firmware/llama.cpp/{su,ve}`.
- `llama-server` dependency and linkage hooks.
- C100 backend directory at `ggml/src/ggml-c100/`.
- Early preflight checks for C100 runtime prerequisites.
- CMake cache variables for the RISC-V firmware toolchain:
  - `C100_RISCV_TOOLCHAIN_DIR`
  - `C100_RISCV_PREFIX`
- Runtime build support for the local extracted toolchain at
  `/home/allen/host_workspace/develop/llama.cpp.sim/ci/toolchains/master-v20251230`.
- Build-local firmware toolchain overlay in `build-c100/c100-riscv-toolchain`
  that supplies the missing `riscv64-unknown-elf-bin2hex` helper expected by
  `scripts/elf2hex-split`.
- C100 simulator/runtime static targets are PIC so they can link into shared
  `libggml-c100.so`.
- `c100_runtime` and C100 extension archives use CMake `WHOLE_ARCHIVE`
  generator expressions so required runtime symbols are pulled into
  `libggml-c100.so` without leaking whole-archive across system libraries.
- C100 backend availability now checks that the runtime API is linked, while
  backend initialization starts the simulator runtime.
- The simulator runtime source supports a `C100_LLAMA_FIRMWARE_DIR` compile
  definition so it uses firmware from the active build tree instead of a
  hard-coded `build/firmware` path.

## Local Files

Changed or added files and directories:

- `CMakeLists.txt`
- `cmake/c100-runtime.cmake`
- `ggml/CMakeLists.txt`
- `ggml/src/CMakeLists.txt`
- `ggml/src/ggml-backend-reg.cpp`
- `ggml/src/ggml-c100/`
- `ggml/src/ggml-c100/CMakeLists.txt`
- `ggml/src/ggml-c100/ggml-c100.c`
- `tools/server/CMakeLists.txt`
- `expt-switch-env.md`

Reference implementation and build files:

- `/home/allen/host_workspace/develop/llama.cpp.sim/ci/Dockerfile`
- `/home/allen/host_workspace/develop/llama.cpp.sim/src/top/CMakeLists.txt`
- `/home/allen/host_workspace/develop/llama.cpp.sim/src/top/llama_cpp.cpp`
- `/home/allen/host_workspace/develop/llama.cpp.sim/firmware/llama.cpp/`
- `/home/allen/host_workspace/develop/llama.cpp.sim/ext/llama.cpp/ggml/src/ggml-c100/`

## External Requirements

The C100 runtime and firmware build expects the same dependency environment as
`llama.cpp.sim/ci/Dockerfile`.

Required host package:

- `device-tree-compiler`, which provides `dtc` for Spike configure.

Required RISC-V bare-metal toolchain:

```bash
ci/toolchains/riscv-toolchain-master-v20251230.tar.gz
```

The Dockerfile installs it to:

```bash
/opt/riscv/master-v20251230
```

Expected environment:

```bash
export RISCV=/opt/riscv/master-v20251230
export RISCV_TOOLCHAIN=/opt/riscv/master-v20251230
export RISCV_PATH=/opt/riscv/master-v20251230
export RISCV_PREFIX=/opt/riscv/master-v20251230/bin/riscv64-unknown-elf-
export PATH=/opt/riscv/master-v20251230/bin:$PATH
```

Current local dependency status:

- `dtc` is available at `/usr/bin/dtc`.
- The toolchain archive is present at
  `/home/allen/host_workspace/develop/llama.cpp.sim/ci/toolchains/riscv-toolchain-master-v20251230.tar.gz`.
- The archive was extracted to
  `/home/allen/host_workspace/develop/llama.cpp.sim/ci/toolchains/master-v20251230`.
- The archive does not include `riscv64-unknown-elf-bin2hex`; the llama.cpp C100
  CMake bridge generates a build-local compatible helper for firmware builds.

## Validation Already Run

Default build validation passed:

```bash
cmake -S . -B build-default
cmake --build build-default --target llama-cli
```

Default build without a simulator root passed:

```bash
cmake -S . -B build-default-nosim \
  -DC100_SIM_ROOT=/tmp/llama-c100-sim-not-present
cmake --build build-default-nosim --target llama-cli
```

C100 backend-only validation passed:

```bash
cmake -S . -B build-c100-backend \
  -DGGML_C100=ON \
  -DLLAMA_C100_RUNTIME=OFF
cmake --build build-c100-backend --target llama-cli
```

C100 runtime configure/build now passes with the local extracted toolchain:

```bash
cmake -S . -B build-c100 \
  -DGGML_C100=ON \
  -DLLAMA_C100_RUNTIME=ON \
  -DC100_SIM_ROOT=/home/allen/host_workspace/develop/llama.cpp.sim \
  -DC100_RISCV_TOOLCHAIN_DIR=/home/allen/host_workspace/develop/llama.cpp.sim/ci/toolchains/master-v20251230 \
  -DC100_RISCV_PREFIX=/home/allen/host_workspace/develop/llama.cpp.sim/ci/toolchains/master-v20251230/bin/riscv64-unknown-elf- \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build-c100 --target llama-server
```

C100 runtime symbol and device checks passed:

```bash
nm -D -A build-c100/bin/libggml-c100.so | c++filt | rg ' c100_llama_| get_simulator_instance| ggml_backend_c100'
build-c100/bin/llama-server --list-devices
```

The `--list-devices` output includes:

```text
Available devices:
  C100: C100 Simulator Backend (512 MiB, 512 MiB free)
```

A minimal `llama-server` startup smoke was recorded under:

```text
experiments/20260702T090734Z-c100-runtime-build-smoke/
```

The smoke reached `/health` with:

```json
{"status":"ok"}
```

`npm test` was not run because this repository root has no `package.json`.

## Next Steps

1. Continue functional validation of actual C100 offload execution. The smoke
   used `--n-gpu-layers 0`; it validates runtime registration/startup but not
   C100 compute correctness.
2. Run a small request/generation through C100 once an appropriately small C100
   workload or model/offload configuration is chosen.
3. Re-run the full CUDA baseline server validation when GPU/NVML is available.

## Current Blocker

Full CUDA-equivalent `llama-server` validation is currently blocked by local GPU
visibility:

```text
Failed to initialize NVML: Unknown Error
```

The C100 runtime build path, firmware build path, symbol linkage, device
enumeration, and minimal server startup smoke have been validated locally.
