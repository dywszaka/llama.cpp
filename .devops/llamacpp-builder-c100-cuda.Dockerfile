# syntax=docker/dockerfile:1

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG RISCV_TOOLCHAIN_ARCHIVE=ci/toolchains/riscv-toolchain-master-v20251230.tar.gz
ARG RISCV_TOOLCHAIN_DIR=/opt/riscv/master-v20251230

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        bash \
        bison \
        build-essential \
        ca-certificates \
        ccache \
        clang-format-15 \
        clang-tidy-15 \
        cmake \
        curl \
        device-tree-compiler \
        doxygen \
        environment-modules \
        flex \
        g++ \
        gcc \
        gdb \
        git \
        graphviz \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libfmt-dev \
        libgmp-dev \
        libmpfr-dev \
        libtool \
        make \
        ninja-build \
        nlohmann-json3-dev \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        rsync \
        valgrind \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

COPY ${RISCV_TOOLCHAIN_ARCHIVE} /tmp/riscv-toolchain.tar.gz

RUN mkdir -p "${RISCV_TOOLCHAIN_DIR}" && \
    tar -xzf /tmp/riscv-toolchain.tar.gz -C "${RISCV_TOOLCHAIN_DIR}" && \
    rm -f /tmp/riscv-toolchain.tar.gz && \
    test -x "${RISCV_TOOLCHAIN_DIR}/bin/riscv64-unknown-elf-gcc"

RUN <<'EOF'
cat > /usr/local/bin/build-llamacpp-c100-cuda <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'HELP'
Configure and build llama.cpp with CUDA and the C100 runtime backend.

Expected mounts:
  -v /path/to/llama.cpp:/workspace/llama.cpp
  -v /path/to/llama.cpp.sim:/workspace/llama.cpp.sim

Environment overrides:
  LLAMA_CPP_ROOT    default: /workspace/llama.cpp
  C100_SIM_ROOT     default: /workspace/llama.cpp.sim
  BUILD_DIR         default: $LLAMA_CPP_ROOT/build-cuda-c100
  BUILD_TARGET      default: llama-server
  BUILD_TYPE        default: Release
  CMAKE_GENERATOR   default: Ninja
  CUDA_STUB_DIR     default: /usr/local/cuda/targets/x86_64-linux/lib/stubs
  CMAKE_EXE_LINKER_FLAGS
                    optional extra executable linker flags
  CMAKE_EXTRA_ARGS  optional extra CMake configure arguments
  CMAKE_EXTRA_ARGS_FILE
                    optional newline-delimited extra CMake configure arguments
HELP
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-/workspace/llama.cpp}"
C100_SIM_ROOT="${C100_SIM_ROOT:-/workspace/llama.cpp.sim}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_CPP_ROOT}/build-cuda-c100}"
BUILD_TARGET="${BUILD_TARGET:-llama-server}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
CUDA_STUB_DIR="${CUDA_STUB_DIR:-/usr/local/cuda/targets/x86_64-linux/lib/stubs}"
CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:-}"

if [[ " ${CMAKE_EXE_LINKER_FLAGS} " != *" -Wl,-rpath-link,${CUDA_STUB_DIR} "* ]]; then
    CMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:+${CMAKE_EXE_LINKER_FLAGS} }-Wl,-rpath-link,${CUDA_STUB_DIR}"
fi

if [[ ! -f "${LLAMA_CPP_ROOT}/CMakeLists.txt" ]]; then
    echo "missing llama.cpp checkout at LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT}" >&2
    exit 2
fi

if [[ ! -f "${C100_SIM_ROOT}/CMakeLists.txt" ]]; then
    echo "missing llama.cpp.sim checkout at C100_SIM_ROOT=${C100_SIM_ROOT}" >&2
    exit 2
fi

cmake_extra_args=()
if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
    read -r -a cmake_extra_args <<< "${CMAKE_EXTRA_ARGS}"
fi
if [[ -n "${CMAKE_EXTRA_ARGS_FILE:-}" ]]; then
    if [[ ! -f "${CMAKE_EXTRA_ARGS_FILE}" ]]; then
        echo "missing CMAKE_EXTRA_ARGS_FILE=${CMAKE_EXTRA_ARGS_FILE}" >&2
        exit 2
    fi

    while IFS= read -r arg || [[ -n "${arg}" ]]; do
        [[ -z "${arg}" ]] && continue
        cmake_extra_args+=("${arg}")
    done < "${CMAKE_EXTRA_ARGS_FILE}"
fi

cmake -S "${LLAMA_CPP_ROOT}" -B "${BUILD_DIR}" -G "${CMAKE_GENERATOR}" \
    -DGGML_CUDA=ON \
    -DGGML_C100=ON \
    -DLLAMA_C100_RUNTIME=ON \
    -DC100_SIM_ROOT="${C100_SIM_ROOT}" \
    -DC100_RISCV_TOOLCHAIN_DIR="${RISCV_TOOLCHAIN}" \
    -DC100_RISCV_PREFIX="${RISCV_PREFIX}" \
    -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}" \
    "${cmake_extra_args[@]}"

cmake --build "${BUILD_DIR}" --target "${BUILD_TARGET}" -j"$(nproc)"
SCRIPT
chmod +x /usr/local/bin/build-llamacpp-c100-cuda
EOF

# The CUDA driver stub ships as libcuda.so, while ggml-cuda records
# libcuda.so.1 as a build-time dependency when linking executables.
RUN ln -sf libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1 && \
    ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

ENV RISCV=/opt/riscv/master-v20251230
ENV RISCV_TOOLCHAIN=/opt/riscv/master-v20251230
ENV RISCV_PATH=/opt/riscv/master-v20251230
ENV RISCV_PREFIX=/opt/riscv/master-v20251230/bin/riscv64-unknown-elf-
ENV PATH=/opt/riscv/master-v20251230/bin:/usr/lib/ccache:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs:/usr/local/cuda/compat
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib
ENV CCACHE_DIR=/workspace/.ccache
ENV CCACHE_COMPRESS=1
ENV CCACHE_MAXSIZE=5G
ENV CMAKE_BUILD_PARALLEL_LEVEL=10

WORKDIR /workspace

CMD ["/bin/bash"]
