#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build llama.cpp CUDA variants with the local builder image.

Usage:
  .devops/llamacpp-cuda-variants.sh [options] [all|cuda|cuda-debug|cuda-c100|cuda-c100-debug]

Defaults:
  all

Options:
  --mount-prefix-from PATH
      Optional host path prefix to rewrite before passing paths to Docker.
  --mount-prefix-to PATH
      Replacement prefix used with --mount-prefix-from.

Environment:
  IMAGE                 default: allen/llamacpp-builder:latest
  LLAMA_CPP_ROOT        default: current repository root
  C100_SIM_ROOT         default: ${LLAMA_CPP_ROOT}/c100-sim
  CUDA_BUILD_DIR        default: $LLAMA_CPP_ROOT/build-cuda
  CUDA_DEBUG_BUILD_DIR  default: $LLAMA_CPP_ROOT/build-cuda-debug
  CUDA_C100_BUILD_DIR   default: $LLAMA_CPP_ROOT/build-cuda-c100
  CUDA_C100_DEBUG_BUILD_DIR
                       default: $LLAMA_CPP_ROOT/build-cuda-c100-debug
  BUILD_TARGET          default: llama-server
  BUILD_TYPE            default: Release
  CMAKE_GENERATOR       default: Ninja
  DOCKER_GPU_ARGS       optional, for example: --gpus all
  CMAKE_EXTRA_ARGS      optional extra CMake configure arguments for both builds
  CUDA_CMAKE_EXTRA_ARGS optional extra CMake configure arguments for CUDA build
  CUDA_DEBUG_CMAKE_EXTRA_ARGS
                       optional extra CMake configure arguments for CUDA Debug build
  C100_CMAKE_EXTRA_ARGS optional extra CMake configure arguments for CUDA+C100 build
  C100_DEBUG_CMAKE_EXTRA_ARGS
                       optional extra CMake configure arguments for CUDA+C100 Debug build
  DOCKER_MOUNT_PREFIX_FROM
                       same as --mount-prefix-from
  DOCKER_MOUNT_PREFIX_TO
                       same as --mount-prefix-to

Examples:
  .devops/llamacpp-cuda-variants.sh cuda
  .devops/llamacpp-cuda-variants.sh cuda-debug
  .devops/llamacpp-cuda-variants.sh cuda-c100-debug
  DOCKER_GPU_ARGS="--gpus all" .devops/llamacpp-cuda-variants.sh all
  CMAKE_EXTRA_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90" .devops/llamacpp-cuda-variants.sh cuda-c100
  .devops/llamacpp-cuda-variants.sh \
      --mount-prefix-from /home/allen/host_workspace \
      --mount-prefix-to /home/anka.zhao \
      all
EOF
}

repo_root() {
    local script_dir
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    cd -- "${script_dir}/.." && pwd
}

require_dir() {
    local name="$1"
    local path="$2"

    if [[ ! -d "${path}" ]]; then
        echo "${name} does not exist: ${path}" >&2
        exit 2
    fi
}

require_file() {
    local name="$1"
    local path="$2"

    if [[ ! -f "${path}" ]]; then
        echo "${name} does not exist: ${path}" >&2
        exit 2
    fi
}

docker_mount_path() {
    local path="$1"
    python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "${path}"
}

rewrite_mount_path() {
    local path="$1"
    local from="${DOCKER_MOUNT_PREFIX_FROM:-}"
    local to="${DOCKER_MOUNT_PREFIX_TO:-}"

    if [[ -z "${from}" && -z "${to}" ]]; then
        printf '%s\n' "${path}"
        return
    fi

    if [[ -z "${from}" || -z "${to}" ]]; then
        echo "both DOCKER_MOUNT_PREFIX_FROM and DOCKER_MOUNT_PREFIX_TO are required for path rewriting" >&2
        exit 2
    fi

    from="${from%/}"
    to="${to%/}"

    case "${path}" in
        "${from}")
            printf '%s\n' "${to}"
            ;;
        "${from}/"*)
            printf '%s%s\n' "${to}" "${path#"${from}"}"
            ;;
        *)
            printf '%s\n' "${path}"
            ;;
    esac
}

parse_args() {
    MODE=all

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            --mount-prefix-from)
                if [[ $# -lt 2 ]]; then
                    echo "--mount-prefix-from requires a value" >&2
                    exit 2
                fi
                DOCKER_MOUNT_PREFIX_FROM="$2"
                shift 2
                ;;
            --mount-prefix-to)
                if [[ $# -lt 2 ]]; then
                    echo "--mount-prefix-to requires a value" >&2
                    exit 2
                fi
                DOCKER_MOUNT_PREFIX_TO="$2"
                shift 2
                ;;
            all|cuda|cuda-debug|cuda-c100|cuda-c100-debug)
                MODE="$1"
                shift
                ;;
            *)
                usage >&2
                exit 2
                ;;
        esac
    done
}

docker_run_base() {
    local -a gpu_args=()
    if [[ -n "${DOCKER_GPU_ARGS:-}" ]]; then
        # Intentionally split like a shell command so DOCKER_GPU_ARGS="--gpus all" works.
        read -r -a gpu_args <<< "${DOCKER_GPU_ARGS}"
    fi

    docker run --rm \
        "${gpu_args[@]}" \
        -v "${LLAMA_CPP_DOCKER_ROOT}:/workspace/llama.cpp" \
        -w /workspace/llama.cpp \
        -e GIT_CONFIG_COUNT=1 \
        -e GIT_CONFIG_KEY_0=safe.directory \
        -e GIT_CONFIG_VALUE_0=/workspace/llama.cpp \
        -e LLAMA_CPP_DOCKER_ROOT="${LLAMA_CPP_DOCKER_ROOT}" \
        -e CUDA_BUILD_DIR="${CUDA_BUILD_DIR}" \
        -e BUILD_TARGET="${BUILD_TARGET}" \
        -e BUILD_TYPE="${BUILD_TYPE}" \
        -e CMAKE_GENERATOR="${CMAKE_GENERATOR}" \
        "${IMAGE}" \
        "$@"
}

configure_and_build_cuda() {
    local -a common_extra=()
    local -a cuda_extra=()
    local -a mode_extra=("$@")

    if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
        read -r -a common_extra <<< "${CMAKE_EXTRA_ARGS}"
    fi
    if [[ -n "${CUDA_CMAKE_EXTRA_ARGS:-}" ]]; then
        read -r -a cuda_extra <<< "${CUDA_CMAKE_EXTRA_ARGS}"
    fi

    docker_run_base bash -lc '
        set -euo pipefail

        if [[ ! -f /workspace/llama.cpp/CMakeLists.txt ]]; then
            echo "missing llama.cpp checkout inside container at /workspace/llama.cpp" >&2
            echo "check Docker bind mount source LLAMA_CPP_DOCKER_ROOT=${LLAMA_CPP_DOCKER_ROOT:-}" >&2
            exit 2
        fi

        cuda_stub_dir="${CUDA_STUB_DIR:-/usr/local/cuda/targets/x86_64-linux/lib/stubs}"
        exe_linker_flags="${CMAKE_EXE_LINKER_FLAGS:-}"
        if [[ " ${exe_linker_flags} " != *" -Wl,-rpath-link,${cuda_stub_dir} "* ]]; then
            exe_linker_flags="${exe_linker_flags:+${exe_linker_flags} }-Wl,-rpath-link,${cuda_stub_dir}"
        fi

        cmake -S /workspace/llama.cpp -B "${CUDA_BUILD_DIR}" -G "${CMAKE_GENERATOR}" \
            -DGGML_CUDA=ON \
            -DGGML_C100=OFF \
            -DLLAMA_BUILD_SERVER=ON \
            -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
            -DCMAKE_EXE_LINKER_FLAGS="${exe_linker_flags}" \
            "$@"

        cmake --build "${CUDA_BUILD_DIR}" --target "${BUILD_TARGET}" -j"$(nproc)"
    ' bash "${mode_extra[@]}" "${common_extra[@]}" "${cuda_extra[@]}"
}

configure_and_build_cuda_debug() {
    local -a debug_extra=(
        "-DCMAKE_CUDA_FLAGS_DEBUG=-G -g -lineinfo"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )
    local -a user_debug_extra=()

    if [[ -n "${CUDA_DEBUG_CMAKE_EXTRA_ARGS:-}" ]]; then
        read -r -a user_debug_extra <<< "${CUDA_DEBUG_CMAKE_EXTRA_ARGS}"
        debug_extra+=("${user_debug_extra[@]}")
    fi

    CUDA_BUILD_DIR="${CUDA_DEBUG_BUILD_DIR}"
    BUILD_TYPE=Debug

    configure_and_build_cuda "${debug_extra[@]}"
}

make_cmake_args_file() {
    local args_file="$1"
    shift

    : > "${args_file}"
    for arg in "$@"; do
        printf '%s\n' "${arg}" >> "${args_file}"
    done
}

configure_and_build_cuda_c100() {
    local -a extra=()
    local -a mode_extra=("$@")
    local -a user_extra=()
    local -a gpu_args=()
    local args_dir
    local args_file
    local container_args_file

    if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
        read -r -a user_extra <<< "${CMAKE_EXTRA_ARGS}"
        extra+=("${user_extra[@]}")
    fi
    if [[ -n "${C100_CMAKE_EXTRA_ARGS:-}" ]]; then
        read -r -a user_extra <<< "${C100_CMAKE_EXTRA_ARGS}"
        extra+=("${user_extra[@]}")
    fi
    extra+=("${mode_extra[@]}")
    if [[ -n "${DOCKER_GPU_ARGS:-}" ]]; then
        # Intentionally split like a shell command so DOCKER_GPU_ARGS="--gpus all" works.
        read -r -a gpu_args <<< "${DOCKER_GPU_ARGS}"
    fi

    args_dir="${LLAMA_CPP_ROOT}/.cache/llamacpp-cuda-variants"
    mkdir -p "${args_dir}"
    args_file="${args_dir}/cmake-extra-args.txt"
    container_args_file="/workspace/llama.cpp/.cache/llamacpp-cuda-variants/cmake-extra-args.txt"
    make_cmake_args_file "${args_file}" "${extra[@]}"

    if ! docker run --rm \
        "${gpu_args[@]}" \
        -v "${LLAMA_CPP_DOCKER_ROOT}:/workspace/llama.cpp" \
        -v "${C100_SIM_DOCKER_ROOT}:/workspace/llama.cpp/c100-sim" \
        -e GIT_CONFIG_COUNT=1 \
        -e GIT_CONFIG_KEY_0=safe.directory \
        -e GIT_CONFIG_VALUE_0=/workspace/llama.cpp \
        -e LLAMA_CPP_ROOT=/workspace/llama.cpp \
        -e C100_SIM_ROOT=/workspace/llama.cpp/c100-sim \
        -e BUILD_DIR="${CUDA_C100_BUILD_DIR}" \
        -e BUILD_TARGET="${BUILD_TARGET}" \
        -e BUILD_TYPE="${BUILD_TYPE}" \
        -e CMAKE_GENERATOR="${CMAKE_GENERATOR}" \
        -e CMAKE_EXTRA_ARGS_FILE="${container_args_file}" \
        "${IMAGE}" \
        build-llamacpp-c100-cuda; then
        rm -f "${args_file}"
        return 1
    fi

    rm -f "${args_file}"
}

configure_and_build_cuda_c100_debug() {
    CUDA_C100_DEVICE_DEBUG="${CUDA_C100_DEVICE_DEBUG:-0}"
    local -a debug_extra=(
        "-DCMAKE_CUDA_FLAGS_DEBUG=-g -lineinfo"
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    )
    local -a user_debug_extra=()

    if [[ "${CUDA_C100_DEVICE_DEBUG}" == "1" ]]; then
        debug_extra[0]="-DCMAKE_CUDA_FLAGS_DEBUG=-G -g -lineinfo"
    fi

    if [[ -n "${C100_DEBUG_CMAKE_EXTRA_ARGS:-}" ]]; then
        read -r -a user_debug_extra <<< "${C100_DEBUG_CMAKE_EXTRA_ARGS}"
        debug_extra+=("${user_debug_extra[@]}")
    fi

    CUDA_C100_BUILD_DIR="${CUDA_C100_DEBUG_BUILD_DIR}"
    BUILD_TYPE=Debug

    configure_and_build_cuda_c100 "${debug_extra[@]}"
}

main() {
    parse_args "$@"

    IMAGE="${IMAGE:-allen/llamacpp-builder:latest}"
    LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$(repo_root)}"
    C100_SIM_ROOT="${C100_SIM_ROOT:-${LLAMA_CPP_ROOT}/c100-sim}"
    CUDA_BUILD_DIR="${CUDA_BUILD_DIR:-/workspace/llama.cpp/build-cuda}"
    CUDA_DEBUG_BUILD_DIR="${CUDA_DEBUG_BUILD_DIR:-/workspace/llama.cpp/build_cuda_debug}"
    CUDA_C100_BUILD_DIR="${CUDA_C100_BUILD_DIR:-/workspace/llama.cpp/build-cuda-c100}"
    CUDA_C100_DEBUG_BUILD_DIR="${CUDA_C100_DEBUG_BUILD_DIR:-/workspace/llama.cpp/build-cuda-c100-debug}"
    BUILD_TARGET="${BUILD_TARGET:-llama-server}"
    BUILD_TYPE="${BUILD_TYPE:-Release}"
    CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"

    LLAMA_CPP_ROOT="$(docker_mount_path "${LLAMA_CPP_ROOT}")"
    C100_SIM_ROOT="$(docker_mount_path "${C100_SIM_ROOT}")"
    LLAMA_CPP_DOCKER_ROOT="$(rewrite_mount_path "${LLAMA_CPP_ROOT}")"
    C100_SIM_DOCKER_ROOT="$(rewrite_mount_path "${C100_SIM_ROOT}")"

    require_file "llama.cpp CMakeLists.txt" "${LLAMA_CPP_ROOT}/CMakeLists.txt"

    case "${MODE}" in
        cuda)
            configure_and_build_cuda
            ;;
        cuda-debug)
            configure_and_build_cuda_debug
            ;;
        cuda-c100)
            require_dir "C100 simulator root" "${C100_SIM_ROOT}"
            require_file "C100 simulator CMakeLists.txt" "${C100_SIM_ROOT}/CMakeLists.txt"
            configure_and_build_cuda_c100
            ;;
        cuda-c100-debug)
            require_dir "C100 simulator root" "${C100_SIM_ROOT}"
            require_file "C100 simulator CMakeLists.txt" "${C100_SIM_ROOT}/CMakeLists.txt"
            configure_and_build_cuda_c100_debug
            ;;
        all)
            configure_and_build_cuda
            require_dir "C100 simulator root" "${C100_SIM_ROOT}"
            require_file "C100 simulator CMakeLists.txt" "${C100_SIM_ROOT}/CMakeLists.txt"
            configure_and_build_cuda_c100
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
