#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-chunk PPL runner for the CUDA+C100 build, with two modes:
#   baseline      CUDA-only baseline PPL run
#   c100-softmax  C100 softmax run (LLAMA_EXPT_C100_SOFT_MAX=1)
#
# Usage: run-ppl-one-chunk.sh <baseline|c100-softmax>
#
# The build-cuda-c100 artifacts link against CUDA 13 runtime libs
# (libcudart.so.13, libcublas.so.13, ...) that are not installed on this host.
# Rather than baking artifacts/models/data into a custom image, this runner
# executes the binary inside the stock `nvidia/cuda:13.0.0-runtime-ubuntu24.04`
# image, mounting the repo (bin + model + data) and installing the small
# runtime packages the stock image lacks at container startup. C100 softmax also
# needs device-tree-compiler because the C100/Spike runtime invokes `dtc`.
#
# Host/Docker path mapping:
#   This host exposes the repo under a bind mount (HOST_MOUNT_PREFIX) that the
#   Docker daemon, living in root's mount namespace, cannot see directly. When
#   run on the host, the repo path is rewritten from HOST_MOUNT_PREFIX to
#   DOCKER_MOUNT_PREFIX for the `docker run -v` source. The script then
#   re-invokes itself inside the container (LLAMA_IN_DOCKER=1), where it runs
#   the perplexity binary directly with container-internal paths.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_REL="c100-sim-scripts/$(basename -- "${BASH_SOURCE[0]}")"
BIN_REL="build-cuda-c100/bin/llama-perplexity"

# Path-prefix rewrite from the host-visible repo root to the Docker-daemon-
# visible source path. Override via env if the layout changes.
HOST_MOUNT_PREFIX="${HOST_MOUNT_PREFIX:-/home/allen/host_workspace}"
DOCKER_MOUNT_PREFIX="${DOCKER_MOUNT_PREFIX:-/home/anka.zhao}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/llama.cpp}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-nvidia/cuda:13.0.0-runtime-ubuntu24.04}"

# --- Mode selection --------------------------------------------------------
MODE="${1:-${LLAMA_PPL_MODE:-}}"
case "${MODE}" in
  baseline)
    EXP_REL="experiments/20260703T075735Z-cuda-c100-release-ppl-one-chunk"
    LOG_REL="${EXP_REL}/logs/ppl.log"
    TIME_REL="${EXP_REL}/results/time.txt"
    DEVICE_ARGS=(--device CUDA0)
    SET_C100_SOFTMAX=0
    SCHED_DEBUG=0
    REQUIRE_PPL=0
    ;;
  c100-softmax)
    EXP_REL="experiments/20260703T084713Z-cuda-c100-softmax-ppl-one-chunk"
    LOG_REL="${EXP_REL}/logs/ppl-c100-softmax.log"
    TIME_REL="${EXP_REL}/results/time-c100-softmax.txt"
    DEVICE_ARGS=(--device CUDA0,C100 --tensor-split 1,0)
    SET_C100_SOFTMAX=1
    SCHED_DEBUG=2
    REQUIRE_PPL=1
    ;;
  *)
    echo "usage: $(basename -- "${BASH_SOURCE[0]}") <baseline|c100-softmax>" >&2
    exit 2
    ;;
esac

# --- Host mode: relaunch this script inside the stock CUDA runtime container.
if [ -z "${LLAMA_IN_DOCKER:-}" ]; then
  host_root="${ROOT_DIR}"
  docker_root="${host_root/#${HOST_MOUNT_PREFIX}/${DOCKER_MOUNT_PREFIX}}"
  if [ "${docker_root}" = "${host_root}" ]; then
    echo "warning: ROOT_DIR '${host_root}' did not start with HOST_MOUNT_PREFIX" \
         "'${HOST_MOUNT_PREFIX}'; docker mount source left unchanged" >&2
  fi
  mkdir -p "${ROOT_DIR}/${EXP_REL}/logs" "${ROOT_DIR}/${EXP_REL}/results"

  exec docker run --rm --runtime=nvidia --gpus all \
      -v "${docker_root}:${CONTAINER_ROOT}" \
      -w "${CONTAINER_ROOT}" \
      -e LLAMA_IN_DOCKER=1 \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e LLAMA_PPL_MODE="${MODE}" \
      --entrypoint "" \
      "${RUNTIME_IMAGE}" \
      bash "${CONTAINER_ROOT}/${SELF_REL}" "${MODE}"
fi

# --- Container mode (LLAMA_IN_DOCKER=1): install missing libs, then run. ---
cd "${CONTAINER_ROOT}"

# Stock CUDA runtime image lacks libcurl4 (direct link dep of llama-perplexity)
# and libgomp1 (link dep of libggml-cpu). The C100/Spike runtime also invokes
# dtc at startup, so install device-tree-compiler for C100 softmax mode.
APT_PACKAGES=(libcurl4 libgomp1)
if [ "${SET_C100_SOFTMAX}" -eq 1 ]; then
  APT_PACKAGES+=(device-tree-compiler)
fi

needs_apt=0
if ! ldconfig -p 2>/dev/null | grep -q 'libcurl\.so\.4'; then
  needs_apt=1
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libgomp\.so\.1'; then
  needs_apt=1
fi
if [ "${SET_C100_SOFTMAX}" -eq 1 ] && ! command -v dtc >/dev/null 2>&1; then
  needs_apt=1
fi
if [ "${needs_apt}" -eq 1 ]; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends "${APT_PACKAGES[@]}"
fi

BIN="${CONTAINER_ROOT}/${BIN_REL}"
LOG="${CONTAINER_ROOT}/${LOG_REL}"
TIME_LOG="${CONTAINER_ROOT}/${TIME_REL}"
mkdir -p "$(dirname "${LOG}")" "$(dirname "${TIME_LOG}")"

start_epoch="$(date +%s)"
status=0

(
  export CUDA_VISIBLE_DEVICES=0
  export PROJECT_ROOT="${CONTAINER_ROOT}/build-cuda-c100"
  export LD_LIBRARY_PATH="${CONTAINER_ROOT}/build-cuda-c100/bin:/usr/local/cuda/lib64"
  if [ "${SET_C100_SOFTMAX}" -eq 1 ]; then
    export LLAMA_EXPT_C100_SOFT_MAX=1
  else
    unset LLAMA_EXPT_C100_SOFT_MAX || true
  fi
  if [ "${SCHED_DEBUG}" -gt 0 ]; then
    export GGML_SCHED_DEBUG="${SCHED_DEBUG}"
  else
    unset GGML_SCHED_DEBUG || true
  fi

  "${BIN}" \
      "${DEVICE_ARGS[@]}" \
      -m "${CONTAINER_ROOT}/data/models/qwen3-8b-nvfp4.gguf" \
      -f "${CONTAINER_ROOT}/data/wikitext/wikitext-2-raw/wiki.test.raw" \
      --cache-type-k f16 \
      --cache-type-v f16 \
      --n_gpu_layers 40 \
      --batch-size 512 \
      --ubatch-size 512 \
      -t 32 \
      -c 512 \
      --kv-unified \
      --chunks 1
) 2>&1 | tee "${LOG}" || status="${PIPESTATUS[0]}"

end_epoch="$(date +%s)"

# The C100-softmax run may fail before producing a PPL estimate; treat the
# absence of a final PPL line as a failure for that mode.
if [ "${REQUIRE_PPL}" -eq 1 ]; then
  if ! grep -q 'Final estimate: PPL' "${LOG}"; then
    status=1
  fi
fi

{
  printf 'start_epoch=%s\n' "${start_epoch}"
  printf 'end_epoch=%s\n' "${end_epoch}"
  printf 'elapsed_seconds=%s\n' "$((end_epoch - start_epoch))"
  printf 'exit_status=%s\n' "${status}"
} > "${TIME_LOG}"

exit "${status}"
