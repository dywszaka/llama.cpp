#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DOCKER_MOUNT_PREFIX_FROM="${DOCKER_MOUNT_PREFIX_FROM:-/home/allen/host_workspace}"
DOCKER_MOUNT_PREFIX_TO="${DOCKER_MOUNT_PREFIX_TO:-/home/anka.zhao}"

exec "${script_dir}/llamacpp-cuda-variants.sh" \
    --mount-prefix-from "${DOCKER_MOUNT_PREFIX_FROM}" \
    --mount-prefix-to "${DOCKER_MOUNT_PREFIX_TO}" \
    "$@"
