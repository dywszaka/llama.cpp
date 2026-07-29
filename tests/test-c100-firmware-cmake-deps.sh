#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE_FILE="${ROOT_DIR}/cmake/c100-runtime.cmake"

grep -Fq 'file(GLOB_RECURSE firmware_deps CONFIGURE_DEPENDS' "${CMAKE_FILE}"
grep -Fq 'DEPENDS ${firmware_deps}' "${CMAKE_FILE}"
