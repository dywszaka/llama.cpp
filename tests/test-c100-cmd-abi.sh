#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cat > "${TMP_DIR}/test-c100-cmd-abi.c" <<'EOF'
#include <stddef.h>
#include <stdint.h>

#include "ggml/src/ggml-c100/llama_cmd_abi.h"

_Static_assert(sizeof(((llama_cmd_header_t*)0)->src0_addr) == sizeof(uint64_t),
               "src0_addr must be 64-bit");
_Static_assert(sizeof(((llama_cmd_header_t*)0)->src1_addr) == sizeof(uint64_t),
               "src1_addr must be 64-bit");
_Static_assert(sizeof(((llama_cmd_header_t*)0)->dst_addr) == sizeof(uint64_t),
               "dst_addr must be 64-bit");
_Static_assert(sizeof(((llama_cmd_header_t*)0)->ext_param_addr) == sizeof(uint64_t),
               "ext_param_addr must be 64-bit");
_Static_assert(LLAMA_CMD_STATUS_OFFSET == 0x34,
               "status offset must match the C100 firmware ABI");
_Static_assert(offsetof(llama_cmd_header_t, status) == LLAMA_CMD_STATUS_OFFSET,
               "status offset macro must match the command struct layout");
_Static_assert(sizeof(llama_cmd_header_t) == 112,
               "command header size must match the 64-bit address ABI");

int main(void) {
    return 0;
}
EOF

cc -I"${ROOT_DIR}" -std=c11 -Wall -Wextra -Werror \
    "${TMP_DIR}/test-c100-cmd-abi.c" -o "${TMP_DIR}/test-c100-cmd-abi"
"${TMP_DIR}/test-c100-cmd-abi"
