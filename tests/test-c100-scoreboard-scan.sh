#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCOREBOARD_C="${ROOT_DIR}/c100-sim/firmware/llama.cpp/su/llama_cmd_scoreboard.c"

if ! grep -Fq 'uint8_t start_idx = sb->head;' "${SCOREBOARD_C}"; then
    echo "scoreboard_check_complete must start scanning at sb->head" >&2
    exit 1
fi
if grep -Fq 'uint8_t start_idx = sb->tail;' "${SCOREBOARD_C}"; then
    echo "scoreboard_check_complete must scan issued entries from head, not tail" >&2
    exit 1
fi
