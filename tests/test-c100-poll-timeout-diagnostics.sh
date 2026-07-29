#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
C100_C="${ROOT_DIR}/ggml/src/ggml-c100/ggml-c100.c"

grep -Fq 'static void c100_dump_cmd_header' "${C100_C}"
grep -Fq 'C100: timeout cmd:' "${C100_C}"
grep -Fq 'c100_llama_read_cmd(&cmd)' "${C100_C}"
grep -Fq 'cmd_id=%u' "${C100_C}"
grep -Fq 'src0=0x%llx/%u' "${C100_C}"
grep -Fq 'ext=0x%llx/%u' "${C100_C}"
