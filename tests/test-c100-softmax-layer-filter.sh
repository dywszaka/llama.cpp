#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPH_CPP="${ROOT_DIR}/src/llama-graph.cpp"

grep -Fq 'static int llama_expt_c100_soft_max_layer()' "${GRAPH_CPP}"
grep -Fq 'LLAMA_C100_SOFTMAX_LAYER' "${GRAPH_CPP}"
grep -Fq 'llama_expt_pin_soft_max_to_c100(sched, kq, il)' "${GRAPH_CPP}"
grep -Fq 'if (target_layer >= 0 && il != target_layer)' "${GRAPH_CPP}"
