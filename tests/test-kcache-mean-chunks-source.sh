#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="${ROOT_DIR}/tools/kcache-mean/kcache-mean.cpp"

grep -q 'int32_t max_chunks' "${SRC}"
grep -q 'decode_recorded_prompt_chunks(ctx, collector, prompt_tokens, params.n_ctx, std::max<int32_t>(1, params.n_batch), params.n_chunks)' "${SRC}"
grep -q 'struct tensor_distribution_stats' "${SRC}"
grep -q 'parse_named_tensor_layer' "${SRC}"
grep -q 'Qcur' "${SRC}"
grep -q 'kq' "${SRC}"
grep -q -- '--tensor-dist-json' "${SRC}"
grep -q -- '--tensor-raw-dump-dir' "${SRC}"
grep -q 'ensure_tensor_raw_open' "${SRC}"
grep -q 'q_raw_f32' "${SRC}"
grep -q 'kq_raw_f32' "${SRC}"
grep -q 'Vcur' "${SRC}"
grep -q 'kqv' "${SRC}"
grep -q 'v_raw_f32' "${SRC}"
grep -q 'vp_raw_f32' "${SRC}"
