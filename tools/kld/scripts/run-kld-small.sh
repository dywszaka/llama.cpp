#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TOOL_DIR="${ROOT_DIR}/tools/kld"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_KLD="${RUN_KLD:-0}"
PARSE_ONLY="${PARSE_ONLY:-0}"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/experiments/kld-baseline-data}"
if [[ -z "${EXP_DIR+x}" ]]; then
  if [[ "${RUN_KLD}" == "1" && "${PARSE_ONLY}" != "1" ]]; then
    EXP_DIR="${ROOT_DIR}/experiments/$(date -u +%Y%m%dT%H%M%SZ)-kld-comparison"
  else
    EXP_DIR="${BASELINE_DIR}"
  fi
fi
BIN="${BIN:-${ROOT_DIR}/build_cuda/bin/llama-perplexity}"
MODEL="${MODEL:-/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf}"
SOURCE_PROMPT="${SOURCE_PROMPT:-${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw}"
PROMPT="${PROMPT:-${BASELINE_DIR}/data/wikitext-small.raw}"
SAMPLE_COUNT="${SAMPLE_COUNT:-8}"
START_DOCUMENT="${START_DOCUMENT:-0}"
MIN_CHARS="${MIN_CHARS:-200}"
CHUNKS="${CHUNKS:-8}"
CTX_SIZE="${CTX_SIZE:-512}"
UBATCH_SIZES="${UBATCH_SIZES:-128 512}"
CASE_MATRIX="${CASE_MATRIX:-nvfp4_outlier:nvfp4:nvfp4:LLAMA_NVFP4_KCACHE_OUTLIER=1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
SPARSE_BASELINE="${SPARSE_BASELINE:-1}"

mkdir -p \
  "${BASELINE_DIR}/baseline-logprobs" \
  "${BASELINE_DIR}/data" \
  "${BASELINE_DIR}/logs" \
  "${EXP_DIR}/baseline-logprobs" \
  "${EXP_DIR}/data" \
  "${EXP_DIR}/diagnostics" \
  "${EXP_DIR}/logs" \
  "${EXP_DIR}/results"

write_baseline_reference() {
  cat > "${BASELINE_DIR}/input-reference.md" <<EOF
# Input Reference

- Baseline contract: ${ROOT_DIR}/expt-baseline.md
- Baseline data directory: ${BASELINE_DIR}
- Binary: ${BIN}
- Model: ${MODEL}
- Source prompt: ${SOURCE_PROMPT}
- Small prompt: ${PROMPT}
- Small prompt bytes: $(wc -c < "${PROMPT}" 2>/dev/null || printf 'not_prepared')
- Dataset manifest: data/wikitext-small.manifest.json
- CUDA device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
- Fixed args: --n_gpu_layers 40, --batch-size 512, -t 32, -c ${CTX_SIZE}, --kv-unified, --no-warmup, --chunks ${CHUNKS}
- Baseline per ubatch: --cache-type-k f16, --cache-type-v f16, --ubatch-size {${UBATCH_SIZES}}, sparse=${SPARSE_BASELINE}, baseline-logprobs/ubatch_<N>.kld
- Tooling: tools/kld
- Diagnostic contract: tools/kld/collection-contract.md
- Diagnostic artifact policy: keep histograms and bounded row samples only; do not dump full K, V, attention-score, probability, or logits tensors.
- NVFP4 V-cache runtime requirements: flash attention disabled by omission, KQV offload enabled by omission, --kv-unified enabled.
- Note: --no-warmup is applied to both baseline and experiment because this is a KLD quality smoke run, not a startup timing run.
EOF
}

write_comparison_reference() {
  cat > "${EXP_DIR}/input-reference.md" <<EOF
# Input Reference

- Baseline contract: ${ROOT_DIR}/expt-baseline.md
- Baseline data directory: ${BASELINE_DIR}
- Baseline log-prob files: ${BASELINE_DIR}/baseline-logprobs/ubatch_<N>.kld
- Dataset manifest: ${BASELINE_DIR}/data/wikitext-small.manifest.json
- Small prompt: ${PROMPT}
- Binary: ${BIN}
- Model: ${MODEL}
- CUDA device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
- Fixed args: --n_gpu_layers 40, --batch-size 512, -t 32, -c ${CTX_SIZE}, --kv-unified, --no-warmup, --chunks ${CHUNKS}
- Experiment matrix: ${CASE_MATRIX}
- Experiment per case: --kl-divergence with the matching baseline-logprobs/ubatch_<N>.kld from ${BASELINE_DIR}
- Tooling: tools/kld
- Diagnostic contract: tools/kld/collection-contract.md
- Diagnostic artifact policy: keep histograms and bounded row samples only; do not dump full K, V, attention-score, probability, or logits tensors.
- NVFP4 V-cache runtime requirements: flash attention disabled by omission, KQV offload enabled by omission, --kv-unified enabled.
EOF
}

prepare_dataset() {
  python3 "${TOOL_DIR}/scripts/prepare-small-wikitext.py" \
    --source "${SOURCE_PROMPT}" \
    --output "${PROMPT}" \
    --manifest "${BASELINE_DIR}/data/wikitext-small.manifest.json" \
    --sample-count "${SAMPLE_COUNT}" \
    --start-document "${START_DOCUMENT}" \
    --min-chars "${MIN_CHARS}"
  write_baseline_reference
}

common_args=(
  -m "${MODEL}"
  -f "${PROMPT}"
  --n_gpu_layers 40
  --batch-size 512
  -t 32
  -c "${CTX_SIZE}"
  --kv-unified
  --no-warmup
  --chunks "${CHUNKS}"
)

clean_env=(
  -u LLAMA_NVFP4_KCACHE_OUTLIER
  -u LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8
  -u LLAMA_KCACHE_HYBRID_FP8_E4M3_E8M0_32_LAYERS
  -u GGML_CUDA_NVFP4_FATTN
  -u GGML_CUDA_NVFP4_FATTN_NO_FALLBACK
  -u GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH
  -u GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH
  -u GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC
  -u GGML_CUDA_NVFP4_FATTN_P_DIRECT
  -u GGML_CUDA_NVFP4_FATTN_DEBUG
)

run_baseline() {
  local ubatch="$1"
  local log_probs="${BASELINE_DIR}/baseline-logprobs/ubatch_${ubatch}.kld"
  local log="${BASELINE_DIR}/logs/baseline_ubatch_${ubatch}.raw.log"
  local kl_base_arg="--kl-divergence-base"
  if [[ "${SPARSE_BASELINE}" == "1" ]]; then
    kl_base_arg="--kl-divergence-base-sparse"
  fi

  {
    echo "case=baseline_ubatch_${ubatch}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "env=CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
    printf 'cmd=%q ' "${BIN}" "${common_args[@]}" \
      --cache-type-k f16 \
      --cache-type-v f16 \
      --ubatch-size "${ubatch}" \
      "${kl_base_arg}" "${log_probs}" \
      ${EXTRA_ARGS}
    printf '\n\n'
  } | tee "${log}"

  env "${clean_env[@]}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${BIN}" "${common_args[@]}" \
      --cache-type-k f16 \
      --cache-type-v f16 \
      --ubatch-size "${ubatch}" \
      "${kl_base_arg}" "${log_probs}" \
      ${EXTRA_ARGS} \
      >> "${log}" 2>&1
}

run_case() {
  local case_spec="$1"
  local ubatch="$2"
  local name k_type v_type env_spec
  IFS=: read -r name k_type v_type env_spec <<< "${case_spec}"
  local env_args=()
  if [[ -n "${env_spec:-}" && "${env_spec}" != "-" ]]; then
    IFS=',' read -r -a env_args <<< "${env_spec}"
  fi
  local log_probs="${BASELINE_DIR}/baseline-logprobs/ubatch_${ubatch}.kld"
  local case_name="kld_${name}_ubatch_${ubatch}"
  local log="${EXP_DIR}/logs/${case_name}.raw.log"
  local diag_dir="${EXP_DIR}/diagnostics/${case_name}"
  mkdir -p "${diag_dir}"

  if [[ ! -s "${log_probs}" ]]; then
    echo "missing baseline log-prob file: ${log_probs}" >&2
    echo "run with RUN_BASELINE=1 first" >&2
    return 2
  fi

  {
    echo "case=${case_name}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "env=CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ${env_spec:-}"
    echo "diagnostics_dir=${diag_dir}"
    printf 'cmd=%q ' "${BIN}" "${common_args[@]}" \
      --cache-type-k "${k_type}" \
      --cache-type-v "${v_type}" \
      --ubatch-size "${ubatch}" \
      --kl-divergence \
      --kl-divergence-base "${log_probs}" \
      ${EXTRA_ARGS}
    printf '\n\n'
  } | tee "${log}"

  env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${env_args[@]}" \
    "${BIN}" "${common_args[@]}" \
      --cache-type-k "${k_type}" \
      --cache-type-v "${v_type}" \
      --ubatch-size "${ubatch}" \
      --kl-divergence \
      --kl-divergence-base "${log_probs}" \
      ${EXTRA_ARGS} \
      >> "${log}" 2>&1
}

if [[ "${PARSE_ONLY}" == "1" ]]; then
  python3 "${TOOL_DIR}/scripts/parse-kld-results.py" --exp-dir "${EXP_DIR}"
  exit 0
fi

prepare_dataset

if [[ "${RUN_BASELINE}" == "1" ]]; then
  for ubatch in ${UBATCH_SIZES}; do
    run_baseline "${ubatch}"
  done
fi

if [[ "${RUN_KLD}" == "1" ]]; then
  write_comparison_reference
  for case_spec in ${CASE_MATRIX}; do
    for ubatch in ${UBATCH_SIZES}; do
      run_case "${case_spec}" "${ubatch}"
    done
  done
  python3 "${TOOL_DIR}/scripts/parse-kld-results.py" --exp-dir "${EXP_DIR}"
fi
