#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/profile-llama-server-ncu.sh [options] [-- extra llama-server args]

Launch llama-server under NVIDIA Nsight Compute, send one fixed completion
request, and save the server logs, request/response, NCU report, and exact run
metadata under experiments/.

Defaults follow expt-baseline.md's llama-server baseline:
  --cache-type-k f16 --cache-type-v f16 --kv-unified

Options:
  --name NAME                 Run name, default: baseline-f16-f16
  --out-dir DIR              Artifact directory, default: experiments/ncu-<timestamp>-<name>
  --bin PATH                 llama-server binary, default: build_cuda/bin/llama-server
  --model PATH               GGUF model, default: qwen3-8b-nvfp4.gguf path from expt-baseline.md
  --host HOST                Server host, default: 127.0.0.1
  --port PORT                Server port, default: 8080
  --cache-type-k TYPE        K cache type, default: f16
  --cache-type-v TYPE        V cache type, default: f16
  --ctx-size N               Context size, default: 2048
  --n-predict N              Request generation tokens, default: 32
  --prompt-file PATH         JSON payload file to use instead of generated payload
  --ncu-bin PATH             ncu binary, default: first ncu in PATH
  --ncu-set SET              NCU section set, default: basic
  --ncu-kernel-name FILTER   Passed to ncu -k, e.g. 'regex:(nvfp4|fp8|gemm)'
  --ncu-launch-count N       Passed to ncu -c when N > 0, default: 0 (unlimited)
  --ncu-launch-skip N        Passed to ncu -s when N > 0, default: 0
  --dry-run                  Create artifacts and print commands, but do not start server
  -h, --help                 Show this help

Useful environment switches:
  CUDA_VISIBLE_DEVICES=0
  LLAMA_NVFP4_VCACHE_LAYER_GLOBAL_SCALE=experiments/qwen3-8b-v-layer-absmax.json
  GGML_CUDA_DISABLE_GRAPHS=1      Diagnostic only; record that this changes baseline env.
EOF
}

quote_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v arg "%q" "$arg"
    out+="${arg} "
  done
  printf '%s\n' "${out% }"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

name="baseline-f16-f16"
out_dir=""
bin="${ROOT_DIR}/build_cuda/bin/llama-server"
model="/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf"
host="127.0.0.1"
port="8080"
cache_type_k="f16"
cache_type_v="f16"
ctx_size="2048"
n_predict="32"
prompt_file=""
ncu_bin="$(command -v ncu || true)"
ncu_set="basic"
ncu_kernel_name=""
ncu_launch_count="0"
ncu_launch_skip="0"
dry_run=0
server_extra=()

while (($# > 0)); do
  case "$1" in
    --name) name="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --bin) bin="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --cache-type-k) cache_type_k="$2"; shift 2 ;;
    --cache-type-v) cache_type_v="$2"; shift 2 ;;
    --ctx-size) ctx_size="$2"; shift 2 ;;
    --n-predict) n_predict="$2"; shift 2 ;;
    --prompt-file) prompt_file="$2"; shift 2 ;;
    --ncu-bin) ncu_bin="$2"; shift 2 ;;
    --ncu-set) ncu_set="$2"; shift 2 ;;
    --ncu-kernel-name) ncu_kernel_name="$2"; shift 2 ;;
    --ncu-launch-count) ncu_launch_count="$2"; shift 2 ;;
    --ncu-launch-skip) ncu_launch_skip="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; server_extra=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${out_dir}" ]]; then
  out_dir="${ROOT_DIR}/experiments/ncu-${timestamp}-${name}"
elif [[ "${out_dir}" != /* ]]; then
  out_dir="${ROOT_DIR}/${out_dir}"
fi

mkdir -p "${out_dir}"

request_payload="${out_dir}/request.json"
server_log="${out_dir}/server.log"
server_stdout="${out_dir}/server.stdout.log"
ncu_stdout="${out_dir}/ncu.stdout.log"
ncu_stderr="${out_dir}/ncu.stderr.log"
response_body="${out_dir}/response.json"
response_meta="${out_dir}/response.meta"
report_base="${out_dir}/profile"
report_file="${report_base}.ncu-rep"

if [[ -n "${prompt_file}" ]]; then
  cp "${prompt_file}" "${request_payload}"
else
  cat > "${request_payload}" <<EOF
{
  "prompt": "You are profiling llama.cpp GPU execution. Summarize why a fixed request is useful for comparing baseline f16 KV cache against K or V cache quantization experiments. Include the terms prefill, decode, K cache, V cache, FP4, FP8, and Tensor Core.",
  "n_predict": ${n_predict},
  "temperature": 0.0,
  "cache_prompt": false,
  "stream": false
}
EOF
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LLAMA_STDOUT_FILE="${LLAMA_STDOUT_FILE:-${server_log}}"

env | LC_ALL=C sort | grep -E '^(CUDA_VISIBLE_DEVICES|GGML_|LLAMA_)=' > "${out_dir}/env.txt" || true

server_cmd=(
  "${bin}"
  -m "${model}"
  --n_gpu_layers 40
  --host "${host}"
  --batch-size 2048
  --ubatch-size 512
  --port "${port}"
  -t 32
  -c "${ctx_size}"
  --cache-type-k "${cache_type_k}"
  --cache-type-v "${cache_type_v}"
  --kv-unified
  --log-file "${server_log}"
)

if ((${#server_extra[@]} > 0)); then
  server_cmd+=("${server_extra[@]}")
fi

ncu_cmd=(
  "${ncu_bin}"
  --mode launch-and-attach
  --forward-signals
  --target-processes application-only
  --profile-from-start yes
  --replay-mode kernel
  --graph-profiling node
  --set "${ncu_set}"
  --force-overwrite
  --export "${report_base}"
)

if [[ "${ncu_launch_count}" != "0" ]]; then
  ncu_cmd+=(--launch-count "${ncu_launch_count}")
fi
if [[ "${ncu_launch_skip}" != "0" ]]; then
  ncu_cmd+=(--launch-skip "${ncu_launch_skip}")
fi
if [[ -n "${ncu_kernel_name}" ]]; then
  ncu_cmd+=(--kernel-name-base demangled --kernel-name "${ncu_kernel_name}")
fi

full_cmd=("${ncu_cmd[@]}" "${server_cmd[@]}")

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd $(printf '%q' "${ROOT_DIR}")"
  echo "export CUDA_VISIBLE_DEVICES=$(printf '%q' "${CUDA_VISIBLE_DEVICES}")"
  echo "export LLAMA_STDOUT_FILE=$(printf '%q' "${LLAMA_STDOUT_FILE}")"
  while IFS='=' read -r k v; do
    [[ -z "${k}" ]] && continue
    [[ "${k}" == "LLAMA_STDOUT_FILE" ]] && continue
    echo "export ${k}=$(printf '%q' "${v}")"
  done < <(env | LC_ALL=C sort | grep -E '^(GGML_|LLAMA_EXPERIMENT_|LLAMA_NVFP4_)=' || true)
  quote_cmd "${full_cmd[@]}"
} > "${out_dir}/run-ncu-server.sh"
chmod +x "${out_dir}/run-ncu-server.sh"

cat > "${out_dir}/summary.md" <<EOF
# NCU llama-server profile: ${name}

- Created: ${timestamp}
- Baseline contract: see \`expt-baseline.md\`
- K cache: \`${cache_type_k}\`
- V cache: \`${cache_type_v}\`
- NCU set: \`${ncu_set}\`
- NCU kernel filter: \`${ncu_kernel_name:-none}\`
- NCU launch count: \`${ncu_launch_count}\`
- NCU launch skip: \`${ncu_launch_skip}\`
- Request: \`request.json\`
- Response: \`response.json\`
- Server log: \`server.log\`
- NCU report: \`profile.ncu-rep\`

## Interpretation Checklist
- Confirm server args and env in \`run-ncu-server.sh\` and \`env.txt\`.
- Confirm cache types and one-shot runtime logs in \`server.log\`.
- Inspect \`ncu-details.csv\` / \`ncu-raw.csv\` for relevant quantization, staging, and GEMM kernels.
- Compare this folder against a f16/f16 baseline folder with the same request and unchanged baseline parameters except the switch or cache type under test.
EOF

echo "artifact directory: ${out_dir}"
echo "server command: $(quote_cmd "${server_cmd[@]}")"
echo "ncu command: $(quote_cmd "${ncu_cmd[@]}")"

if ((dry_run)); then
  echo "dry run only"
  exit 0
fi

if [[ -z "${ncu_bin}" || ! -x "${ncu_bin}" ]]; then
  echo "ncu binary not found or not executable: ${ncu_bin}" >&2
  exit 1
fi
if [[ ! -x "${bin}" ]]; then
  echo "llama-server binary not found or not executable: ${bin}" >&2
  exit 1
fi
if [[ ! -f "${model}" ]]; then
  echo "model not found: ${model}" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${ncu_pid:-}" ]] && kill -0 "${ncu_pid}" 2>/dev/null; then
    kill -INT "${ncu_pid}" 2>/dev/null || true
    sleep 2
    kill -TERM "${ncu_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${full_cmd[@]}" > >(tee "${ncu_stdout}") 2> >(tee "${ncu_stderr}" >&2) &
ncu_pid=$!

ready=0
for _ in $(seq 1 "${SERVER_READY_TIMEOUT:-300}"); do
  if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${ncu_pid}" 2>/dev/null; then
    echo "ncu/llama-server exited before becoming ready" >&2
    wait "${ncu_pid}" || true
    exit 1
  fi
  sleep 1
done

if ((ready == 0)); then
  echo "server did not become ready before timeout" >&2
  exit 1
fi

http_code="$(
  curl -sS \
    -H 'Content-Type: application/json' \
    -X POST \
    --data @"${request_payload}" \
    -o "${response_body}" \
    -w '%{http_code}' \
    "http://${host}:${port}/completion"
)"
echo "http_code=${http_code}" > "${response_meta}"

sleep "${POST_REQUEST_PROFILE_SECONDS:-3}"
cleanup
trap - EXIT

wait "${ncu_pid}" || true

if [[ -f "${report_file}" ]]; then
  "${ncu_bin}" --import "${report_file}" --csv --page details --print-details all > "${out_dir}/ncu-details.csv" 2> "${out_dir}/ncu-details.stderr.log" || true
  "${ncu_bin}" --import "${report_file}" --csv --page raw > "${out_dir}/ncu-raw.csv" 2> "${out_dir}/ncu-raw.stderr.log" || true
fi

{
  echo
  echo "## Result"
  echo "- HTTP code: ${http_code}"
  if [[ -f "${report_file}" ]]; then
    echo "- NCU report captured: yes"
  else
    echo "- NCU report captured: no"
  fi
} >> "${out_dir}/summary.md"

echo "completed: ${out_dir}"
