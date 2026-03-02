#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/compare_cpu_cuda_logits.sh [options]

Options:
  --model PATH                    Model path
  --prompt TEXT                   Prompt text
  --prompt-file PATH              Prompt file path
  --cpu-bin PATH                  CPU llama-server binary
  --cuda-bin PATH                 CUDA llama-server binary
  --cpu-port N                    CPU server port (default: 18080)
  --cuda-port N                   CUDA server port (default: 18081)
  --cpu-host HOST                 CPU server host (default: 127.0.0.1)
  --cuda-host HOST                CUDA server host (default: 127.0.0.1)
  --n-gpu-layers N                CUDA n_gpu_layers (default: 40)
  --ctx-size N                    Context size (default: 8192)
  --threads N                     Threads (default: 16)
  --batch-size N                  Batch size (default: 2048)
  --ubatch-size N                 Ubatch size (default: 512)
  --seed N                        RNG seed (default: 12345)
  --max-tokens N                  max_tokens (default: 1)
  --topn N                        top-N logits to compare (default: 100)
  --temperature FLOAT             temperature (default: 0)
  --top-p FLOAT                   top_p (default: 1)
  --top-k N                       top_k (default: 1)
  --min-p FLOAT                   min_p (default: 0)
  --ready-timeout-sec N           Service ready timeout (default: 300)
  --request-timeout-sec N         Request timeout (default: 180)
  --out-dir DIR                   Output directory
  --no-start                      Do not start/stop servers automatically
  --cpu-url URL                   CPU base URL (used with --no-start)
  --cuda-url URL                  CUDA base URL (used with --no-start)
  --keep-running                  Keep started servers alive after run
  -h, --help                      Show help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MODEL="/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf"
DEFAULT_CPU_BIN="${ROOT_DIR}/build/bin/llama-server"
DEFAULT_CUDA_BIN="${ROOT_DIR}/build_cuda_release/bin/llama-server"
DEFAULT_PROMPT="请用三句话解释为什么量化会影响语言模型输出的一致性。"

MODEL_PATH="${DEFAULT_MODEL}"
PROMPT="${DEFAULT_PROMPT}"
PROMPT_FILE=""
CPU_BIN="${DEFAULT_CPU_BIN}"
CUDA_BIN="${DEFAULT_CUDA_BIN}"
CPU_PORT="18080"
CUDA_PORT="18081"
CPU_HOST="127.0.0.1"
CUDA_HOST="127.0.0.1"
N_GPU_LAYERS="40"
CTX_SIZE="8192"
THREADS="16"
BATCH_SIZE="2048"
UBATCH_SIZE="512"
SEED="12345"
MAX_TOKENS="1"
TOPN="100"
TEMPERATURE="0"
TOP_P="1"
TOP_K="1"
MIN_P="0"
READY_TIMEOUT_SEC="300"
REQUEST_TIMEOUT_SEC="180"
OUT_DIR="${ROOT_DIR}/artifacts/logits/$(date -u +%Y%m%dT%H%M%SZ)"
NO_START="0"
CPU_URL=""
CUDA_URL=""
KEEP_RUNNING="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="${2:-}"; shift 2 ;;
        --prompt) PROMPT="${2:-}"; shift 2 ;;
        --prompt-file) PROMPT_FILE="${2:-}"; shift 2 ;;
        --cpu-bin) CPU_BIN="${2:-}"; shift 2 ;;
        --cuda-bin) CUDA_BIN="${2:-}"; shift 2 ;;
        --cpu-port) CPU_PORT="${2:-}"; shift 2 ;;
        --cuda-port) CUDA_PORT="${2:-}"; shift 2 ;;
        --cpu-host) CPU_HOST="${2:-}"; shift 2 ;;
        --cuda-host) CUDA_HOST="${2:-}"; shift 2 ;;
        --n-gpu-layers) N_GPU_LAYERS="${2:-}"; shift 2 ;;
        --ctx-size) CTX_SIZE="${2:-}"; shift 2 ;;
        --threads) THREADS="${2:-}"; shift 2 ;;
        --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
        --ubatch-size) UBATCH_SIZE="${2:-}"; shift 2 ;;
        --seed) SEED="${2:-}"; shift 2 ;;
        --max-tokens) MAX_TOKENS="${2:-}"; shift 2 ;;
        --topn) TOPN="${2:-}"; shift 2 ;;
        --temperature) TEMPERATURE="${2:-}"; shift 2 ;;
        --top-p) TOP_P="${2:-}"; shift 2 ;;
        --top-k) TOP_K="${2:-}"; shift 2 ;;
        --min-p) MIN_P="${2:-}"; shift 2 ;;
        --ready-timeout-sec) READY_TIMEOUT_SEC="${2:-}"; shift 2 ;;
        --request-timeout-sec) REQUEST_TIMEOUT_SEC="${2:-}"; shift 2 ;;
        --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
        --no-start) NO_START="1"; shift ;;
        --cpu-url) CPU_URL="${2:-}"; shift 2 ;;
        --cuda-url) CUDA_URL="${2:-}"; shift 2 ;;
        --keep-running) KEEP_RUNNING="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -n "${PROMPT_FILE}" ]]; then
    if [[ ! -f "${PROMPT_FILE}" ]]; then
        echo "Error: prompt file not found: ${PROMPT_FILE}" >&2
        exit 1
    fi
    PROMPT="$(cat "${PROMPT_FILE}")"
fi

if [[ -z "${PROMPT}" ]]; then
    echo "Error: prompt is empty." >&2
    exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: model not found: ${MODEL_PATH}" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required." >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required." >&2
    exit 1
fi

if [[ "${NO_START}" == "0" ]]; then
    if [[ ! -x "${CPU_BIN}" ]]; then
        echo "Error: cpu binary not executable: ${CPU_BIN}" >&2
        exit 1
    fi
    if [[ ! -x "${CUDA_BIN}" ]]; then
        echo "Error: cuda binary not executable: ${CUDA_BIN}" >&2
        exit 1
    fi
fi

if [[ "${NO_START}" == "1" ]]; then
    CPU_URL="${CPU_URL:-http://${CPU_HOST}:${CPU_PORT}}"
    CUDA_URL="${CUDA_URL:-http://${CUDA_HOST}:${CUDA_PORT}}"
else
    CPU_URL="http://${CPU_HOST}:${CPU_PORT}"
    CUDA_URL="http://${CUDA_HOST}:${CUDA_PORT}"
fi

mkdir -p "${OUT_DIR}"
CPU_LOG="${OUT_DIR}/cpu_server.log"
CUDA_LOG="${OUT_DIR}/cuda_server.log"
PAYLOAD_JSON="${OUT_DIR}/request_payload.json"
CPU_RESP="${OUT_DIR}/cpu_logits.json"
CUDA_RESP="${OUT_DIR}/cuda_logits.json"
DIFF_JSON="${OUT_DIR}/diff_report.json"
DIFF_MD="${OUT_DIR}/diff_report.md"

CPU_PID=""
CUDA_PID=""

cleanup() {
    if [[ "${KEEP_RUNNING}" == "1" ]]; then
        return 0
    fi
    if [[ -n "${CPU_PID}" ]]; then
        kill "${CPU_PID}" >/dev/null 2>&1 || true
        wait "${CPU_PID}" >/dev/null 2>&1 || true
        CPU_PID=""
    fi
    if [[ -n "${CUDA_PID}" ]]; then
        kill "${CUDA_PID}" >/dev/null 2>&1 || true
        wait "${CUDA_PID}" >/dev/null 2>&1 || true
        CUDA_PID=""
    fi
}
trap cleanup EXIT

wait_ready() {
    local base_url="$1"
    local timeout="$2"
    local start_ts
    start_ts="$(date +%s)"
    while true; do
        local health_code=""
        if health_code="$(curl -sS --max-time 3 -o /tmp/llama_health.$$ -w "%{http_code}" "${base_url}/health" 2>/dev/null)"; then
            rm -f /tmp/llama_health.$$ >/dev/null 2>&1 || true
            if [[ "${health_code}" =~ ^2 ]]; then
                return 0
            fi
        fi
        rm -f /tmp/llama_health.$$ >/dev/null 2>&1 || true
        if (( $(date +%s) - start_ts > timeout )); then
            return 1
        fi
        sleep 1
    done
}

post_json() {
    local url="$1"
    local payload_file="$2"
    local out_file="$3"
    local code
    code="$(curl -sS --max-time "${REQUEST_TIMEOUT_SEC}" -o "${out_file}" -w "%{http_code}" \
        -H "Content-Type: application/json" \
        --data @"${payload_file}" \
        "${url}")"
    if [[ ! "${code}" =~ ^2 ]]; then
        echo "Error: request failed for ${url}, http=${code}" >&2
        cat "${out_file}" >&2 || true
        exit 1
    fi
}

python3 - "${MODEL_PATH}" "${PROMPT}" "${MAX_TOKENS}" "${SEED}" "${TEMPERATURE}" "${TOP_P}" "${TOP_K}" "${MIN_P}" "${TOPN}" > "${PAYLOAD_JSON}" <<'PY'
import json
import sys

model = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])
seed = int(sys.argv[4])
temperature = float(sys.argv[5])
top_p = float(sys.argv[6])
top_k = int(sys.argv[7])
min_p = float(sys.argv[8])
topn = int(sys.argv[9])

payload = {
    "model": model,
    "prompt": prompt,
    "max_tokens": max_tokens,
    "n_predict": max_tokens,
    "seed": seed,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "min_p": min_p,
    "n_probs": topn,
    "return_logits": True,
    "cache_prompt": False,
    "stream": False,
}
print(json.dumps(payload, ensure_ascii=False))
PY

if [[ "${NO_START}" == "0" ]]; then
    echo "[1/4] start CPU server..."
    "${CPU_BIN}" \
        -m "${MODEL_PATH}" \
        --host "${CPU_HOST}" \
        --port "${CPU_PORT}" \
        --n-gpu-layers 0 \
        --ctx-size "${CTX_SIZE}" \
        --threads "${THREADS}" \
        --batch-size "${BATCH_SIZE}" \
        --ubatch-size "${UBATCH_SIZE}" \
        --no-warmup >"${CPU_LOG}" 2>&1 &
    CPU_PID="$!"

    if ! wait_ready "${CPU_URL}" "${READY_TIMEOUT_SEC}"; then
        echo "Error: CPU server not ready within timeout. Log: ${CPU_LOG}" >&2
        exit 1
    fi

    echo "[2/4] start CUDA server..."
    CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_DISABLE_GRAPHS=1 \
    "${CUDA_BIN}" \
        -m "${MODEL_PATH}" \
        --host "${CUDA_HOST}" \
        --port "${CUDA_PORT}" \
        --n-gpu-layers "${N_GPU_LAYERS}" \
        --ctx-size "${CTX_SIZE}" \
        --threads "${THREADS}" \
        --batch-size "${BATCH_SIZE}" \
        --ubatch-size "${UBATCH_SIZE}" \
        --no-warmup >"${CUDA_LOG}" 2>&1 &
    CUDA_PID="$!"

    if ! wait_ready "${CUDA_URL}" "${READY_TIMEOUT_SEC}"; then
        echo "Error: CUDA server not ready within timeout. Log: ${CUDA_LOG}" >&2
        exit 1
    fi
fi

echo "[3/4] query /completion on CPU and CUDA..."
post_json "${CPU_URL}/completion" "${PAYLOAD_JSON}" "${CPU_RESP}"
post_json "${CUDA_URL}/completion" "${PAYLOAD_JSON}" "${CUDA_RESP}"

echo "[4/4] analyze diff..."
python3 "${ROOT_DIR}/scripts/analyze_logits_diff.py" \
    --cpu-json "${CPU_RESP}" \
    --cuda-json "${CUDA_RESP}" \
    --out-json "${DIFF_JSON}" \
    --out-md "${DIFF_MD}"

cat <<EOF
saved: ${CPU_RESP}
saved: ${CUDA_RESP}
saved: ${DIFF_JSON}
saved: ${DIFF_MD}
EOF
