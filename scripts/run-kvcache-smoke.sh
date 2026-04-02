#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 <cache_type_k> <cache_type_v> <port> <tag>" >&2
    exit 1
fi

cache_type_k="$1"
cache_type_v="$2"
port="$3"
tag="$4"

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${root_dir}/build_cuda/bin/llama-server"
model_path="/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf"
log_path="${root_dir}/logs/${tag}.server.log"
resp_path="${root_dir}/logs/${tag}.response.json"
health_path="${root_dir}/logs/${tag}.health.json"
pid_path="${root_dir}/logs/${tag}.pid"

mkdir -p "${root_dir}/logs"
rm -f "${log_path}" "${resp_path}" "${health_path}" "${pid_path}"

cleanup() {
    if [[ -f "${pid_path}" ]]; then
        pid="$(cat "${pid_path}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
        rm -f "${pid_path}"
    fi
}

trap cleanup EXIT

env \
    CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_DISABLE_GRAPHS=1 \
    CUDA_LAUNCH_BLOCKING=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    "${server_bin}" \
    -m "${model_path}" \
    --n_gpu_layers 40 \
    --host 127.0.0.1 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --port "${port}" \
    -t 32 \
    -c 2048 \
    --cache-type-k "${cache_type_k}" \
    --cache-type-v "${cache_type_v}" \
    --no-warmup \
    --log-file "${log_path}" \
    >"${log_path}.stdout" 2>&1 &

server_pid=$!
echo "${server_pid}" > "${pid_path}"

for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" > "${health_path}" 2>/dev/null; then
        break
    fi

    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "server exited before becoming healthy; see ${log_path}.stdout" >&2
        exit 1
    fi

    sleep 1
done

if [[ ! -s "${health_path}" ]]; then
    echo "server did not become healthy within timeout; see ${log_path}.stdout" >&2
    exit 1
fi

curl -fsS "http://127.0.0.1:${port}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{
        "messages": [
            {
                "role": "user",
                "content": "用一句话解释 KV cache 的作用。"
            }
        ],
        "stream": false,
        "cache_prompt": true,
        "reasoning_format": "none",
        "temperature": 0,
        "max_tokens": 32,
        "timings_per_token": true
    }' \
    > "${resp_path}"

cleanup
trap - EXIT

echo "tag=${tag}"
echo "cache_type_k=${cache_type_k}"
echo "cache_type_v=${cache_type_v}"
echo "port=${port}"
echo "log=${log_path}"
echo "response=${resp_path}"
