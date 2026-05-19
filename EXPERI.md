# Experiment Baseline Runtime Parameters

This document is the baseline runtime contract for local experiments. Future
experiments must start from the parameters below, change only the parameters
required by the experiment, and compare the result against this baseline.

If an experiment changes any baseline parameter, record the reason in that
experiment's folder. Results with changed baseline parameters must not be
treated as direct A/B comparisons unless the changed parameter is the explicit
subject of the experiment.

## Baseline Rules

- Build every new experiment run script from the commands in this document.
- Keep model path, request/prompt input, context size, batch sizes, cache types,
  GPU layer count, thread count, CUDA device, and server KV mode unchanged
  unless the experiment explicitly targets one of them.
- For PPL experiments, run and record both:
  - a baseline run using the parameters in this document;
  - an experiment run that changes only the experiment parameter or switch.
- Store each experiment under a dedicated folder in `experiments/`, including
  scripts, request/input data, raw outputs, responses, and summarized results.
- Environment variables not listed in a baseline command are not part of the
  baseline. Add them only when an experiment needs them, and document why.

## Shared Baseline

- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- Native NVFP4 matmul: `GGML_CUDA_NVFP4_NATIVE=1`
- GPU layers: `--n_gpu_layers 40`
- CPU threads: `-t 32`
- K cache type: `--cache-type-k f16`
- V cache type: `--cache-type-v f16`

## llama-server Baseline

Use this baseline for NVFP4 CUDA `llama-server` startup validation:

```bash
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_NVFP4_NATIVE=1 \
LLAMA_STDOUT_FILE="${WORKSPACE}/gpu.log" \
  "${WORKSPACE}/build_cuda/bin/llama-server" \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    --n_gpu_layers 40 \
    --host 127.0.0.1 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --port 8080 \
    -t 32 \
    -c 2048 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --kv-unified \
    --log-file "${WORKSPACE}/gpu.log"
```

Fixed `llama-server` arguments:

- Binary: `${WORKSPACE}/build_cuda/bin/llama-server`
- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- Host: `--host 127.0.0.1`
- Port: `--port 8080`
- GPU layers: `--n_gpu_layers 40`
- Batch size: `--batch-size 2048`
- UBatch size: `--ubatch-size 512`
- Threads: `-t 32`
- Context size: `-c 2048`
- K cache type: `--cache-type-k f16`
- V cache type: `--cache-type-v f16`
- KV mode: `--kv-unified`
- Log file: `--log-file ${WORKSPACE}/gpu.log`

Fixed `llama-server` environment:

- `CUDA_VISIBLE_DEVICES=0`
- `GGML_CUDA_NVFP4_NATIVE=1`
- `LLAMA_STDOUT_FILE=${WORKSPACE}/gpu.log`

For each `llama-server` validation, save the startup script, request payload,
server response, server log, and validation summary in the experiment folder.

## llama-perplexity Baseline

Use this baseline for PPL experiments:

```bash
CUDA_VISIBLE_DEVICES=0 \
GGML_CUDA_NVFP4_NATIVE=1 \
  "${ROOT_DIR}/build_cuda/bin/llama-perplexity" \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f "${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw" \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --n_gpu_layers 40 \
    --batch-size 512 \
    --ubatch-size 512 \
    -t 32 \
    -c 512
```

Fixed `llama-perplexity` arguments:

- Binary default: `${ROOT_DIR}/build_cuda/bin/llama-perplexity`
- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- Prompt file: `${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw`
- K cache type: `--cache-type-k f16`
- V cache type: `--cache-type-v f16`
- GPU layers: `--n_gpu_layers 40`
- Batch size: `--batch-size 512`
- UBatch size: `--ubatch-size 512`
- Threads: `-t 32`
- Context size: `-c 512`

Fixed `llama-perplexity` environment:

- `CUDA_VISIBLE_DEVICES=0`
- `GGML_CUDA_NVFP4_NATIVE=1`

For each PPL experiment, save the baseline run script, experiment run script,
prompt/input reference, raw logs, parsed PPL metrics, and comparison summary in
the experiment folder.

## Comparison Rule

Direct comparisons are valid only when the baseline run and experiment run use
the same fixed parameters and differ only by the named experiment switch or
parameter. In summaries, report both the baseline result and experiment result,
then state the delta.
