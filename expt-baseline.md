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
- Name experiment folders with a UTC timestamp first, using the
  `YYYYMMDDThhmmssZ` format such as `20260520T024947Z`, followed by a short
  experiment identifier, so records sort chronologically and stay easy to scan.
- Environment variables not listed in a baseline command are not part of the
  baseline. Add them only when an experiment needs them, and document why.

## Shared Baseline

- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- GPU layers: `--n_gpu_layers 40`
- CPU threads: `-t 32`
- K cache type: `--cache-type-k f16`
- V cache type: `--cache-type-v f16`
- Flash attention: disabled (`flash_attn=0`; do not pass `--flash-attn`
  unless the experiment targets flash attention)
- KQV offload: enabled (`offload_kqv=1`; do not pass `--no-kv-offload`)
- KV mode: unified (`--kv-unified`)

## NVFP4 V-Cache Runtime Requirement

The default baseline keeps these requirements enabled so experiments can switch
V cache type to `--cache-type-v nvfp4` without changing unrelated runtime
plumbing. The current runtime requires all of the following for NVFP4 V-cache:

- Flash attention disabled: `flash_attn=0`.
- KQV offload enabled: `offload_kqv=1`.
- Unified KV cache enabled: `--kv-unified`.

If these requirements are not met, context creation fails with:

```text
NVFP4 V cache requires flash_attn=0, offload_kqv=1, and kv_unified=1
```

For PPL, server, or benchmark experiments that test `--cache-type-v nvfp4`,
keep these defaults in place and document that they are runtime compatibility
requirements for NVFP4 V-cache. Only change them when the experiment explicitly
targets flash attention, KQV offload, or KV layout behavior.

## llama-server Baseline

Use this baseline for NVFP4 CUDA `llama-server` startup validation:

```bash
CUDA_VISIBLE_DEVICES=0 \
LLAMA_STDOUT_FILE="${WORKSPACE}/gpu.log" \
  "${WORKSPACE}/build_cuda/bin/llama-server" \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    --n_gpu_layers 40 \
    --host 127.0.0.1 \
    --batch-size 512 \
    --ubatch-size 512 \
    --port 8080 \
    -t 32 \
    -c 8192 \
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
- Batch size: `--batch-size 512`
- UBatch size: `--ubatch-size 512`
- Threads: `-t 32`
- Context size: `-c 8192`
- K cache type: `--cache-type-k f16`
- V cache type: `--cache-type-v f16`
- Flash attention: disabled by default; do not pass `--flash-attn`
- KQV offload: enabled by default; do not pass `--no-kv-offload`
- KV mode: `--kv-unified`
- Log file: `--log-file ${WORKSPACE}/gpu.log`

Fixed `llama-server` environment:

- `CUDA_VISIBLE_DEVICES=0`
- `LLAMA_STDOUT_FILE=${WORKSPACE}/gpu.log`

For each `llama-server` validation, save the startup script, request payload,
server response, server log, and validation summary in the experiment folder.

## llama-perplexity Baseline

Use this baseline for PPL experiments:

```bash
CUDA_VISIBLE_DEVICES=0 \
  "${ROOT_DIR}/build_cuda/bin/llama-perplexity" \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f "${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw" \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --n_gpu_layers 40 \
    --batch-size 512 \
    --ubatch-size 512 \
    -t 32 \
    -c 8192 \
    --kv-unified
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
- Context size: `-c 8192`
- Flash attention: disabled by default; do not pass `--flash-attn`
- KQV offload: enabled by default; do not pass `--no-kv-offload`
- KV mode: `--kv-unified`

Fixed `llama-perplexity` environment:

- `CUDA_VISIBLE_DEVICES=0`

For each PPL experiment, save the baseline run script, experiment run script,
prompt/input reference, raw logs, parsed PPL metrics, and comparison summary in
the experiment folder.

## KLD Baseline Data

Reusable KLD baseline data is managed in `experiments/kld-baseline-data`.
Future invariant KLD assets should be added there rather than to dated
comparison experiment folders.

Keep these assets in `experiments/kld-baseline-data`:

- small fixed evaluation datasets and dataset manifests;
- baseline f16/f16 log-prob files generated from those datasets;
- baseline generation logs;
- baseline input/config references.

Do not store experiment-group comparison logs, parsed metrics, summaries, or
diagnostic outputs in `experiments/kld-baseline-data`. For each comparison
against this baseline data, create a dated experiment folder under
`experiments/` using the standard `YYYYMMDDThhmmssZ-<identifier>` convention and
store the experiment-group logs, parsed metrics, diagnostics, and summary there.

Current KLD baseline data:

- Directory: `experiments/kld-baseline-data`
- Dataset: `experiments/kld-baseline-data/data/wikitext-small.raw`
- Dataset manifest: `experiments/kld-baseline-data/data/wikitext-small.manifest.json`
- Baseline log-prob file: `experiments/kld-baseline-data/baseline-logprobs/ubatch_512.kld`
- Baseline command log: `experiments/kld-baseline-data/logs/baseline_ubatch_512.raw.log`
- Tooling: `tools/kld`

## llama-bench Baseline

Use this baseline for local `llama-bench` throughput measurements:

```bash
CUDA_VISIBLE_DEVICES=0 \
  "${ROOT_DIR}/build_cuda/bin/llama-bench" \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --n-gpu-layers 40 \
    --batch-size 2048 \
    --ubatch-size 512 \
    --kv-unified 1 \
    --flash-attn 0 \
    --no-kv-offload 0 \
    -t 32 \
    -p 512 \
    -n 128
```

Fixed `llama-bench` arguments:

- Binary default: `${ROOT_DIR}/build_cuda/bin/llama-bench`
- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- K cache type: `--cache-type-k f16`
- V cache type: `--cache-type-v f16`
- GPU layers: `--n-gpu-layers 40`
- Batch size: `--batch-size 2048`
- UBatch size: `--ubatch-size 512`
- Flash attention: `--flash-attn 0`
- KQV offload: `--no-kv-offload 0`
- KV mode: `--kv-unified 1`
- Threads: `-t 32`
- Prompt tokens: `-p 512`
- Generation tokens: `-n 128`

Fixed `llama-bench` environment:

- `CUDA_VISIBLE_DEVICES=0`

For each benchmark experiment, save the baseline run script, experiment run
script, raw benchmark output, parsed throughput metrics, and comparison summary
in the experiment folder.

## Comparison Rule

Direct comparisons are valid only when the baseline run and experiment run use
the same fixed parameters and differ only by the named experiment switch or
parameter. In summaries, report both the baseline result and experiment result,
then state the delta.
