# CUDA PPL Baseline Quickstart

This guide is for developers who already know llama.cpp and need to run the
local CUDA perplexity baseline in this experiment environment.

`expt-baseline.md` is the authoritative source for the baseline model, data,
runtime parameters, and experiment-record rules. If anything here conflicts with
`expt-baseline.md`, use `expt-baseline.md` and update this quickstart.

## Prerequisites

- CUDA toolkit and an NVIDIA CUDA-capable GPU are available.
- The baseline model path from `expt-baseline.md` exists:
  `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`.
- The baseline PPL input from `expt-baseline.md` exists:
  `data/wikitext/wikitext-2-raw/wiki.test.raw`.
- Run from the repository root unless the command explicitly sets `ROOT_DIR`.

## Build `llama-perplexity` With CUDA

The upstream CUDA build option is documented in `docs/build.md#cuda`; this
environment uses the repository-local `build_cuda` directory.

Copyable build commands:

```bash
cd /home/allen/host_workspace/develop/llama.cpp

cmake -S . -B build_cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_cuda --config Release -j "$(nproc)" --target llama-perplexity
```

If CUDA GPU architecture detection is wrong for the host, add
`-DCMAKE_CUDA_ARCHITECTURES="<arch>"` to the `cmake -S ...` command as described
in `docs/build.md#override-compute-capability-specifications`.

## Run The Baseline PPL

This command intentionally mirrors the `llama-perplexity Baseline` section of
`expt-baseline.md`. Do not add `--flash-attn` or `--no-kv-offload` for the
baseline. The command below uses the current CLI spelling `--n-gpu-layers` for
the baseline GPU layer value.

```bash
cd /home/allen/host_workspace/develop/llama.cpp
ROOT_DIR="$(pwd)"

CUDA_VISIBLE_DEVICES=0 \
  "${ROOT_DIR}/build_cuda/bin/llama-perplexity" \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f "${ROOT_DIR}/data/wikitext/wikitext-2-raw/wiki.test.raw" \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --n-gpu-layers 40 \
    --batch-size 512 \
    --ubatch-size 512 \
    -t 32 \
    -c 512 \
    --kv-unified
```

For a recorded PPL experiment, create a dedicated folder under `experiments/`
and save the script, input references, raw output, parsed PPL metric, and
summary as required by `expt-baseline.md` and
`docs/development/experiment-records.md`.

## Confirm Success

A successful run exits with status `0` and prints a final perplexity estimate
near the end of stdout. Look for a line in this form:

```text
Final estimate: PPL = <score> +/- <error>
```

The `<score>` value on that final estimate line is the baseline PPL score to
record or compare against. If the binary reports missing model/data files,
CUDA initialization failure, unsupported cache/runtime options, or exits
non-zero before printing `Final estimate`, the baseline run did not complete.
