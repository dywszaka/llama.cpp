# NVFP4 KV Cache Smoke Experiment (2026-04-02)

## Scope

- Goal: verify that the current model can run through `llama-server` with `KV cache = NVFP4/NVFP4` on CUDA.
- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- Server config source: `.vscode/launch.json`, task `Debug llama-server (nvfp4 cuda)`
- Experiment level: smoke only
- Context size: `2048`
- Batch size: `2048`
- Ubatch size: `512`
- GPU layers: `40`
- Flash attention: disabled

## Repro Script

- Script: [`scripts/run-kvcache-smoke.sh`](/home/allen/host_workspace/develop/llama.cpp/scripts/run-kvcache-smoke.sh)

It starts `llama-server`, waits for `/health`, sends one `/v1/chat/completions` request, then saves:

- server stdout log
- server log file
- response JSON

## Commands

Baseline:

```bash
./scripts/run-kvcache-smoke.sh f16 f16 8082 kv-f16-baseline
```

Target:

```bash
./scripts/run-kvcache-smoke.sh nvfp4 nvfp4 8083 kv-nvfp4-target
```

## Environment Used

The script keeps the launch config's CUDA-oriented runtime shape, but drops the high-volume NVFP4 debug env vars to keep the smoke run readable:

```bash
CUDA_VISIBLE_DEVICES=0
GGML_CUDA_DISABLE_GRAPHS=1
CUDA_LAUNCH_BLOCKING=1
GGML_CUDA_TRUNC_ENABLE=0
GGML_CUDA_TRUNC_LOG=0
GGML_CUDA_NVFP4_NATIVE=1
```

## First Failure and Fix

The first `NVFP4/NVFP4` run failed after prompt prefill and crashed on the first decode token.

Observed stack:

- `ggml_cuda_mul_mat_vec_q()`
- `ggml/src/ggml-cuda/mmvq.cu`
- `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printf_fifo)`
- CUDA error: `invalid argument`

Root cause:

- `ggml/src/ggml-cuda/mmvq.cu` contained temporary NVFP4 MMVQ kernel `printf` debug code from a local `tmp` commit.
- That debug code unconditionally called `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, ...)` inside the decode path.
- `NVFP4 K` decode enters `ggml_cuda_mul_mat_vec_q()`, so the smoke experiment hit that path immediately.

Fix applied:

- Removed the temporary MMVQ NVFP4 `printf` instrumentation from [`ggml/src/ggml-cuda/mmvq.cu`](/home/allen/host_workspace/develop/llama.cpp/ggml/src/ggml-cuda/mmvq.cu)
- Rebuilt `llama-server`
- Re-ran the experiment

## Artifacts

Baseline:

- [`logs/kv-f16-baseline.server.log`](/home/allen/host_workspace/develop/llama.cpp/logs/kv-f16-baseline.server.log)
- [`logs/kv-f16-baseline.server.log.stdout`](/home/allen/host_workspace/develop/llama.cpp/logs/kv-f16-baseline.server.log.stdout)
- [`logs/kv-f16-baseline.response.json`](/home/allen/host_workspace/develop/llama.cpp/logs/kv-f16-baseline.response.json)

Target:

- [`logs/kv-nvfp4-target.server.log`](/home/allen/host_workspace/develop/llama.cpp/logs/kv-nvfp4-target.server.log)
- [`logs/kv-nvfp4-target.server.log.stdout`](/home/allen/host_workspace/develop/llama.cpp/logs/kv-nvfp4-target.server.log.stdout)
- [`logs/kv-nvfp4-target.response.json`](/home/allen/host_workspace/develop/llama.cpp/logs/kv-nvfp4-target.response.json)

## Results

### 1. Startup and request execution

`F16/F16`:

- server startup: success
- `/health`: success
- `/v1/chat/completions`: HTTP 200
- completion finished: yes

`NVFP4/NVFP4`:

- server startup: success
- `/health`: success
- `/v1/chat/completions`: HTTP 200
- completion finished: yes

Conclusion:

- `KV cache NVFP4` is functionally live in `llama-server` for the current model after removing the MMVQ debug crash path.

### 2. KV memory

From server logs:

- `F16/F16`: `CUDA0 KV buffer size = 288.00 MiB`
- `NVFP4/NVFP4`: `CUDA0 KV buffer size = 81.00 MiB`

Derived:

- NVFP4 KV uses `28.1%` of the F16 KV memory
- memory reduction factor: `288 / 81 = 3.56x`
- memory saved: `207 MiB`

Conclusion:

- KV memory reduction matches the expected NVFP4 compression behavior for this smoke setup.

### 3. Prompt and decode timing

From response timings and server logs:

`F16/F16`

- prompt: `188.233 ms / 16 tok = 85.00 tok/s`
- decode: `1029.769 ms / 32 tok = 31.07 tok/s`

`NVFP4/NVFP4`

- prompt: `282.524 ms / 16 tok = 56.63 tok/s`
- decode: `1152.008 ms / 32 tok = 27.78 tok/s`

Delta:

- prompt throughput: `-33.4%`
- decode throughput: `-10.6%`
- decode latency per token: `+11.9%`

Conclusion:

- Current smoke result shows clear KV memory gain, but throughput regresses.
- The decode regression is moderate.
- The prompt regression is larger than expected for a final-quality implementation.

### 4. Output quality

Prompt:

```text
用一句话解释 KV cache 的作用。
```

`F16/F16` first output fragment:

```text
<think>
好的，用户让我用一句话解释KV cache的作用。首先，我需要回忆一下KV cache是什么。KV cache应该是指键值缓存，
```

`NVFP4/NVFP4` first output fragment:

```text
 fld解放�,output{{-\', charset三千 或:def Ungêsco-TRS MTई乎filesizegartঐ unavoidaker Bracket国际机场不下onas hydro一个职业睫
```

Observation:

- With `temperature = 0`, the first generated token already diverges.
- The `NVFP4/NVFP4` output is not semantically usable on this prompt.

Conclusion:

- The smoke experiment proves the runtime path works and KV memory reduction is real.
- The current implementation does **not** pass a minimal output-quality gate on this model.

## Final Assessment

For the current model and this smoke setup:

- Functional bring-up: **Pass**
- KV memory reduction: **Pass**
- Stable decode completion: **Pass**
- Output quality: **Fail**

So the current state is:

- `NVFP4 KV cache` is experimentally runnable
- `NVFP4 KV cache` is not yet quality-ready for this model

## Recommended Next Check

Before wider benchmarking, the next highest-value check is:

1. compare `F16/F16`, `NVFP4/F16`, `F16/NVFP4`, `NVFP4/NVFP4`
2. identify whether divergence is dominated by `K` or `V`
3. if needed, inspect the `K` native consume path first, because the first-token divergence appears immediately in decode
