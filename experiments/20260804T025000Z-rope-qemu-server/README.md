# ROPE qemu_cuda llama-server validation

This startup/request validation follows `expt-baseline.md` with these necessary
local changes:

- the same Qwen3-8B NVFP4 model is available at
  `/home/lerong.chen/qwen3-8b-nvfp4.gguf` rather than the baseline path;
- port `58082` is used to avoid conflicts with other local services;
- `GGML_CUDA_ROPE_QEMU_MODE=qemu_cuda` and the static table path enable the
  experiment under test.

All other baseline server parameters are retained: 40 GPU layers, 32 CPU
threads, batch/ubatch 512, context 8192, F16 K/V cache, unified KV mode, flash
attention disabled, and KQV offload enabled.

Validation result: `/health` returned HTTP 200 with `{"status":"ok"}` and the
saved `/completion` request returned HTTP 200 after evaluating one prompt token
and generating one token. The server log confirms the qemu_cuda RoPE path and a
single 4 MiB table load on CUDA device 0. The container was stopped after the
request.
