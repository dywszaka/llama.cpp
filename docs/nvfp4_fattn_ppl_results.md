# NVFP4 FATTN PPL Results

Created: 2026-05-11

Dataset:

- `/home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw`

Model:

- `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`

Common parameters:

- `-ngl 99`
- `-c 512`
- `-b 512`
- `-ub 512`
- Wikitext run used `584 chunks`, `n_ctx=512`, `batch_size=512`, `n_seq=1`

## Results Table

| Experiment | Log | Key flags | PPL | tok/s |
|---|---|---|---:|---:|
| Flash attention on, native baseline log | `/tmp/ppl_nvfp4_fattn_native.log` | `-fa`, no NVFP4 FATTN experiment flags observed in log extraction | `10.4010 +/- 0.08131` | `4722.97` |
| Flash attention off comparison log | `/tmp/ppl_nvfp4_fattn_off.log` | no FA comparison | `10.4010 +/- 0.08131` | `4726.68` |
| NVFP4 FATTN, two-level P scale | `/tmp/ppl_nvfp4_fattn_p_twolevel.log` | `GGML_CUDA_NVFP4_FATTN=1`, `P_DIRECT=0`, Q+K smooth | `10.3666 +/- 0.08002` | `539.41` |
| NVFP4 FATTN, direct raw P | `/tmp/ppl_nvfp4_fattn_p_direct.log` | `GGML_CUDA_NVFP4_FATTN=1`, `P_DIRECT=1`, Q+K smooth | `11.6735 +/- 0.09296` | `545.55` |
| NVFP4 FATTN, no Q smooth | `/tmp/ppl_nvfp4_fattn_no_q_smooth.log` | `GGML_CUDA_NVFP4_FATTN=1`, `NO_Q_SMOOTH=1` | `11.4336 +/- 0.08888` | `550.43` |
| NVFP4 FATTN, no K smooth | `/tmp/ppl_nvfp4_fattn_no_k_smooth.log` | `GGML_CUDA_NVFP4_FATTN=1`, `NO_K_SMOOTH=1` | `10.4057 +/- 0.08035` | `542.28` |
| NVFP4 FATTN, no K smooth, K cache NVFP4 direct | `/tmp/ppl_nvfp4_fattn_no_k_smooth_kcache_nvfp4.log` | `GGML_CUDA_NVFP4_FATTN=1`, `NO_K_SMOOTH=1`, `--cache-type-k nvfp4`, direct compressed K cache QK | `10.5143 +/- 0.08132` | `493.03` |
| NVFP4 FATTN, no Q/K smooth, K cache NVFP4 direct, dynamic Q | `/tmp/ppl_nvfp4_fattn_no_q_no_k_smooth_kcache_nvfp4_q_dynamic.log` | `NO_Q_SMOOTH=1`, `NO_K_SMOOTH=1`, `Q_DYNAMIC=1`, `--cache-type-k nvfp4` | `10.5940 +/- 0.08204` | `493.32` |
| NVFP4 FATTN, no Q/K smooth, K cache NVFP4 direct, Q input scale | `/tmp/ppl_nvfp4_fattn_no_q_no_k_smooth_kcache_nvfp4_q_input_scale.log` | `NO_Q_SMOOTH=1`, `NO_K_SMOOTH=1`, `Q_DYNAMIC` unset, `--cache-type-k nvfp4` | `10.9134 +/- 0.08527` | `493.81` |

## Main Accuracy Observations

- Two-level P scale is important.
  - Direct raw P worsened PPL from `10.3666` to `11.6735`.
- Q smoothing is important.
  - Disabling Q smooth worsened PPL to `11.4336`.
- K smoothing is much less important in the current setup.
  - Disabling K smooth changed PPL from `10.3666` to `10.4057`.
- Storing K cache as NVFP4 with no K smoothing worsened PPL modestly versus F16 K cache with no K smoothing.
  - PPL changed from `10.4057` to `10.5143`.
  - This experiment uses the existing K-cache NVFP4 quantization, with per-token global scale, and does not smooth K.
  - QK directly consumes compressed K-cache `block_nvfp4` data; K is not dequantized to F32 and requantized before QK.
- With NVFP4 K cache and no K smoothing, also disabling Q smoothing and using native dynamic per-row Q quantization worsened PPL to `10.5940`.
  - This leaves K cache direct and changes the Q side of QK to the same per-row global-scale quantization mode used by `ggml_cuda_mul_mat_nvfp4_native()` when no input scale is bound.
- With the same no-Q/no-K-smooth NVFP4 K-cache setup, using the static `q_input_scale` binding instead of dynamic Q worsened PPL further to `10.9134`.
  - This is the default QK path when `GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC` is unset.

## Commands

### NVFP4 FATTN, Two-Level P Scale

```bash
env CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_p_twolevel.log
```

Observed:

- `Final estimate: PPL = 10.3666 +/- 0.08002`
- `prompt eval time = 554319.70 ms / 299008 tokens`
- `539.41 tokens per second`
- `graphs reused = 583`

### NVFP4 FATTN, Direct Raw P

```bash
env CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=1 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_p_direct.log
```

Observed:

- `Final estimate: PPL = 11.6735 +/- 0.09296`
- `prompt eval time = 548086.15 ms / 299008 tokens`
- `545.55 tokens per second`
- `graphs reused = 583`

### NVFP4 FATTN, No Q Smooth

```bash
env CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=1 \
    GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=0 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_no_q_smooth.log
```

Observed:

- `Final estimate: PPL = 11.4336 +/- 0.08888`
- `prompt eval time = 543227.66 ms / 299008 tokens`
- `550.43 tokens per second`
- `graphs reused = 583`

### NVFP4 FATTN, No K Smooth

```bash
env CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=0 \
    GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_no_k_smooth.log
```

Observed:

- `Final estimate: PPL = 10.4057 +/- 0.08035`
- `prompt eval time = 551389.82 ms / 299008 tokens`
- `542.28 tokens per second`
- `graphs reused = 583`

### NVFP4 FATTN, No K Smooth, K Cache NVFP4 Direct

This experiment stores the runtime K cache as `GGML_TYPE_NVFP4`, using the existing per-token global-scale K-cache quantization. K smoothing is disabled. For QK, the native FATTN path directly consumes the compressed K-cache `block_nvfp4` data:

- It copies the compressed blocks for the current `(batch, kv_head)` into a contiguous temporary NVFP4 matrix because the cache view is strided by head.
- Native FP4 matmul uses the in-band NVFP4 block scale bytes directly.
- The per-token K-cache global scale is applied to QK after matmul.
- The Q-mean correction term reads K directly from NVFP4 cache with the attached scale tensor.
- K is not dequantized to F32 and is not requantized before QK.

```bash
env CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=0 \
    GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    --cache-type-k nvfp4 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_no_k_smooth_kcache_nvfp4.log
```

Observed:

- `Final estimate: PPL = 10.5143 +/- 0.08132`
- `prompt eval time = 606472.01 ms / 299008 tokens`
- `493.03 tokens per second`
- `graphs reused = 583`

### NVFP4 FATTN, No Q/K Smooth, K Cache NVFP4 Direct, Dynamic Q

This experiment extends the direct NVFP4 K-cache case by disabling Q smoothing and letting QK use `ggml_cuda_mul_mat_nvfp4_native()` dynamic RHS quantization. In practice, the QK destination does not bind an NVFP4 input scale when `GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC=1`, so the native matmul computes per-row Q amax/global scale and applies the matching dynamic column scale after matmul.

```bash
env CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=1 \
    GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1 \
    GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC=1 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    --cache-type-k nvfp4 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_no_q_no_k_smooth_kcache_nvfp4_q_dynamic.log
```

Observed:

- `Final estimate: PPL = 10.5940 +/- 0.08204`
- `prompt eval time = 606118.48 ms / 299008 tokens`
- `493.32 tokens per second`
- `graphs reused = 583`

### NVFP4 FATTN, No Q/K Smooth, K Cache NVFP4 Direct, Q Input Scale

This experiment keeps the same no-Q/no-K-smooth direct NVFP4 K-cache setup, but does not enable `GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC`. QK therefore binds the static `q_input_scale` tensor and uses the non-dynamic RHS quantization path in `ggml_cuda_mul_mat_nvfp4_native()`.

```bash
env -u GGML_CUDA_NVFP4_FATTN_Q_DYNAMIC \
    CUDA_VISIBLE_DEVICES=0 \
    GGML_CUDA_NVFP4_NATIVE=1 \
    GGML_CUDA_TRUNC_ENABLE=0 \
    GGML_CUDA_TRUNC_LOG=0 \
    GGML_CUDA_NVFP4_FATTN=1 \
    GGML_CUDA_NVFP4_FATTN_NO_FALLBACK=1 \
    GGML_CUDA_NVFP4_FATTN_P_DIRECT=0 \
    GGML_CUDA_NVFP4_FATTN_NO_Q_SMOOTH=1 \
    GGML_CUDA_NVFP4_FATTN_NO_K_SMOOTH=1 \
    ./build_cuda/bin/llama-perplexity \
    -m /home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf \
    -f /home/allen/host_workspace/develop/llama.cpp/data/wikitext/wikitext-2-raw/wiki.test.raw \
    -ngl 99 -fa -c 512 -b 512 -ub 512 \
    --cache-type-k nvfp4 \
    2>&1 | tee /tmp/ppl_nvfp4_fattn_no_q_no_k_smooth_kcache_nvfp4_q_input_scale.log
```

Observed:

- `Final estimate: PPL = 10.9134 +/- 0.08527`
- `prompt eval time = 605513.39 ms / 299008 tokens`
- `493.81 tokens per second`
- `graphs reused = 583`
