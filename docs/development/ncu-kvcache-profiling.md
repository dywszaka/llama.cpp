# NCU profiling for llama-server KV-cache experiments

This guide describes how to use NVIDIA Nsight Compute (`ncu`) to confirm which
CUDA compute path `llama-server` uses during KV-cache experiments. The goal is
not only to collect timings, but to prove whether a K-cache or V-cache
quantization path is actually active compared with the f16/f16 baseline.

## Baseline rule

Start every run from the `llama-server` baseline in `expt-baseline.md`:

- Model: `/home/allen/host_workspace/develop/models/qwen3-8b-nvfp4.gguf`
- `CUDA_VISIBLE_DEVICES=0`
- `--n_gpu_layers 40`
- `--batch-size 2048`
- `--ubatch-size 512`
- `-t 32`
- `-c 2048`
- `--cache-type-k f16`
- `--cache-type-v f16`
- `--kv-unified`

For a direct A/B comparison, change only the cache type or experiment switch
under test. If you add diagnostic variables such as `GGML_CUDA_DISABLE_GRAPHS=1`
or change the request size, record that in the experiment folder and do not use
that run as a strict throughput baseline.

## Reusable script

Use:

```bash
scripts/profile-llama-server-ncu.sh --help
```

The script starts `llama-server` under `ncu`, sends one fixed `/completion`
request, then stores artifacts under `experiments/ncu-<timestamp>-<name>/`:

- `run-ncu-server.sh`: exact command and environment
- `env.txt`: captured CUDA/GGML/LLAMA environment
- `request.json`: request payload
- `response.json` and `response.meta`: server response and HTTP status
- `server.log`: llama.cpp logs
- `profile.ncu-rep`: Nsight Compute report
- `ncu-details.csv` and `ncu-raw.csv`: exported report pages
- `summary.md`: run metadata and checklist

Before expensive captures, do a dry run:

```bash
scripts/profile-llama-server-ncu.sh \
  --name baseline-f16-f16 \
  --dry-run
```

## Recommended run sequence

### 1. f16/f16 baseline

```bash
scripts/profile-llama-server-ncu.sh \
  --name baseline-f16-f16 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --ncu-launch-count 80
```

Expected evidence:

- `server.log` shows `type_k = f16` and `type_v = f16`.
- No NVFP4 V-cache switch enablement log should appear.
- `ncu` captures ordinary f16/F32 attention and matmul kernels plus model
  weight matmul kernels.

### 2. K-cache NVFP4 experiment

```bash
scripts/profile-llama-server-ncu.sh \
  --name k-nvfp4-v-f16 \
  --cache-type-k nvfp4 \
  --cache-type-v f16 \
  --ncu-kernel-name 'regex:(nvfp4|cublas|gemm|set_rows)' \
  --ncu-launch-count 120
```

Expected evidence:

- `server.log` shows `type_k = nvfp4` and `type_v = f16`.
- `ncu` should include NVFP4 K-cache write/read support kernels such as
  `set_rows` or NVFP4 staging kernels.
- During attention KQ, native FP4 evidence is a combination of:
  - NVFP4 native matmul logs from `ggml/src/ggml-cuda/expt/nvfp4/nvfp4-matmul.cu`;
  - quantization/staging kernels for F32 Q to NVFP4;
  - cuBLASLt GEMM kernels in `ncu` for the same request window.
- Absence of the NVFP4 native logs or absence of NVFP4 staging kernels usually
  means the run fell back to a non-native path or did not hit the intended K
  cache path.

### 3. V-cache NVFP4 FP4 P*V

The current NVFP4 V-cache runtime requires `flash_attn=0`, `--kv-unified`,
KQV offload enabled, and the V-cache experiment switch. Its p*v matmul
dynamically quantizes P rows to NVFP4 by default:

```bash
LLAMA_EXPERIMENT_NVFP4_VCACHE=1 \
scripts/profile-llama-server-ncu.sh \
  --name v-nvfp4-fp4-pv \
  --cache-type-k f16 \
  --cache-type-v nvfp4 \
  --ncu-kernel-name 'regex:(vcache|nvfp4|quantize|matmul|set_rows)' \
  --ncu-launch-count 160
```

Expected evidence:

- `server.log` contains:
  - `LLAMA_EXPERIMENT_NVFP4_VCACHE=1 -> enabled`;
  - `ggml_cuda_vcache_nvfp4_log_fp4_pv_once: CUDA NVFP4 V-cache p*v quantizes
    P to dynamic NVFP4 by default`.
- `server.log` shows `type_v = nvfp4`.
- `ncu` includes V-cache NVFP4 store kernels and P quantization kernels.
- If cuBLASLt FP4 is unavailable for the shape or toolkit, expect custom CUDA
  kernels from `vcache-nvfp4-matmul.cu`; this still proves P is dynamically
  quantized to FP4 before dotting with NVFP4 V.

### 4. V-cache NVFP4 cuBLASLt FP4 P*V

```bash
LLAMA_EXPERIMENT_NVFP4_VCACHE=1 \
scripts/profile-llama-server-ncu.sh \
  --name v-nvfp4-fp4-pv-lt \
  --cache-type-k f16 \
  --cache-type-v nvfp4 \
  --ncu-kernel-name 'regex:(vcache|nvfp4|cublas|gemm|matmul|quantize)' \
  --ncu-launch-count 200
```

Expected evidence:

- `LLAMA_EXPERIMENT_NVFP4_VCACHE=1 -> enabled` appears in `server.log`.
- A one-shot active log from `vcache-nvfp4-matmul.cu` indicates successful
  cuBLASLt FP4 P*V use. If Lt is unavailable or returns unsupported, the code can
  fall back to the custom CUDA dot path, so check logs before trusting only a
  kernel name.
- `ncu` should show staging/quantization kernels plus cuBLASLt GEMM kernels.
  cuBLASLt kernel names are driver/toolkit dependent; use the source logs and
  the presence of `cublasLtMatmul`-driven kernels together.

### 5. V-cache FP8

```bash
scripts/profile-llama-server-ncu.sh \
  --name v-fp8-e4m3-e8m0-32 \
  --cache-type-k f16 \
  --cache-type-v fp8_e4m3_e8m0_32 \
  --ncu-kernel-name 'regex:(fp8|e8m0|cublas|gemm|matmul|set_rows)' \
  --ncu-launch-count 160
```

Expected evidence:

- `server.log` shows `type_v = fp8_e4m3_e8m0_32`.
- `ncu` includes FP8 quantization/repack kernels and cuBLASLt GEMM kernels when
  the native FP8 path is available.
- If testing the E4M2 masking experiment, include
  `GGML_FP8_E4M3_E8M0_32_EXPERIMENT_E4M2=1` and verify the one-shot quantizer or
  copy log confirms the switch state.

## Reading the evidence

Use three independent signals before concluding that a path is active:

1. Runtime configuration: `run-ncu-server.sh`, `env.txt`, and `server.log` must
   show the intended cache types and experiment switches.
2. llama.cpp path logs: once-only logs must confirm enabled/disabled state and,
   when available, successful active path use.
3. NCU report: `ncu-details.csv` or `ncu-raw.csv` must include the expected
   quantization, staging, custom matmul, or cuBLASLt kernels during the request.

Do not infer "FP4*FP4 is active" from a cuBLASLt kernel name alone. cuBLASLt
renames and selects kernels internally. For FP4 or FP8 Tensor Core paths, combine
the code-path log with the NCU report showing the associated quantize/stage work
and GEMM launches in the same captured request.

## Suggested comparisons

For each code change in a KV-cache experiment, keep at least two sibling folders:

- `baseline-f16-f16`: exact baseline cache types and request.
- `experiment-name`: same command and request, with only the experiment cache
  type and switch changed.

In the experiment summary, report:

- cache types and switches;
- whether switch logs confirm enabled or disabled state;
- whether the expected NCU kernels appeared;
- prompt/decode throughput or latency from `server.log`;
- any changed baseline parameter that invalidates strict A/B comparison.

## Practical notes

- `ncu --set basic` is still expensive on full server runs. Start with
  `--ncu-launch-count` and a focused `--ncu-kernel-name` filter.
- If kernel names are hidden by CUDA graphs, rerun a diagnostic capture with
  `GGML_CUDA_DISABLE_GRAPHS=1`. Record it as diagnostic only unless the baseline
  is captured with the same environment.
- For decode-path V-cache checks, keep `n_predict` high enough to execute several
  decode iterations. The script default is 32 generated tokens.
- For prefill K-cache write checks, keep a non-trivial prompt. The generated
  default prompt intentionally contains enough text to exercise prompt processing
  and decode in one request.
