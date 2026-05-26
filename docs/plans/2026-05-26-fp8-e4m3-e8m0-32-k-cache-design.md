# FP8 E4M3 E8M0 32 K Cache Design

## Goal

Add experimental CUDA K-cache support for `--cache-type-k fp8_e4m3_e8m0_32`.
During KQ, the stored K cache uses `FP8(E4M3)+E8M0` block32 format and the F32
Q operand is quantized to the same FP8 block32 format inside the native CUDA
matmul path before cuBLASLt multiplication.

## Scope

- CUDA KQV-offload path first.
- Non-flash attention first.
- `GGML_TYPE_FP8_E4M3_E8M0_32` only.
- PPL validation with the repository baseline parameters.

Out of scope:

- Flash attention with FP8 K cache.
- CPU KQ execution with FP8 K cache.
- New FP8 block formats.
- A new dedicated KQ kernel.

## Approach

Reuse the existing FP8 block32 storage type, CUDA set_rows path, and native
FP8 matmul path. The attention graph keeps the existing non-flash structure:

1. `kq = ggml_mul_mat(k, q)`
2. CUDA dispatch sees `src0=GGML_TYPE_FP8_E4M3_E8M0_32`, `src1=F32`,
   `dst=F32`.
3. `ggml_cuda_mul_mat_fp8_e8m0_native()` quantizes the F32 Q operand into
   temporary FP8 block32 data and dispatches cuBLASLt with FP8 inputs and FP32
   accumulation.

The user-facing switch is the existing cache type argument:
`--cache-type-k fp8_e4m3_e8m0_32`. Because the cache type defaults to F16, the
experiment remains off by default.

## Runtime Rules

- Allow `fp8_e4m3_e8m0_32` in the K-cache type parser.
- Reject FP8 K cache when `offload_kqv=0`, because the intended KQ path is CUDA.
- Reject FP8 K cache with flash attention until a dedicated FP8-K flash-attn path
  is implemented.
- Keep FP8 V-cache behavior unchanged.

## Testing

- Add a CUDA smoke script that starts `llama-cli` with
  `--cache-type-k fp8_e4m3_e8m0_32`, `--cache-type-v f16`, non-flash attention,
  KQV offload, and unified KV cache.
- Extend the focused FP8 CUDA test with a KQ-shaped
  `mul_mat(K_fp8, Q_f32)` case. The test sets
  `GGML_CUDA_FP8_E8M0_NATIVE_NO_FALLBACK=1` so unsupported native dispatch is a
  hard failure instead of silently falling back.
- Run PPL with baseline parameters, comparing `f16/f16` against
  `fp8_e4m3_e8m0_32/f16`, and save scripts, raw logs, metrics, and summary in a
  dedicated `experiments/` folder.
