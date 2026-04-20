# Non-Flash-Attn Decode-Only FP8 P/V Design

## Goal

Add a CUDA path that supports non-`flash_attn` decode for `--cache-type-v fp8_e4m3_e8m0_32`, with both `V` and `P` quantized to `FP8(E4M3)` using `E8M0` scales, and with `P x V` computed directly through FP8 tensor operations instead of dequantizing to `F16`/`F32` first.

## Scope

- CUDA only.
- Non-`flash_attn` path only.
- Decode only: `q_len = 1`, `bs = 1`.
- Qwen3-8B-NVFP4 main path first, starting with `n_embd_head_v = 128`.
- Experimental mode only. No silent fallback to `flash_attn`.

Out of scope:

- Prefill.
- `bs > 1`.
- Generic multi-shape `mul_mat`.
- CPU, Metal, Vulkan, HIP, SYCL.
- Full `flash_attn` parity.

## Current Problem

Today the experimental FP8 `V` cache support is only wired into the CUDA `ggml_flash_attn_ext()` path. Non-`flash_attn` attention uses:

1. `kq = mul_mat(k, q)`
2. `kq = soft_max_ext(kq)`
3. `kqv = mul_mat(v, kq)`

That path materializes `P = softmax(KQ)` as a tensor and stores `V` in the non-`flash_attn` transposed KV layout. The current experimental implementation does not provide a direct FP8 `mul_mat(v_fp8, p_fp8)` kernel for that layout, so the code currently forces or requires `flash_attn`.

## Chosen Approach

Implement a decode-only experimental branch in the non-`flash_attn` path:

1. Keep the existing non-`flash_attn` `KQ -> softmax` logic.
2. After `soft_max_ext`, quantize the materialized `P` vector online into a block FP8 format compatible with the existing `V` experimental format semantics: `E4M3` payload and `E8M0` block scale.
3. Add a CUDA decode-only `mul_mat(v_fp8, p_fp8)` path that consumes the non-`flash_attn` transposed `V` layout directly and computes `FP8 * FP8 -> FP32 accumulate`.
4. Dispatch that path only when all experimental constraints are satisfied.
5. Remove the current environment-variable override and rely on the standard V cache type parameter.

## Data Representation

### V cache

- Reuse `GGML_TYPE_FP8_E4M3_E8M0_32` as the stored `V` type.
- Preserve non-`flash_attn` KV layout, where `V` is stored transposed relative to the `flash_attn` layout.
- Block size remains 32 values per block.
- Each block stores:
  - 32 `E4M3` values
  - 1 `E8M0` scale

### P vector

- `P` remains a temporary tensor produced by `soft_max_ext`.
- Introduce an experimental CUDA-side packing step for decode-only use:
  - Read the `F32`/`F16` `P` values for the current decode step
  - Pack contiguous groups into `FP8(E4M3) + E8M0 scale` blocks
- `P` is not written into KV cache.

## Kernel Strategy

### Why a dedicated kernel

The generic CUDA `mul_mat` path does not currently provide a decode-specialized `FP8(V) * FP8(P)` implementation for the non-`flash_attn` `V` layout. The new path will be narrow and explicit instead of trying to generalize all `mul_mat` forms.

### Decode-only kernel contract

- Input `V`: non-`flash_attn` transposed KV cache view in `GGML_TYPE_FP8_E4M3_E8M0_32`
- Input `P`: temporary packed decode vector in block FP8 format
- Output: `FP32` accumulator, converted to the expected downstream output format
- Shape assumptions:
  - single query token
  - single batch item
  - first implementation for head dimension 128

### Compute requirement

- Use CUDA FP8 tensor operations directly for the inner products.
- Do not dequantize the entire `V` or `P` tensors into `F16` or `F32` before multiplication.
- Accumulate into `FP32`.

The implementation can still unpack or rearrange fragments into tensor-core-compatible tile fragments, but the multiply path itself must remain FP8-input MMA rather than a scalar dequantize-and-multiply loop.

## Dispatch Rules

Enable the experimental non-`flash_attn` path only when all of the following hold:

- `--cache-type-v fp8_e4m3_e8m0_32`
- CUDA backend
- `flash_attn == false`
- decode-only graph shape
- supported head dimension
- supported architecture for the chosen FP8 tensor op path

If any requirement is not met, fail explicitly with a targeted error message. Do not silently switch to `flash_attn`.

## API and Behavior Changes

- The FP8 V cache type is selected through `--cache-type-v fp8_e4m3_e8m0_32` or `LLAMA_ARG_CACHE_TYPE_V=fp8_e4m3_e8m0_32`.
- `llama_init_from_model()` should reject unsupported combinations directly:
  - unsupported model architecture
  - unsupported non-`flash_attn` shape
  - unsupported backend or GPU capability
- Existing `flash_attn` experimental path remains available when the caller explicitly requests `-fa`.

## Testing Strategy

### Red/green tests

1. Add a failing smoke test for non-`flash_attn` decode with `--cache-type-v fp8_e4m3_e8m0_32` and no `-fa`.
2. Make it pass once the new dispatch and kernel path are in place.

### Numeric tests

Add a focused CUDA regression test for decode-only attention:

- baseline: non-`flash_attn`, `V=f16`, `P=f32`
- experimental: non-`flash_attn`, `V=fp8_e4m3_e8m0_32`, `P=fp8_e4m3_e8m0_32`

Measure:

- `nmse`
- `max_abs`
- optionally greedy token agreement in a short decode loop

### Smoke verification

Run both:

- `llama-cli` short generation without `-fa`
- `llama-server` startup without `-fa`

Both must show:

- non-`flash_attn` configuration remains in effect
- experimental `V` type is active
- no forced `flash_attn`

## Risks

- The non-`flash_attn` `V` layout is different from the current `flash_attn` experimental path, so indexing errors are the main correctness risk.
- Decode-only specialization reduces surface area but still requires careful graph-shape gating.
- FP8 tensor-core fragment layout constraints may force additional packing logic for `P`, which is acceptable as long as the multiplication itself remains FP8-input MMA.

## Acceptance Criteria

- Debug and Release `llama-cli` and `llama-server` can run non-`flash_attn` decode with `--cache-type-v fp8_e4m3_e8m0_32`.
- The path does not auto-enable `flash_attn`.
- `P x V` uses direct FP8-input tensor operations with `FP32` accumulation.
- Focused CUDA tests and smoke tests pass on the RTX 5090 environment.
