# FP8 E4M3 E8M0 16 V Cache Design

## Goal

Add `--cache-type-v fp8_e4m3_e8m0_16` for non-flash CUDA attention. The V cache and the materialized attention probability matrix `P` use the same block-16 `FP8(E4M3)+E8M0` quantization before `V * P`.

## Scope

Flash attention is out of scope. The new type is CUDA-oriented for non-flash attention and must require KQV offload in the same way as the existing block-32 FP8 V cache path.

## Approach

Introduce a new ggml quantized type, `GGML_TYPE_FP8_E4M3_E8M0_16`, with a 16-value block containing one E8M0 scale byte and 16 E4M3 payload bytes. Wire it into the CLI V cache type list, CPU reference quantize/dequantize helpers, CUDA copy/set-rows helpers, and KV cache layout selection.

For non-flash `V * P`, add a native CUDA cuBLASLt path for `src0=fp8_e4m3_e8m0_16`, `src1=f32`, `dst=f32`. The path first quantizes each contiguous 16-value `P` block with the same E8M0 scale and E4M3 payload semantics as V. Because cuBLASLt exposes `VEC32_UE8M0` scale channels, each logical block16 V/P block is repacked as one vec32 group by appending 16 zero FP8 payload values, preserving block16 quantization semantics while using cuBLASLt for accumulation.

## Testing

Extend `tests/test-vcache-fp8-e4m3-e8m0.cu` with block16 roundtrip and CUDA `mul_mat` cases. The regression compares block16 `V * quantized(P)` against the existing F32 baseline with tolerances similar to the block32 path.
