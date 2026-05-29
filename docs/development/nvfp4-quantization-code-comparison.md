# NVFP4 Quantization Code Comparison

This note compares the standalone files `QuantBF16_NvFp4.hpp`,
`QuantBF16_NvFp4.cpp`, and `nvfp.hpp` with the current llama.cpp NVFP4
quantization implementation.

## Scope

The three standalone files contain two different models:

- `nvfp.hpp` is a compact float reference implementation for the same logical
  NVFP4 block format used by llama.cpp.
- `QuantBF16_NvFp4.*` is a BF16-input, fixed-point, hardware-style behavior
  model that produces a packed NVFP4 block reference.

The current llama.cpp implementation is broader than both. It includes CPU
reference quantization/dequantization, CPU activation roundtrip for NVFP4
weights, CUDA KV-cache storage quantization, CUDA native matmul RHS
quantization, cuBLASLt scale-channel repacking, and experiment paths such as
NVFP4 K-cache outlier sidecars.

## Common NVFP4 Format

Both `nvfp.hpp` and llama.cpp use the same basic `block_nvfp4` layout:

- block size: `QK_NVFP4 = 16`;
- scale: one byte `e`, interpreted as E4M3 and then multiplied by `0.5`;
- payload: `QK_NVFP4 / 2 = 8` bytes, each byte packing two E2M1 values;
- value table: `{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}`;
- packing order: low nibble is element `2*j`, high nibble is element `2*j+1`.

Relevant code:

- `nvfp.hpp`: `block_nvfp4`, `kvalues_nvfp4`, packing in
  `quantize_row_nvfp4_ref()`.
- llama.cpp: `ggml/src/ggml-common.h` defines `block_nvfp4` and
  `kvalues_nvfp4`; `ggml/src/ggml-quants.c` packs/dequantizes in the same
  low-nibble-first order.

## `nvfp.hpp` Versus llama.cpp Reference Quantization

The high-level algorithm in `nvfp.hpp` matches llama.cpp's CPU reference path:

1. Split each row into 16-value blocks.
2. Compute `vmax = max(abs(x))` for the block.
3. Compute the unquantized block scale:

   ```text
   scale = global_scale * (vmax / 6)
   ```

4. Quantize `scale` to a finite E4M3 byte.
5. Decode that E4M3 byte with the NVFP4 half-scale convention:

   ```text
   scale_f = E4M3(e) * 0.5
   inv_scale = global_scale / scale_f
   ```

6. Quantize each scaled value `x * inv_scale` to the nearest E2M1 table entry.
7. Dequantize as:

   ```text
   x_deq = kvalues_nvfp4[nibble] * (scale_f / global_scale)
   ```

This is the same mathematical form used by llama.cpp's
`quantize_row_nvfp4_ref()` and `dequantize_row_nvfp4()`.

The concrete differences are small:

| Area | `nvfp.hpp` | llama.cpp |
|---|---|---|
| Function linkage | Header-only standalone helpers | Exported ggml quant/dequant functions |
| Input precision | FP32 reference plus local BF16 conversion helpers | FP32 reference; BF16 conversion lives in ggml where needed |
| E2M1 tie handling | Exact ties prefer larger magnitude | Strict `<` update; first matching table entry wins |
| Default global scale | Optional default `1.0f` | Default overload aborts; caller must pass global scale |
| QK=8 variant | Not present | `GGML_TYPE_NVFP4_8` reference functions exist |

The tie-handling difference is normally low impact. For ordinary half-way
thresholds between positive magnitudes, the larger magnitude appears later in
the table, so both implementations usually choose it. The duplicate zero
entries are a visible semantic difference: `nvfp.hpp` can prefer the later zero
when the error and magnitude are both tied, while llama.cpp keeps the first zero.

## `QuantBF16_NvFp4.*` Algorithm

`QuantBF16_NvFp4_Model::process()` is not the same style of reference as
`nvfp.hpp`. It models a BF16-to-NVFP4 hardware datapath:

1. Rejects unsupported input: null pointer, non-positive `s_tensor`, or block
   size other than 16.
2. Converts the tensor scale into a fixed-point global scale:

   ```text
   global_scale_q = fixed_q16(1 / s_tensor)
   ```

3. Finds block absmax by comparing BF16 absolute-value bit patterns.
4. Computes a fixed-point block scale by multiplying the BF16 absmax by
   `global_scale_q`, then approximating division by 6 with shift-add terms:

   ```text
   (x + 3) >> 3
 + (x + 3) >> 5
 + (x + 3) >> 7
 + (x + 3) >> 9
 + (x + 3) >> 11
 + (x + 3) >> 13
   ```

5. Converts that fixed-point block scale to an E4M3-like byte with explicit
   MSB detection, exponent field construction, mantissa rounding, carry,
   floor, and clamp behavior.
6. Reconstructs a half-scale fixed-point value from the scale byte.
7. Quantizes each BF16 magnitude by comparing `2 * target` against integer
   thresholds corresponding to nearest-neighbor boundaries for
   `{0, 1, 2, 3, 4, 6, 8, 12}`.
8. Reattaches sign with bit 3 and packs low element in the low nibble, high
   element in the high nibble.

This means `QuantBF16_NvFp4.*` is intended to answer a different question:
"what packed NVFP4 bytes would this fixed-point BF16 hardware-like path emit?"
It is not just a shorter version of llama.cpp's FP32 reference quantizer.

## Main Algorithmic Differences

### Input Domain

`QuantBF16_NvFp4.*` consumes BF16 bit patterns directly. It never converts the
whole block to FP32 for the main path. llama.cpp consumes FP32 values in the
reference quantizer, CPU activation roundtrip, CUDA K-cache set_rows path, and
CUDA native matmul RHS quantization.

This affects exact behavior near BF16 rounding boundaries. If a caller starts
from FP32 activations and uses `QuantBF16_NvFp4.*`, the FP32-to-BF16 conversion
step is outside the model and can change the final packed bytes.

### Global Scale Convention

llama.cpp's quantizer takes `global_scale` directly. In model graph binding,
stored input scale tensors are converted to `global_scale = 1 / input_scale`
before activation quantization. CPU activation roundtrip does this in
`src/llama-nvfp4.cpp`; CUDA native matmul does the same through
`ggml_cuda_nvfp4_input_global_scale()`.

`QuantBF16_NvFp4_Model::process()` takes `s_tensor` and internally uses
`1 / s_tensor` as the Q16 global scale. If `s_tensor` corresponds to
llama.cpp's `input_scale`, the conventions align conceptually. If a caller
passes llama.cpp's already-inverted `global_scale`, the result will be inverted
again and will not match.

### Block Scale Quantization

llama.cpp reference code computes a floating-point `scale` and chooses the
nearest finite E4M3 byte by scanning all 256 encodings.

`QuantBF16_NvFp4.*` computes the scale in integer/fixed-point form. Its division
by 6, E4M3 exponent construction, mantissa rounding, and saturation behavior
are all explicit bit-level rules. These rules can disagree with nearest finite
E4M3 search by one scale code near boundaries. That disagreement will shift all
16 E2M1 decisions in the block.

### E2M1 Quantization

llama.cpp quantizes each scaled FP32 value by nearest absolute error against the
signed `kvalues_nvfp4` table.

`QuantBF16_NvFp4.*` quantizes magnitudes by threshold comparisons in fixed-point
space:

```text
0/1:   target * 2 < 1 * half_scale
1/2:   target * 2 < 3 * half_scale
2/3:   target * 2 < 5 * half_scale
3/4:   target * 2 < 7 * half_scale
4/6:   target * 2 < 10 * half_scale
6/8:   target * 2 < 14 * half_scale
8/12:  target * 2 < 20 * half_scale
```

Those thresholds are the nearest-neighbor boundaries for the positive E2M1
levels, but implemented with integer arithmetic and strict `<` comparisons.
Values exactly on a threshold round upward because the lower bucket uses `<`,
not `<=`. llama.cpp also usually rounds upward at exact positive midpoints
because the larger magnitude appears later and has smaller-or-equal error only
after the equality point, but duplicate zero and floating-point precision can
still produce byte-level differences.

### Scale Byte Meaning for cuBLASLt

llama.cpp's CUDA native path stores `block_nvfp4.e` in ggml's logical E4M3
encoding, then splits data and scale channels for cuBLASLt. During splitting,
scale bytes may be converted or tiled for cuBLASLt's expected FP4 scale-channel
layout.

`QuantBF16_NvFp4.*` returns an `NvFp4BlockRef` with `block_scale_e4m3` and
packed E2M1 data. It does not perform cuBLASLt channel splitting, 128x4 scale
tiling, source-weight repacking, row splitting, or `alpha` compensation.

### Runtime Integration

llama.cpp has runtime paths that do not exist in the standalone code:

- model graph nodes bind NVFP4 input and weight scale tensors to
  `GGML_OP_MUL_MAT`;
- CPU graph execution can roundtrip activations through NVFP4 before matmul;
- CUDA native matmul dynamically quantizes the RHS activation matrix to NVFP4;
- CUDA KV-cache set_rows stores NVFP4 K/V cache rows and side scale tensors;
- NVFP4 K-cache outlier experiments can remove outliers before quantization and
  apply sparse correction after KQ;
- V-cache and flash-attention experiments have specialized quantization and
  matmul paths.

The standalone files are useful for byte-level block behavior, not for proving
end-to-end llama.cpp runtime behavior.

## Expected Match Cases

The standalone `nvfp.hpp` reference should generally match llama.cpp for:

- FP32 input;
- block size 16;
- the same `global_scale`;
- normal finite values;
- no K-cache outlier zeroing;
- no dynamic per-row scale mode;
- comparing `block_nvfp4` bytes before cuBLASLt repacking.

The `QuantBF16_NvFp4.*` model should only be expected to match llama.cpp if the
llama.cpp path is modified or configured to use the same BF16 input rounding,
fixed-point global-scale conversion, fixed-point block-scale quantization, and
threshold rules. With the current code, it should be treated as a related
hardware-behavior model, not as an exact oracle for current llama.cpp output.

## Practical Implications

- For current llama.cpp CPU/CUDA validation, `nvfp.hpp` is the closer algorithmic
  reference.
- For hardware alignment work, `QuantBF16_NvFp4.*` is valuable because it
  exposes fixed-point scale rounding and threshold decisions that llama.cpp's
  float nearest-neighbor implementation does not model.
- If byte-for-byte parity with `QuantBF16_NvFp4.*` becomes a requirement, the
  main implementation gap is the block-scale quantizer. The current llama.cpp
  code would need a switch-gated BF16/fixed-point scale path or a dedicated
  comparator test that accepts the expected differences.
- Any parity test must explicitly state the scale convention: `s_tensor` in the
  BF16 model corresponds to a tensor/input scale whose reciprocal is the
  quantizer global scale.
