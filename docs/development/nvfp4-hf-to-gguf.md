# NVFP4 HF to GGUF Conversion Notes

This document explains how `llama.cpp` converts Hugging Face / ModelOpt NVFP4 tensors into GGUF, based on the implementation merged by PR [#19769](https://github.com/ggml-org/llama.cpp/pull/19769) and the current `master` branch.

Relevant code:

- [convert_hf_to_gguf.py](/convert_hf_to_gguf.py)
- [ggml/src/ggml-common.h](/ggml/src/ggml-common.h)
- [ggml/src/ggml-quants.c](/ggml/src/ggml-quants.c)
- [src/llama-model.cpp](/src/llama-model.cpp)
- [src/llama-graph.cpp](/src/llama-graph.cpp)

## Summary

The NVFP4 conversion path does not dequantize weights to floating point and then re-quantize them.

Instead, it:

1. Detects that the input model uses ModelOpt NVFP4.
2. Skips the normal generic dequantization path.
3. Repackages the original packed FP4 values into ggml's NVFP4 block layout.
4. Repackages the per-16-value scales into ggml's `UE4M3` scale bytes.
5. Stores the extra `weight_scale_2` factor as a separate `.scale` tensor when it is not trivially `1.0`.

In other words, the main work is a layout transform, not a value-domain transform.

## Detection and control flow

The dedicated path is enabled when `quant_algo == "NVFP4"` is found in either:

- `config.json` -> `quantization_config.quant_algo`
- `hf_quant_config.json` -> `quantization.quant_algo`

See:

- [convert_hf_to_gguf.py:677](/convert_hf_to_gguf.py#L677)

Once detected:

- `dequant_model()` returns immediately for NVFP4.
- `modify_tensors()` skips `.weight_scale`, `.weight_scale_2`, `.input_scale`, `.k_scale`, `.v_scale`, and the paired `.weight` tensors that have NVFP4 metadata.
- `_generate_nvfp4_tensors()` writes the final GGUF tensors directly.

See:

- [convert_hf_to_gguf.py:522](/convert_hf_to_gguf.py#L522)
- [convert_hf_to_gguf.py:677](/convert_hf_to_gguf.py#L677)

## HF-side tensor layout

For a typical NVFP4 linear tensor, ModelOpt provides:

- `weight`: shape `[out_features, in_features / 2]`, dtype `uint8`
- `weight_scale`: shape `[out_features, in_features / 16]`, float8-like per-block scale
- `weight_scale_2`: optional outer scale, usually scalar per tensor, or scalar per expert for MoE

Interpretation:

- one byte in `weight` stores two 4-bit values
- one `weight_scale[..., b]` applies to 16 logical weight values
- those 16 logical values are stored in 8 bytes

That is why the converter derives:

- `n_blocks = scale.shape[1]`
- `weight.reshape(out_features, n_blocks, 8)`

See:

- [convert_hf_to_gguf.py:572](/convert_hf_to_gguf.py#L572)

## HF packed nibble order

ModelOpt stores each 16-value block as 8 bytes in this order:

```text
byte[i] = val[2*i] | (val[2*i + 1] << 4),  i = 0..7
```

So the 16 logical FP4 codes are packed as adjacent pairs:

```text
byte0 = val0  | (val1  << 4)
byte1 = val2  | (val3  << 4)
byte2 = val4  | (val5  << 4)
byte3 = val6  | (val7  << 4)
byte4 = val8  | (val9  << 4)
byte5 = val10 | (val11 << 4)
byte6 = val12 | (val13 << 4)
byte7 = val14 | (val15 << 4)
```

The converter first unpacks these nibbles:

```python
w = weight.reshape(out_features, n_blocks, 8)
vals = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(out_features, n_blocks, 16)
```

See:

- [convert_hf_to_gguf.py:575](/convert_hf_to_gguf.py#L575)

## ggml NVFP4 block layout

In ggml, one NVFP4 block represents 64 logical values:

- `QK_NVFP4 = 64`
- `QK_NVFP4_SUB = 16`

The in-memory block is:

```c
typedef struct {
    uint8_t d[4];   // 4 UE4M3 scales, one per 16-value sub-block
    uint8_t qs[32]; // 32 bytes of packed 4-bit values
} block_nvfp4;
```

So one ggml block is:

- 4 scale bytes
- 32 data bytes
- total 36 bytes for 64 logical values

See:

- [ggml/src/ggml-common.h:200](/ggml/src/ggml-common.h#L200)

## Reordering from HF layout to ggml layout

ggml expects each 16-value sub-block in this nibble arrangement:

```text
qs[j] = val[j] | (val[j + 8] << 4),  j = 0..7
```

This means the first half and second half of the 16-value block are interleaved by position:

```text
ggml_byte0 = val0 | (val8  << 4)
ggml_byte1 = val1 | (val9  << 4)
ggml_byte2 = val2 | (val10 << 4)
ggml_byte3 = val3 | (val11 << 4)
ggml_byte4 = val4 | (val12 << 4)
ggml_byte5 = val5 | (val13 << 4)
ggml_byte6 = val6 | (val14 << 4)
ggml_byte7 = val7 | (val15 << 4)
```

That is why the converter rebuilds `qs` as:

```python
qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).to(torch.uint8)
```

See:

- [convert_hf_to_gguf.py:581](/convert_hf_to_gguf.py#L581)

## Scale conversion

There are two distinct scale concepts in the HF source format.

### `weight_scale`

`weight_scale` is the per-16-value local scale.

The converter does not decode it to float and then re-encode it numerically.
Instead, it preserves the source float8 bit pattern and maps it into ggml's unsigned `UE4M3` byte representation by clearing the sign bit:

```python
d_ue = scale.view(torch.uint8).numpy().reshape(out_features, n_blocks) & 0x7F
```

This is intentional. The resulting bytes are written directly into `block_nvfp4.d`.

See:

- [convert_hf_to_gguf.py:579](/convert_hf_to_gguf.py#L579)

On the ggml side, each `UE4M3` byte is decoded by `ggml_ue4m3_to_fp32()`, which returns the scale already multiplied by `0.5` to match the doubled `kvalues_mxfp4` convention.

See:

- [ggml/src/ggml-impl.h:494](/ggml/src/ggml-impl.h#L494)
- [ggml/src/ggml-quants.c:472](/ggml/src/ggml-quants.c#L472)

### `weight_scale_2`

`weight_scale_2` is the outer scale factor.

In the final implementation it is not baked into the packed NVFP4 block data.

Instead:

- if it is missing, the converter assumes `1.0`
- if it is scalar and equal to `1.0`, no extra tensor is written
- otherwise it is emitted as a separate `.scale` tensor in `F32`

Per-tensor case:

- `blk.X.attn_q.weight`
- `blk.X.attn_q.scale`

Per-expert case:

- `blk.X.ffn_gate_exps.weight`
- `blk.X.ffn_gate_exps.scale`

See:

- [convert_hf_to_gguf.py:590](/convert_hf_to_gguf.py#L590)
- [convert_hf_to_gguf.py:599](/convert_hf_to_gguf.py#L599)
- [convert_hf_to_gguf.py:667](/convert_hf_to_gguf.py#L667)

## 16-value sub-block diagram

One `weight_scale[..., b]` aligns with one 16-value logical block:

```text
HF source
---------
weight_scale[o, b]
    |
    +-- weight[o, b, 0..7 bytes]
            |
            +-- 16 logical FP4 codes

Logical values in one block:

    val0  val1  val2  val3  val4  val5  val6  val7
    val8  val9  val10 val11 val12 val13 val14 val15
```

Byte packing before and after conversion:

```text
HF byte packing:
    b0 = val0 | (val1  << 4)
    b1 = val2 | (val3  << 4)
    b2 = val4 | (val5  << 4)
    b3 = val6 | (val7  << 4)
    b4 = val8 | (val9  << 4)
    b5 = val10| (val11 << 4)
    b6 = val12| (val13 << 4)
    b7 = val14| (val15 << 4)

ggml sub-block packing:
    q0 = val0 | (val8  << 4)
    q1 = val1 | (val9  << 4)
    q2 = val2 | (val10 << 4)
    q3 = val3 | (val11 << 4)
    q4 = val4 | (val12 << 4)
    q5 = val5 | (val13 << 4)
    q6 = val6 | (val14 << 4)
    q7 = val7 | (val15 << 4)
```

## 64-value super-block diagram

ggml groups four 16-value sub-blocks into one 64-value block:

```text
sub-block 0: values   0..15   -> scale d[0], bytes qs[ 0.. 7]
sub-block 1: values  16..31   -> scale d[1], bytes qs[ 8..15]
sub-block 2: values  32..47   -> scale d[2], bytes qs[16..23]
sub-block 3: values  48..63   -> scale d[3], bytes qs[24..31]
```

Physical byte layout in one ggml NVFP4 block:

```text
offset 0..3    : d[0], d[1], d[2], d[3]
offset 4..35   : qs[0] ... qs[31]
```

The converter builds this with:

```python
n_super = n_blocks // 4
d_grouped  = d_ue.reshape(out_features, n_super, 4)
qs_grouped = qs.reshape(out_features, n_super, 4, 8).reshape(out_features, n_super, 32)
raw = np.concatenate([d_grouped, qs_grouped], axis=-1).reshape(out_features, n_super * 36)
```

The logical tensor shape reported to GGUF stays in element units, not byte units:

```text
[out_features, n_super * 64]
```

See:

- [convert_hf_to_gguf.py:583](/convert_hf_to_gguf.py#L583)

## Alignment rules

The alignment relationship can be summarized as:

```text
HF:
    1 weight_scale entry  <-> 16 logical weights <-> 8 packed bytes

GGUF / ggml NVFP4:
    4 scale bytes         <-> 64 logical weights <-> 32 packed bytes
```

Or, equivalently:

```text
4 HF 16-value blocks = 1 ggml 64-value block
```

This is a pure regrouping of already-quantized data.

## MoE expert tensors

For expert tensors matching:

```text
.experts.<expert_id>.(gate_proj|up_proj|down_proj).weight
```

the converter does not immediately write each expert independently.

Instead it:

1. repacks each expert into raw ggml NVFP4 bytes
2. groups them by `(layer_id, proj_type)`
3. sorts by `expert_id`
4. stacks them into one GGUF tensor with an expert dimension
5. emits a sibling `.scale` tensor containing one `scale2` value per expert when needed

See:

- [convert_hf_to_gguf.py:625](/convert_hf_to_gguf.py#L625)
- [convert_hf_to_gguf.py:655](/convert_hf_to_gguf.py#L655)

At runtime, the model loader performs a generic pass that loads optional `.scale` tensors for many weights, including NVFP4.

See:

- [src/llama-model.cpp:7447](/src/llama-model.cpp#L7447)

During graph construction, the selected expert outputs are multiplied by the loaded `.scale` tensor using `ggml_mul()`.

See:

- [src/llama-graph.cpp:1376](/src/llama-graph.cpp#L1376)
- [src/llama-graph.cpp:1400](/src/llama-graph.cpp#L1400)
- [src/llama-graph.cpp:1421](/src/llama-graph.cpp#L1421)
- [src/llama-graph.cpp:1508](/src/llama-graph.cpp#L1508)

## Runtime dequantization view

Conceptually, ggml reconstructs each logical value as:

```text
value = kvalues_mxfp4[fp4_code] * ue4m3_scale
```

where:

- `fp4_code` comes from the packed nibble in `qs`
- `ue4m3_scale` comes from `d[s]`
- optional `scale2` is applied outside the packed block as a separate tensor multiply

The reference quantizer and dequantizer are:

- [ggml/src/ggml-quants.c:307](/ggml/src/ggml-quants.c#L307)
- [ggml/src/ggml-quants.c:472](/ggml/src/ggml-quants.c#L472)

## Practical takeaway

If you are tracing a single HF NVFP4 tensor through conversion:

1. Find `weight`, `weight_scale`, and optional `weight_scale_2`.
2. Split `weight` into 16-value logical blocks.
3. Reorder the nibble pairs from ModelOpt order to ggml order.
4. Group every four 16-value blocks into one 64-value ggml block.
5. Store the four local scale bytes in `d[0..3]`.
6. Store the 32 packed FP4 bytes in `qs[0..31]`.
7. Store outer `scale2` separately as `.scale` if it is not `1.0`.

That is the full HF to GGUF conversion logic for NVFP4 in `llama.cpp`.
