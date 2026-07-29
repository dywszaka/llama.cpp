# verify-rope.py RoPE reconstruction algorithm

`verify-rope.py` validates an exported `GGML_OP_ROPE` destination tensor by
reconstructing the expected output from three inputs:

- the exported ROPE input tensor, `src0`;
- the exported I32 position tensor, `src1`;
- a precomputed F32 little-endian cos/sin table, usually
  `rope-cos-sin-f32.bin`.

The validator does not recompute trigonometric values. It only looks up
precomputed `(cos, sin)` pairs from the cos/sin binary table.

## Inputs

The result path passed on the command line is the exported ROPE `dst` binary.
The script loads `manifest.json` from the result directory unless
`--manifest` is provided. From that manifest it resolves all records with the
same ROPE `node_index`:

- `dst`: exported ROPE result to validate;
- `src0`: input tensor before RoPE;
- `src1`: I32 position tensor;
- `op_params`: RoPE parameters recorded on the `dst` manifest record.

The cos/sin manifest defaults to `rope-cos-sin-manifest.json` next to the
script unless `--cos-sin-manifest` is provided. That manifest points to the
actual data file.

## cos/sin table layout

The table manifest must have:

```json
{
  "format": "llama_cuda_rope_cos_sin_v1",
  "dtype": "f32_le",
  "component_order": ["cos", "sin"],
  "shape": [context_size, channels, 2]
}
```

`channels` must equal `n_dims / 2`. The binary layout is position-major, then
channel-major, then component:

```text
offset_bytes = ((position * channels + channel_idx) * 2 + component) * 4
component 0 = cos
component 1 = sin
```

So the script reads each pair as:

```python
cos_value, sin_value = struct.unpack_from("<ff", raw, offset_bytes)
```

## Compatibility checks

The validator accepts only one-dimensional GPT-NeoX RoPE:

- `mode` must include `ROPE_TYPE_NEOX`;
- MROPE and VISION modes are rejected;
- a frequency-factor `src2` is rejected because the static table was generated
  without per-position frequency factors.

The following RoPE parameters from the exported node must match the cos/sin
table manifest:

- exact integer match: `n_dims`, `n_ctx_orig`;
- exact floating value match: `freq_base`, `freq_scale`, `ext_factor`,
  `attn_factor`, `beta_fast`, `beta_slow`.

The script also requires:

- `dst` and `src0` have the same shape and both are F16 or both are F32;
- `src1` is I32;
- `n_dims` is positive, even, and no larger than `src0.ne[0]`;
- the number of positions equals `src0.ne[2] * src0.ne[3]`.

## Reconstruction formula

For each token, the script first maps tensor indices to a position id:

```python
token_index = i3 * src0.ne[2] + i2
position = src1[token_index]
```

For every `i1` lane and every rotary channel in the first half of `n_dims`,
the script loads the GPT-NeoX half-split pair:

```python
x0 = src0[channel_idx,              i1, i2, i3]
x1 = src0[channel_idx + n_dims / 2, i1, i2, i3]
```

Then it reads the table pair for that token position and channel:

```python
cos_value, sin_value = table[position, channel_idx]
```

The expected rotated values are:

```python
expected0 = x0 * cos_value - x1 * sin_value
expected1 = x0 * sin_value + x1 * cos_value
```

These are compared against:

```python
dst[channel_idx,              i1, i2, i3]
dst[channel_idx + n_dims / 2, i1, i2, i3]
```

Channels outside the rotary dimension are not rotated. For every
`i0 >= n_dims`, the expected output is the original input value:

```python
expected = src0[i0, i1, i2, i3]
```

## Result rounding

The expected values are optionally rounded before comparison:

- `--result-rounding f32`: no extra rounding;
- `--result-rounding bf16-rne`: round the expected value to BF16 with
  round-to-nearest-even, then convert back to F32;
- `--result-rounding auto`: use `bf16-rne` when `command.txt` contains
  `GGML_CUDA_TRUNC_ENABLE=1`, otherwise use `f32`.

## Error check

Each reconstructed value is compared with the exported `dst` value using:

```python
abs(actual - expected) <= atol + rtol * abs(expected)
```

The script reports mismatch details up to `--max-mismatches`, plus maximum
absolute and relative error over all checked elements.

## How the bundled cos/sin table is generated

`generate-rope-cos-sin.cpp` builds a CUDA `ggml_rope_ext()` graph with:

- an F32 input tensor shaped `[n_dims, 1, context_size, 1]`;
- positions `[0, context_size)`;
- first-half channels set to `1.0`;
- second-half channels set to `0.0`.

With this basis input, GPT-NeoX RoPE produces:

```text
first half  = cos(position, channel)
second half = sin(position, channel)
```

The generator then writes those values as interleaved F32 pairs:

```text
[position][channel][cos, sin]
```
