# First Decode Attention Softmax Dump

This debug path dumps the first non-flash attention softmax reached during the
first single-token generation decode step.

## Build

```bash
cmake --build build --target llama-cli -j2
```

Use the active build directory for the local configuration. The dump path is in
the core `llama` target and does not require a separate tool.

## Run

Set `LLAMA_DUMP_FIRST_DECODE_ATTN_SOFTMAX=1` and run a generation that produces
at least two predicted tokens. The first generated token is sampled from prompt
logits; the second token causes the first single-token generation decode.

Example:

```bash
LLAMA_DUMP_FIRST_DECODE_ATTN_SOFTMAX=1 \
./build/bin/llama-cli \
  -m /path/to/model.gguf \
  -ngl 0 -c 64 -b 16 -ub 16 -t 2 -n 2 \
  -p 'Hello' --no-display-prompt --no-warmup
```

Flash attention must be disabled so the explicit attention `GGML_OP_SOFT_MAX`
node exists. The default is disabled unless `--flash-attn` or
`LLAMA_ARG_FLASH_ATTN` enables it.

## Output

The fixed output directory is:

```text
experiments/first-decode-attn-softmax-dump/
```

Files:

- `attn_softmax_input.bin`: raw F32 bytes captured from the attention softmax
  `src0` input node before softmax execution.
- `attn_softmax_mask.bin`: raw F16 or F32 bytes captured from the attention
  softmax `src1` mask node, when the softmax has a mask. Graph-input masks may
  be copied at output dump time because they are not produced by a scheduler
  eval callback.
- `attn_softmax_sinks.bin`: raw F32 bytes captured from optional attention
  sink values on softmax `src2`, when the model uses attention sinks.
- `attn_softmax_output.bin`: raw F32 bytes captured from the attention softmax
  output node after execution.
- `metadata.json`: schema version, tensor ids, dtype, shape, byte strides,
  byte sizes, paths, attention layer, and softmax `scale` / `max_bias`
  parameters.

Only the first matching attention softmax is dumped. Prompt/prefill,
position-zero single-token prompt decode, later generation tokens, and final
logits softmax are not captured.

## Parse

```bash
python3 scripts/parse-first-decode-attn-softmax-dump.py \
  experiments/first-decode-attn-softmax-dump --limit 16
```

Add `--plain` to include all tensor values in the JSON output.

## Compare

Recompute softmax from the exported `input` and optional `mask`, then compare it
with the exported `output`:

```bash
python3 scripts/compare-first-decode-attn-softmax-dump.py \
  experiments/first-decode-attn-softmax-dump --limit 16
```

The comparison script follows `ggml_compute_forward_soft_max_f32`: it uses
metadata byte strides, applies softmax scale, mask broadcasting, ALiBi
`max_bias` slope calculation, and optional attention sinks in the softmax
denominator. It reports `max_abs_diff`, `mean_abs_diff`, and `rmse`.
