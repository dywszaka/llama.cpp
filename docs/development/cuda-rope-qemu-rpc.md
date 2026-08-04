# CUDA ROPE QEMU experiment protocol

## Supported llama.cpp node

The experiment only intercepts forward `GGML_OP_ROPE` nodes that exactly match
the static lookup table:

- GPT-NeoX mode `2`;
- `n_dims=128`, `n_ctx_orig=40960`;
- `freq_base=1000000`, `freq_scale=1`, `ext_factor=0`, `attn_factor=1`;
- `beta_fast=32`, `beta_slow=1`;
- no `freq_factors` tensor and all M-RoPE sections zero;
- F32 or F16 source and matching destination type;
- dense destination and the source layout expected by the upstream CUDA RoPE
  kernel.

All other RoPE nodes continue through the original CUDA dispatch even when the
experiment mode is enabled. The lookup table covers positions `[0, 8192)` and
has shape `[8192, 64, 2]` with component order `[cos, sin]`.

## Canonical numerical model

The source tensor is packed on the caller CUDA stream into dense BF16 using raw
F32-high-word truncation. F16 sources are first converted to F32, then truncated
to BF16. Positions remain I32.

For each GPT-NeoX pair and table values `c` and `s`:

```text
x0 = bf16_to_f32(input[channel])
x1 = bf16_to_f32(input[channel + 64])
p1 = fp32_mul(x1, s)
y0 = fp32_fma(x0, c, -p1)
p0 = fp32_mul(x0, s)
y1 = fp32_fma(x1, c, p0)
```

`y0` and `y1` are narrowed to BF16 with RNE. This matches the RVV
`vfmul` + `vfmsac` / `vfmacc` sequence. No runtime sine/cosine function or SFU
is used by the RVV or qemu_cuda operator kernels.

## ZMQ RPC

The client uses REQ and the daemon uses REP. Requests contain three frames:

1. packed 152-byte `rope_fp32_rpc_request_v1`;
2. dense canonical BF16 source bytes;
3. dense I32 positions, with `ne2*ne3` elements.

Responses contain a packed 40-byte header followed by BF16 output bytes. The
header includes magic `0x31505252`, version, request id, status, error code,
output byte count, and daemon elapsed time. Both sides use `static_assert` to
freeze the packed layouts.

The daemon validates frame count and sizes, dtype, shape multiplication,
complete static-table parameters, and globalram bounds. The resident firmware
also validates the mailbox and every position before table access.

## Globalram layout

```text
0x000000: 208-byte mailbox
0x001000: 4 MiB immutable F32 cos/sin table
0x401000: per-request, 64-byte-aligned source / positions / destination
```

The daemon loads and size-checks the complete table before binding the QEMU
globalram service. The table is therefore not retransmitted in every RPC.

## Runtime modes

- `cuda`: original llama.cpp CUDA RoPE.
- `qemu`: canonical BF16 packing, D2H/RPC/RVV, H2D and native output conversion;
  the QEMU result is used downstream.
- `qemu_cuda`: canonical packing, static-table FP32 rotation and output
  conversion on the CUDA device. The immutable table is loaded to each CUDA
  device once; subsequent operator calls are device-only and do not use ZMQ.
- `compare`: runs all three models, records error and BF16-bit metrics, and
  copies the original CUDA result to the real destination.

All non-`cuda` modes disable CUDA graph capture for graphs containing RoPE.
