# CUDA ADD QEMU I/O protocol

This document freezes the `GGML_OP_ADD` boundary used by the experimental
QEMU/RVV and qemu_cuda replacement.

## llama.cpp operator boundary

The owning CUDA entry point is `ggml_cuda_op_add()` in
`ggml/src/ggml-cuda/binbcast.cu`.

- `src0`: F32 or F16 tensor.
- `src1`: F32 or F16 tensor; `ggml_can_repeat(src1, src0)` must hold.
- `dst`: F32 or F16 tensor with the same shape as `src0`.
- Shape: four ggml dimensions `ne[0..3]` from `dst`.
- Strides: byte strides `nb[0..3]` from both inputs and the output.
- Broadcast: every logical destination coordinate `(i0,i1,i2,i3)` reads
  `src1[i0 % ne10, i1 % ne11, i2 % ne12, i3 % ne13]`.
- Output: the original CUDA path computes one native floating-point addition
  per logical destination element.
- There are no `op_params`, scalar parameters, optional inputs, or fused bias
  semantics beyond the two tensors.
- The experiment requires contiguous `dst`, matching the current CUDA backend
  allocation for this operator. Input strides remain fully supported.

## Canonical tensor layout

CUDA preprocessing resolves all source strides and broadcast coordinates before
offload. It produces two identically shaped dense arrays in ggml logical order,
with `ne0` varying fastest:

```text
linear = (((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0)
```

Each native F32/F16 value is converted to F32 and packed as canonical BF16 by
direct high-16-bit truncation:

```text
bf16_bits = f32_bits >> 16
```

This input conversion is RZ/truncation, including NaN, Inf, subnormal, and
signed-zero bit patterns. No RNE increment is applied at this boundary.

## FP32-compute numerical contract

The llama.cpp replacement uses `call_add_fp32`:

```text
left_f32  = BF16_TO_FP32_EXACT(left_bf16)
right_f32 = BF16_TO_FP32_EXACT(right_bf16)
sum_f32   = FP32_ADD_RNE(left_f32, right_f32)
dst_bf16  = FP32_TO_BF16_RNE(sum_f32)
```

The RVV implementation uses `e16m4 -> e32m8` widening so both groups contain
128 lanes at VLEN=512. NI900 BF16 mode remains enabled for the e16 boundary
instructions, and `frm` is set to RNE before the final `vfncvt.f.f.w`.

The qemu_cuda kernel consumes the exact same canonical BF16 inputs, performs
`__fadd_rn`, and applies the same BF16 RNE/NaN rule. QEMU and qemu_cuda outputs
are compared as raw `uint16_t` values and target zero mismatches.

## Runtime modes

`GGML_CUDA_ADD_QEMU_MODE` supports:

- `cuda`: original llama.cpp CUDA ADD; default.
- `qemu`: canonical preprocess, QEMU/RVV ADD, BF16-to-native CUDA output.
- `qemu_cuda`: device-only canonical preprocess, FP32 ADD model, and output.
- `compare`: all three paths run; the original CUDA result remains downstream.

The RPC request and response layouts mirror
`call_add_fp32/include/add_fp32_rpc_protocol.h`. Requests contain only dense
BF16 `src0`, dense BF16 `src1`, shape, byte counts, and request metadata.
