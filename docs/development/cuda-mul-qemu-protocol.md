# CUDA MUL QEMU protocol

`GGML_OP_MUL` consumes `src0`, `src1` and produces `dst`. The supported native
types are F32/F16 operands and F32/F16 output. `src0` and `dst` have the same
logical `ne[4]`; `src1` may repeat along every dimension. Native byte strides
for both sources are preserved in the narrow hook; the current experimental
path requires a contiguous destination. The CUDA preprocess maps output
coordinate `i[d]` to `src1` coordinate `i[d] % src1.ne[d]`, loads both native
values, and packs two dense row-major BF16 tensors with RZ/high-16-bit
truncation: `bf16_bits = f32_bits >> 16`.

The canonical arithmetic contract is:

```text
canonical_dst[i] = BF16_RNE(
    BF16_TO_F32(canonical_src0[i]) *
    BF16_TO_F32(canonical_src1[i]))
```

The NI900 BF16 arithmetic preserves BF16 subnormal inputs. qemu_cuda mirrors
that behavior and canonicalizes NaN multiplication results to BF16 `0x7fc0`.

RPC request frames are header, dense `src0`, dense `src1`. Response frames are
header and dense output. Headers have fixed packed layouts, magic/version/header
size/request id, four output dimensions, all dtype fields, and exact byte
counts. All tensors are BF16. The daemon validates frame count, dimensions,
overflow, byte counts, mailbox readiness and globalram ranges.

The mailbox allocates 64-byte-aligned regions for both inputs and output and
uses `BOOTING/IDLE/REQUEST/RUNNING/DONE/ERROR`. The firmware is resident and
publishes `[ READY ]` once. `qemu_cuda` consumes the same device BF16 buffers;
pure `qemu_cuda` performs no ZMQ, D2H or H2D operation.

In `compare`, the original CUDA output is copied back to the real destination.
The QEMU output is compared with it using MSE/RMSE/max absolute error, while
QEMU and qemu_cuda outputs are compared as raw `uint16_t` bits.
