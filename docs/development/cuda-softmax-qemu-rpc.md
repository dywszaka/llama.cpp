# CUDA softmax QEMU RPC and globalram mailbox

This document defines the transport used by
`ggml/src/ggml-cuda/expt/softmax-qemu.cu` and
`call_softmax/daemon/softmax_qemu_daemon.cpp`. The CUDA tensor and algorithm
semantics remain defined by `cuda-softmax-io-protocol.md`.

## ZMQ RPC

The llama.cpp client uses a ZMQ `REQ` socket. The daemon binds a `REP` socket at
`tcp://127.0.0.1:15580` by default. Integer and float fields are little-endian;
both processes run in the same x86-64 container.

Each request has four frames:

1. packed `softmax_rpc_request_v1` metadata;
2. raw effective-logit bytes copied from CUDA;
3. raw mask tensor bytes, or an empty frame;
4. raw sink tensor bytes, or an empty frame.

Each response has two frames:

1. packed `softmax_rpc_response_v1` with status, error code, byte count, and
   QEMU elapsed time;
2. dense output bytes, or an empty frame on error.

The llama.cpp integration uses `SOFTMAX_REQUEST_BF16_IO`: frame 2 and the
response are BF16 bit patterns, frame 3 is empty because scale/mask/ALiBi have
already been folded into the effective logits on CUDA, and optional frame 4
contains BF16 attention sinks. The legacy request form without this flag keeps
the original F32 input/mask/sink/output layout.

The protocol magic is `0x31584d53` and the current version is 1. The canonical
layout definitions are duplicated with compile-time size checks in:

- `ggml/src/ggml-cuda/expt/softmax-qemu-protocol.h`;
- `call_softmax/include/softmax_rpc_protocol.h`.

## Globalram mailbox

The RVV firmware accesses the HIF window at CPU address `0x100000000`. The
mailbox begins at globalram offset 0 and request data begins at offset `0x1000`.
The daemon writes input, optional mask, optional sinks, and output regions on
64-byte boundaries, then publishes `SOFTMAX_MAILBOX_REQUEST`. The resident
firmware transitions through `RUNNING` and finally `DONE` or `ERROR`. For BF16
requests, the firmware uses the deterministic integer core shared with the
qemu_cuda experiment and includes optional BF16 sinks in the row maximum and
normalization denominator.

QEMU exposes four interleaved 64-byte HIF banks. A request received on bank `b`
with bank-local line address `L` maps to the logical byte offset:

```text
logical_offset = (L * 4 + b) * 64
```

The daemon implements the four globalram REP endpoints itself, so llama.cpp,
QEMU, the globalram backing store, and the RVV mailbox all run in one Docker
container without a separate model-bringup memory process.

## Synchronization and failure behavior

Only one softmax request occupies the mailbox at a time. The daemon serializes
RPC requests, validates frame sizes and tensor dimensions, and times out if the
firmware does not finish. llama.cpp treats transport, daemon, firmware, and
response-size errors as fatal in QEMU modes rather than silently substituting a
CUDA or zero-filled result.

## Per-call timing log

Set `GGML_CUDA_SOFT_MAX_QEMU_TIMING=1` to emit one llama.cpp log line after each
RVV request completes. The line starts with `RVV_SOFTMAX_TIMING` and contains:

- CUDA device-to-host staging time;
- ZMQ request/response round-trip time;
- daemon-side request time reported in the RPC response;
- optional host-to-device return-copy time;
- total external softmax time.

The workspace launcher enables this switch by default and captures llama.cpp
stdout/stderr in `${SOFTMAX_LOG_DIR}/llama.log`.
