# NVFP4 CUDA Fast Path

This document records the design plan for adding a native NVFP4 matmul path to the CUDA backend, together with the minimal functional evaluation flow.

The goal is to make the implementation and the evaluation criteria explicit before wiring the fast path into the CUDA execution chain.

## Goals

- Use `cublasLtMatmul` as the native NVFP4 GEMM path on supported NVIDIA hardware.
- Keep the current CUDA kernels as a fallback path.
- Keep the model format unchanged and reuse the existing NVFP4 GGUF payload.
- Validate functionality with a single request only:
  - server starts
  - request succeeds
  - response text is readable
  - response is not garbled

## Design Plan

### 1. Capability Gating

- Detect whether the runtime environment supports NVFP4 execution.
- Enable the fast path only when all of the following are true:
  - the GPU is Blackwell-class or newer (`sm_100+`)
  - the CUDA / cuBLASLt stack exposes the required FP4 and block-scale APIs
  - the tensor shape and layout satisfy the NVFP4 constraints
- If any prerequisite is missing, fall back to the existing CUDA implementation.

### 2. Data Layout

- Preserve the packed FP4 payload and its block-scale metadata from model loading to device memory.
- Avoid re-packing weights on every token or layer call.
- Keep the existing GGUF model format intact; do not introduce a new on-disk format for this work.

### 3. Matmul Dispatch

- Add a CUDA dispatch branch for eligible linear layers:
  - attention projections
  - FFN up / down / gate projections
  - any other GEMM that already uses the CUDA backend
- Configure `cublasLtMatmul` with the FP4 tensor type and the scale mode required by NVFP4.
- Cache matmul descriptors, heuristics, and workspace selection by shape so the fast path does not rebuild descriptors repeatedly.

### 4. Activation Quantization Bridge

- Add a CUDA-side quantization bridge for activations before NVFP4 GEMM.
- Reuse scratch buffers for the packed activation payload and its scale data.
- Keep the bridge narrow and deterministic:
  - quantize only when the next GEMM can consume NVFP4
  - otherwise fall back to the existing FP16 / BF16 path

### 5. Output Policy

- Default the GEMM output to BF16 / FP16 for numerical safety.
- Only keep FP4 output when the following op can consume it and the graph has been audited for that path.

### 6. Fallback and Logging

- Fallback on:
  - unsupported hardware
  - unsupported tensor shapes
  - incompatible strides or alignment
  - missing CUDA / cuBLASLt features
- Add minimal debug logging so it is obvious whether a layer used the NVFP4 path or fell back.

## Functional Evaluation Plan

### 1. Model and Server

- Use the following model:

```text
~/host_workspace/develop/models/qwen3-8b-nvfp4-official.gguf
```

- Start the server with `llama-server`.
- Keep the server bound to the local host and the default evaluation port expected by `request.sh`.

Example:

```bash
llama-server -m ~/host_workspace/develop/models/qwen3-8b-nvfp4-official.gguf
```

### 2. Request Script

- Use the existing `request.sh` as the evaluation driver.
- Send a single request only.
- The current script already targets `http://localhost:8080/v1/chat/completions` and uses a short Chinese prompt (`你好`), which is sufficient for this functional check.

Example:

```bash
./request.sh
```

### 3. Pass Criteria

- The server starts successfully.
- The request returns successfully.
- The response body is structurally valid.
- The returned text is readable and not garbled.
- There is no obvious truncation, empty response, or server crash.

### 4. Non-Goals

- No throughput benchmark.
- No multi-request stability test.
- No long-context stress test.
- No quality comparison against a baseline model.

## Notes

- This document intentionally keeps the evaluation surface small so it can be used as a quick functional gate during bring-up.
- If the evaluation script or the server port changes later, keep the same semantics: one request, one response, readable output, no garbling.
