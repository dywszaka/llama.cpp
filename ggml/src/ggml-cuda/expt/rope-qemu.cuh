#pragma once

#include "../common.cuh"

bool ggml_cuda_rope_qemu_enabled();

// Returns true once a QEMU implementation handled the operation. The current
// interface-only implementation deliberately returns false so the caller uses
// the existing CUDA kernel as a correctness-preserving fallback.
bool ggml_cuda_rope_qemu_try_run(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);
