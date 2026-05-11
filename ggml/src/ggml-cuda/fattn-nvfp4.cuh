#pragma once

#include "common.cuh"

bool ggml_cuda_flash_attn_ext_nvfp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_nvfp4_fattn_no_fallback();
