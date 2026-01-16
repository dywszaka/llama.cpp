#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_debug_nvfp4_copy(struct ggml_tensor * src, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
