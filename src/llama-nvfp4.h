#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_nvfp4_act_roundtrip_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        int ith,
        int nth,
        void * userdata);

#ifdef __cplusplus
}
#endif
