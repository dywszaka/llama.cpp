#include "ggml-debug.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

bool nvfp4_copy_debug_enabled() {
    static bool initialized = false;
    static bool enabled = false;
    if (!initialized) {
        enabled = getenv("LLAMA_NVFP4_COPY_DEBUG") != nullptr;
        initialized = true;
    }
    return enabled;
}

} // namespace

void ggml_debug_nvfp4_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (!nvfp4_copy_debug_enabled()) {
        return;
    }

    if (!dst || dst->type != GGML_TYPE_F32) {
        return;
    }

    const char * name = ggml_get_name(dst);
    if (!name || strstr(name, "token_embd.weight") == nullptr) {
        return;
    }

    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;

    const size_t max_bytes    = (size_t) ggml_nbytes(dst);
    const size_t sample_bytes = std::min<size_t>(max_bytes, 1 * 1024 * 1024);
    const size_t n            = sample_bytes / sizeof(float);

    std::vector<float> buf(n);
    ggml_backend_tensor_get(dst, buf.data(), 0, sample_bytes);

    int64_t nan = 0;
    int64_t inf = 0;
    int64_t finite = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        const float v = buf[i];
        if (std::isnan(v)) {
            nan++;
            continue;
        }
        if (std::isinf(v)) {
            inf++;
            continue;
        }
        finite++;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }

    const float min_out = finite ? min_v : 0.0f;
    const float max_out = finite ? max_v : 0.0f;
    const char * src_buf = src && src->buffer ? ggml_backend_buffer_name(src->buffer) : "none";
    const char * dst_buf = dst->buffer ? ggml_backend_buffer_name(dst->buffer) : "none";
    fprintf(stderr,
            "copy_debug: %s sample bytes=%zu min=%.6g max=%.6g nan=%" PRId64 " inf=%" PRId64 " src_buf=%s dst_buf=%s\n",
            name, sample_bytes, min_out, max_out, nan, inf, src_buf, dst_buf);
}
