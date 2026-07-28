#include "rope-qemu.cuh"

#include <atomic>
#include <cstdlib>
#include <cstring>

static const char * GGML_CUDA_ROPE_QEMU_ENABLED_ENV = "GGML_CUDA_ROPE_QEMU_ENABLED";

static bool parse_enabled(const char * value) {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0 ||
            std::strcmp(value, "false") == 0 || std::strcmp(value, "off") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
            std::strcmp(value, "on") == 0) {
        return true;
    }
    GGML_LOG_WARN("%s: unknown %s=%s; using disabled\n",
            __func__, GGML_CUDA_ROPE_QEMU_ENABLED_ENV, value);
    return false;
}

bool ggml_cuda_rope_qemu_enabled() {
    static const bool enabled = parse_enabled(std::getenv(GGML_CUDA_ROPE_QEMU_ENABLED_ENV));
    static std::atomic<bool> logged(false);
    if (!logged.exchange(true)) {
        GGML_LOG_INFO("%s: %s=%s; RoPE QEMU dispatch hook %s\n",
                __func__, GGML_CUDA_ROPE_QEMU_ENABLED_ENV,
                enabled ? "1" : "0", enabled ? "enabled" : "disabled");
    }
    return enabled;
}

bool ggml_cuda_rope_qemu_try_run(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    static std::atomic<bool> logged(false);
    if (!logged.exchange(true)) {
        GGML_LOG_INFO(
                "%s: RoPE QEMU hook reached for tensor '%s'; using the CUDA fallback until "
                "the QEMU operator implementation is connected\n",
                __func__, ggml_get_name(dst));
    }
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}
