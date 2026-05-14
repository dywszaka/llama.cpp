#include "llama-vcache-nvfp4.h"

#include "llama-cparams.h"
#include "llama-impl.h"

#include <atomic>
#include <cstdlib>

namespace {

constexpr const char * LLAMA_EXPERIMENT_NVFP4_VCACHE_ENV = "LLAMA_EXPERIMENT_NVFP4_VCACHE";

bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

}

bool llama_vcache_nvfp4_experiment_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = env_flag_enabled(LLAMA_EXPERIMENT_NVFP4_VCACHE_ENV) ? 1 : 0;
    }
    return cached != 0;
}

void llama_vcache_nvfp4_log_once() {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    LLAMA_LOG_INFO(
            "%s: %s=%s -> %s\n",
            __func__,
            LLAMA_EXPERIMENT_NVFP4_VCACHE_ENV,
            llama_vcache_nvfp4_experiment_enabled() ? "1" : "0",
            llama_vcache_nvfp4_experiment_enabled()
                ? "enabled (experimental NVFP4 V-cache path)"
                : "disabled");
}

bool llama_vcache_nvfp4_type_supported(ggml_type type_v) {
    return type_v == GGML_TYPE_NVFP4;
}

bool llama_vcache_nvfp4_runtime_supported(const llama_cparams & cparams, ggml_type type_v) {
    if (!llama_vcache_nvfp4_type_supported(type_v)) {
        return false;
    }

    if (!llama_vcache_nvfp4_experiment_enabled()) {
        return false;
    }

    if (cparams.flash_attn) {
        return false;
    }

    if (!cparams.offload_kqv) {
        return false;
    }

    if (!cparams.kv_unified) {
        return false;
    }

    return true;
}

bool llama_vcache_nvfp4_should_transpose_store(const llama_cparams & cparams, ggml_type type_v) {
    return llama_vcache_nvfp4_runtime_supported(cparams, type_v);
}

bool llama_vcache_nvfp4_uses_padded_tokens(const llama_cparams & cparams, ggml_type type_v) {
    return llama_vcache_nvfp4_runtime_supported(cparams, type_v);
}

uint32_t llama_vcache_nvfp4_token_padding(const llama_cparams & cparams, ggml_type type_v) {
    if (!llama_vcache_nvfp4_uses_padded_tokens(cparams, type_v)) {
        return 1u;
    }

    return 16u;
}
