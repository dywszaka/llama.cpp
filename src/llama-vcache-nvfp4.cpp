#include "llama-vcache-nvfp4.h"

#include "llama-cparams.h"
#include "llama-impl.h"

#include <atomic>
#include <cstdlib>

namespace {

constexpr const char * LLAMA_EXPERIMENT_NVFP4_VCACHE_ENV = "LLAMA_EXPERIMENT_NVFP4_VCACHE";
constexpr const char * LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE_ENV = "LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE";
constexpr const char * LLAMA_EXPERIMENT_NVFP4_VCACHE_PER_BLOCK_SCALE_ENV = "LLAMA_EXPERIMENT_NVFP4_VCACHE_PER_BLOCK_SCALE";
constexpr const char * LLAMA_NVFP4_VCACHE_LAYER_GLOBAL_SCALE_DEFAULT_PATH = "experiments/qwen3-8b-v-layer-absmax.json";
constexpr float LLAMA_NVFP4_VCACHE_FP4_MAX = 6.0f;
constexpr float LLAMA_NVFP4_VCACHE_E4M3_HALF_MAX = 224.0f;
constexpr float LLAMA_NVFP4_VCACHE_GLOBAL_SCALE_MAX = LLAMA_NVFP4_VCACHE_FP4_MAX * LLAMA_NVFP4_VCACHE_E4M3_HALF_MAX;
constexpr float LLAMA_NVFP4_VCACHE_DEFAULT_V_GLOBAL_ABSMAX = 80.428f;

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

const char * llama_vcache_nvfp4_layer_global_scale_path() {
    const char * env = getenv(LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE_ENV);
    if (env == nullptr || env[0] == '\0' || env[0] == '0') {
        return nullptr;
    }
    if (env[0] == '1' && env[1] == '\0') {
        return LLAMA_NVFP4_VCACHE_LAYER_GLOBAL_SCALE_DEFAULT_PATH;
    }
    return env;
}

bool llama_vcache_nvfp4_per_block_scale_enabled() {
    return env_flag_enabled(LLAMA_EXPERIMENT_NVFP4_VCACHE_PER_BLOCK_SCALE_ENV);
}

float llama_vcache_nvfp4_default_v_global_absmax() {
    return LLAMA_NVFP4_VCACHE_DEFAULT_V_GLOBAL_ABSMAX;
}

float llama_vcache_nvfp4_default_v_global_scale() {
    return LLAMA_NVFP4_VCACHE_GLOBAL_SCALE_MAX / LLAMA_NVFP4_VCACHE_DEFAULT_V_GLOBAL_ABSMAX;
}

void llama_vcache_nvfp4_log_scale_mode_once(bool nvfp4_vcache_active) {
    static std::atomic<bool> logged(false);
    if (logged.exchange(true)) {
        return;
    }

    const char * layer_path = llama_vcache_nvfp4_layer_global_scale_path();
    const bool per_layer = nvfp4_vcache_active && layer_path != nullptr;
    const bool per_block = nvfp4_vcache_active && !per_layer && llama_vcache_nvfp4_per_block_scale_enabled();

    LLAMA_LOG_INFO(
            "%s: %s=%s, %s=%s -> %s\n",
            __func__,
            LLAMA_EXPERIMENT_NVFP4_VCACHE_LAYER_GLOBAL_SCALE_ENV,
            layer_path != nullptr ? layer_path : "(unset)",
            LLAMA_EXPERIMENT_NVFP4_VCACHE_PER_BLOCK_SCALE_ENV,
            llama_vcache_nvfp4_per_block_scale_enabled() ? "1" : "0",
            per_layer ? "enabled, NVFP4 V-cache uses experimental per-layer JSON global scales"
                      : per_block ? "enabled, NVFP4 V-cache uses experimental per-block external scales"
                                  : nvfp4_vcache_active ? "disabled, NVFP4 V-cache uses default per-tensor global scale"
                                                        : "inactive, NVFP4 V-cache scale mode not used");
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
