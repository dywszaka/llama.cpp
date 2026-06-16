#include "../src/llama-kv-cache-nvfp4-outlier-config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void require(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void set_env_var(const char * name, const char * value) {
    if (value == nullptr) {
#if defined(_WIN32)
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    } else {
#if defined(_WIN32)
        _putenv_s(name, value);
#else
        setenv(name, value, 1);
#endif
    }
}

int main() {
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE", nullptr);
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD", nullptr);
    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_threshold_profile(false), "balanced") == 0,
            "default full-NVFP4 profile should be balanced");
    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_capacity_profile_for_mode(512, false), "balanced") == 0,
            "default full-NVFP4 capacity profile should be balanced");

    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE", "new");
    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_threshold_profile(false), "new") == 0,
            "new full-NVFP4 threshold profile should be selectable");
    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_capacity_profile_for_mode(512, false), "new") == 0,
            "new full-NVFP4 capacity profile should be selectable");

    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE", "bf16");
    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_threshold_profile(false), "bf16") == 0,
            "bf16 full-NVFP4 threshold profile should be selectable");
    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_capacity_profile_for_mode(512, false), "bf16") == 0,
            "bf16 full-NVFP4 capacity profile should be selectable");
    require(llama_nvfp4_kcache_outlier_layer_capacity_count_for_mode(512, false) == 36,
            "bf16 full-NVFP4 capacity profile should have one entry per layer");

    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_threshold_profile(true), "balanced") == 0,
            "hybrid FP8 mode should keep the balanced threshold profile");
    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_capacity_profile_for_mode(512, true), "ctx512") == 0,
            "hybrid FP8 mode should keep the context-specific capacity profile");

    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD", "20");
    require(std::strcmp(llama_nvfp4_kcache_outlier_layer_threshold_profile(false), "env-override") == 0,
            "threshold override should still take precedence over selected profile");

    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD", nullptr);
    set_env_var("LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE", nullptr);

    std::puts("test-nvfp4-kcache-outlier-profile: ok");
    return 0;
}
