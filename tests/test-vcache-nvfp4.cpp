#include "../src/llama-vcache-nvfp4.h"
#include "../src/llama-cparams.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static bool expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "%s\n", msg);
        return false;
    }
    return true;
}

static bool expect_close(float actual, float expected, float tol, const char * msg) {
    if (std::fabs(actual - expected) > tol) {
        std::fprintf(stderr, "%s: actual=%g expected=%g\n", msg, (double) actual, (double) expected);
        return false;
    }
    return true;
}

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

int main() {
    llama_cparams cparams = {};

    cparams.flash_attn = false;
    cparams.offload_kqv = true;
    cparams.kv_unified = false;

    if (!expect(llama_vcache_nvfp4_type_supported(GGML_TYPE_NVFP4), "NVFP4 type should be supported")) {
        return 1;
    }

    if (!expect(!llama_vcache_nvfp4_type_supported(GGML_TYPE_NVFP4_8), "NVFP4_8 V-cache should remain unsupported")) {
        return 1;
    }

    if (!expect(llama_vcache_nvfp4_token_padding(cparams, GGML_TYPE_F16) == 1u, "non-NVFP4 token padding should stay 1")) {
        return 1;
    }

    if (!expect(llama_vcache_nvfp4_token_padding(cparams, GGML_TYPE_NVFP4) == 1u, "kv_unified=0 should keep NVFP4 V-cache token padding disabled")) {
        return 1;
    }

    if (!expect(!llama_vcache_nvfp4_runtime_supported(cparams, GGML_TYPE_NVFP4), "kv_unified=0 must disable NVFP4 V-cache runtime path")) {
        return 1;
    }

    cparams.kv_unified = true;
    if (!expect(llama_vcache_nvfp4_token_padding(cparams, GGML_TYPE_NVFP4) == 16u, "NVFP4 V-cache token padding should use 16 slots")) {
        return 1;
    }

    if (!expect(llama_vcache_nvfp4_runtime_supported(cparams, GGML_TYPE_NVFP4), "type_v=NVFP4 should enable the V-cache runtime path when runtime settings are compatible")) {
        return 1;
    }

    cparams.flash_attn = true;
    if (!expect(!llama_vcache_nvfp4_runtime_supported(cparams, GGML_TYPE_NVFP4), "flash attention must disable NVFP4 V-cache runtime path")) {
        return 1;
    }

    cparams.flash_attn = false;
    cparams.offload_kqv = false;
    if (!expect(!llama_vcache_nvfp4_runtime_supported(cparams, GGML_TYPE_NVFP4), "offload_kqv=0 must disable NVFP4 V-cache runtime path")) {
        return 1;
    }

    set_env_var("LLAMA_NVFP4_VCACHE_LAYER_GLOBAL_SCALE", nullptr);
    set_env_var("LLAMA_NVFP4_VCACHE_PER_BLOCK_SCALE", nullptr);
    if (!expect(llama_vcache_nvfp4_layer_global_scale_path() == nullptr, "default V-cache scale path should not use a per-layer JSON file")) {
        return 1;
    }
    if (!expect(!llama_vcache_nvfp4_per_block_scale_enabled(), "default V-cache scale path should not use per-block external scales")) {
        return 1;
    }
    if (!expect_close(llama_vcache_nvfp4_default_v_global_absmax(), 80.428f, 1e-6f, "default V-cache absmax should use wiki P90")) {
        return 1;
    }
    if (!expect_close(llama_vcache_nvfp4_default_v_global_scale(), 1344.0f / 80.428f, 1e-5f, "default V-cache global scale should derive from wiki P90")) {
        return 1;
    }

    set_env_var("LLAMA_NVFP4_VCACHE_PER_BLOCK_SCALE", "1");
    if (!expect(llama_vcache_nvfp4_per_block_scale_enabled(), "per-block V-cache scale switch should enable the old scale path")) {
        return 1;
    }

    set_env_var("LLAMA_NVFP4_VCACHE_LAYER_GLOBAL_SCALE", "1");
    if (!expect(llama_vcache_nvfp4_layer_global_scale_path() != nullptr, "per-layer V-cache scale switch should select a JSON path")) {
        return 1;
    }

    std::puts("test-vcache-nvfp4: ok");
    return 0;
}
