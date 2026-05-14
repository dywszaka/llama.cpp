#include "../src/llama-vcache-nvfp4.h"
#include "../src/llama-cparams.h"

#include <cstdio>

static bool expect(bool cond, const char * msg) {
    if (!cond) {
        std::fprintf(stderr, "%s\n", msg);
        return false;
    }
    return true;
}

int main() {
    llama_cparams cparams = {};

    cparams.flash_attn = false;
    cparams.offload_kqv = true;

    if (!expect(llama_vcache_nvfp4_type_supported(GGML_TYPE_NVFP4), "NVFP4 type should be supported")) {
        return 1;
    }

    if (!expect(!llama_vcache_nvfp4_type_supported(GGML_TYPE_NVFP4_8), "NVFP4_8 V-cache should remain unsupported")) {
        return 1;
    }

    if (!expect(llama_vcache_nvfp4_token_padding(cparams, GGML_TYPE_F16) == 1u, "non-NVFP4 token padding should stay 1")) {
        return 1;
    }

    if (!expect(llama_vcache_nvfp4_token_padding(cparams, GGML_TYPE_NVFP4) == 1u, "runtime-disabled NVFP4 token padding should stay 1")) {
        return 1;
    }

    if (!expect(!llama_vcache_nvfp4_runtime_supported(cparams, GGML_TYPE_NVFP4), "runtime path should stay disabled when experiment switch is off")) {
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

    std::puts("test-vcache-nvfp4: ok");
    return 0;
}
