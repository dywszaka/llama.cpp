#pragma once

#include "ggml.h"

struct llama_cparams;
struct ggml_tensor;

bool llama_vcache_nvfp4_experiment_enabled();
void llama_vcache_nvfp4_log_once();

bool llama_vcache_nvfp4_runtime_supported(const llama_cparams & cparams, ggml_type type_v);

bool llama_vcache_nvfp4_type_supported(ggml_type type_v);
bool llama_vcache_nvfp4_should_transpose_store(const llama_cparams & cparams, ggml_type type_v);
bool llama_vcache_nvfp4_uses_padded_tokens(const llama_cparams & cparams, ggml_type type_v);
uint32_t llama_vcache_nvfp4_token_padding(const llama_cparams & cparams, ggml_type type_v);
