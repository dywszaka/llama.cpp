#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llama_expt {

struct nvfp4_k_offline_channel_order {
    std::string path;
    std::vector<std::vector<int32_t>> per_layer;
};

const char * nvfp4_k_offline_channel_order_env();
bool nvfp4_k_offline_channel_order_enabled();
const nvfp4_k_offline_channel_order * nvfp4_k_offline_channel_order_get();
const nvfp4_k_offline_channel_order * nvfp4_k_offline_channel_order_load_from_env();
nvfp4_k_offline_channel_order nvfp4_k_offline_channel_order_load(
        const std::string & path,
        size_t expected_layers,
        size_t expected_head_dim);
std::vector<int32_t> nvfp4_k_offline_channel_order_gqa_indices(
        const std::vector<int32_t> & order,
        int64_t head_dim,
        int64_t n_head_kv);
void nvfp4_k_offline_channel_order_maybe_log_enabled();

} // namespace llama_expt
