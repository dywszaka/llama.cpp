#include "nvfp4-k-offline-channel-order.h"

#include "llama-impl.h"

#include "../../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace llama_expt {
namespace {

using json = nlohmann::ordered_json;

constexpr const char * ENV_ORDER = "LLAMA_EXPT_NVFP4_K_OFFLINE_CHANNEL_ORDER";

std::string env_str(const char * name) {
    const char * value = std::getenv(name);
    return value ? value : "";
}

bool parse_kcur_layer_name(const std::string & name, size_t & layer) {
    constexpr const char * prefix = "Kcur-";
    constexpr size_t prefix_len = 5;
    if (name.rfind(prefix, 0) != 0 || name.size() == prefix_len) {
        return false;
    }

    size_t parsed = 0;
    for (size_t i = prefix_len; i < name.size(); ++i) {
        const char ch = name[i];
        if (ch < '0' || ch > '9') {
            return false;
        }
        parsed = parsed*10 + (size_t) (ch - '0');
    }

    layer = parsed;
    return true;
}

void validate_permutation(const std::vector<int32_t> & order, size_t expected_head_dim, const std::string & label) {
    if (order.size() != expected_head_dim) {
        throw std::runtime_error(label + " channel_order length " + std::to_string(order.size()) +
                " != expected head dim " + std::to_string(expected_head_dim));
    }

    std::vector<bool> seen(expected_head_dim, false);
    for (int32_t idx : order) {
        if (idx < 0 || (size_t) idx >= expected_head_dim) {
            throw std::runtime_error(label + " channel_order index " + std::to_string(idx) +
                    " is outside [0," + std::to_string(expected_head_dim) + ")");
        }
        if (seen[(size_t) idx]) {
            throw std::runtime_error(label + " channel_order repeats index " + std::to_string(idx));
        }
        seen[(size_t) idx] = true;
    }
}

} // namespace

const char * nvfp4_k_offline_channel_order_env() {
    return ENV_ORDER;
}

bool nvfp4_k_offline_channel_order_enabled() {
    return !env_str(ENV_ORDER).empty();
}

nvfp4_k_offline_channel_order nvfp4_k_offline_channel_order_load(
        const std::string & path,
        size_t expected_layers,
        size_t expected_head_dim) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }

    json root;
    in >> root;

    if (!root.contains("records") || !root["records"].is_array()) {
        throw std::runtime_error(path + " does not contain a records array");
    }

    nvfp4_k_offline_channel_order result;
    result.path = path;
    result.per_layer.resize(expected_layers);
    std::vector<bool> seen(expected_layers, false);

    for (const auto & rec : root["records"]) {
        const std::string name = rec.value("name", "");
        size_t il = 0;
        if (!parse_kcur_layer_name(name, il)) {
            continue;
        }
        if (il >= expected_layers) {
            throw std::runtime_error(path + " has unexpected layer " + std::to_string(il) +
                    " for expected layer count " + std::to_string(expected_layers));
        }
        if (seen[il]) {
            throw std::runtime_error(path + " has duplicate order for layer " + std::to_string(il));
        }
        if (!rec.contains("channel_order") || !rec["channel_order"].is_array()) {
            throw std::runtime_error(path + " record " + name + " does not contain a channel_order array");
        }

        auto & order = result.per_layer[il];
        order.reserve(rec["channel_order"].size());
        for (const auto & value : rec["channel_order"]) {
            order.push_back(value.get<int32_t>());
        }

        validate_permutation(order, expected_head_dim, path + " record " + name);
        seen[il] = true;
    }

    for (size_t il = 0; il < expected_layers; ++il) {
        if (!seen[il]) {
            throw std::runtime_error(path + " is missing Kcur-" + std::to_string(il));
        }
    }

    return result;
}

std::vector<int32_t> nvfp4_k_offline_channel_order_gqa_indices(
        const std::vector<int32_t> & order,
        int64_t head_dim,
        int64_t n_head_kv) {
    if (head_dim <= 0 || n_head_kv <= 0) {
        throw std::runtime_error("invalid K head dimensions for offline channel order");
    }
    if ((int64_t) order.size() != head_dim) {
        throw std::runtime_error("offline channel order length does not match K head dim");
    }

    std::vector<int32_t> indices;
    indices.reserve((size_t) (head_dim*n_head_kv));
    for (int64_t ih = 0; ih < n_head_kv; ++ih) {
        const int64_t head_offset = ih*head_dim;
        for (int32_t idx : order) {
            indices.push_back((int32_t) (head_offset + idx));
        }
    }

    return indices;
}

const nvfp4_k_offline_channel_order * nvfp4_k_offline_channel_order_load_from_env() {
    static std::once_flag once;
    static nvfp4_k_offline_channel_order order;
    static bool loaded = false;
    static std::string error;

    std::call_once(once, [] {
        const std::string path = env_str(ENV_ORDER);
        if (path.empty()) {
            return;
        }

        try {
            order = nvfp4_k_offline_channel_order_load(path, 36, 128);
            loaded = true;
        } catch (const std::exception & e) {
            error = e.what();
        }
    });

    if (!error.empty()) {
        throw std::runtime_error(error);
    }

    return loaded ? &order : nullptr;
}

const nvfp4_k_offline_channel_order * nvfp4_k_offline_channel_order_get() {
    return nvfp4_k_offline_channel_order_load_from_env();
}

void nvfp4_k_offline_channel_order_maybe_log_enabled() {
    static std::once_flag once;
    std::call_once(once, [] {
        const nvfp4_k_offline_channel_order * order = nvfp4_k_offline_channel_order_get();
        if (order) {
            LLAMA_LOG_INFO("%s: enabled %s='%s' layers=%zu head_dim=%zu; K cache and Q KQ operands use matching offline channel order\n",
                    __func__, ENV_ORDER, order->path.c_str(), order->per_layer.size(),
                    order->per_layer.empty() ? 0 : order->per_layer.front().size());
        }
    });
}

} // namespace llama_expt
