#include "llama-kv-cache-unified.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-vcache-nvfp4.h"
#include "llama-kv-cache-nvfp4-outlier-config.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

//
// llama_kv_cache_unified
//

static constexpr float LLAMA_NVFP4_VCACHE_FP4_MAX = 6.0f;
static constexpr float LLAMA_NVFP4_VCACHE_E4M3_HALF_MAX = 224.0f;
static constexpr float LLAMA_NVFP4_VCACHE_GLOBAL_SCALE_MAX = LLAMA_NVFP4_VCACHE_FP4_MAX * LLAMA_NVFP4_VCACHE_E4M3_HALF_MAX;

static std::set<uint32_t> llama_kcache_hybrid_fp8_high_medium_layers_local() {
    std::set<uint32_t> result;
    for (size_t i = 0; i < llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layer_count; ++i) {
        result.insert(llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layers[i]);
    }
    return result;
}

static std::set<uint32_t> llama_kcache_hybrid_fp8_layers_local() {
    const char * value = llama_nvfp4_kcache_hybrid_fp8_layers_env();
    if (value == nullptr || value[0] == '\0') {
        return {};
    }

    if (std::strcmp(value, "high_medium") == 0) {
        return llama_kcache_hybrid_fp8_high_medium_layers_local();
    }

    std::set<uint32_t> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(item.c_str(), &end, 10);
        if (end == item.c_str() || parsed > std::numeric_limits<uint32_t>::max()) {
            return {};
        }
        result.insert((uint32_t) parsed);
    }

    return result;
}

static std::string llama_kcache_hybrid_fp8_layers_to_string_local(const std::set<uint32_t> & layers) {
    std::ostringstream ss;
    bool first = true;
    for (const uint32_t layer : layers) {
        if (!first) {
            ss << ",";
        }
        ss << layer;
        first = false;
    }
    return ss.str();
}

static bool llama_nvfp4_parse_json_number_after_key(
        const std::string & text,
        size_t & pos,
        const char * key,
        double & value) {
    const std::string needle = std::string("\"") + key + "\"";
    pos = text.find(needle, pos);
    if (pos == std::string::npos) {
        return false;
    }

    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;

    while (pos < text.size() && std::isspace((unsigned char) text[pos])) {
        ++pos;
    }

    const char * begin = text.c_str() + pos;
    char * end = nullptr;
    value = std::strtod(begin, &end);
    if (end == begin) {
        return false;
    }

    pos = (size_t) (end - text.c_str());
    return true;
}

static std::vector<float> llama_nvfp4_vcache_load_layer_absmax(
        const char * path,
        uint32_t n_layer) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(std::string("failed to open NVFP4 V-cache layer absmax file: ") + path);
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    const std::string text = ss.str();

    std::vector<float> result(n_layer, 0.0f);
    std::vector<bool> seen(n_layer, false);
    size_t pos = 0;
    while (true) {
        double layer_f = 0.0;
        if (!llama_nvfp4_parse_json_number_after_key(text, pos, "layer", layer_f)) {
            break;
        }

        double absmax_f = 0.0;
        if (!llama_nvfp4_parse_json_number_after_key(text, pos, "absmax", absmax_f)) {
            break;
        }

        const int layer = (int) layer_f;
        if (layer >= 0 && (uint32_t) layer < n_layer && absmax_f > 0.0 && std::isfinite(absmax_f)) {
            result[(size_t) layer] = (float) absmax_f;
            seen[(size_t) layer] = true;
        }
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!seen[il]) {
            throw std::runtime_error("missing NVFP4 V-cache layer absmax entry for layer " + std::to_string(il));
        }
    }

    return result;
}

llama_kv_cache_unified::llama_kv_cache_unified(
        const llama_model &  model,
          layer_filter_cb && filter,
                ggml_type    type_k,
                ggml_type    type_v,
                     bool    v_trans,
                     bool    offload,
                     bool    unified,
                 uint32_t    kv_size,
                 uint32_t    n_seq_max,
                 uint32_t    n_pad,
                 uint32_t    n_swa,
           llama_swa_type    swa_type) :
    model(model), hparams(model.hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    GGML_ASSERT(kv_size % n_pad == 0);

    type_v_cache = type_v;
    has_k_scale = type_k == GGML_TYPE_NVFP4 || type_k == GGML_TYPE_NVFP4_8;
    if (type_k == GGML_TYPE_NVFP4) {
        kcache_hybrid_fp8_layers = llama_kcache_hybrid_fp8_layers_local();
    }
    non_flash_fp8_e8m0 = (type_v == GGML_TYPE_FP8_E4M3_E8M0_32 ||
                          type_v == GGML_TYPE_FP8_E4M3_E8M0_16) && !v_trans;
    nvfp4_vcache = llama_vcache_nvfp4_type_supported(type_v);
    const char * nvfp4_layer_scale_path = llama_vcache_nvfp4_layer_global_scale_path();
    nvfp4_vcache_layer_global_scale = nvfp4_vcache && v_trans && nvfp4_layer_scale_path != nullptr;
    nvfp4_vcache_per_block_scale = nvfp4_vcache && v_trans && !nvfp4_vcache_layer_global_scale && llama_vcache_nvfp4_per_block_scale_enabled();
    llama_vcache_nvfp4_log_scale_mode_once(nvfp4_vcache && v_trans);
    nvfp4_kcache_outlier = type_k == GGML_TYPE_NVFP4 && llama_nvfp4_kcache_outlier_enabled();
    const bool kcache_hybrid_fp8 = !kcache_hybrid_fp8_layers.empty();
    const uint32_t * nvfp4_kcache_outlier_layer_capacities =
            llama_nvfp4_kcache_outlier_layer_capacities_for_mode(kv_size, kcache_hybrid_fp8);
    const size_t nvfp4_kcache_outlier_layer_capacity_count =
            llama_nvfp4_kcache_outlier_layer_capacity_count_for_mode(kv_size, kcache_hybrid_fp8);
    const char * nvfp4_kcache_outlier_layer_capacity_profile =
            llama_nvfp4_kcache_outlier_layer_capacity_profile_for_mode(kv_size, kcache_hybrid_fp8);
    const char * nvfp4_kcache_outlier_layer_threshold_profile =
            llama_nvfp4_kcache_outlier_layer_threshold_profile(kcache_hybrid_fp8);
    if (nvfp4_kcache_outlier) {
        static std::atomic<bool> logged(false);
        if (!logged.exchange(true)) {
            LLAMA_LOG_INFO(
                    "%s: NVFP4 K-cache compact outlier sidecar enabled: threshold_profile=%s layer_capacity_profile=%s layer_capacities=%zu\n",
                    __func__,
                    nvfp4_kcache_outlier_layer_threshold_profile,
                    nvfp4_kcache_outlier_layer_capacity_profile,
                    nvfp4_kcache_outlier_layer_capacity_count);
        }
    }
    if (!kcache_hybrid_fp8_layers.empty()) {
        LLAMA_LOG_INFO("%s: hybrid FP8(E4M3+E8M0 block32) K-cache layers enabled: %s\n",
                __func__, llama_kcache_hybrid_fp8_layers_to_string_local(kcache_hybrid_fp8_layers).c_str());
    }

    // TODO: this is temporary until we support passing reuse layer filters [KV_REUSE]
    auto n_layer_cache = hparams.n_layer;
    if (model.arch == LLM_ARCH_GEMMA3N) {
        n_layer_cache = 20;
    }
    if (model.arch == LLM_ARCH_GLM4_MOE) {
        // GLM-4.5: Only process up to last layer, skip final NextN layer
        n_layer_cache = hparams.n_layer - hparams.nextn_predict_layers;
    }

    std::vector<float> nvfp4_layer_absmax;
    if (nvfp4_vcache_layer_global_scale) {
        nvfp4_layer_absmax = llama_nvfp4_vcache_load_layer_absmax(nvfp4_layer_scale_path, n_layer_cache);
        nvfp4_vcache_layer_global_scales.resize(n_layer_cache, 0.0f);
        for (uint32_t il = 0; il < n_layer_cache; ++il) {
            nvfp4_vcache_layer_global_scales[il] = LLAMA_NVFP4_VCACHE_GLOBAL_SCALE_MAX / nvfp4_layer_absmax[(size_t) il];
        }
    }

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            const uint32_t outlier_tensors_per_layer = nvfp4_kcache_outlier
                    ? 5u*(1 + n_stream)
                    : 0u;
            const uint32_t recent_f16_tensors_per_layer = nvfp4_kcache_outlier && llama_nvfp4_kcache_recent_f16_enabled()
                    ? 3u*(1 + n_stream)
                    : 0u;
            const uint32_t tensors_per_layer = 4u*(1 + n_stream) + outlier_tensors_per_layer + recent_f16_tensors_per_layer;
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(tensors_per_layer*n_layer_cache*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    size_t nvfp4_kcache_outlier_sidecar_bytes = 0;
    size_t nvfp4_kcache_recent_f16_sidecar_bytes = 0;

    for (uint32_t il = 0; il < n_layer_cache; il++) {
        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, il);
            continue;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();
        const ggml_type layer_type_k = kcache_hybrid_fp8_layers.count(il) != 0
                ? GGML_TYPE_FP8_E4M3_E8M0_32
                : type_k;
        const bool layer_has_k_scale = layer_type_k == GGML_TYPE_NVFP4 || layer_type_k == GGML_TYPE_NVFP4_8;
        const bool layer_nvfp4_kcache_outlier = nvfp4_kcache_outlier && layer_type_k == GGML_TYPE_NVFP4;
        const bool layer_nvfp4_kcache_recent_f16 =
                layer_nvfp4_kcache_outlier && llama_nvfp4_kcache_recent_f16_enabled();
        const uint32_t kcache_recent_f16_window = layer_nvfp4_kcache_recent_f16
                ? llama_nvfp4_kcache_recent_f16_window()
                : 0u;
        const uint32_t kv_size_v = use_nvfp4_vcache_layout() ? GGML_PAD(kv_size, 16u) : kv_size;
        const uint32_t n_v_scale = use_nvfp4_vcache_layout()
                ? (use_nvfp4_vcache_single_global_scale() ? 1u : n_embd_v_gqa * (kv_size_v / 16u))
                : 0u;

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        ggml_tensor * k;
        ggml_tensor * v;
        ggml_tensor * k_scale = nullptr;
        ggml_tensor * v_scale = nullptr;
        ggml_tensor * k_outlier_count = nullptr;
        ggml_tensor * k_outlier_offset = nullptr;
        ggml_tensor * k_outlier_cursor = nullptr;
        ggml_tensor * k_outlier_index = nullptr;
        ggml_tensor * k_outlier_value = nullptr;
        ggml_tensor * k_recent_f16 = nullptr;
        ggml_tensor * k_recent_f16_active = nullptr;
        ggml_tensor * k_recent_f16_pos = nullptr;

        k = ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa, kv_size, n_stream);
        if (use_nvfp4_vcache_layout()) {
            v = ggml_new_tensor_3d(ctx, type_v, kv_size_v, n_embd_v_gqa, n_stream);
        } else {
            v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
        }

        if (layer_has_k_scale) {
            k_scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) kv_size * n_stream);
            ggml_format_name(k_scale, "cache_k_gscale_l%d", il);
        }
        if (layer_nvfp4_kcache_outlier) {
            k_outlier_count = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) kv_size * n_stream);
            const uint32_t configured_capacity =
                    (size_t) il < nvfp4_kcache_outlier_layer_capacity_count
                            ? nvfp4_kcache_outlier_layer_capacities[(size_t) il]
                            : 1u;
            const uint32_t outlier_row_capacity = std::max<uint32_t>(configured_capacity, 1u);
            const int64_t outlier_capacity = (int64_t) outlier_row_capacity * kv_size;
            k_outlier_offset = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) kv_size * n_stream);
            k_outlier_cursor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_stream);
            k_outlier_index = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, outlier_capacity, n_stream);
            k_outlier_value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, outlier_capacity, n_stream);
            ggml_format_name(k_outlier_offset, "cache_k_outlier_offset_l%d", il);
            ggml_format_name(k_outlier_cursor, "cache_k_outlier_cursor_l%d", il);
            ggml_format_name(k_outlier_count, "cache_k_outlier_count_l%d", il);
            ggml_format_name(k_outlier_index, "cache_k_outlier_index_l%d", il);
            ggml_format_name(k_outlier_value, "cache_k_outlier_value_l%d", il);
            nvfp4_kcache_outlier_sidecar_bytes += ggml_nbytes(k_outlier_count);
            nvfp4_kcache_outlier_sidecar_bytes += ggml_nbytes(k_outlier_offset);
            nvfp4_kcache_outlier_sidecar_bytes += ggml_nbytes(k_outlier_cursor);
            nvfp4_kcache_outlier_sidecar_bytes += ggml_nbytes(k_outlier_index);
            nvfp4_kcache_outlier_sidecar_bytes += ggml_nbytes(k_outlier_value);
        }
        if (layer_nvfp4_kcache_recent_f16) {
            k_recent_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd_k_gqa, kcache_recent_f16_window, n_stream);
            k_recent_f16_active = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) kv_size * n_stream);
            k_recent_f16_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) kv_size * n_stream);
            ggml_format_name(k_recent_f16, "cache_k_recent_f16_l%d", il);
            ggml_format_name(k_recent_f16_active, "cache_k_recent_f16_active_l%d", il);
            ggml_format_name(k_recent_f16_pos, "cache_k_recent_f16_pos_l%d", il);
            nvfp4_kcache_recent_f16_sidecar_bytes += ggml_nbytes(k_recent_f16);
            nvfp4_kcache_recent_f16_sidecar_bytes += ggml_nbytes(k_recent_f16_active);
            nvfp4_kcache_recent_f16_sidecar_bytes += ggml_nbytes(k_recent_f16_pos);
        }
        if (use_nvfp4_vcache_layout()) {
            v_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_v_scale, n_stream);
            ggml_format_name(v_scale, use_nvfp4_vcache_single_global_scale() ? "cache_v_lgscale_l%d" : "cache_v_gscale_l%d", il);
        }

        ggml_format_name(k, "cache_k_l%d", il);
        ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
        std::vector<ggml_tensor *> k_scale_stream;
        std::vector<ggml_tensor *> v_scale_stream;
        std::vector<ggml_tensor *> k_outlier_count_stream;
        std::vector<ggml_tensor *> k_outlier_offset_stream;
        std::vector<ggml_tensor *> k_outlier_cursor_stream;
        std::vector<ggml_tensor *> k_outlier_index_stream;
        std::vector<ggml_tensor *> k_outlier_value_stream;
        std::vector<ggml_tensor *> k_recent_f16_stream;
        std::vector<ggml_tensor *> k_recent_f16_active_stream;
        std::vector<ggml_tensor *> k_recent_f16_pos_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]));
            if (use_nvfp4_vcache_layout()) {
                v_stream.push_back(ggml_view_2d(ctx, v, kv_size_v, n_embd_v_gqa, v->nb[1], s*v->nb[2]));
            } else {
                v_stream.push_back(ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]));
            }
            if (k_scale) {
                k_scale_stream.push_back(ggml_view_1d(ctx, k_scale, kv_size, (int64_t) s * kv_size * sizeof(float)));
            }
            if (k_outlier_count) {
                const int64_t stream_off = (int64_t) s * kv_size;
                k_outlier_count_stream.push_back(ggml_view_1d(ctx, k_outlier_count, kv_size, stream_off * sizeof(int32_t)));
                GGML_ASSERT(k_outlier_offset != nullptr);
                GGML_ASSERT(k_outlier_cursor != nullptr);
                k_outlier_offset_stream.push_back(ggml_view_1d(ctx, k_outlier_offset, kv_size, stream_off * sizeof(int32_t)));
                k_outlier_cursor_stream.push_back(ggml_view_1d(ctx, k_outlier_cursor, 1, (int64_t) s * sizeof(int32_t)));
                k_outlier_index_stream.push_back(ggml_view_1d(ctx, k_outlier_index, k_outlier_index->ne[0], (int64_t) s * k_outlier_index->nb[1]));
                k_outlier_value_stream.push_back(ggml_view_1d(ctx, k_outlier_value, k_outlier_index->ne[0], (int64_t) s * k_outlier_value->nb[1]));
            }
            if (k_recent_f16) {
                const int64_t stream_off = (int64_t) s * kv_size;
                k_recent_f16_stream.push_back(ggml_view_2d(ctx, k_recent_f16, n_embd_k_gqa, kcache_recent_f16_window, k_recent_f16->nb[1], (int64_t) s*k_recent_f16->nb[2]));
                k_recent_f16_active_stream.push_back(ggml_view_1d(ctx, k_recent_f16_active, kv_size, stream_off * sizeof(int32_t)));
                k_recent_f16_pos_stream.push_back(ggml_view_1d(ctx, k_recent_f16_pos, kv_size, stream_off * sizeof(int32_t)));
            }
            if (v_scale) {
                v_scale_stream.push_back(ggml_view_1d(ctx, v_scale, n_v_scale, (int64_t) s * n_v_scale * sizeof(float)));
            }
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({
                il, k, v, k_scale, v_scale,
                k_outlier_count, k_outlier_offset, k_outlier_cursor, k_outlier_index, k_outlier_value,
                k_recent_f16, k_recent_f16_active, k_recent_f16_pos,
                k_stream, v_stream, k_scale_stream, v_scale_stream,
                k_outlier_count_stream, k_outlier_offset_stream, k_outlier_cursor_stream, k_outlier_index_stream, k_outlier_value_stream,
                k_recent_f16_stream, k_recent_f16_active_stream, k_recent_f16_pos_stream,
        });
    }

    // TODO: this is temporary until we support passing reuse layer filters [KV_REUSE]
    if (model.arch == LLM_ARCH_GEMMA3N) {
        LLAMA_LOG_DEBUG("%s: GEMMA3N: reuse layers [%d, %d]\n", __func__, n_layer_cache, hparams.n_layer - 1);

        for (uint32_t il = n_layer_cache; il < hparams.n_layer; il++) {
            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, il);
                continue;
            }

            const bool     is_swa   = hparams.is_swa(il);
            const uint32_t il_reuse = n_layer_cache - (is_swa ? 2 : 1);

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());
            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: layer %3d: reuse layer %d, isw = %d\n", __func__, il, il_reuse, is_swa);
        }
    }

    if (nvfp4_kcache_outlier) {
        LLAMA_LOG_INFO("%s: NVFP4 K-cache compact outlier sidecar size = %8.2f MiB (%zu bytes)\n",
                __func__, nvfp4_kcache_outlier_sidecar_bytes/1024.0/1024.0, nvfp4_kcache_outlier_sidecar_bytes);
    }
    if (nvfp4_kcache_recent_f16_sidecar_bytes > 0) {
        LLAMA_LOG_INFO("%s: NVFP4 K-cache recent-F16 sidecar enabled: window=%u size = %8.2f MiB (%zu bytes)\n",
                __func__,
                llama_nvfp4_kcache_recent_f16_window(),
                nvfp4_kcache_recent_f16_sidecar_bytes/1024.0/1024.0,
                nvfp4_kcache_recent_f16_sidecar_bytes);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto * buft = it.first;
        auto * ctx  = it.second;

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    if (use_nvfp4_vcache_single_global_scale()) {
        for (const auto & layer : layers) {
            GGML_ASSERT(layer.v_scale != nullptr);
            if (use_nvfp4_vcache_layer_global_scale()) {
                const float absmax = nvfp4_layer_absmax[(size_t) layer.il];
                const float global_scale = nvfp4_vcache_layer_global_scales[(size_t) layer.il];
                LLAMA_LOG_INFO("%s: layer %u NVFP4 V-cache global_scale=%g from absmax=%g\n",
                        __func__, layer.il, (double) global_scale, (double) absmax);
            } else {
                LLAMA_LOG_INFO("%s: layer %u NVFP4 V-cache global_scale=%g from default per-tensor P90 absmax=%g\n",
                        __func__, layer.il,
                        (double) llama_vcache_nvfp4_default_v_global_scale(),
                        (double) llama_vcache_nvfp4_default_v_global_absmax());
            }
        }
        init_nvfp4_vcache_layer_global_scales();
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        const std::string k_type_name = kcache_hybrid_fp8_layers.empty()
                ? std::string(ggml_type_name(layers.empty() ? GGML_TYPE_F16 : layers.front().k->type))
                : std::string(ggml_type_name(type_k)) + "+fp8_e4m3_e8m0_32";
        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                k_type_name.c_str(), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;

    const char * LLAMA_SET_ROWS = getenv("LLAMA_SET_ROWS");
    supports_set_rows = LLAMA_SET_ROWS ? atoi(LLAMA_SET_ROWS) != 0 : supports_set_rows;

    if (!supports_set_rows) {
        // ref: https://github.com/ggml-org/llama.cpp/pull/14363
        GGML_ASSERT(unified && "cannot use non-unified KV cache without ggml_set_rows() support");
    }

    if (!supports_set_rows) {
        LLAMA_LOG_WARN("%s: LLAMA_SET_ROWS=0, using old ggml_cpy() method for backwards compatibility\n", __func__);
    }
}

void llama_kv_cache_unified::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & buf : bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
        init_nvfp4_vcache_layer_global_scales();
    }
}

bool llama_kv_cache_unified::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache_unified::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache_unified::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache_unified::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache_unified::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache_unified::seq_pos_min(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_unified::seq_pos_max(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

llama_memory_context_ptr llama_kv_cache_unified::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_unified_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_unified_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_unified::init_full() {
    return std::make_unique<llama_kv_cache_unified_context>(this);
}

llama_memory_context_ptr llama_kv_cache_unified::init_update(llama_context * lctx, bool optimize) {
    bool do_shift = get_has_shift();

    defrag_info dinfo;

    if (use_nvfp4_vcache_layout()) {
        return std::make_unique<llama_kv_cache_unified_context>(this, lctx, do_shift, std::move(dinfo), std::move(sc_info));
    }

    // see if we need to defrag
    if (n_stream == 1) {
        // note : for now do not consider defrag for n_stream > 1
        const auto & cells = v_cells[seq_to_stream[0]];

        bool do_defrag = optimize;

        const auto thold = lctx->get_cparams().defrag_thold;

        if (!do_defrag && thold > 0.0f) {
            const auto n_kv = cells.used_max_p1();

            // - do not defrag small contexts (i.e. < 2048 tokens)
            // - count the padding towards the number of used tokens
            const float fragmentation = n_kv >= 2048 ? std::max(0.0f, 1.0f - (float(cells.get_used() + n_pad)/n_kv)) : 0.0f;

            if (fragmentation > thold) {
                LLAMA_LOG_DEBUG("%s: fragmentation: %.2f - requesting defrag\n", __func__, fragmentation);

                do_defrag = true;
            }
        }

        if (do_defrag) {
            dinfo = defrag_prepare(lctx->graph_max_nodes());
        }
    }

    return std::make_unique<llama_kv_cache_unified_context>(this, lctx, do_shift, std::move(dinfo), std::move(sc_info));
}

llama_kv_cache_unified::slot_info_vec_t llama_kv_cache_unified::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache_unified::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells_unified> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        const bool cont = use_contiguous_v_slots();

        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, cont);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remeber the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache_unified::update(llama_context * lctx, bool do_shift, const defrag_info & dinfo, const stream_copy_info & sc_info) {
    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);
                ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                if (layer.k_scale) {
                    ggml_backend_tensor_copy(layer.k_scale_stream[ssrc], layer.k_scale_stream[sdst]);
                }
                if (layer.k_outlier_count) {
                    ggml_backend_tensor_copy(layer.k_outlier_count_stream[ssrc], layer.k_outlier_count_stream[sdst]);
                    if (layer.k_outlier_offset) {
                        ggml_backend_tensor_copy(layer.k_outlier_offset_stream[ssrc], layer.k_outlier_offset_stream[sdst]);
                        ggml_backend_tensor_copy(layer.k_outlier_cursor_stream[ssrc], layer.k_outlier_cursor_stream[sdst]);
                        ggml_backend_tensor_copy(layer.k_outlier_index_stream[ssrc], layer.k_outlier_index_stream[sdst]);
                        ggml_backend_tensor_copy(layer.k_outlier_value_stream[ssrc], layer.k_outlier_value_stream[sdst]);
                    } else {
                        ggml_backend_tensor_copy(layer.k_outlier_index_stream[ssrc], layer.k_outlier_index_stream[sdst]);
                        ggml_backend_tensor_copy(layer.k_outlier_value_stream[ssrc], layer.k_outlier_value_stream[sdst]);
                    }
                }
                if (layer.k_recent_f16) {
                    ggml_backend_tensor_copy(layer.k_recent_f16_stream[ssrc], layer.k_recent_f16_stream[sdst]);
                    ggml_backend_tensor_copy(layer.k_recent_f16_active_stream[ssrc], layer.k_recent_f16_active_stream[sdst]);
                    ggml_backend_tensor_copy(layer.k_recent_f16_pos_stream[ssrc], layer.k_recent_f16_pos_stream[sdst]);
                }
                if (layer.v_scale) {
                    ggml_backend_tensor_copy(layer.v_scale_stream[ssrc], layer.v_scale_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    if (!dinfo.empty()) {
        LLAMA_LOG_DEBUG("%s: defragmenting KV cache\n", __func__);

        // note: for now do not consider defrag for n_stream > 1
        auto & cells = v_cells[seq_to_stream[0]];
        auto & head  = v_heads[seq_to_stream[0]];

        // apply moves:
        {
            const auto n_kv = dinfo.ids.size();

            for (uint32_t i = 0; i < n_kv; ++i) {
                assert(dinfo.ids[i] <= n_kv);

                if (dinfo.ids[i] == n_kv || dinfo.ids[i] == i) {
                    continue;
                }

                cells.mv(i, dinfo.ids[i]);
            }

            // reset the head so we can find the first free slot during the next ubatch
            head = 0;
        }

        ggml_backend_sched_reset(sched);

        auto * res = lctx->get_gf_res_reserve();

        res->reset();

        auto * gf = build_graph_defrag(res, lctx, dinfo);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute graph for defrag\n", __func__);
            return updated;
        }

        res->set_inputs(nullptr);

        if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: failed to compute defrag\n", __func__);
            return updated;
        }

        updated = true;
    }

    return updated;
}

llama_kv_cache_unified::slot_info llama_kv_cache_unified::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<llama_seq_id>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<llama_seq_id>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (is_masked_swa(pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache_unified::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache_unified::get_can_shift() const {
    return !has_k_scale;
}

uint32_t llama_kv_cache_unified::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache_unified::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache_unified::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

uint32_t llama_kv_cache_unified::get_n_kv() const {
    uint32_t result = 0;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        result = std::max(std::min(cells.size(), std::max(n_pad, GGML_PAD(cells.used_max_p1(), n_pad))), result);
    }

    return result;
}

uint32_t llama_kv_cache_unified::get_n_kv_tensor(uint32_t n_kv) const {
    if (nvfp4_vcache) {
        return GGML_PAD(n_kv, 16u);
    }

    if (!non_flash_fp8_e8m0) {
        return n_kv;
    }

    return GGML_PAD(n_kv, 32);
}

bool llama_kv_cache_unified::use_contiguous_v_slots() const {
    return nvfp4_vcache || !supports_set_rows;
}

bool llama_kv_cache_unified::use_nvfp4_vcache_layout() const {
    return nvfp4_vcache && v_trans;
}

bool llama_kv_cache_unified::use_nvfp4_vcache_layer_global_scale() const {
    return use_nvfp4_vcache_layout() && nvfp4_vcache_layer_global_scale;
}

bool llama_kv_cache_unified::use_nvfp4_vcache_single_global_scale() const {
    return use_nvfp4_vcache_layout() && !nvfp4_vcache_per_block_scale;
}

void llama_kv_cache_unified::init_nvfp4_vcache_layer_global_scales() {
    if (!use_nvfp4_vcache_single_global_scale()) {
        return;
    }

    for (const auto & layer : layers) {
        GGML_ASSERT(layer.v_scale != nullptr);
        const float global_scale = use_nvfp4_vcache_layer_global_scale()
                ? nvfp4_vcache_layer_global_scales[(size_t) layer.il]
                : llama_vcache_nvfp4_default_v_global_scale();
        for (uint32_t s = 0; s < n_stream; ++s) {
            ggml_backend_tensor_set(layer.v_scale_stream[s], &global_scale, 0, sizeof(float));
        }
    }
}

uint32_t llama_kv_cache_unified::get_v_cache_kv_padded() const {
    return use_nvfp4_vcache_layout() ? GGML_PAD(get_size(), 16u) : get_size();
}

uint32_t llama_kv_cache_unified::get_v_cache_block_count() const {
    return get_v_cache_kv_padded() / 16u;
}

uint32_t llama_kv_cache_unified::get_v_cache_row_count(int32_t il) const {
    return hparams.n_embd_v_gqa(il);
}

uint32_t llama_kv_cache_unified::get_v_cache_scale_offset(uint32_t row, uint32_t block) const {
    return row * get_v_cache_block_count() + block;
}

bool llama_kv_cache_unified::get_supports_set_rows() const {
    return supports_set_rows;
}

ggml_tensor * llama_kv_cache_unified::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;
    auto * k_scale = layers[ikv].k_scale;
    auto * k_outlier_count = layers[ikv].k_outlier_count;
    auto * k_outlier_offset = layers[ikv].k_outlier_offset;
    auto * k_outlier_cursor = layers[ikv].k_outlier_cursor;
    auto * k_outlier_index = layers[ikv].k_outlier_index;
    auto * k_outlier_value = layers[ikv].k_outlier_value;
    auto * k_recent_f16 = layers[ikv].k_recent_f16;
    auto * k_recent_f16_active = layers[ikv].k_recent_f16_active;
    auto * k_recent_f16_pos = layers[ikv].k_recent_f16_pos;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];
    n_kv = get_n_kv_tensor(n_kv);

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    ggml_tensor * res = ggml_view_4d(ctx, k,
            hparams.n_embd_head_k, hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);

    if (k_scale) {
        ggml_tensor * scale = ggml_view_4d(ctx, k_scale,
                n_kv, 1, 1, ns,
                0,
                0,
                (int64_t) kv_size * sizeof(float),
                (int64_t) sinfo.s0 * kv_size * sizeof(float));
        ggml_tensor_set_nvfp4_scale(res, scale);
    }
    if (k_outlier_count) {
        ggml_tensor * outlier_count = ggml_view_4d(ctx, k_outlier_count,
                n_kv, 1, 1, ns,
                0,
                0,
                (int64_t) kv_size * sizeof(int32_t),
                (int64_t) sinfo.s0 * kv_size * sizeof(int32_t));
        GGML_ASSERT(k_outlier_offset != nullptr);
        GGML_ASSERT(k_outlier_cursor != nullptr);
        ggml_tensor * outlier_offset = ggml_view_4d(ctx, k_outlier_offset,
                n_kv, 1, 1, ns,
                0,
                0,
                (int64_t) kv_size * sizeof(int32_t),
                (int64_t) sinfo.s0 * kv_size * sizeof(int32_t));
        ggml_tensor * outlier_cursor = ggml_view_1d(ctx, k_outlier_cursor, ns, (int64_t) sinfo.s0 * sizeof(int32_t));
        ggml_tensor * outlier_index = ggml_view_2d(ctx, k_outlier_index,
                k_outlier_index->ne[0], ns,
                k_outlier_index->nb[1],
                (int64_t) sinfo.s0 * k_outlier_index->nb[1]);
        ggml_tensor * outlier_value = ggml_view_2d(ctx, k_outlier_value,
                k_outlier_index->ne[0], ns,
                k_outlier_value->nb[1],
                (int64_t) sinfo.s0 * k_outlier_value->nb[1]);
        ggml_tensor_set_nvfp4_kcache_outliers_compact(res, outlier_count, outlier_offset, outlier_index, outlier_value);
        ggml_tensor_set_nvfp4_kcache_outlier_cursor(res, outlier_cursor);
    }
    if (k_recent_f16) {
        GGML_ASSERT(k_recent_f16_active != nullptr);
        GGML_ASSERT(k_recent_f16_pos != nullptr);
        ggml_tensor * recent_f16 = ggml_view_4d(ctx, k_recent_f16,
                hparams.n_embd_head_k, hparams.n_head_kv(il), k_recent_f16->ne[1], ns,
                ggml_row_size(k_recent_f16->type, hparams.n_embd_head_k),
                ggml_row_size(k_recent_f16->type, n_embd_k_gqa),
                ggml_row_size(k_recent_f16->type, n_embd_k_gqa*k_recent_f16->ne[1]),
                ggml_row_size(k_recent_f16->type, n_embd_k_gqa*k_recent_f16->ne[1])*sinfo.s0);
        ggml_tensor * recent_active = ggml_view_4d(ctx, k_recent_f16_active,
                n_kv, 1, 1, ns,
                0,
                0,
                (int64_t) kv_size * sizeof(int32_t),
                (int64_t) sinfo.s0 * kv_size * sizeof(int32_t));
        ggml_tensor * recent_pos = ggml_view_4d(ctx, k_recent_f16_pos,
                n_kv, 1, 1, ns,
                0,
                0,
                (int64_t) kv_size * sizeof(int32_t),
                (int64_t) sinfo.s0 * kv_size * sizeof(int32_t));
        ggml_tensor_set_nvfp4_kcache_recent_f16(res, recent_f16, recent_active, recent_pos);
    }

    return res;
}

ggml_tensor * llama_kv_cache_unified::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;
    auto * v_scale = layers[ikv].v_scale;

    const uint64_t kv_size      = get_size();
    const uint64_t kv_size_v    = get_v_cache_kv_padded();
    const uint64_t n_embd_v_gqa_phys = use_nvfp4_vcache_layout() ? v->ne[1] : v->ne[0];
    const uint64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);
    n_kv = get_n_kv_tensor(n_kv);

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa_phys >= n_embd_v_gqa);

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        ggml_tensor * res = ggml_view_4d(ctx, v,
                hparams.n_embd_head_v, hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v),            // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa_phys),         // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa_phys*kv_size), // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa_phys*kv_size)*sinfo.s0);

        if (v_scale) {
            ggml_tensor * scale = ggml_view_4d(ctx, v_scale,
                    n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v, ns,
                    ggml_row_size(v_scale->type, kv_size*hparams.n_embd_head_v),
                    ggml_row_size(v_scale->type, kv_size),
                    ggml_row_size(v_scale->type, kv_size*n_embd_v_gqa),
                    ggml_row_size(v_scale->type, kv_size*n_embd_v_gqa)*sinfo.s0);
            ggml_tensor_set_nvfp4_scale(res, scale);
        }

        return res;
    }

    if (use_nvfp4_vcache_layout()) {
        if (v_scale && use_nvfp4_vcache_single_global_scale()) {
            ggml_tensor * res = ggml_view_4d(ctx, v,
                        n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v, ns,
                        ggml_row_size(v->type, kv_size_v*hparams.n_embd_head_v),
                        ggml_row_size(v->type, kv_size_v),
                        ggml_row_size(v->type, kv_size_v*n_embd_v_gqa_phys),
                        ggml_row_size(v->type, kv_size_v*n_embd_v_gqa_phys)*sinfo.s0);
            ggml_tensor * scale = ggml_view_1d(ctx, v_scale, ns, (int64_t) sinfo.s0 * sizeof(float));
            ggml_tensor_set_nvfp4_scale(res, scale);
            return res;
        }

        ggml_tensor * res = ggml_view_4d(ctx, v,
                    n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v, ns,
                    ggml_row_size(v->type, kv_size_v*hparams.n_embd_head_v),
                    ggml_row_size(v->type, kv_size_v),
                    ggml_row_size(v->type, kv_size_v*n_embd_v_gqa_phys),
                    ggml_row_size(v->type, kv_size_v*n_embd_v_gqa_phys)*sinfo.s0);

        if (v_scale) {
            ggml_tensor * scale = ggml_view_4d(ctx, v_scale,
                    n_kv / 16, hparams.n_head_kv(il), hparams.n_embd_head_v, ns,
                    (int64_t) get_v_cache_block_count() * hparams.n_embd_head_v * sizeof(float),
                    (int64_t) get_v_cache_block_count() * sizeof(float),
                    (int64_t) get_v_cache_block_count() * n_embd_v_gqa_phys * sizeof(float),
                    (int64_t) sinfo.s0 * get_v_cache_block_count() * n_embd_v_gqa_phys * sizeof(float));
            ggml_tensor_set_nvfp4_scale(res, scale);
        }

        return res;
    }

    // note: v->nb[1] > v->nb[2]
    ggml_tensor * res = ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v, ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v),    // v->nb[1]
            ggml_row_size(v->type, kv_size),                          // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa), // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);

    return res;
}

ggml_tensor * llama_kv_cache_unified::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo, const llama_ubatch * ubatch) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const int64_t n_embd_k_gqa = k->ne[0];
    const int64_t n_tokens = k_cur->ne[2];

    k_cur = ggml_reshape_2d(ctx, k_cur, k->ne[0], n_tokens);

    if (k_idxs && supports_set_rows) {
        if (k->ne[2] > 1) {
            k = ggml_reshape_2d(ctx, k, k->ne[0], k->ne[1]*k->ne[2]);
        }

        ggml_tensor * res = ggml_set_rows(ctx, k, k_cur, k_idxs);
        if (nvfp4_kcache_outlier) {
            const float threshold = llama_nvfp4_kcache_outlier_layer_threshold((uint32_t) il, !kcache_hybrid_fp8_layers.empty());
            std::memcpy(&res->op_params[0], &threshold, sizeof(threshold));
        }
        if (layers[ikv].k_scale) {
            ggml_tensor_set_nvfp4_scale(res, layers[ikv].k_scale);
        }
        if (layers[ikv].k_outlier_count) {
            GGML_ASSERT(layers[ikv].k_outlier_offset != nullptr);
            GGML_ASSERT(layers[ikv].k_outlier_cursor != nullptr);
            ggml_tensor_set_nvfp4_kcache_outliers_compact(
                    res,
                    layers[ikv].k_outlier_count,
                    layers[ikv].k_outlier_offset,
                    layers[ikv].k_outlier_index,
                    layers[ikv].k_outlier_value);
            ggml_tensor_set_nvfp4_kcache_outlier_cursor(res, layers[ikv].k_outlier_cursor);
        }
        if (layers[ikv].k_recent_f16) {
            GGML_ASSERT(layers[ikv].k_recent_f16_active != nullptr);
            GGML_ASSERT(layers[ikv].k_recent_f16_pos != nullptr);
            int32_t query_pos = n_tokens > 0 ? (int32_t) n_tokens - 1 : 0;
            if (ubatch != nullptr && ubatch->pos != nullptr) {
                for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                    query_pos = std::max<int32_t>(query_pos, ubatch->pos[i]);
                }
            }
            const int32_t recent_window = (int32_t) llama_nvfp4_kcache_recent_f16_window();
            std::memcpy(&res->op_params[1], &recent_window, sizeof(recent_window));
            std::memcpy(&res->op_params[2], &query_pos, sizeof(query_pos));
            ggml_tensor_set_nvfp4_kcache_recent_f16(
                    res,
                    layers[ikv].k_recent_f16,
                    layers[ikv].k_recent_f16_active,
                    layers[ikv].k_recent_f16_pos);
        }
        return res;
    }

    // TODO: fallback to old ggml_cpy() method for backwards compatibility
    //       will be removed when ggml_set_rows() is adopted by all backends

    GGML_ASSERT(n_stream == 1 && "n_stream > 1 not supported without LLAMA_SET_ROWS");

    ggml_tensor * k_view = ggml_view_1d(ctx, k,
            n_tokens*n_embd_k_gqa,
            ggml_row_size(k->type, n_embd_k_gqa)*sinfo.head());

    return ggml_cpy(ctx, k_cur, k_view);
}

ggml_tensor * llama_kv_cache_unified::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;
    auto * v_scale = layers[ikv].v_scale;

    const int64_t n_embd_v_gqa = v_cur->ne[0]*v_cur->ne[1];
    const int64_t n_tokens     = v_cur->ne[2];
    const int64_t n_embd_v_gqa_phys = use_nvfp4_vcache_layout() ? v->ne[1] : v->ne[0];

    v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_v_gqa, n_tokens);

    if (v_idxs && supports_set_rows) {
        if (!v_trans) {
            if (v->ne[2] > 1) {
                v = ggml_reshape_2d(ctx, v, v->ne[0], v->ne[1]*v->ne[2]);
            }

            ggml_tensor * res = ggml_set_rows(ctx, v, v_cur, v_idxs);
            if (v_scale) {
                ggml_tensor_set_nvfp4_scale(res, v_scale);
            }
            return res;
        }

        if (use_nvfp4_vcache_layout()) {
            if (n_embd_v_gqa < n_embd_v_gqa_phys) {
                v_cur = ggml_pad(ctx, v_cur, n_embd_v_gqa_phys - n_embd_v_gqa, 0, 0, 0);
            }

            GGML_ASSERT(v->ne[0] % 16 == 0);
            GGML_ASSERT(v_cur->ne[0] % 16 == 0);

            ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 16, v->ne[0]*v->ne[1]*v->ne[2] / 16);
            v_cur = ggml_reshape_2d(ctx, v_cur, 16, v_cur->ne[0]*v_cur->ne[1] / 16);

            ggml_tensor * res = ggml_set_rows(ctx, v_view, v_cur, v_idxs);
            if (v_scale) {
                ggml_tensor_set_nvfp4_scale(res, v_scale);
            }
            return res;
        }

        // [TAG_V_CACHE_VARIABLE]
        if (n_embd_v_gqa < v->ne[0]) {
            v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_v_gqa, 0, 0, 0);
        }

        // the row becomes a single element
        ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, v->ne[0]*v->ne[1]*v->ne[2]);

        v_cur = ggml_reshape_2d(ctx, v_cur, 1, v_cur->ne[0]*v_cur->ne[1]);

        ggml_tensor * res = ggml_set_rows(ctx, v_view, v_cur, v_idxs);
        if (v_scale) {
            ggml_tensor_set_nvfp4_scale(res, v_scale);
        }
        return res;
    }

    // TODO: fallback to old ggml_cpy() method for backwards compatibility
    //       will be removed when ggml_set_rows() is adopted by all backends

    GGML_ASSERT(n_stream == 1 && "n_stream > 1 not supported without LLAMA_SET_ROWS");

    ggml_tensor * v_view = nullptr;

    if (!v_trans) {
        v_view = ggml_view_1d(ctx, v,
                n_tokens*n_embd_v_gqa,
                ggml_row_size(v->type, n_embd_v_gqa)*sinfo.head());
    } else {
        v_cur = ggml_transpose(ctx, v_cur);

        v_view = ggml_view_2d(ctx, v, n_tokens, n_embd_v_gqa,
                (v->ne[1]    )*ggml_element_size(v),
                (sinfo.head())*ggml_element_size(v));
    }

    return ggml_cpy(ctx, v_cur, v_view);
}

ggml_tensor * llama_kv_cache_unified::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache_unified::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else if (use_nvfp4_vcache_layout()) {
        GGML_ASSERT(hparams.n_embd_v_gqa_max() % 16 == 0);
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max()/16);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

void llama_kv_cache_unified::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    if (!supports_set_rows) {
        return;
    }

    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache_unified::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    if (!supports_set_rows) {
        return;
    }

    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else if (use_nvfp4_vcache_layout()) {
        // note: the NVFP4 V cache stores 16 token slots per block.
        const int64_t kv_size = get_v_cache_kv_padded();
        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();
        GGML_ASSERT(n_embd_v_gqa % 16 == 0);

        const int64_t n_row_groups = n_embd_v_gqa / 16;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (int64_t j = 0; j < n_embd_v_gqa; j += 16) {
                    data[s*sinfo.size()*n_row_groups + i*n_row_groups + j/16] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache_unified::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

void llama_kv_cache_unified::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    float * data = (float *) dst->data;

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps     = n_tokens/n_stream;
    const int64_t n_tps_pad = GGML_PAD(n_tps, GGML_KQ_MASK_PAD);

    std::fill(data, data + ggml_nelements(dst), -INFINITY);

    // Use only the previous KV cells of the correct sequence for each token of the ubatch.
    // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
    // Example with a cache of 10 tokens, 2 tokens populated in cache and 3 tokens in batch:
    //   Causal mask:
    //      xxx-------
    //      xxxx------
    //      xxxxx-----
    //   Non-causal mask:
    //      xxxxx-----
    //      xxxxx-----
    //      xxxxx-----
    // To visualize the mask, see https://github.com/ggml-org/llama.cpp/pull/12615
    // TODO: optimize this section
    for (uint32_t h = 0; h < 1; ++h) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            for (uint32_t ii = 0; ii < n_tps; ++ii) {
                const uint32_t i = s*n_tps + ii;

                const llama_seq_id seq_id = ubatch->seq_id[i][0];

                const auto & cells = v_cells[seq_to_stream[seq_id]];

                const llama_pos p1 = ubatch->pos[i];

                const uint64_t idst = n_kv*(h*n_stream*n_tps_pad + s*n_tps_pad + ii);

                for (uint32_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j)) {
                        continue;
                    }

                    // mask the token if not the same sequence
                    if (!cells.seq_has(j, seq_id)) {
                        continue;
                    }

                    const llama_pos p0 = cells.pos_get(j);

                    // mask future tokens
                    if (causal_attn && p0 > p1) {
                        continue;
                    }

                    // apply SWA if any
                    if (is_masked_swa(p0, p1)) {
                        continue;
                    }

                    data[idst + j] = hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f;
                }
            }
        }
    }
}

void llama_kv_cache_unified::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

size_t llama_kv_cache_unified::total_size() const {
    size_t size = 0;

    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache_unified::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache_unified::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += ggml_nbytes(layer.v);
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache_unified::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast  = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow  = cparams.yarn_beta_slow;

    const auto & n_rot     = hparams.n_rot;
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;

    // See llm_build_deepseek2() for why attn_factor has to be scaled for YaRN RoPE to work correctly.
    // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
    const float yarn_attn_factor = model.arch == LLM_ARCH_DEEPSEEK2
                                    ? 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale))
                                    : cparams.yarn_attn_factor;

    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache_unified * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    const llama_kv_cache_unified * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }
}

ggml_cgraph * llama_kv_cache_unified::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    const auto & n_embd_head_k = hparams.n_embd_head_k;
  //const auto & n_embd_head_v = hparams.n_embd_head_v;

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_embd_head_k, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                0);

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, rope_factors, freq_base_l, freq_scale_l);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

ggml_cgraph * llama_kv_cache_unified::build_graph_defrag(
         llm_graph_result * res,
            llama_context * lctx,
        const defrag_info & dinfo) const {
    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    GGML_ASSERT(n_stream == 1 && "n_stream > 1 does not support defrag");

    const auto & cells = v_cells[0];

    const auto & ids = dinfo.ids;

    const auto & cparams = lctx->get_cparams();

#if 0
    // CPU defrag
    //
    // TODO: optimizations are possible:
    //       - multiple threads
    //       - avoid copying to the host memory when already there
    //
    // likely not worth the effort, as we have ggml_graph based defrag
    //

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    const uint32_t kv_size = size;

    std::vector<uint8_t> buf_k;
    std::vector<uint8_t> buf_v;

    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t k_size_row = ggml_row_size(k_l[il]->type, n_embd_k_gqa);
        const size_t k_size     = ggml_row_size(k_l[il]->type, n_embd_k_gqa*kv_size);

        const size_t v_size_el = ggml_type_size(v_l[il]->type);
        const size_t v_size    = ggml_row_size (v_l[il]->type, n_embd_v_gqa*kv_size);

        buf_k.resize(k_size);
        buf_v.resize(v_size);

        ggml_backend_tensor_get(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_get(v_l[il], buf_v.data(), 0, buf_v.size());

        // batch move [i, i+nm) to [id, id+nm)
        // note: cells can move only to a lower index
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == n_kv) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < n_kv && ids[i + nm] == id + nm) {
                nm++;
            }

            // move keys
            {
                const int64_t os =  i*k_size_row;
                const int64_t od = id*k_size_row;

                memcpy(buf_k.data() + od, buf_k.data() + os, nm*k_size_row);
            }

            // move values (note: they are transposed)
            {
                const int64_t os =  i;
                const int64_t od = id;

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    memcpy(buf_v.data() + (od + j*kv_size)*v_size_el, buf_v.data() + (os + j*kv_size)*v_size_el, nm*v_size_el);
                }
            }

            i += nm - 1;
        }

        ggml_backend_tensor_set(k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_set(v_l[il], buf_v.data(), 0, buf_v.size());
    }
#else
    for (uint32_t i = 0; i < ids.size(); ++i) {
        const uint32_t id = ids[i];

        if (i == id || id == ids.size()) {
            continue;
        }

        uint32_t nm = 1;

        while (i + nm < ids.size() && ids[i + nm] == id + nm) {
            nm++;
        }

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            ggml_tensor * view_k_src = ggml_view_2d(ctx, layer.k,
                    n_embd_k_gqa, nm,
                    ggml_row_size(layer.k->type, n_embd_k_gqa),
                    ggml_row_size(layer.k->type, n_embd_k_gqa*i));

            ggml_tensor * view_k_dst = ggml_view_2d(ctx, layer.k,
                    n_embd_k_gqa, nm,
                    ggml_row_size(layer.k->type, n_embd_k_gqa),
                    ggml_row_size(layer.k->type, n_embd_k_gqa*id));

            ggml_tensor * view_v_src;
            ggml_tensor * view_v_dst;
            ggml_tensor * view_k_scale_src = nullptr;
            ggml_tensor * view_k_scale_dst = nullptr;
            ggml_tensor * view_v_scale_src = nullptr;
            ggml_tensor * view_v_scale_dst = nullptr;
            ggml_tensor * view_k_outlier_count_src = nullptr;
            ggml_tensor * view_k_outlier_count_dst = nullptr;
            ggml_tensor * view_k_outlier_offset_src = nullptr;
            ggml_tensor * view_k_outlier_offset_dst = nullptr;
            ggml_tensor * view_k_recent_f16_active_src = nullptr;
            ggml_tensor * view_k_recent_f16_active_dst = nullptr;
            ggml_tensor * view_k_recent_f16_pos_src = nullptr;
            ggml_tensor * view_k_recent_f16_pos_dst = nullptr;

            if (cparams.flash_attn) {
                // NOTE: the V cache is not transposed when using flash attention
                view_v_src = ggml_view_2d(ctx, layer.v,
                        n_embd_v_gqa, nm,
                        ggml_row_size(layer.v->type, n_embd_v_gqa),
                        ggml_row_size(layer.v->type, n_embd_v_gqa*i));

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        n_embd_v_gqa, nm,
                        ggml_row_size(layer.v->type, n_embd_v_gqa),
                        ggml_row_size(layer.v->type, n_embd_v_gqa*id));
            } else {
                view_v_src = ggml_view_2d(ctx, layer.v,
                        nm, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, cells.size()),
                        ggml_row_size(layer.v->type, i));

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        nm, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, cells.size()),
                        ggml_row_size(layer.v->type, id));
            }

            if (use_nvfp4_vcache_layout()) {
                const int64_t kv_size_v = get_v_cache_kv_padded();
                view_v_src = ggml_view_2d(ctx, layer.v,
                        kv_size_v, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, kv_size_v),
                        ggml_row_size(layer.v->type, kv_size_v*n_embd_v_gqa)*0 + ggml_row_size(layer.v->type, 1)*i);

                view_v_dst = ggml_view_2d(ctx, layer.v,
                        kv_size_v, n_embd_v_gqa,
                        ggml_row_size(layer.v->type, kv_size_v),
                        ggml_row_size(layer.v->type, kv_size_v*n_embd_v_gqa)*0 + ggml_row_size(layer.v->type, 1)*id);
            }

            ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_src, view_k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, view_v_src, view_v_dst));

            if (layer.k_scale) {
                view_k_scale_src = ggml_view_1d(ctx, layer.k_scale, nm, i * sizeof(float));
                view_k_scale_dst = ggml_view_1d(ctx, layer.k_scale, nm, id * sizeof(float));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_scale_src, view_k_scale_dst));
            }
            if (layer.k_outlier_count) {
                view_k_outlier_count_src = ggml_view_1d(ctx, layer.k_outlier_count, nm, i * sizeof(int32_t));
                view_k_outlier_count_dst = ggml_view_1d(ctx, layer.k_outlier_count, nm, id * sizeof(int32_t));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_outlier_count_src, view_k_outlier_count_dst));

                GGML_ASSERT(layer.k_outlier_offset != nullptr);
                view_k_outlier_offset_src = ggml_view_1d(ctx, layer.k_outlier_offset, nm, i * sizeof(int32_t));
                view_k_outlier_offset_dst = ggml_view_1d(ctx, layer.k_outlier_offset, nm, id * sizeof(int32_t));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_outlier_offset_src, view_k_outlier_offset_dst));
            }
            if (layer.k_recent_f16) {
                GGML_ASSERT(layer.k_recent_f16_active != nullptr);
                GGML_ASSERT(layer.k_recent_f16_pos != nullptr);
                view_k_recent_f16_active_src = ggml_view_1d(ctx, layer.k_recent_f16_active, nm, i * sizeof(int32_t));
                view_k_recent_f16_active_dst = ggml_view_1d(ctx, layer.k_recent_f16_active, nm, id * sizeof(int32_t));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_recent_f16_active_src, view_k_recent_f16_active_dst));

                view_k_recent_f16_pos_src = ggml_view_1d(ctx, layer.k_recent_f16_pos, nm, i * sizeof(int32_t));
                view_k_recent_f16_pos_dst = ggml_view_1d(ctx, layer.k_recent_f16_pos, nm, id * sizeof(int32_t));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, view_k_recent_f16_pos_src, view_k_recent_f16_pos_dst));
            }
            if (layer.v_scale) {
                if (use_nvfp4_vcache_single_global_scale()) {
                    // The scalar global scale is independent of cell positions.
                } else if (use_nvfp4_vcache_layout()) {
                    const int64_t n_blk = get_v_cache_block_count();
                    const int64_t src_block = i / 16;
                    const int64_t dst_block = id / 16;
                    if (src_block != dst_block) {
                        view_v_scale_src = ggml_view_2d(ctx, layer.v_scale,
                                n_blk, n_embd_v_gqa,
                                ggml_row_size(layer.v_scale->type, n_blk),
                                0);
                        view_v_scale_dst = ggml_view_2d(ctx, layer.v_scale,
                                n_blk, n_embd_v_gqa,
                                ggml_row_size(layer.v_scale->type, n_blk),
                                0);
                    }
                } else {
                    view_v_scale_src = ggml_view_2d(ctx, layer.v_scale,
                            nm, n_embd_v_gqa,
                            ggml_row_size(layer.v_scale->type, cells.size()),
                            ggml_row_size(layer.v_scale->type, i));
                    view_v_scale_dst = ggml_view_2d(ctx, layer.v_scale,
                            nm, n_embd_v_gqa,
                            ggml_row_size(layer.v_scale->type, cells.size()),
                            ggml_row_size(layer.v_scale->type, id));
                }
                if (view_v_scale_src && view_v_scale_dst) {
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, view_v_scale_src, view_v_scale_dst));
                }
            }
        }

        i += nm - 1;
    }

    //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);
#endif

    return gf;
}

llama_kv_cache_unified::defrag_info llama_kv_cache_unified::defrag_prepare(int32_t n_max_nodes) const {
    GGML_ASSERT(n_stream == 1 && "n_stream > 1 does not support defrag");

    const auto & cells = v_cells[0];

    const uint32_t n_layer = layers.size();

    const uint32_t n_kv   = cells.used_max_p1();
    const uint32_t n_used = cells.get_used();

    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();

    // number of cells moved
    uint32_t n_moves = 0;

    // each move requires 6*n_layer tensors (see graph_build_kv_self_defrag)
    //   - source view, destination view, copy operation
    //   - x2 for keys and values
    //const uint32_t max_moves = max_nodes()/(6*n_layer);
    // TODO: tmp fix https://github.com/ggerganov/llama.cpp/issues/6685#issuecomment-2057579516
    const uint32_t max_moves = (n_max_nodes - 2*n_layer)/(6*n_layer);

    // determine which KV cells to move where
    defrag_info res;
    auto & ids = res.ids;

    ids.resize(n_kv, n_kv);

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        if (!cells.is_empty(i0)) {
            ids[i0] = i0;

            continue;
        }

        // found a hole - fill it with data from the end of the cache

        uint32_t nh = 1;

        // determine the size of the hole
        while (i0 + nh < n_used && cells.is_empty(i0 + nh)) {
            nh++;
        }

        uint32_t nf = 0;
        uint32_t is = n_kv - 1;

        // starting from the end, find nh non-empty cells
        for (; is > i0; --is) {
            if (cells.is_empty(is) || ids[is] != n_kv) {
                continue;
            }

            // non-empty cell which is not yet moved
            nf++;

            if (nf == nh) {
                break;
            }
        }

        // this can only happen if `n_used` is not accurate, which would be a bug
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;

        uint32_t i1 = is;

        // are we moving a continuous block of memory?
        bool cont = false;

        // should we stop searching for the next move?
        bool stop = false;

        // go back and move the nf cells to the hole
        for (; i1 < n_kv; ++i1) {
            if (cells.is_empty(i1) || ids[i1] != n_kv) {
                if (n_moves == max_moves) {
                    stop = true;
                    break;
                }

                cont = false;
                continue;
            }

            // this cell goes to (i0 + nf)
            ids[i1] = i0 + nf;

            if (!cont) {
                n_moves++;
                cont = true;
            }

            nf++;

            if (nf == nh) {
                break;
            }
        }

        if (stop || n_moves == max_moves) {
            break;
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;
    }

    if (n_moves == 0) {
        return {};
    }

    LLAMA_LOG_DEBUG("%s: (tmp log) KV defrag cell moves: %u\n", __func__, n_moves);

    LLAMA_LOG_DEBUG("%s: expected gf nodes: %u\n", __func__, 6*n_moves*n_layer);

    return res;
}

bool llama_kv_cache_unified::is_masked_swa(llama_pos p0, llama_pos p1) const {
    assert(p0 >= 0 && p1 >= 0);

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:
            {
            } break;
        case LLAMA_SWA_TYPE_STANDARD:
            {
                if (p1 - p0 >= (int32_t) n_swa) {
                    return true;
                }
            } break;
        case LLAMA_SWA_TYPE_CHUNKED:
            {
                const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                if (p0 < pos_chunk_start) {
                    return true;
                }
            } break;
    }

    return false;
}

void llama_kv_cache_unified::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    if (use_nvfp4_vcache_layout()) {
        throw std::runtime_error("NVFP4 V-cache state export is not supported");
    }
    if (nvfp4_kcache_outlier) {
        throw std::runtime_error("NVFP4 K-cache outlier state export is not supported");
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id))) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache_unified::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    if (use_nvfp4_vcache_layout()) {
        throw std::runtime_error("NVFP4 V-cache state import is not supported");
    }
    if (nvfp4_kcache_outlier) {
        throw std::runtime_error("NVFP4 K-cache outlier state import is not supported");
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read_to(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read_to(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, seq_id);
        res = res && state_read_data(io, strm, cell_count);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache_unified::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache_unified::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    std::vector<uint8_t> tmp_buf;

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length each into tmp_buf and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }

        const uint32_t has_scale = layer.k_scale ? 1u : 0u;
        io.write(&has_scale, sizeof(has_scale));
        if (layer.k_scale) {
            auto * k_scale = layer.k_scale_stream[cr.strm];
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                io.write_tensor(k_scale, range.first * sizeof(float), range_size * sizeof(float));
            }
        }
    }

    const uint32_t has_v_scale = nvfp4_vcache ? 1u : 0u;
    io.write(&has_v_scale, sizeof(has_v_scale));
    if (nvfp4_vcache) {
        for (const auto & layer : layers) {
            auto * v_scale = layer.v_scale_stream[cr.strm];
            GGML_ASSERT(v_scale != nullptr);
            if (use_nvfp4_vcache_single_global_scale()) {
                io.write_tensor(v_scale, 0, sizeof(float));
                continue;
            }
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(layer.il);
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + (size_t) j * cells.size()) * sizeof(float);
                    io.write_tensor(v_scale, src_offset, range_size * sizeof(float));
                }
            }
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length each into tmp_buf and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }

        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length each into tmp_buf and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }

        }
    }
}

bool llama_kv_cache_unified::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        const auto sinfo = find_slot(ubatch, true);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        apply_ubatch(sinfo, ubatch);

        const auto head_cur = sinfo.head();

        // keep the head at the old position because we will read the KV data into it in state_read_data()
        head = head_cur;

        LLAMA_LOG_DEBUG("%s: head_cur = %d, head = %d, cell_count = %d, dest_seq_id = %d\n", __func__, head_cur, head, cell_count, dest_seq_id);

        // DEBUG CHECK: head_cur should be our first cell, head_cur + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        GGML_ASSERT(head_cur + cell_count <= cells.size());
        GGML_ASSERT(cells.pos_get(head_cur)                  == ubatch.pos[0]);
        GGML_ASSERT(cells.pos_get(head_cur + cell_count - 1) == ubatch.pos[cell_count - 1]);
        GGML_ASSERT(cells.seq_has(head_cur,                  dest_seq_id));
        GGML_ASSERT(cells.seq_has(head_cur + cell_count - 1, dest_seq_id));
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache_unified::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            ggml_backend_tensor_set(k, io.read(cell_count * k_size_row), head * k_size_row, cell_count * k_size_row);
        }

        uint32_t has_scale_ref = 0;
        io.read_to(&has_scale_ref, sizeof(has_scale_ref));
        if ((layer.k_scale != nullptr) != (has_scale_ref != 0)) {
            LLAMA_LOG_ERROR("%s: mismatched key scale sidecar presence (layer %d)\n", __func__, il);
            return false;
        }
        if (cell_count && layer.k_scale) {
            ggml_backend_tensor_set(layer.k_scale_stream[strm], io.read(cell_count * sizeof(float)), head * sizeof(float), cell_count * sizeof(float));
        }
    }

    uint32_t has_v_scale_ref = 0;
    io.read_to(&has_v_scale_ref, sizeof(has_v_scale_ref));
    if ((nvfp4_vcache ? 1u : 0u) != has_v_scale_ref) {
        LLAMA_LOG_ERROR("%s: mismatched value scale sidecar presence\n", __func__);
        return false;
    }
    if (cell_count && nvfp4_vcache) {
        for (const auto & layer : layers) {
            if (layer.v_scale == nullptr) {
                LLAMA_LOG_ERROR("%s: missing value scale sidecar for layer %d\n", __func__, layer.il);
                return false;
            }
            if (use_nvfp4_vcache_single_global_scale()) {
                ggml_backend_tensor_set(layer.v_scale_stream[strm], io.read(sizeof(float)), 0, sizeof(float));
                continue;
            }
            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(layer.il);
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                ggml_backend_tensor_set(
                        layer.v_scale_stream[strm],
                        io.read(cell_count * sizeof(float)),
                        (head + (size_t) j * cells.size()) * sizeof(float),
                        cell_count * sizeof(float));
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                ggml_backend_tensor_set(v, io.read(cell_count * v_size_row), head * v_size_row, cell_count * v_size_row);
            }

        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    const size_t dst_offset = (head + j * cells.size()) * v_size_el;
                    ggml_backend_tensor_set(v, io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                }
            }

        }
    }

    return true;
}

//
// llama_kv_cache_unified_context
//

llama_kv_cache_unified_context::llama_kv_cache_unified_context(llama_memory_status status) : status(status) {}

llama_kv_cache_unified_context::llama_kv_cache_unified_context(
        llama_kv_cache_unified * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_n_kv_tensor(kv->get_size());

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_unified_context::llama_kv_cache_unified_context(
        llama_kv_cache_unified * kv,
        llama_context * lctx,
        bool do_shift,
        defrag_info dinfo,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), dinfo(std::move(dinfo)), sc_info(std::move(sc_info)) {
    if (!do_shift && this->dinfo.empty() && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_unified_context::llama_kv_cache_unified_context(
        llama_kv_cache_unified * kv,
        llama_kv_cache_unified::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_unified_context::~llama_kv_cache_unified_context() = default;

bool llama_kv_cache_unified_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_unified_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, dinfo, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);

    n_kv = kv->get_n_kv_tensor(kv->get_n_kv());

    return true;
}

llama_memory_status llama_kv_cache_unified_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_unified_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_unified_context::get_n_kv() const {
    return n_kv;
}

bool llama_kv_cache_unified_context::get_supports_set_rows() const {
    return kv->get_supports_set_rows();
}

ggml_tensor * llama_kv_cache_unified_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_unified_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_unified_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    const llama_ubatch * ubatch = i_cur < ubatches.size() ? &ubatches[i_cur] : nullptr;
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur], ubatch);
}

ggml_tensor * llama_kv_cache_unified_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_unified_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_unified_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

void llama_kv_cache_unified_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_unified_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_unified_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_unified_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_unified_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

uint32_t llama_kv_cache_unified::get_padding(const llama_cparams & cparams) {
    if (llama_vcache_nvfp4_runtime_supported(cparams, GGML_TYPE_NVFP4)) {
        return llama_vcache_nvfp4_token_padding(cparams, GGML_TYPE_NVFP4);
    }

    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}
