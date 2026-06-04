#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

// Model-dependent NVFP4 K-cache outlier configuration.
//
// The values below were derived for the model/run captured in:
//   experiments/20260601T042900Z-kcache-outlier-threshold-hybrid-ppl/
// using:
//   scripts/run_hybrid_threshold16_compact_min.sh
// and the `hybrid_threshold16_compact_min` result in summary.md.
//
// Recompute these values when changing model, context size, prompt/data mix, or
// K-cache layout. The layer capacities are compact sidecar entry counts per
// layer. Layers listed in hybrid_fp8_e4m3_e8m0_32_layers do not allocate NVFP4
// outlier sidecars when hybrid mode is enabled.
//
// The ctx8192 table was re-derived for the balanced threshold profile from:
//   experiments/20260604T085500Z-balanced-threshold-ctx8192-capacity-derive/
// using the full fourth-case PPL profile at n_ctx=8192. Values are
// max(base_ctx512_capacity, ceil(observed_peak_compact_used * 1.5)). For this
// balanced-threshold run, the derived ctx8192 capacities match the ctx512
// compact-min capacities and validate with no compact overflow.

static constexpr float llama_nvfp4_kcache_outlier_threshold = 16.0f;

// Balanced full-NVFP4 profile derived from:
//   experiments/20260603T032033Z-kcache-outlier-balanced-threshold-config/
// using:
//   scripts/derive-kcache-outlier-balanced-config.py
//
// This profile targets closer per-layer outlier counts than the earlier
// density-only profile while avoiding global thresholds that showed large PPL
// regression in the sweep data. It is used when NVFP4 K-cache outlier sidecar is
// enabled without hybrid FP8 K-cache layers.
static constexpr float llama_nvfp4_kcache_outlier_layer_thresholds_balanced[] = {
      256.0f,  48.0f,  24.0f,  32.0f,  48.0f, 192.0f,  32.0f,  24.0f, 192.0f,
       24.0f, 192.0f,  32.0f,  24.0f,  24.0f,  24.0f,  24.0f,  24.0f,  32.0f,
       32.0f,  24.0f,  32.0f,  32.0f,  24.0f,  24.0f,  24.0f,  32.0f,  32.0f,
       32.0f,  32.0f,  32.0f,  32.0f,  32.0f,  32.0f,  32.0f,  32.0f,  32.0f,
};

static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities_balanced[] = {
        1,  2,  2,  1, 14,  1,  2,  5,  1,
        2,  1,  5, 24,  5,  5,  3,  3,  1,
        1,  4,  1,  1,  4, 29,  3,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1, 25,
};

static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities[] = {
        0,   0, 418,  72,   0,   0,   0,  68,   0,
       14,   0,   0,   0,  29,   0,  46, 174,  31,
      294,  16, 129, 321, 221,   0,  17,  61,  26,
       48,  28,  29, 883,  30,   8,  18, 751,   0,
};

static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities_ctx8192[] = {
        0,   0, 418,  72,   0,   0,   0,  68,   0,
       14,   0,   0,   0,  29,   0,  46, 174,  31,
      294,  16, 129, 321, 221,   0,  17,  61,  26,
       48,  28,  29, 883,  30,   8,  18, 751,   0,
};

static constexpr uint32_t llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layers[] = {
        0, 1, 4, 5, 6, 8, 10, 11, 12, 14, 23, 35,
};

static constexpr size_t llama_nvfp4_kcache_outlier_layer_capacity_count =
        sizeof(llama_nvfp4_kcache_outlier_layer_capacities) /
        sizeof(llama_nvfp4_kcache_outlier_layer_capacities[0]);

static constexpr size_t llama_nvfp4_kcache_outlier_layer_capacity_ctx8192_count =
        sizeof(llama_nvfp4_kcache_outlier_layer_capacities_ctx8192) /
        sizeof(llama_nvfp4_kcache_outlier_layer_capacities_ctx8192[0]);

static constexpr size_t llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layer_count =
        sizeof(llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layers) /
        sizeof(llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layers[0]);

static constexpr size_t llama_nvfp4_kcache_outlier_layer_thresholds_balanced_count =
        sizeof(llama_nvfp4_kcache_outlier_layer_thresholds_balanced) /
        sizeof(llama_nvfp4_kcache_outlier_layer_thresholds_balanced[0]);

static constexpr size_t llama_nvfp4_kcache_outlier_layer_capacities_balanced_count =
        sizeof(llama_nvfp4_kcache_outlier_layer_capacities_balanced) /
        sizeof(llama_nvfp4_kcache_outlier_layer_capacities_balanced[0]);

static inline bool llama_nvfp4_kcache_outlier_enabled() {
    const char * value = std::getenv("LLAMA_NVFP4_KCACHE_OUTLIER");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static inline const char * llama_nvfp4_kcache_hybrid_fp8_layers_env() {
    const char * value = std::getenv("LLAMA_NVFP4_KCACHE_OUTLIER_HYBRID_FP8");
    if (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) {
        return "high_medium";
    }

    return std::getenv("LLAMA_KCACHE_HYBRID_FP8_E4M3_E8M0_32_LAYERS");
}

static inline const uint32_t * llama_nvfp4_kcache_outlier_layer_capacities_for_ctx(uint32_t kv_size) {
    return kv_size >= 8192 ? llama_nvfp4_kcache_outlier_layer_capacities_ctx8192
                           : llama_nvfp4_kcache_outlier_layer_capacities;
}

static inline size_t llama_nvfp4_kcache_outlier_layer_capacity_count_for_ctx(uint32_t kv_size) {
    return kv_size >= 8192 ? llama_nvfp4_kcache_outlier_layer_capacity_ctx8192_count
                           : llama_nvfp4_kcache_outlier_layer_capacity_count;
}

static inline const char * llama_nvfp4_kcache_outlier_layer_capacity_profile_for_ctx(uint32_t kv_size) {
    return kv_size >= 8192 ? "ctx8192" : "ctx512";
}

static inline const uint32_t * llama_nvfp4_kcache_outlier_layer_capacities_for_mode(
        uint32_t kv_size,
        bool hybrid_fp8) {
    return hybrid_fp8 ? llama_nvfp4_kcache_outlier_layer_capacities_for_ctx(kv_size)
                      : llama_nvfp4_kcache_outlier_layer_capacities_balanced;
}

static inline size_t llama_nvfp4_kcache_outlier_layer_capacity_count_for_mode(
        uint32_t kv_size,
        bool hybrid_fp8) {
    return hybrid_fp8 ? llama_nvfp4_kcache_outlier_layer_capacity_count_for_ctx(kv_size)
                      : llama_nvfp4_kcache_outlier_layer_capacities_balanced_count;
}

static inline const char * llama_nvfp4_kcache_outlier_layer_capacity_profile_for_mode(
        uint32_t kv_size,
        bool hybrid_fp8) {
    return hybrid_fp8 ? llama_nvfp4_kcache_outlier_layer_capacity_profile_for_ctx(kv_size)
                      : "balanced";
}

static inline float llama_nvfp4_kcache_outlier_layer_threshold(uint32_t layer, bool hybrid_fp8) {
    (void) hybrid_fp8;
    return (size_t) layer < llama_nvfp4_kcache_outlier_layer_thresholds_balanced_count
                   ? llama_nvfp4_kcache_outlier_layer_thresholds_balanced[layer]
                   : llama_nvfp4_kcache_outlier_threshold;
}

static inline const char * llama_nvfp4_kcache_outlier_layer_threshold_profile(bool hybrid_fp8) {
    (void) hybrid_fp8;
    return "balanced";
}
