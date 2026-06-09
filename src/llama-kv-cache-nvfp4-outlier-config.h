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
// K-cache layout. The layer capacities are compact sidecar slot counts per KV
// row for each layer; allocation multiplies this by kv_size so outlier entries
// persist for the lifetime of their KV row. Layers listed in
// hybrid_fp8_e4m3_e8m0_32_layers do not allocate NVFP4 outlier sidecars when
// hybrid mode is enabled.
//
// The ctx8192 table was re-derived for the balanced threshold profile from:
//   docs/development/nvfp4-kcache-outlier-thresholds/profiles/ctx8192-capacity/
// with raw evidence in:
//   experiments/20260604T085500Z-balanced-threshold-ctx8192-capacity-derive/
// using the full fourth-case PPL profile at n_ctx=8192. Values are
// max(base_ctx512_capacity, ceil(observed_peak_row_outliers * 1.5)). For this
// balanced-threshold run, the derived ctx8192 capacities match the ctx512
// compact-min capacities and validate with no compact overflow.

static constexpr float llama_nvfp4_kcache_outlier_threshold = 16.0f;

// Balanced full-NVFP4 profile derived from:
//   docs/development/nvfp4-kcache-outlier-thresholds/profiles/balanced/
// using:
//   docs/development/nvfp4-kcache-outlier-thresholds/scripts/derive-kcache-outlier-balanced-config.py
//
// This profile targets closer per-layer outlier counts than the earlier
// density-only profile while avoiding global thresholds that showed large PPL
// regression in the sweep data. It is used when NVFP4 K-cache outlier sidecar is
// enabled without hybrid FP8 K-cache layers and no alternate profile is selected.
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

// New full-NVFP4 profile selected by:
//   LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE=new
//
// It was derived from the real-data max-batch outlier-ratio sweep in:
//   docs/development/nvfp4-kcache-outlier-thresholds/profiles/ratio-1e4/
// Capacity peaks were calibrated with the full PPL run in:
//   docs/development/nvfp4-kcache-outlier-thresholds/profiles/ratio-1e4/full-ppl-capacity-observed.csv
//
// This profile targets per-layer max 512-row batch outlier_ratio ~= 1e-4 using
// the Wikitext PPL baseline input. Capacities are ceil(observed full-PPL
// max-batch outliers * 2.0) from the selected thresholds.
static constexpr float llama_nvfp4_kcache_outlier_layer_thresholds_new[] = {
      214.00f,  42.00f,  19.00f,  16.25f,  42.00f,  72.00f,  26.00f,  15.25f,  40.00f,
       13.00f,  38.00f,  27.00f,  23.00f,  14.50f,  21.00f,  14.50f,  17.00f,  13.50f,
       17.75f,  12.50f,  13.50f,  18.25f,  17.75f,  23.00f,  13.00f,  16.25f,  14.50f,
       15.50f,  14.50f,  15.50f,  20.25f,  15.00f,  15.75f,  13.00f,  20.25f,  30.00f,
};

static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities_new[] = {
      138,  98, 126, 124, 292, 100, 190, 252, 234,
      102, 164, 124,  94, 106, 116, 188, 194, 222,
      234, 234, 322, 170, 156,  82, 164,  84, 124,
      112, 100,  74, 100,  96,  32, 238,  96, 164,
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

static constexpr size_t llama_nvfp4_kcache_outlier_layer_thresholds_new_count =
        sizeof(llama_nvfp4_kcache_outlier_layer_thresholds_new) /
        sizeof(llama_nvfp4_kcache_outlier_layer_thresholds_new[0]);

static constexpr size_t llama_nvfp4_kcache_outlier_layer_capacities_new_count =
        sizeof(llama_nvfp4_kcache_outlier_layer_capacities_new) /
        sizeof(llama_nvfp4_kcache_outlier_layer_capacities_new[0]);

static inline bool llama_nvfp4_kcache_outlier_enabled() {
    const char * value = std::getenv("LLAMA_NVFP4_KCACHE_OUTLIER");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static inline const char * llama_nvfp4_kcache_outlier_threshold_override_env() {
    const char * value = std::getenv("LLAMA_NVFP4_KCACHE_OUTLIER_THRESHOLD");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

static inline bool llama_nvfp4_kcache_outlier_new_profile_enabled() {
    const char * value = std::getenv("LLAMA_NVFP4_KCACHE_OUTLIER_PROFILE");
    return value != nullptr && std::strcmp(value, "new") == 0;
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
    if (hybrid_fp8) {
        return llama_nvfp4_kcache_outlier_layer_capacities_for_ctx(kv_size);
    }
    return llama_nvfp4_kcache_outlier_new_profile_enabled()
                   ? llama_nvfp4_kcache_outlier_layer_capacities_new
                   : llama_nvfp4_kcache_outlier_layer_capacities_balanced;
}

static inline size_t llama_nvfp4_kcache_outlier_layer_capacity_count_for_mode(
        uint32_t kv_size,
        bool hybrid_fp8) {
    if (hybrid_fp8) {
        return llama_nvfp4_kcache_outlier_layer_capacity_count_for_ctx(kv_size);
    }
    return llama_nvfp4_kcache_outlier_new_profile_enabled()
                   ? llama_nvfp4_kcache_outlier_layer_capacities_new_count
                   : llama_nvfp4_kcache_outlier_layer_capacities_balanced_count;
}

static inline const char * llama_nvfp4_kcache_outlier_layer_capacity_profile_for_mode(
        uint32_t kv_size,
        bool hybrid_fp8) {
    if (hybrid_fp8) {
        return llama_nvfp4_kcache_outlier_layer_capacity_profile_for_ctx(kv_size);
    }
    return llama_nvfp4_kcache_outlier_new_profile_enabled() ? "new" : "balanced";
}

static inline float llama_nvfp4_kcache_outlier_layer_threshold(uint32_t layer, bool hybrid_fp8) {
    (void) hybrid_fp8;
    const char * override_value = llama_nvfp4_kcache_outlier_threshold_override_env();
    if (override_value != nullptr) {
        char * end = nullptr;
        const float parsed = std::strtof(override_value, &end);
        if (end != override_value && parsed > 0.0f) {
            return parsed;
        }
    }

    if (!hybrid_fp8 && llama_nvfp4_kcache_outlier_new_profile_enabled()) {
        return (size_t) layer < llama_nvfp4_kcache_outlier_layer_thresholds_new_count
                       ? llama_nvfp4_kcache_outlier_layer_thresholds_new[layer]
                       : llama_nvfp4_kcache_outlier_threshold;
    }

    return (size_t) layer < llama_nvfp4_kcache_outlier_layer_thresholds_balanced_count
                   ? llama_nvfp4_kcache_outlier_layer_thresholds_balanced[layer]
                   : llama_nvfp4_kcache_outlier_threshold;
}

static inline const char * llama_nvfp4_kcache_outlier_layer_threshold_profile(bool hybrid_fp8) {
    if (llama_nvfp4_kcache_outlier_threshold_override_env() != nullptr) {
        return "env-override";
    }
    if (!hybrid_fp8 && llama_nvfp4_kcache_outlier_new_profile_enabled()) {
        return "new";
    }
    return "balanced";
}
