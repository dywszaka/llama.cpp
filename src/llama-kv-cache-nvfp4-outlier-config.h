#pragma once

#include <cstdint>
#include <cstddef>

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

static constexpr float llama_nvfp4_kcache_outlier_threshold = 16.0f;

static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities[] = {
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

static constexpr size_t llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layer_count =
        sizeof(llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layers) /
        sizeof(llama_nvfp4_kcache_hybrid_fp8_e4m3_e8m0_32_layers[0]);
