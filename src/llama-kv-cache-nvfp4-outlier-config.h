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
//
// The ctx8192 table was derived from:
//   experiments/20260602T061500Z-kcache-outlier-capacity-nctx-scaling/
// using:
//   scripts/run_hybrid_capacity_nctx.sh 8192
// with V=nvfp4, --no-warmup, --chunks 1, and capacity scale=16 to avoid
// clipping. Values are max(base_ctx512_capacity, ceil(ctx8192_peak * 1.25)).

static constexpr float llama_nvfp4_kcache_outlier_threshold = 16.0f;

static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities[] = {
        0,   0, 418,  72,   0,   0,   0,  68,   0,
       14,   0,   0,   0,  29,   0,  46, 174,  31,
      294,  16, 129, 321, 221,   0,  17,  61,  26,
       48,  28,  29, 883,  30,   8,  18, 751,   0,
};

static constexpr uint32_t llama_nvfp4_kcache_outlier_layer_capacities_ctx8192[] = {
        0,   0, 695,  72,   0,   0,   0,  68,   0,
       14,   0,   0,   0,  29,   0,  46, 174,  31,
      409,  16, 129, 564, 580,   0,  17, 128,  26,
       77,  28,  29,1664,  30,   8,  18,1667,   0,
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
