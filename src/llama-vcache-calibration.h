#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

struct llama_vcache_layer_recommendation {
    float range_p99   = 0.0f;
    float range_p999  = 0.0f;
    float range_p9999 = 0.0f;

    float scale_p99   = 0.0f;
    float scale_p999  = 0.0f;
    float scale_p9999 = 0.0f;
};

struct llama_vcache_layer_stats {
    static constexpr int HISTOGRAM_BINS = 2048;

    uint64_t sample_count = 0;
    uint64_t zero_count = 0;
    uint64_t nonfinite_count = 0;
    uint64_t overflow_count = 0;

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float  abs_max = 0.0f;

    std::array<uint64_t, HISTOGRAM_BINS> histogram = {};

    void add(float v);
    void add(const float * data, size_t n);

    double quantile_abs(double q) const;
    llama_vcache_layer_recommendation recommendation() const;
};

bool llama_vcache_parse_vcur_layer(const char * name, int & layer);
