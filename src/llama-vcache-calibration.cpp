#include "llama-vcache-calibration.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

constexpr double kHistMinLog2 = -32.0;
constexpr double kHistMaxLog2 =  16.0;
constexpr double kHistLog2Span = kHistMaxLog2 - kHistMinLog2;

static int histogram_bin_for_abs(double av) {
    if (av <= 0.0) {
        return -1;
    }

    const double l2 = std::log2(av);
    if (l2 >= kHistMaxLog2) {
        return llama_vcache_layer_stats::HISTOGRAM_BINS - 1;
    }
    if (l2 <= kHistMinLog2) {
        return 0;
    }

    const double pos = (l2 - kHistMinLog2) / kHistLog2Span;
    int idx = (int) std::floor(pos * llama_vcache_layer_stats::HISTOGRAM_BINS);
    idx = std::max(0, std::min(llama_vcache_layer_stats::HISTOGRAM_BINS - 1, idx));
    return idx;
}

static double histogram_upper_bound(int idx) {
    const double pos_hi = (double) (idx + 1) / llama_vcache_layer_stats::HISTOGRAM_BINS;
    return std::exp2(kHistMinLog2 + pos_hi * kHistLog2Span);
}

static float scale_for_range(double range) {
    return range > 0.0 ? (float) (448.0 / range) : 0.0f;
}

} // namespace

void llama_vcache_layer_stats::add(float v) {
    ++sample_count;

    if (!std::isfinite(v)) {
        ++nonfinite_count;
        return;
    }

    const double av = std::fabs((double) v);
    if (av == 0.0) {
        ++zero_count;
        return;
    }

    sum_abs += av;
    sum_sq += av * av;
    abs_max = std::max(abs_max, (float) av);

    if (std::log2(av) > kHistMaxLog2) {
        ++overflow_count;
    }

    histogram[(size_t) histogram_bin_for_abs(av)]++;
}

void llama_vcache_layer_stats::add(const float * data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        add(data[i]);
    }
}

double llama_vcache_layer_stats::quantile_abs(double q) const {
    if (sample_count == 0) {
        return 0.0;
    }

    q = std::max(0.0, std::min(1.0, q));
    const uint64_t target = (uint64_t) std::ceil(q * sample_count);

    if (target <= zero_count) {
        return 0.0;
    }

    uint64_t seen = zero_count;
    for (int i = 0; i < HISTOGRAM_BINS; ++i) {
        seen += histogram[(size_t) i];
        if (seen >= target) {
            return std::min<double>(histogram_upper_bound(i), abs_max);
        }
    }

    return abs_max;
}

llama_vcache_layer_recommendation llama_vcache_layer_stats::recommendation() const {
    llama_vcache_layer_recommendation rec;

    rec.range_p99   = (float) quantile_abs(0.99);
    rec.range_p999  = (float) quantile_abs(0.999);
    rec.range_p9999 = (float) quantile_abs(0.9999);

    rec.scale_p99   = scale_for_range(rec.range_p99);
    rec.scale_p999  = scale_for_range(rec.range_p999);
    rec.scale_p9999 = scale_for_range(rec.range_p9999);

    return rec;
}

bool llama_vcache_parse_vcur_layer(const char * name, int & layer) {
    layer = -1;

    if (name == nullptr) {
        return false;
    }

    static constexpr const char * prefix = "Vcur-";
    static constexpr size_t prefix_len = 5;
    if (std::strncmp(name, prefix, prefix_len) != 0) {
        return false;
    }

    char * end = nullptr;
    const long parsed = std::strtol(name + prefix_len, &end, 10);
    if (end == name + prefix_len || *end != '\0' || parsed < 0) {
        return false;
    }

    layer = (int) parsed;
    return true;
}
