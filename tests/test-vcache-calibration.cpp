#include "../src/llama-vcache-calibration.h"

#include <cmath>
#include <cstdio>
#include <vector>

static bool approx(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

int main() {
    int layer = -1;

    if (!llama_vcache_parse_vcur_layer("Vcur-7", layer) || layer != 7) {
        std::fprintf(stderr, "failed to parse Vcur-7\n");
        return 1;
    }

    if (llama_vcache_parse_vcur_layer("Qcur-7", layer)) {
        std::fprintf(stderr, "unexpected parse success for Qcur-7\n");
        return 1;
    }

    llama_vcache_layer_stats stats;
    const std::vector<float> values = { 0.0f, -0.25f, 0.5f, -1.0f, 2.0f, -4.0f, 8.0f };
    stats.add(values.data(), values.size());

    if (stats.sample_count != values.size()) {
        std::fprintf(stderr, "unexpected sample count: %llu\n", (unsigned long long) stats.sample_count);
        return 1;
    }

    if (!approx(stats.abs_max, 8.0, 1e-6)) {
        std::fprintf(stderr, "unexpected abs max: %f\n", stats.abs_max);
        return 1;
    }

    const double p50 = stats.quantile_abs(0.50);
    const double p99 = stats.quantile_abs(0.99);

    if (p50 < 0.4 || p50 > 2.5) {
        std::fprintf(stderr, "unexpected p50: %f\n", p50);
        return 1;
    }

    if (p99 < 4.0 || p99 > 8.5) {
        std::fprintf(stderr, "unexpected p99: %f\n", p99);
        return 1;
    }

    const auto rec = stats.recommendation();
    if (rec.range_p999 < rec.range_p99) {
        std::fprintf(stderr, "unexpected recommendation ordering: p99=%f p999=%f\n", rec.range_p99, rec.range_p999);
        return 1;
    }
    if (rec.scale_p999 <= 0.0f) {
        std::fprintf(stderr, "unexpected p999 scale: %f\n", rec.scale_p999);
        return 1;
    }

    std::puts("test-vcache-calibration: ok");
    return 0;
}
