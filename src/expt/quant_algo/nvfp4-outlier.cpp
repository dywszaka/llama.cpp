#include "nvfp4-outlier.h"

#include "../../llama-kv-cache-nvfp4-outlier-config.h"

#include "../../../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llama_expt {
namespace {

constexpr float NVFP4_GLOBAL_SCALE_MAX = 1344.0f;

float nvfp4_global_scale_from_amax(float amax) {
    return (amax > 0.0f && std::isfinite(amax)) ? (NVFP4_GLOBAL_SCALE_MAX / amax) : 0.0f;
}

std::vector<float> nvfp4_roundtrip_rows_dynamic(
        const std::vector<float> & input,
        size_t row_size,
        std::vector<float> * row_global_scales = nullptr) {
    if (row_size == 0 || row_size % QK_NVFP4 != 0 || input.size() % row_size != 0) {
        throw std::runtime_error("NVFP4 dynamic row roundtrip requires rows divisible by block size");
    }

    const size_t rows = input.size() / row_size;
    std::vector<float> output(input.size());
    std::vector<block_nvfp4> quantized(row_size / QK_NVFP4);
    if (row_global_scales) {
        row_global_scales->resize(rows);
    }

    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * row_size;
        float amax = 0.0f;
        for (size_t col = 0; col < row_size; ++col) {
            amax = std::max(amax, std::fabs(input[offset + col]));
        }
        const float global_scale = nvfp4_global_scale_from_amax(amax);
        if (row_global_scales) {
            (*row_global_scales)[row] = global_scale;
        }
        quantize_row_nvfp4_ref(input.data() + offset, quantized.data(), (int64_t) row_size, global_scale);
        dequantize_row_nvfp4(quantized.data(), output.data() + offset, (int64_t) row_size, global_scale);
    }

    return output;
}

std::vector<float> nvfp4_roundtrip_k_outlier(
        const std::vector<float> & input,
        size_t row_size,
        float threshold,
        float global_scale,
        size_t & outlier_count) {
    if (row_size == 0 || row_size % QK_NVFP4 != 0 || input.size() % row_size != 0) {
        throw std::runtime_error("NVFP4 K outlier roundtrip requires rows divisible by block size");
    }

    outlier_count = 0;
    std::vector<float> residual(input);
    for (float & value : residual) {
        if (std::fabs(value) > threshold) {
            value = 0.0f;
            ++outlier_count;
        }
    }

    std::vector<float> output(input.size());
    std::vector<block_nvfp4> quantized(row_size / QK_NVFP4);
    const size_t rows = input.size() / row_size;
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * row_size;
        quantize_row_nvfp4_ref(residual.data() + offset, quantized.data(), (int64_t) row_size, global_scale);
        dequantize_row_nvfp4(quantized.data(), output.data() + offset, (int64_t) row_size, global_scale);
        for (size_t col = 0; col < row_size; ++col) {
            const float original = input[offset + col];
            if (std::fabs(original) > threshold) {
                output[offset + col] = original;
            }
        }
    }

    return output;
}

class nvfp4_outlier_attention_quant_round_algo : public attention_quant_round_algo {
public:
    std::string name() const override {
        return "nvfp4_outlier";
    }

    attention_quant_round_result quant_round(const attention_quant_round_input & input) const override {
        attention_quant_round_result result;

        const float k_threshold = llama_nvfp4_kcache_outlier_layer_threshold(input.layer, false);
        const float k_global_scale = nvfp4_global_scale_from_amax(k_threshold);
        size_t k_outlier_count = 0;

        result.k.values = nvfp4_roundtrip_k_outlier(
                input.k_values,
                (size_t) input.k_record.ne[0],
                k_threshold,
                k_global_scale,
                k_outlier_count);
        result.k.metadata.mode = "nvfp4_outlier_threshold_layer0";
        result.k.metadata.integer_fields["layer"] = input.layer;
        result.k.metadata.number_fields["threshold"] = k_threshold;
        result.k.metadata.number_fields["global_scale"] = k_global_scale;
        result.k.metadata.integer_fields["outlier_count"] = k_outlier_count;

        result.q.values = nvfp4_roundtrip_rows_dynamic(input.q_values, (size_t) input.q_record.ne[0]);
        result.q.metadata.mode = "nvfp4_dynamic_row_amax";
        result.q.metadata.string_fields["global_scale"] = "dynamic_per_row_amax";

        return result;
    }
};

} // namespace

std::unique_ptr<attention_quant_round_algo> make_nvfp4_outlier_attention_quant_round_algo() {
    return std::unique_ptr<attention_quant_round_algo>(new nvfp4_outlier_attention_quant_round_algo());
}

} // namespace llama_expt
