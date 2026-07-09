#include "fp8-e4m3-e8m0.h"

#include "../../../ggml/src/ggml-quants.h"

#include <stdexcept>

namespace llama_expt {
namespace {

std::vector<float> fp8_e4m3_e8m0_32_roundtrip_rows(
        const std::vector<float> & input,
        size_t row_size) {
    if (row_size == 0 || row_size % QK_FP8_E4M3_E8M0_32 != 0 || input.size() % row_size != 0) {
        throw std::runtime_error("FP8 E4M3+E8M0 block32 roundtrip requires rows divisible by block size");
    }

    std::vector<float> output(input.size());
    std::vector<block_fp8_e4m3_e8m0_32> quantized(row_size / QK_FP8_E4M3_E8M0_32);
    const size_t rows = input.size() / row_size;
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * row_size;
        quantize_row_fp8_e4m3_e8m0_32_ref(input.data() + offset, quantized.data(), (int64_t) row_size);
        dequantize_row_fp8_e4m3_e8m0_32(quantized.data(), output.data() + offset, (int64_t) row_size);
    }

    return output;
}

quant_round_tensor_metadata make_fp8_e4m3_e8m0_32_metadata(const char * role) {
    quant_round_tensor_metadata metadata;
    metadata.mode = std::string("fp8_e4m3_e8m0_32_") + role;
    metadata.integer_fields["block_size"] = QK_FP8_E4M3_E8M0_32;
    metadata.string_fields["scale"] = "e8m0_per_block";
    metadata.string_fields["value_format"] = "e4m3";
    return metadata;
}

class fp8_e4m3_e8m0_attention_quant_round_algo : public attention_quant_round_algo {
public:
    std::string name() const override {
        return "fp8_e4m3_e8m0_32";
    }

    attention_quant_round_result quant_round(const attention_quant_round_input & input) const override {
        attention_quant_round_result result;

        result.k.values = fp8_e4m3_e8m0_32_roundtrip_rows(input.k_values, (size_t) input.k_record.ne[0]);
        result.k.metadata = make_fp8_e4m3_e8m0_32_metadata("k");

        result.q.values = fp8_e4m3_e8m0_32_roundtrip_rows(input.q_values, (size_t) input.q_record.ne[0]);
        result.q.metadata = make_fp8_e4m3_e8m0_32_metadata("q");

        return result;
    }
};

} // namespace

std::unique_ptr<attention_quant_round_algo> make_fp8_e4m3_e8m0_attention_quant_round_algo() {
    return std::unique_ptr<attention_quant_round_algo>(new fp8_e4m3_e8m0_attention_quant_round_algo());
}

} // namespace llama_expt
