#pragma once

#include "../tensor-export-eval.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llama_expt {

struct attention_quant_round_input {
    const tensor_record & k_record;
    const tensor_record & q_record;
    const std::vector<float> & k_values;
    const std::vector<float> & q_values;
    uint32_t layer = 0;
};

struct attention_quant_round_tensor_result {
    std::vector<float> values;
    quant_round_tensor_metadata metadata;
};

struct attention_quant_round_result {
    attention_quant_round_tensor_result k;
    attention_quant_round_tensor_result q;
};

class attention_quant_round_algo {
public:
    virtual ~attention_quant_round_algo() = default;

    virtual std::string name() const = 0;
    virtual attention_quant_round_result quant_round(const attention_quant_round_input & input) const = 0;
};

} // namespace llama_expt
