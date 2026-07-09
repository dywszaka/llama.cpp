#pragma once

#include "attention-quant-round.h"

#include <memory>

namespace llama_expt {

std::unique_ptr<attention_quant_round_algo> make_fp8_e4m3_e8m0_attention_quant_round_algo();

} // namespace llama_expt
