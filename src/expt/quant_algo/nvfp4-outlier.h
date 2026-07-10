#pragma once

#include "attention-quant-round.h"

#include <memory>

namespace llama_expt {

std::unique_ptr<attention_quant_round_algo> make_nvfp4_outlier_attention_quant_round_algo();

} // namespace llama_expt
