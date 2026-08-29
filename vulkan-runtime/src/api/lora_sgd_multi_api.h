#pragma once
#include <cstdint>
extern "C" int spaceslug_lora_sgd_multi(float* a, float* b, float const* da, float const* db, float learning_rate, std::uint32_t rank);
