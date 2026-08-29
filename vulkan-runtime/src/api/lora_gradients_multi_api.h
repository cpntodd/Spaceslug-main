#pragma once
#include <cstdint>
// Four packed adapter sets: A[4,64,R], B[4,R,64], outputs dA/dB same layout.
extern "C" int spaceslug_lora_gradients_multi(float const* x, float const* dy, float const* a, float const* b, float* da, float* db, std::uint32_t rows, std::uint32_t hidden, std::uint32_t rank, std::uint32_t target);
