#pragma once
#include <cstdint>
extern "C" int spaceslug_projection_backward(float const* d_output, float const* weight, float* d_input, std::uint32_t rows, std::uint32_t input_size, std::uint32_t output_size);
