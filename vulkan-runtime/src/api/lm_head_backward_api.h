#pragma once
#include <cstdint>
extern "C" int spaceslug_lm_head_backward(float const* dlogits, float const* weight, float* dprojected, std::uint32_t rows, std::uint32_t vocab, std::uint32_t hidden);
