#pragma once
#include <cstdint>
extern "C" int spaceslug_attention_causal_backward(float const* q, float const* k, float const* v, float const* d_output, float* output, std::uint32_t tokens, std::uint32_t hidden, std::uint32_t mode);
