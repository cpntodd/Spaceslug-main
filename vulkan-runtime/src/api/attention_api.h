#pragma once

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

// Fixed-shape fp32 scaled-dot-product attention:
//   output = softmax(Q * K^T / sqrt(D)) * V
// with T=128, D=64, row-major Q/K/V/output arrays.
// Returns 0 on success, 1 for invalid arguments, or 2 for a Vulkan/runtime error.
int spaceslug_attention(float const* q, float const* k, float const* v, float* output);

#ifdef __cplusplus
}

constexpr std::uint32_t SPACESLUG_ATTENTION_T = 128;
constexpr std::uint32_t SPACESLUG_ATTENTION_D = 64;
constexpr std::uint32_t SPACESLUG_ATTENTION_FLOATS = SPACESLUG_ATTENTION_T * SPACESLUG_ATTENTION_D;
static_assert(SPACESLUG_ATTENTION_FLOATS == 8192);
#endif
