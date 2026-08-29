#pragma once

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

// Padded fp32 causal attention. q/k/v/output point to 128*64 row-major
// floats; only the prefix [0,T) is valid and each row t attends to k <= t.
// D must be 64 and T must be in [1,128]. Returns 0 on success, 1 for invalid
// arguments, or 2 for a Vulkan/runtime error.
int spaceslug_attention_causal(float const* q, float const* k, float const* v, float* output, uint32_t T, uint32_t D);

#ifdef __cplusplus
}
constexpr std::uint32_t SPACESLUG_ATTENTION_CAUSAL_T = 128;
constexpr std::uint32_t SPACESLUG_ATTENTION_CAUSAL_D = 64;
constexpr std::uint32_t SPACESLUG_ATTENTION_CAUSAL_FLOATS = SPACESLUG_ATTENTION_CAUSAL_T * SPACESLUG_ATTENTION_CAUSAL_D;
#endif
