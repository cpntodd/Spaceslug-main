#pragma once

#include <cstdint>

// Computes Y[M,64] = X[M,64] * A[64,R] * B[R,64], all row-major fp32.
// Valid shapes: 1 <= M <= 128 and 1 <= R <= 8.
// Returns 0 on success, 1 for invalid arguments, 2 for Vulkan failure.
extern "C" int spaceslug_lora_delta(float const* x,
                                    float const* a,
                                    float const* b,
                                    float* y,
                                    std::uint32_t m,
                                    std::uint32_t rank);
