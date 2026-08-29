#pragma once
#include <cstdint>

// Executes one fixed Tiny LoRA GPU update sequence: delta forward, dA/dB, SGD.
// X/dY are [M,64], A[64,R], B[R,64]; A/B are updated in place. Forward delta
// is returned in Y[M,64]. Valid: 1<=M<=128, 1<=R<=8, fp32 row-major.
// Returns 0 success, 1 invalid input, 2 if a constituent Vulkan operation fails.
extern "C" int spaceslug_lora_train_step(float const* x, float const* dy,
                                         float* a, float* b, float* y,
                                         float learning_rate, std::uint32_t m,
                                         std::uint32_t rank);
