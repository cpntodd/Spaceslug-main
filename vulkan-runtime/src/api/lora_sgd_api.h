#pragma once
#include <cstdint>

// Updates A[64,R] and B[R,64] in place: parameter -= learning_rate * gradient.
// Row-major fp32; valid 1 <= R <= 8. Returns 0 success, 1 invalid input, 2 GPU failure.
extern "C" int spaceslug_lora_sgd(float* a, float* b, float const* da, float const* db,
                                  float learning_rate, std::uint32_t rank);
