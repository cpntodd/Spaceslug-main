#pragma once
#include <cstddef>
#include <cstdint>

// Validated fp32 row-major C[MxN] = A[MxK] * B[KxN] operation.
// Dimensions must satisfy M,N divisible by 64 and K divisible by 32.
extern "C" int spaceslug_sgemm(float const* a, float const* b, float* c,
                                std::uint32_t m, std::uint32_t n, std::uint32_t k,
                                float* max_relative_error);
