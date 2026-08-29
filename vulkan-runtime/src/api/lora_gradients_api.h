#pragma once
#include <cstdint>

// X/dY are [M,64], A[64,R], B[R,64]; dA[64,R], dB[R,64], row-major fp32.
// Computes dA=X^T*dY*B^T and dB=(X*A)^T*dY for 1<=M<=128, 1<=R<=8.
extern "C" int spaceslug_lora_gradients(float const* x, float const* dy, float const* a,
                                        float const* b, float* da, float* db,
                                        std::uint32_t m, std::uint32_t rank);
