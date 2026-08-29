#pragma once
#include <cstdint>

// Computes masked causal CE and dLogits for logits[rows,vocab]. dLogits is
// unnormalized per-row; callers divide by included target count. Vocab <= 320.
extern "C" int spaceslug_causal_loss(float const* logits, std::uint32_t const* targets,
                                     std::uint32_t const* mask, float* dlogits,
                                     float* row_loss, std::uint32_t rows,
                                     std::uint32_t vocab);
