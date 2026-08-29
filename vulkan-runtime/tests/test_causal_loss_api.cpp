#include "api/causal_loss_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
int main() {
    constexpr std::uint32_t rows = 3, vocab = 8;
    std::vector<float> logits(rows * vocab), grad(rows * vocab), grad_ref(rows * vocab, 0.0f), loss(rows), loss_ref(rows, 0.0f);
    std::vector<std::uint32_t> targets{2, 7, 1}, mask{1, 0, 1};
    for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = std::sin(float(i) * 0.17f);
    for (std::uint32_t row = 0; row < rows; ++row) { float maximum = *std::max_element(logits.begin() + row * vocab, logits.begin() + (row + 1) * vocab); float normalizer = 0.0f; for (std::uint32_t col = 0; col < vocab; ++col) normalizer += std::exp(logits[row * vocab + col] - maximum); for (std::uint32_t col = 0; col < vocab; ++col) grad_ref[row * vocab + col] = mask[row] ? std::exp(logits[row * vocab + col] - maximum) / normalizer - (col == targets[row]) : 0.0f; if (mask[row]) loss_ref[row] = std::log(normalizer) + maximum - logits[row * vocab + targets[row]]; }
    if (spaceslug_causal_loss(logits.data(), targets.data(), mask.data(), grad.data(), loss.data(), rows, vocab) != 0) return 1;
    float max_error = 0.0f; for (std::size_t i = 0; i < grad.size(); ++i) max_error = std::max(max_error, std::abs(grad[i] - grad_ref[i])); for (std::size_t i = 0; i < loss.size(); ++i) max_error = std::max(max_error, std::abs(loss[i] - loss_ref[i])); if (max_error > 3e-5f) { std::cerr << "causal loss mismatch " << max_error << '\n'; return 1; } std::cout << "Causal loss: PASS max_abs_error=" << max_error << '\n';
}
