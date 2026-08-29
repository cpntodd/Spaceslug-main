#include "api/lora_delta_api.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t m = 3, rank = 4;
    std::vector<float> x(m * 64), a(64 * rank), b(rank * 64), y(m * 64), reference(m * 64, 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.07f);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::cos(float(i) * 0.11f) * 0.1f;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(float(i) * 0.13f) * 0.1f;
    for (std::uint32_t row = 0; row < m; ++row) for (std::uint32_t column = 0; column < 64; ++column) for (std::uint32_t r = 0; r < rank; ++r) for (std::uint32_t k = 0; k < 64; ++k) reference[row * 64 + column] += x[row * 64 + k] * a[k * rank + r] * b[r * 64 + column];
    if (spaceslug_lora_delta(x.data(), a.data(), b.data(), y.data(), m, rank) != 0) return 1;
    float max_error = 0.0f;
    for (std::size_t i = 0; i < y.size(); ++i) max_error = std::max(max_error, std::abs(y[i] - reference[i]));
    if (max_error > 2e-4f) { std::cerr << "LoRA delta mismatch: " << max_error << '\n'; return 1; }
    std::cout << "LoRA delta: PASS max_abs_error=" << max_error << '\n';
}
