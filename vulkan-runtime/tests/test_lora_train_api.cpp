#include "api/lora_train_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t m = 3, rank = 4;
    constexpr float learning_rate = 0.05f;
    std::vector<float> x(m * 64), dy(m * 64), a(64 * rank), b(rank * 64), y(m * 64), a_ref(64 * rank), b_ref(rank * 64), y_ref(m * 64, 0.0f), da(64 * rank, 0.0f), db(rank * 64, 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = std::sin(float(i) * 0.07f); dy[i] = std::cos(float(i) * 0.03f); }
    for (std::size_t i = 0; i < a.size(); ++i) a_ref[i] = a[i] = std::cos(float(i) * 0.11f) * 0.1f;
    for (std::size_t i = 0; i < b.size(); ++i) b_ref[i] = b[i] = std::sin(float(i) * 0.13f) * 0.1f;
    for (std::uint32_t row = 0; row < m; ++row) for (std::uint32_t column = 0; column < 64; ++column) for (std::uint32_t r = 0; r < rank; ++r) for (std::uint32_t k = 0; k < 64; ++k) y_ref[row * 64 + column] += x[row * 64 + k] * a_ref[k * rank + r] * b_ref[r * 64 + column];
    for (std::uint32_t k = 0; k < 64; ++k) for (std::uint32_t r = 0; r < rank; ++r) for (std::uint32_t row = 0; row < m; ++row) for (std::uint32_t column = 0; column < 64; ++column) da[k * rank + r] += x[row * 64 + k] * dy[row * 64 + column] * b_ref[r * 64 + column];
    for (std::uint32_t r = 0; r < rank; ++r) for (std::uint32_t column = 0; column < 64; ++column) for (std::uint32_t row = 0; row < m; ++row) { float reduced = 0.0f; for (std::uint32_t k = 0; k < 64; ++k) reduced += x[row * 64 + k] * a_ref[k * rank + r]; db[r * 64 + column] += reduced * dy[row * 64 + column]; }
    for (std::size_t i = 0; i < a_ref.size(); ++i) a_ref[i] -= learning_rate * da[i];
    for (std::size_t i = 0; i < b_ref.size(); ++i) b_ref[i] -= learning_rate * db[i];
    if (spaceslug_lora_train_step(x.data(), dy.data(), a.data(), b.data(), y.data(), learning_rate, m, rank) != 0) return 1;
    float max_error = 0.0f; for (std::size_t i = 0; i < y.size(); ++i) max_error = std::max(max_error, std::abs(y[i] - y_ref[i])); for (std::size_t i = 0; i < a.size(); ++i) max_error = std::max(max_error, std::abs(a[i] - a_ref[i])); for (std::size_t i = 0; i < b.size(); ++i) max_error = std::max(max_error, std::abs(b[i] - b_ref[i]));
    if (max_error > 3e-4f) { std::cerr << "LoRA train-step mismatch: " << max_error << '\n'; return 1; }
    std::cout << "LoRA train step: PASS max_abs_error=" << max_error << '\n';
}
