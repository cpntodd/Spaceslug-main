#include "api/lora_gradients_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t m = 3, rank = 4;
    std::vector<float> x(m * 64), dy(m * 64), a(64 * rank), b(rank * 64), da(64 * rank), db(rank * 64), da_ref(64 * rank, 0.0f), db_ref(rank * 64, 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = std::sin(float(i) * 0.07f); dy[i] = std::cos(float(i) * 0.03f); }
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::cos(float(i) * 0.11f) * 0.1f;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = std::sin(float(i) * 0.13f) * 0.1f;
    for (std::uint32_t k = 0; k < 64; ++k) for (std::uint32_t r = 0; r < rank; ++r) for (std::uint32_t row = 0; row < m; ++row) for (std::uint32_t column = 0; column < 64; ++column) da_ref[k * rank + r] += x[row * 64 + k] * dy[row * 64 + column] * b[r * 64 + column];
    for (std::uint32_t r = 0; r < rank; ++r) for (std::uint32_t column = 0; column < 64; ++column) for (std::uint32_t row = 0; row < m; ++row) { float reduced = 0.0f; for (std::uint32_t k = 0; k < 64; ++k) reduced += x[row * 64 + k] * a[k * rank + r]; db_ref[r * 64 + column] += reduced * dy[row * 64 + column]; }
    if (spaceslug_lora_gradients(x.data(), dy.data(), a.data(), b.data(), da.data(), db.data(), m, rank) != 0) return 1;
    float max_error = 0.0f; for (std::size_t i = 0; i < da.size(); ++i) max_error = std::max(max_error, std::abs(da[i] - da_ref[i])); for (std::size_t i = 0; i < db.size(); ++i) max_error = std::max(max_error, std::abs(db[i] - db_ref[i]));
    if (max_error > 3e-4f) { std::cerr << "LoRA gradients mismatch: " << max_error << '\n'; return 1; }
    std::cout << "LoRA gradients: PASS max_abs_error=" << max_error << '\n';
}
