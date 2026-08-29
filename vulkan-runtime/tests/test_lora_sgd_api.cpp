#include "api/lora_sgd_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t rank = 4;
    constexpr float learning_rate = 0.05f;
    std::vector<float> a(64 * rank), b(64 * rank), da(64 * rank), db(64 * rank), a_ref(64 * rank), b_ref(64 * rank);
    for (std::size_t i = 0; i < a.size(); ++i) { a[i] = std::sin(float(i) * 0.07f); b[i] = std::cos(float(i) * 0.03f); da[i] = std::cos(float(i) * 0.11f); db[i] = std::sin(float(i) * 0.13f); a_ref[i] = a[i] - learning_rate * da[i]; b_ref[i] = b[i] - learning_rate * db[i]; }
    if (spaceslug_lora_sgd(a.data(), b.data(), da.data(), db.data(), learning_rate, rank) != 0) return 1;
    float max_error = 0.0f; for (std::size_t i = 0; i < a.size(); ++i) { max_error = std::max(max_error, std::abs(a[i] - a_ref[i])); max_error = std::max(max_error, std::abs(b[i] - b_ref[i])); }
    if (max_error > 1e-6f) { std::cerr << "LoRA SGD mismatch: " << max_error << '\n'; return 1; }
    std::cout << "LoRA SGD: PASS max_abs_error=" << max_error << '\n';
}
