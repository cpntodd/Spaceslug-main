#include "api/lora_sgd_multi_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
int main() {
    constexpr std::uint32_t rank = 4;
    constexpr float lr = .05f;
    std::vector<float> a(4 * 64 * rank), b(4 * rank * 64), da(a.size()), db(b.size()), ra, rb;
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = std::sin(i * .07f);
        da[i] = std::cos(i * .11f);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = std::cos(i * .03f);
        db[i] = std::sin(i * .13f);
    }
    ra = a;
    rb = b;
    for (std::size_t i = 0; i < ra.size(); ++i)
        ra[i] -= lr * da[i];
    for (std::size_t i = 0; i < rb.size(); ++i)
        rb[i] -= lr * db[i];
    if (spaceslug_lora_sgd_multi(a.data(), b.data(), da.data(), db.data(), lr, rank) != 0)
        return 1;
    float e = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - ra[i]));
    for (std::size_t i = 0; i < b.size(); ++i)
        e = std::max(e, std::abs(b[i] - rb[i]));
    if (e > 1e-6f)
        return 1;
    std::cout << "Multi LoRA SGD: PASS max_abs_error=" << e << '\n';
}
