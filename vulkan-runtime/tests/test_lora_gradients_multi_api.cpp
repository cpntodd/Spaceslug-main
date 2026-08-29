#include "api/lora_gradients_multi_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
int main() {
    constexpr std::uint32_t rows = 3, d = 64, r = 4, target = 2;
    std::vector<float> x(rows * d), dy(rows * d), a(4 * d * r), b(4 * r * d), da(4 * d * r), db(4 * r * d),
        ra(4 * d * r, 0), rb(4 * r * d, 0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(i * .07f);
        dy[i] = std::cos(i * .03f);
    }
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = std::cos(i * .11f) * .1f;
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = std::sin(i * .13f) * .1f;
    for (std::uint32_t k = 0; k < d; ++k)
        for (std::uint32_t rr = 0; rr < r; ++rr)
            for (std::uint32_t row = 0; row < rows; ++row)
                for (std::uint32_t col = 0; col < d; ++col)
                    ra[target * d * r + k * r + rr] +=
                        x[row * d + k] * dy[row * d + col] * b[target * r * d + rr * d + col];
    for (std::uint32_t rr = 0; rr < r; ++rr)
        for (std::uint32_t col = 0; col < d; ++col)
            for (std::uint32_t row = 0; row < rows; ++row) {
                float z = 0;
                for (std::uint32_t k = 0; k < d; ++k)
                    z += x[row * d + k] * a[target * d * r + k * r + rr];
                rb[target * r * d + rr * d + col] += z * dy[row * d + col];
            }
    if (spaceslug_lora_gradients_multi(
            x.data(), dy.data(), a.data(), b.data(), da.data(), db.data(), rows, d, r, target) != 0)
        return 1;
    float e = 0;
    for (std::size_t i = 0; i < da.size(); ++i)
        e = std::max(e, std::abs(da[i] - ra[i]));
    for (std::size_t i = 0; i < db.size(); ++i)
        e = std::max(e, std::abs(db[i] - rb[i]));
    if (e > 4e-4f)
        return 1;
    std::cout << "Multi LoRA gradients: PASS max_abs_error=" << e << '\n';
}
