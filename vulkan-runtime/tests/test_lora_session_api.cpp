#include "api/lora_session_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr uint32_t m = 3, rank = 4;
    constexpr float lr = 0.05f;
    std::vector<float> x(m * 64), dy(m * 64), a(64 * rank), b(64 * rank), y(m * 64);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(float(i) * .07f);
        dy[i] = std::cos(float(i) * .03f);
    }
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = std::cos(float(i) * .11f) * .1f;
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = std::sin(float(i) * .13f) * .1f;
    std::vector<float> ar = a, br = b;
    spaceslug_lora_session* session = nullptr;
    if (spaceslug_lora_session_create(m, rank, lr, a.data(), b.data(), &session) != 0 || !session)
        return 1;
    for (int step = 0; step < 2; ++step) {
        std::vector<float> yr(m * 64, 0), da(64 * rank, 0), db(64 * rank, 0), expected(m * 64, 0);
        for (uint32_t row = 0; row < m; ++row)
            for (uint32_t col = 0; col < 64; ++col)
                for (uint32_t r = 0; r < rank; ++r)
                    for (uint32_t k = 0; k < 64; ++k)
                        expected[row * 64 + col] += x[row * 64 + k] * ar[k * rank + r] * br[r * 64 + col];
        for (uint32_t k = 0; k < 64; ++k)
            for (uint32_t r = 0; r < rank; ++r)
                for (uint32_t row = 0; row < m; ++row)
                    for (uint32_t col = 0; col < 64; ++col)
                        da[k * rank + r] += x[row * 64 + k] * dy[row * 64 + col] * br[r * 64 + col];
        for (uint32_t r = 0; r < rank; ++r)
            for (uint32_t col = 0; col < 64; ++col)
                for (uint32_t row = 0; row < m; ++row) {
                    float z = 0;
                    for (uint32_t k = 0; k < 64; ++k)
                        z += x[row * 64 + k] * ar[k * rank + r];
                    db[r * 64 + col] += z * dy[row * 64 + col];
                }
        for (std::size_t i = 0; i < ar.size(); ++i)
            ar[i] -= lr * da[i];
        for (std::size_t i = 0; i < br.size(); ++i)
            br[i] -= lr * db[i];
        if (spaceslug_lora_session_step(session, x.data(), dy.data(), yr.data()) != 0)
            return 2;
        float error = 0;
        for (std::size_t i = 0; i < yr.size(); ++i)
            error = std::max(error, std::abs(yr[i] - expected[i]));
        if (error > 3e-4f) {
            std::cerr << "LoRA session mismatch: " << error << '\n';
            return 3;
        }
    }
    if (spaceslug_lora_session_destroy(session) != 0)
        return 4;
    std::cout << "LoRA session: PASS\n";
}
