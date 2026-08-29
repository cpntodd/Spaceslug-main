#include "api/attention_causal_api.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t t = 8;
    constexpr std::uint32_t d = 64;
    std::vector<float> q(128u * d), k(128u * d), v(128u * d), out(128u * d, 0.0f);
    for (std::uint32_t row = 0; row < t; ++row) {
        for (std::uint32_t col = 0; col < d; ++col) {
            q[row * d + col] = (row == col % t) ? 1.0f : 0.0f;
            k[row * d + col] = (row == col % t) ? 1.0f : 0.0f;
            v[row * d + col] = static_cast<float>(row + 1);
        }
    }
    int rc = spaceslug_attention_causal(q.data(), k.data(), v.data(), out.data(), t, d);
    if (rc != 0) {
        std::cerr << "spaceslug_attention_causal returned " << rc << "\n";
        return 1;
    }
    // Query row zero has exactly one legal key, so this checks the causal mask
    // directly. Later rows must be finite and remain within the prefix's V range.
    for (std::uint32_t col = 0; col < d; ++col) {
        if (!std::isfinite(out[col]) || std::fabs(out[col] - 1.0f) > 1e-4f) {
            std::cerr << "causal row-zero mismatch at column " << col << ": " << out[col] << "\n";
            return 1;
        }
    }
    for (std::uint32_t row = 1; row < t; ++row) {
        float value = out[row * d];
        if (!std::isfinite(value) || value < 1.0f || value > static_cast<float>(row + 1)) {
            std::cerr << "causal output out of prefix range at row " << row << ": " << value << "\n";
            return 1;
        }
    }
    std::cout << "Causal attention ABI: PASS\n";
    return 0;
}
