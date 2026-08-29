#include "api/embedding_training_api.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t V = 259, H = 64, R = 9;
    constexpr float lr = 0.03125f;
    std::vector<float> weights(V * H);
    std::vector<std::uint32_t> ids{3, 17, 3, 258, 17, 999, 3, 258, 17};
    std::vector<std::uint8_t> mask{1, 1, 0, 1, 1, 1, 1, 0, 1};
    std::vector<float> dstate(R * H);
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i) * 0.013f) * 0.1f;
    for (std::size_t i = 0; i < dstate.size(); ++i) dstate[i] = std::cos(float(i) * 0.071f) * 0.2f;
    std::vector<float> ref_gradient(V * H, 0.0f), ref_weights = weights;
    for (std::uint32_t r = 0; r < R; ++r) {
        if (!mask[r] || ids[r] >= V) continue;
        for (std::uint32_t h = 0; h < H; ++h) ref_gradient[ids[r] * H + h] += dstate[r * H + h];
    }
    for (std::size_t i = 0; i < ref_weights.size(); ++i) ref_weights[i] -= lr * ref_gradient[i];

    auto* handle = spaceslug_embedding_training_create(weights.data(), V, H);
    if (!handle) { std::cerr << "create failed\n"; return 1; }
    if (spaceslug_embedding_training_step(handle, ids.data(), dstate.data(), mask.data(), R, lr) != 0) return 2;
    std::vector<float> actual_weights(V * H), actual_gradient(V * H);
    if (spaceslug_embedding_training_readback(handle, actual_weights.data(), actual_gradient.data()) != 0) return 3;
    float max_error = 0.0f;
    for (std::size_t i = 0; i < actual_weights.size(); ++i)
        max_error = std::max(max_error, std::max(std::abs(actual_weights[i] - ref_weights[i]), std::abs(actual_gradient[i] - ref_gradient[i])));
    if (max_error > 2e-6f) { std::cerr << "parity max error " << max_error << "\n"; return 4; }

    // Finite difference of the linear sparse loss confirms the gradient sign.
    constexpr std::size_t probe = 3 * H + 7;
    constexpr float delta = 1e-3f;
    auto loss = [&](float const* w) {
        float sum = 0.0f;
        for (std::uint32_t r = 0; r < R; ++r) if (mask[r] && ids[r] == 3) sum += w[probe] * dstate[r * H + 7];
        return sum;
    };
    auto plus = actual_weights, minus = actual_weights;
    plus[probe] += delta; minus[probe] -= delta;
    float finite_difference = (loss(plus.data()) - loss(minus.data())) / (2.0f * delta);
    if (std::abs(finite_difference - actual_gradient[probe]) > 2e-4f) return 5;
    spaceslug_embedding_training_destroy(handle);
    std::cout << "embedding_training PASS max_error=" << max_error << " finite_difference=" << finite_difference << "\n";
    return 0;
}
