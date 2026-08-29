#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    constexpr std::size_t w1 = H * 4 * H, b1 = 4 * H, w2 = 4 * H * H, b2 = H;
    constexpr std::size_t group = w1 + b1 + w2 + b2, total = 3 * group;
    std::vector<float> embeddings(V * H, 0.0f), positions(Tcap * H, 0.0f), lm_head(H * Vp, 0.0f);
    std::vector<float> state(total);
    for (std::size_t i = 0; i < total; ++i) state[i] = 0.0001f * static_cast<float>(i + 1);
    auto context = vulkan_runtime::core::create_context("ffn-state-roundtrip");
    ForwardResourceGraph graph(context, embeddings.data(), positions.data(), lm_head.data());
    if (graph.update_ffn_state(state.data(), total, 17) != 0) {
        std::cerr << "FFN state update failed\n";
        return 1;
    }
    std::vector<float> restored(total, 0.0f);
    std::uint64_t step = 0;
    if (graph.readback_ffn_state(restored.data(), total, &step) != 0 || step != 17) {
        std::cerr << "FFN state readback failed step=" << step << "\n";
        return 1;
    }
    double max_error = 0.0;
    for (std::size_t i = 0; i < total; ++i)
        max_error = std::max(max_error, std::abs(static_cast<double>(restored[i]) - state[i]));
    if (max_error > 1.0e-6) {
        std::cerr << "FFN state roundtrip mismatch max=" << max_error << "\n";
        return 1;
    }
    std::cout << "FFN state roundtrip: PASS max=" << max_error << " step=" << step << "\n";
    return 0;
}
