#include "api/tiny_forward_persistent.h"
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> e(V * H, 0.01f), p(Tcap * H, 0.0f), lm(H * Vp, 0.0f);
    std::vector<float> q(H * H, 0.0f), k(H * H, 0.0f), v(H * H, 0.0f), o(H * H, 0.0f);
    for (std::uint32_t i = 0; i < H; ++i)
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
    auto ctx = vulkan_runtime::core::create_context("tiny-metrics-test");
    float gpu_loss = 0.0f;
    std::uint32_t gpu_count = 0;
    double cpu_loss = 0.0;
    std::uint32_t cpu_count = 0;
    {
        ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
        std::array<std::uint32_t, Tcap> tokens{}, targets{}, masks{};
        for (std::uint32_t i = 0; i < Tcap; ++i) {
            tokens[i] = i % V;
            targets[i] = (i * 3) % V;
            masks[i] = 1;
        }
        graph.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &gpu_loss, &gpu_count);
        std::vector<float> logits(Tcap * Vp), rows(Tcap);
        graph.forward_loss_fixed_retained(tokens.data(), targets.data(), masks.data(), logits.data(), rows.data());
        for (std::uint32_t r = 0; r < Tcap; ++r)
            if (masks[r]) {
                double mx = -INFINITY, z = 0.0;
                for (std::uint32_t c = 0; c < V; ++c)
                    mx = std::max(mx, static_cast<double>(logits[r * Vp + c]));
                for (std::uint32_t c = 0; c < V; ++c)
                    z += std::exp(static_cast<double>(logits[r * Vp + c]) - mx);
                cpu_loss += std::log(z) + mx - logits[r * Vp + targets[r]];
                ++cpu_count;
            }
    }
    vulkan_runtime::core::destroy_context(ctx);
    if (gpu_count != cpu_count ||
        std::abs(gpu_loss - static_cast<float>(cpu_loss)) > 2e-3f * (1.0f + std::abs(gpu_loss))) {
        std::cerr << "Tiny GPU metrics parity mismatch: loss " << gpu_loss << " vs " << cpu_loss << ", count "
                  << gpu_count << " vs " << cpu_count << '\n';
        return 1;
    }
    std::cout << "tiny GPU scalar metrics parity passed\n";
    return 0;
}
