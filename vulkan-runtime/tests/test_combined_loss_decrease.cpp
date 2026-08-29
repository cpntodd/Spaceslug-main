#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> e(V * H, 0.1f), p(Tcap * H, 0.02f);
    std::vector<float> q(H * H, 0.0f), k(H * H, 0.0f), v(H * H, 0.0f), o(H * H, 0.0f), lm(H * Vp, 0.0f);
    for (std::uint32_t i = 0; i < H; ++i) {
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.01f;
    }
    auto context = vulkan_runtime::core::create_context("combined-loss-decrease");
    ForwardResourceGraph graph(context, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    constexpr std::uint32_t rows = 4;
    std::vector<std::uint32_t> tokens(Tcap, 1), targets(Tcap, 2), masks(Tcap, 0);
    for (std::uint32_t i = 0; i < rows; ++i) { tokens[i] = i + 1; targets[i] = i + 2; masks[i] = 1; }
    constexpr float lr = 1.0e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1.0e-8f, decay = 0.0f;
    const std::size_t group = H * 4 * H + 4 * H + 4 * H * H + H;
    std::vector<float> ffn(3 * group, 0.0f);
    for (std::size_t i = 0; i < group; ++i) ffn[i] = 0.001f;
    if (graph.update_ffn_state(ffn.data(), ffn.size(), 0) != 0) return 1;
    std::vector<float> gamma(H, 1.0f), gm(H, 0.0f), gv(H, 0.0f);
    if (graph.update_gamma_state(gamma.data(), gm.data(), gv.data(), 0) != 0) return 1;
    const std::vector<float> ffn_before = ffn;
    const std::vector<float> gamma_before = gamma;
    float before = 0.0f, after = 0.0f;
    std::uint32_t count = 0;
    graph.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &before, &count);
    if (count != rows || !std::isfinite(before)) return 1;
    if (graph.train_positions_adamw(tokens.data(), targets.data(), masks.data(), rows, lr, beta1, beta2, eps, decay) != 0) return 1;
    graph.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &after, &count);
    if (count != rows || !std::isfinite(after) || !(after < before)) {
        std::cerr << "combined loss did not decrease: before=" << before << " after=" << after << " count=" << count << "\n";
        return 1;
    }
    std::vector<float> gamma_after(H), gamma_m(H), gamma_v(H), ffn_after(3 * group);
    std::uint64_t gamma_step = 0, ffn_step = 0;
    if (graph.readback_gamma_state(gamma_after.data(), gamma_m.data(), gamma_v.data(), &gamma_step) != 0 ||
        graph.readback_ffn_state(ffn_after.data(), ffn_after.size(), &ffn_step) != 0 || gamma_step != 1 || ffn_step != 1)
        return 1;
    bool gamma_changed = false, ffn_changed = false;
    for (std::size_t i = 0; i < gamma.size(); ++i) gamma_changed |= std::abs(gamma_after[i] - gamma_before[i]) > 1.0e-8f;
    for (std::size_t i = 0; i < ffn.size(); ++i) ffn_changed |= std::abs(ffn_after[i] - ffn_before[i]) > 1.0e-8f;
    if (!ffn_changed) {
        float max_gamma_m = 0.0f;
        for (float value : gamma_m) max_gamma_m = std::max(max_gamma_m, std::abs(value));
        std::cerr << "integrated FFN state did not change after loss decrease; max_gamma_m=" << max_gamma_m << "\n";
        return 1;
    }
    std::cout << "Bounded integrated FFN loss decrease: PASS before=" << before << " after=" << after
              << " gamma_changed=" << gamma_changed << " ffn_changed=" << ffn_changed << "\n";
    return 0;
}
