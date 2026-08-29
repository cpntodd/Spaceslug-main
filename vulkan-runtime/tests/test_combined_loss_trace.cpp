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
    auto context = vulkan_runtime::core::create_context("combined-loss-trace");
    ForwardResourceGraph graph(context, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    constexpr std::uint32_t rows = 4, steps = 10;
    std::vector<std::uint32_t> tokens(Tcap, 1), targets(Tcap, 2), masks(Tcap, 0);
    for (std::uint32_t i = 0; i < rows; ++i) { tokens[i] = i + 1; targets[i] = i + 2; masks[i] = 1; }
    constexpr float lr = 1.0e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1.0e-8f, decay = 0.0f;
    const std::size_t group = H * 4 * H + 4 * H + 4 * H * H + H;
    std::vector<float> ffn(3 * group, 0.001f), gamma(H, 1.0f), gm(H, 0.0f), gv(H, 0.0f);
    if (graph.update_ffn_state(ffn.data(), ffn.size(), 0) != 0 || graph.update_gamma_state(gamma.data(), gm.data(), gv.data(), 0) != 0) return 1;
    float initial = 0.0f, loss = 0.0f;
    std::uint32_t count = 0;
    graph.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &initial, &count);
    if (count != rows || !std::isfinite(initial)) return 1;
    float minimum = initial, maximum = initial;
    for (std::uint32_t step = 0; step < steps; ++step) {
        if (graph.train_positions_adamw(tokens.data(), targets.data(), masks.data(), rows, lr, beta1, beta2, eps, decay) != 0) return 1;
        graph.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &loss, &count);
        if (count != rows || !std::isfinite(loss)) return 1;
        minimum = std::min(minimum, loss);
        maximum = std::max(maximum, loss);
    }
    if (!(loss < initial) || maximum > initial * 1.01f) {
        std::cerr << "combined loss trace failed: initial=" << initial << " final=" << loss << " min=" << minimum << " max=" << maximum << '\n';
        return 1;
    }
    std::vector<float> gamma_after(H), gamma_m(H), gamma_v(H), ffn_after(3 * group);
    std::uint64_t gamma_step = 0, ffn_step = 0;
    if (graph.readback_gamma_state(gamma_after.data(), gamma_m.data(), gamma_v.data(), &gamma_step) != 0 ||
        graph.readback_ffn_state(ffn_after.data(), ffn_after.size(), &ffn_step) != 0 || gamma_step != steps || ffn_step != steps)
        return 1;
    for (float value : gamma_after) if (!std::isfinite(value)) return 1;
    for (float value : ffn_after) if (!std::isfinite(value)) return 1;
    std::cout << "Combined loss trace: PASS steps=" << steps << " initial=" << initial << " final=" << loss
              << " min=" << minimum << " max=" << maximum << " gamma_step=" << gamma_step << " ffn_step=" << ffn_step << '\n';
    return 0;
}
