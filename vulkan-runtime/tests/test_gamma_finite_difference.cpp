#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> e(V * H, 0.1f), p(Tcap * H, 0.02f), q(H * H, 0.0f), k(H * H, 0.0f), v(H * H, 0.0f), o(H * H, 0.0f), lm(H * Vp, 0.0f);
    for (std::uint32_t i = 0; i < H; ++i) {
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.01f;
    }
    auto context = vulkan_runtime::core::create_context("gamma-finite-difference");
    ForwardResourceGraph graph(context, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    constexpr std::uint32_t rows = 2;
    std::vector<std::uint32_t> tokens(Tcap, 1), targets(Tcap, 2), masks(Tcap, 0);
    tokens[0] = 1; tokens[1] = 2; targets[0] = 2; targets[1] = 3; masks[0] = masks[1] = 1;
    std::vector<float> gamma(H, 1.0f), zero(H, 0.0f), dgamma(H), dffn(H * 4 * H + 4 * H + 4 * H * H + H), dx(rows * H), x(rows * H), scaled(rows * H), dstate(rows * H);
    const std::size_t group = dffn.size();
    std::vector<float> ffn(3 * group, 0.001f);
    if (graph.update_ffn_state(ffn.data(), ffn.size(), 0) != 0 || graph.update_gamma_state(gamma.data(), zero.data(), zero.data(), 0) != 0)
        return 1;
    auto loss_for_gamma = [&](std::vector<float> const& candidate) {
        ForwardResourceGraph probe(context, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
        if (probe.update_ffn_state(ffn.data(), ffn.size(), 0) != 0 || probe.update_gamma_state(candidate.data(), zero.data(), zero.data(), 0) != 0)
            return std::numeric_limits<float>::quiet_NaN();
        float value = 0.0f; std::uint32_t included = 0;
        probe.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &value, &included);
        return included == rows ? value : std::numeric_limits<float>::quiet_NaN();
    };
    float loss = 0.0f; std::uint32_t count = 0;
    graph.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &loss, &count);
    constexpr float h = 1.0e-3f;
    std::vector<float> gamma_plus = gamma, gamma_minus = gamma;
    gamma_plus[0] += h; gamma_minus[0] -= h;
    float loss_plus = loss_for_gamma(gamma_plus), loss_minus = loss_for_gamma(gamma_minus);
    if (count != rows || !std::isfinite(loss_plus) || !std::isfinite(loss_minus) || graph.train_positions_adamw(tokens.data(), targets.data(), masks.data(), rows, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 0 ||
        graph.readback_combined_gradients(dgamma.data(), dffn.data(), group, dx.data(), x.data(), scaled.data(), dstate.data(), rows) != 0)
        return 1;
    float max_error = 0.0f;
    for (std::uint32_t c = 0; c < H; ++c) {
        float expected = 0.0f;
        for (std::uint32_t r = 0; r < rows; ++r) expected += scaled[r * H + c] * dstate[r * H + c];
        max_error = std::max(max_error, std::abs(dgamma[c] - expected));
    }
    const float finite_difference = (loss_plus - loss_minus) / (2.0f * h);
    const float finite_difference_error = std::abs(finite_difference - dgamma[0]);
    if (!std::isfinite(loss) || !std::isfinite(finite_difference) || !std::isfinite(finite_difference_error) ||
        max_error > 2.0e-3f || finite_difference_error > 5.0e-2f)
        return 1;
    std::cout << "Gamma finite difference contract: PASS max_gradient_error=" << max_error
              << " finite_difference_error=" << finite_difference_error << "\n";
    return 0;
}
