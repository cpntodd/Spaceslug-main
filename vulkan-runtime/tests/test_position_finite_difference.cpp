#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
void adamw(float& p, float& m, float& v, float g, std::uint32_t step) {
    m = 0.9f * m + 0.1f * g;
    v = 0.999f * v + 0.001f * g * g;
    const float mh = m / (1.0f - std::pow(0.9f, float(step)));
    const float vh = v / (1.0f - std::pow(0.999f, float(step)));
    p -= 1.0e-3f * mh / (std::sqrt(vh) + 1.0e-8f);
}
}

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> e(V * H, 0.1f), p(Tcap * H, 0.02f), q(H * H, 0.0f), k(H * H, 0.0f), v(H * H, 0.0f), o(H * H, 0.0f), lm(H * Vp, 0.0f);
    for (std::size_t i = 0; i < e.size(); ++i) e[i] = 0.07f + 0.013f * float(i % 11);
    for (std::uint32_t i = 0; i < H; ++i) {
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.2f;
    }
    auto context = vulkan_runtime::core::create_context("position-finite-difference");
    constexpr std::uint32_t rows = 3, active_rows = 2;
    std::vector<std::uint32_t> tokens(Tcap, 1), targets(Tcap, 2), masks(Tcap, 0);
    tokens[0] = 1; tokens[1] = 2; tokens[2] = 3; targets[0] = 2; targets[1] = 3; targets[2] = 4; masks[0] = masks[1] = 1;
    std::vector<float> gamma(H, 1.0f), zero(H, 0.0f);
    gamma[0] = 1.7f;
    auto loss_for_position = [&](std::vector<float> const& candidate) {
        ForwardResourceGraph probe(context, e.data(), candidate.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
        if (probe.update_gamma_state(gamma.data(), zero.data(), zero.data(), 0) != 0)
            return std::numeric_limits<float>::quiet_NaN();
        float value = 0.0f; std::uint32_t included = 0;
        probe.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &value, &included);
        return included == active_rows ? value : std::numeric_limits<float>::quiet_NaN();
    };
    constexpr float h = 1.0e-1f, learning_rate = 1.0f;
    std::vector<float> plus = p, minus = p;
    plus[0] += h; minus[0] -= h;
    const float loss_plus = loss_for_position(plus), loss_minus = loss_for_position(minus);
    ForwardResourceGraph graph(context, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    if (graph.update_gamma_state(gamma.data(), zero.data(), zero.data(), 0) != 0)
        return 1;
    float before = 0.0f; std::uint32_t included = 0;
    graph.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &before, &included);
    std::vector<float> position_before(Tcap * H), after(Tcap * H), position_gradient(Tcap * H);
    if (graph.readback_positions(position_before.data()) != 0)
        return 1;
    if (included != active_rows || !std::isfinite(loss_plus) || !std::isfinite(loss_minus) ||
        graph.train_positions_sgd(tokens.data(), targets.data(), masks.data(), active_rows, learning_rate) != 0 ||
        graph.readback_positions(after.data()) != 0 || graph.readback_position_gradient(position_gradient.data(), position_gradient.size()) != 0)
        return 1;
    const float finite_difference = (loss_plus - loss_minus) / (2.0f * h);
    const float observed_gradient = (position_before[0] - after[0]) / learning_rate;
    const float direct_gradient_error = std::abs(position_gradient[0] - finite_difference);
    const float error = std::abs(observed_gradient - finite_difference);
    if (!std::isfinite(before) || !std::isfinite(finite_difference) || !std::isfinite(observed_gradient) || !std::isfinite(direct_gradient_error) || direct_gradient_error > 5.0e-2f || error > 5.0e-2f)
        return 1;
    float max_masked_gradient = 0.0f;
    for (std::size_t i = rows * H; i < position_gradient.size(); ++i)
        max_masked_gradient = std::max(max_masked_gradient, std::abs(position_gradient[i]));
    if (max_masked_gradient != 0.0f)
        return 1;
    const float adamw_before = after[0];
    std::vector<float> adamw_positions(Tcap * H, 0.0f), adamw_m(Tcap * H, 0.0f), adamw_v(Tcap * H, 0.0f), adamw_gradient(Tcap * H);
    std::uint64_t adamw_step = 0;
    if (graph.train_positions_adamw(tokens.data(), targets.data(), masks.data(), active_rows, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 0 ||
        graph.readback_position_gradient(adamw_gradient.data(), adamw_gradient.size()) != 0 ||
        graph.readback_base_train_positions_adamw_state(adamw_positions.data(), adamw_m.data(), adamw_v.data(), &adamw_step) != 0 || adamw_step != 1)
        return 1;
    float expected_position = adamw_before, expected_m = 0.0f, expected_v = 0.0f;
    adamw(expected_position, expected_m, expected_v, adamw_gradient[0], 1);
    const float adamw_error = std::abs(expected_position - adamw_positions[0]);
    if (!std::isfinite(adamw_error) || adamw_error > 2.0e-3f)
        return 1;
    std::cout << "Position finite difference contract: PASS finite_difference=" << finite_difference
              << " observed_gradient=" << observed_gradient << " direct_gradient=" << position_gradient[0] << " direct_gradient_error=" << direct_gradient_error << " error=" << error
              << " before0=" << position_before[0] << " after0=" << after[0] << " adamw_expected0=" << expected_position << " adamw_gpu0=" << adamw_positions[0] << " adamw_error=" << adamw_error << "\n";
    return 0;
}
