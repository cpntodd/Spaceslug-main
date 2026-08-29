#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> e(V * H, 0.01f), p(Tcap * H, 0.0f), q(H * H, 0.0f), k(H * H, 0.0f), v(H * H, 0.0f), o(H * H, 0.0f), lm(H * Vp, 0.0f);
    for (std::uint32_t i = 0; i < H; ++i) {
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.02f;
    }
    auto context = vulkan_runtime::core::create_context("gamma-forward-effect");
    ForwardResourceGraph graph(context, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    std::vector<std::uint32_t> tokens(Tcap, 1);
    std::vector<float> baseline(Tcap * Vp), scaled(Tcap * Vp);
    graph.forward(tokens.data(), Tcap, baseline.data());
    std::vector<float> gamma(H, 2.0f);
    if (graph.update_trainable_gamma(gamma.data()) != 0) return 1;
    graph.forward(tokens.data(), Tcap, scaled.data());
    float max_difference = 0.0f;
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        if (!std::isfinite(baseline[i]) || !std::isfinite(scaled[i])) return 1;
        max_difference = std::max(max_difference, std::abs(scaled[i] - baseline[i]));
    }
    // The ordinary forward API remains frozen; bounded training owns the
    // post-attention true-RMSNorm chain and is exercised through its training ABI.
    if (ForwardResourceGraph::trainable_normalization_supported || max_difference > 1.0e-6f) {
        std::cerr << "primary gate expectation failed max=" << max_difference << "\n";
        return 1;
    }
    std::vector<float> staged_raw(Tcap * H), staged_inv(Tcap);
    if (graph.seed_rmsnorm_staged_gamma(gamma.data(), gamma.size()) != 0 ||
        graph.run_rmsnorm_state_only_staged(Tcap) != 0 ||
        graph.readback_rmsnorm_state_staged(staged_raw.data(), staged_raw.size(), staged_inv.data(), staged_inv.size()) != 0) {
        std::cerr << "staged RMSNorm state setup failed\n";
        return 1;
    }
    float max_staged_reference_error = 0.0f;
    for (std::uint32_t r = 0; r < Tcap; ++r) {
        double sum_sq = 0.0;
        for (std::uint32_t c = 0; c < H; ++c) {
            float x = e[tokens[r] * H + c] + p[r * H + c];
            sum_sq += double(x) * double(x);
            max_staged_reference_error = std::max(max_staged_reference_error, std::abs(staged_raw[r * H + c] - x));
        }
        max_staged_reference_error = std::max(max_staged_reference_error,
                                               std::abs(staged_inv[r] - float(1.0 / std::sqrt(sum_sq / H + 1.0e-5))));
    }
    if (max_staged_reference_error > 3.0e-5f) return 1;
    std::vector<float> staged_states(Tcap * H);
    if (graph.readback_rmsnorm_states_staged(staged_states.data(), staged_states.size()) != 0) return 1;
    float max_staged_gamma_effect = 0.0f;
    for (std::uint32_t r = 0; r < Tcap; ++r)
        for (std::uint32_t c = 0; c < H; ++c) {
            float x = staged_raw[r * H + c];
            float expected = x * staged_inv[r] * gamma[c];
            max_staged_gamma_effect = std::max(max_staged_gamma_effect, std::abs(staged_states[r * H + c] - expected));
        }
    if (max_staged_gamma_effect > 3.0e-5f) return 1;
    std::cout << "Primary gamma forward effect: PASS gated max=" << max_difference
              << " staged_state_ref=" << max_staged_reference_error
              << " staged_gamma_effect=" << max_staged_gamma_effect << "\n";
    return 0;
}
