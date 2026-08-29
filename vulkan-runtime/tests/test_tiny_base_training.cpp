#include "api/tiny_base_training_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main() {
    if (std::string(spaceslug_tiny_base_training_capability()).find(
            "fp32_lm_head_only_sgd_adamw_standalone_caller_supplied_activations_persistent_m_v_step") != 0)
        return 1;
    constexpr std::uint32_t H = 4, V = 5, VP = 8, Tcap = 7, rows = 3;
    constexpr float lr = 0.03125f;
    std::vector<float> w(H * VP), a(Tcap * H), dl(Tcap * VP), expected_w, expected_g(H * VP, 0.0f);
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = std::sin(float(i) * 0.13f);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = std::cos(float(i) * 0.17f);
    for (std::size_t i = 0; i < dl.size(); ++i) dl[i] = std::sin(float(i) * 0.07f);
    for (std::uint32_t h = 0; h < H; ++h)
        for (std::uint32_t v = 0; v < V; ++v)
            for (std::uint32_t t = 0; t < Tcap; ++t)
                if (t < rows) expected_g[h * VP + v] += a[t * H + h] * dl[t * VP + v];
    expected_w = w;
    for (std::uint32_t h = 0; h < H; ++h) {
        for (std::uint32_t v = 0; v < V; ++v) expected_w[h * VP + v] -= lr * expected_g[h * VP + v];
        for (std::uint32_t v = V; v < VP; ++v) expected_w[h * VP + v] = 0.0f;
    }
    // Finite-difference check for the CPU scalar objective whose analytic
    // derivative is the same fixed-Tcap accumulation sent to the GPU.
    constexpr float eps = 1.0e-3f;
    auto objective = [&](std::vector<float> const& weight, std::uint32_t h, std::uint32_t v) {
        float value = 0.0f;
        for (std::uint32_t t = 0; t < Tcap; ++t) if (t < rows)
            value += a[t * H + h] * weight[h * VP + v] * dl[t * VP + v];
        return value;
    };
    for (std::uint32_t h = 0; h < H; ++h) for (std::uint32_t v = 0; v < V; ++v) {
        auto plus = w, minus = w; plus[h * VP + v] += eps; minus[h * VP + v] -= eps;
        float numerical = (objective(plus, h, v) - objective(minus, h, v)) / (2.0f * eps);
        if (std::abs(numerical - expected_g[h * VP + v]) > 2.0e-4f) return 1;
    }
    spaceslug_tiny_base_training* session = spaceslug_tiny_base_training_create(w.data(), H, V, VP, Tcap);
    if (!session || spaceslug_tiny_base_training_step(session, a.data(), dl.data(), rows, lr) != 0) return 1;
    std::vector<float> got_w(w.size()), got_g(w.size());
    if (spaceslug_tiny_base_training_readback(session, got_w.data(), got_g.data()) != 0) return 1;
    float ew = 0.0f, eg = 0.0f;
    for (std::size_t i = 0; i < got_w.size(); ++i) { ew = std::max(ew, std::abs(got_w[i] - expected_w[i])); eg = std::max(eg, std::abs(got_g[i] - expected_g[i])); }
    // Padded lanes are explicitly protected and remain at their initial value only
    // for this generic API; gradients are required to be zero there.
    for (std::uint32_t h = 0; h < H; ++h) for (std::uint32_t v = V; v < VP; ++v) if (got_g[h * VP + v] != 0.0f) return 1;
    spaceslug_tiny_base_training_destroy(session);
    if (ew > 3e-5f || eg > 3e-5f) { std::cerr << "base training mismatch w=" << ew << " g=" << eg << '\n'; return 1; }
    std::cout << "Tiny base training: PASS max_weight_error=" << ew << " max_gradient_error=" << eg << '\n';

    constexpr float alr = 0.01f, beta1 = 0.8f, beta2 = 0.9f, ae = 1.0e-5f, decay = 0.03f;
    std::vector<float> aw = w, am(w.size(), 0.0f), av(w.size(), 0.0f), ag(w.size());
    for (std::uint32_t h = 0; h < H; ++h) for (std::uint32_t v = V; v < VP; ++v) aw[h * VP + v] = 0.0f;
    auto adam_ref = [&](std::uint64_t step) {
        for (std::size_t i = 0; i < aw.size(); ++i) {
            am[i] = beta1 * am[i] + (1.0f - beta1) * ag[i];
            av[i] = beta2 * av[i] + (1.0f - beta2) * ag[i] * ag[i];
            float mhat = am[i] / (1.0f - std::pow(beta1, float(step)));
            float vhat = av[i] / (1.0f - std::pow(beta2, float(step)));
            aw[i] = (1.0f - alr * decay) * aw[i] - alr * mhat / (std::sqrt(vhat) + ae);
        }
    };
    session = spaceslug_tiny_base_training_create(w.data(), H, V, VP, Tcap);
    if (!session) return 1;
    for (std::uint32_t h = 0; h < H; ++h) for (std::uint32_t v = 0; v < V; ++v) {
        ag[h * VP + v] = 0.0f;
        for (std::uint32_t t = 0; t < rows; ++t) ag[h * VP + v] += a[t * H + h] * dl[t * VP + v];
    }
    adam_ref(1);
    if (spaceslug_tiny_base_training_adamw_step(session, a.data(), dl.data(), rows, alr, beta1, beta2, ae, decay) != 0) return 1;
    std::vector<float> rw(w.size()), rg(w.size()), rm(w.size()), rv(w.size()); std::uint64_t step = 0;
    if (spaceslug_tiny_base_training_readback_adamw_state(session, rw.data(), rg.data(), rm.data(), rv.data(), &step) != 0 || step != 1) return 1;
    float ea = 0.0f; for (std::size_t i = 0; i < w.size(); ++i) ea = std::max(ea, std::abs(rw[i] - aw[i]));
    auto checkpoint_w = rw, checkpoint_m = rm, checkpoint_v = rv;
    adam_ref(2);
    if (spaceslug_tiny_base_training_adamw_step(session, a.data(), dl.data(), rows, alr, beta1, beta2, ae, decay) != 0) return 1;
    auto uninterrupted = std::vector<float>(w.size()); if (spaceslug_tiny_base_training_readback_adamw_state(session, uninterrupted.data(), rg.data(), rm.data(), rv.data(), &step) != 0 || step != 2) return 1;
    auto resumed = spaceslug_tiny_base_training_create(w.data(), H, V, VP, Tcap); if (!resumed) return 1;
    if (spaceslug_tiny_base_training_update_adamw_state(resumed, checkpoint_w.data(), checkpoint_m.data(), checkpoint_v.data(), 1) != 0 || spaceslug_tiny_base_training_adamw_step(resumed, a.data(), dl.data(), rows, alr, beta1, beta2, ae, decay) != 0) return 1;
    std::vector<float> continued(w.size()), tmp(w.size()), tmpm(w.size()), tmpv(w.size()); if (spaceslug_tiny_base_training_readback_adamw_state(resumed, continued.data(), tmp.data(), tmpm.data(), tmpv.data(), &step) != 0 || step != 2) return 1;
    float ep = 0.0f; for (std::size_t i = 0; i < w.size(); ++i) ep = std::max(ep, std::abs(continued[i] - uninterrupted[i]));
    spaceslug_tiny_base_training_destroy(resumed); spaceslug_tiny_base_training_destroy(session);
    session = spaceslug_tiny_base_training_create(w.data(), H, V, VP, Tcap);
    if (!session) return 1;
    auto invalid_state = w;
    invalid_state[0] = std::numeric_limits<float>::quiet_NaN();
    if (spaceslug_tiny_base_training_update_adamw_state(session, invalid_state.data(), am.data(), av.data(), 0) == 0 ||
        spaceslug_tiny_base_training_update_adamw_state(session, w.data(), am.data(), av.data(), UINT64_MAX) == 0) return 1;
    spaceslug_tiny_base_training_destroy(session);
    if (ea > 5.0e-5f || ep > 5.0e-5f) { std::cerr << "AdamW parity mismatch one=" << ea << " checkpoint=" << ep << '\n'; return 1; }
    session = spaceslug_tiny_base_training_create(w.data(), H, V, VP, Tcap);
    if (!session) return 1;
    std::vector<float> rejected_w(w.size()), rejected_g(w.size()), rejected_m(w.size()), rejected_v(w.size());
    std::uint64_t rejected_step = 0;
    if (spaceslug_tiny_base_training_adamw_step(session, nullptr, dl.data(), rows, alr, beta1, beta2, ae, decay) == 0 ||
        spaceslug_tiny_base_training_adamw_step(session, a.data(), nullptr, rows, alr, beta1, beta2, ae, decay) == 0 ||
        spaceslug_tiny_base_training_adamw_step(session, a.data(), dl.data(), 0, alr, beta1, beta2, ae, decay) == 0 ||
        spaceslug_tiny_base_training_readback_adamw_state(session, rejected_w.data(), rejected_g.data(), rejected_m.data(), rejected_v.data(), &rejected_step) != 0 ||
        spaceslug_tiny_base_training_adamw_step(session, a.data(), dl.data(), rows, -alr, beta1, beta2, ae, decay) == 0 ||
        spaceslug_tiny_base_training_adamw_step(session, a.data(), dl.data(), rows, alr, beta1, beta2, ae, -decay) == 0 ||
        spaceslug_tiny_base_training_adamw_step(session, a.data(), dl.data(), rows, NAN, beta1, beta2, ae, decay) == 0 ||
        spaceslug_tiny_base_training_adamw_step(session, a.data(), dl.data(), rows, alr, 1.0f, beta2, ae, decay) == 0) return 1;
    std::vector<float> rejected_w_after(w.size()), rejected_g_after(w.size()), rejected_m_after(w.size()), rejected_v_after(w.size());
    std::uint64_t rejected_step_after = 0;
    if (spaceslug_tiny_base_training_readback_adamw_state(session, rejected_w_after.data(), rejected_g_after.data(), rejected_m_after.data(), rejected_v_after.data(), &rejected_step_after) != 0 ||
        rejected_step_after != rejected_step || rejected_w_after != rejected_w || rejected_g_after != rejected_g || rejected_m_after != rejected_m || rejected_v_after != rejected_v) return 1;
    spaceslug_tiny_base_training_destroy(session);
    std::cout << "Standalone AdamW checkpoint parity: PASS max_error=" << std::max(ea, ep) << '\n';
    return 0;
}
