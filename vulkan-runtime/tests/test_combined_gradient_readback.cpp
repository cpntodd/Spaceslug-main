#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace {
float gelu(float z) {
    constexpr float a = 0.7978845608028654f;
    return 0.5f * z * (1.0f + std::tanh(a * (z + 0.044715f * z * z * z)));
}
float gelu_grad(float z) {
    constexpr float a = 0.7978845608028654f;
    float t = std::tanh(a * (z + 0.044715f * z * z * z));
    return 0.5f * (1.0f + t) + 0.5f * z * (1.0f - t * t) * a * (1.0f + 3.0f * 0.044715f * z * z);
}
void adamw_update(float& w, float& m, float& v, float g, std::uint32_t step, float lr, float b1, float b2, float eps, float decay) {
    m = b1 * m + (1.0f - b1) * g;
    v = b2 * v + (1.0f - b2) * g * g;
    const float mhat = m / (1.0f - std::pow(b1, float(step)));
    const float vhat = v / (1.0f - std::pow(b2, float(step)));
    w = (1.0f - lr * decay) * w - lr * mhat / (std::sqrt(vhat) + eps);
}
}

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> e(V * H, 0.1f), p(Tcap * H, 0.02f);
    std::vector<float> q(H * H, 0.0f), k(H * H, 0.0f), v(H * H, 0.0f), o(H * H, 0.0f), lm(H * Vp, 0.0f);
    for (std::uint32_t i = 0; i < H; ++i) {
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.01f;
    }
    auto context = vulkan_runtime::core::create_context("combined-gradient-readback");
    ForwardResourceGraph graph(context, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    constexpr std::uint32_t rows = 2;
    std::vector<std::uint32_t> tokens(Tcap, 1), targets(Tcap, 2), masks(Tcap, 0);
    tokens[0] = 1; tokens[1] = 2; targets[0] = 2; targets[1] = 3; masks[0] = masks[1] = 1;
    const std::size_t w1n = H * 4 * H, b1n = 4 * H, w2n = 4 * H * H, b2n = H, group = w1n + b1n + w2n + b2n;
    std::vector<float> state(3 * group, 0.0f);
    for (std::size_t i = 0; i < group; ++i) state[i] = 0.001f;
    std::vector<float> gamma(H, 1.0f), zero_gamma(H, 0.0f);
    if (graph.update_ffn_state(state.data(), state.size(), 0) != 0 || graph.update_gamma_state(gamma.data(), zero_gamma.data(), zero_gamma.data(), 0) != 0)
        return 1;
    float before = 0.0f;
    std::uint32_t count = 0;
    graph.forward_loss_fixed_metrics(tokens.data(), targets.data(), masks.data(), &before, &count);
    if (count != rows ||
        graph.train_positions_adamw(tokens.data(), targets.data(), masks.data(), rows, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 0)
        return 1;
    std::vector<float> dg(H), dffn(group), dx(rows * H), x(rows * H), scaled(rows * H), dstate(rows * H);
    if (graph.readback_combined_gradients(dg.data(), dffn.data(), dffn.size(), dx.data(), x.data(), scaled.data(), dstate.data(), rows) != 0)
        return 1;
    // Independent CPU reconstruction of the FFN parameter gradients from the
    // graph-readback activation and input-gradient tensors. This deliberately
    // checks every packed FFN group rather than comparing GPU output to itself.
    std::vector<float> weights(state.begin(), state.begin() + group);
    auto const* w1 = weights.data();
    auto const* b1 = w1 + w1n;
    auto const* w2 = b1 + b1n;
    std::vector<float> cpu_w1(w1n, 0.0f), cpu_b1(b1n, 0.0f), cpu_w2(w2n, 0.0f), cpu_b2(b2n, 0.0f);
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<float> z(b1n);
        for (std::uint32_t j = 0; j < 4 * H; ++j) {
            z[j] = b1[j];
            for (std::uint32_t i = 0; i < H; ++i) z[j] += x[r * H + i] * w1[i * 4 * H + j];
            float upstream = 0.0f;
            for (std::uint32_t o2 = 0; o2 < H; ++o2) upstream += dx[r * H + o2] * w2[j * H + o2];
            for (std::uint32_t i = 0; i < H; ++i) cpu_w1[i * 4 * H + j] += x[r * H + i] * gelu_grad(z[j]) * upstream;
            cpu_b1[j] += gelu_grad(z[j]) * upstream;
            for (std::uint32_t o2 = 0; o2 < H; ++o2) cpu_w2[j * H + o2] += gelu(z[j]) * dx[r * H + o2];
        }
        for (std::uint32_t o2 = 0; o2 < H; ++o2) cpu_b2[o2] += dx[r * H + o2];
    }
    float max_error = 0.0f;
    for (std::size_t i = 0; i < w1n; ++i) max_error = std::max(max_error, std::abs(dffn[i] - cpu_w1[i]));
    for (std::size_t i = 0; i < b1n; ++i) max_error = std::max(max_error, std::abs(dffn[w1n + i] - cpu_b1[i]));
    for (std::size_t i = 0; i < w2n; ++i) max_error = std::max(max_error, std::abs(dffn[w1n + b1n + i] - cpu_w2[i]));
    for (std::size_t i = 0; i < b2n; ++i) max_error = std::max(max_error, std::abs(dffn[w1n + b1n + w2n + i] - cpu_b2[i]));
    float gamma_error = 0.0f;
    for (std::uint32_t c = 0; c < H; ++c) {
        float expected = 0.0f;
        for (std::uint32_t r = 0; r < rows; ++r)
            // True RMSNorm dgamma uses the normalized FFN input and the
            // gradient entering RMSNorm (the FFN input-gradient), not the
            // downstream post-normalization gradient consumed by attention.
            expected += (x[r * H + c] / gamma[c]) * dx[r * H + c];
        gamma_error = std::max(gamma_error, std::abs(dg[c] - expected));
    }
    std::vector<float> gamma_after(H), gamma_m(H), gamma_v(H), ffn_after(3 * group), ffn_m(3 * group), ffn_v(3 * group);
    float recurrence_probe_w = 0.7f, recurrence_probe_m = 0.0f, recurrence_probe_v = 0.0f;
    adamw_update(recurrence_probe_w, recurrence_probe_m, recurrence_probe_v, 0.25f, 1, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f);
    std::uint64_t gamma_step = 0, ffn_step = 0;
    if (graph.readback_gamma_state(gamma_after.data(), gamma_m.data(), gamma_v.data(), &gamma_step) != 0 ||
        graph.readback_ffn_state(ffn_after.data(), ffn_after.size(), &ffn_step) != 0 || gamma_step != 1 || ffn_step != 1)
        return 1;
    float gamma_state_error = 0.0f;
    for (std::size_t i = 0; i < H; ++i) {
        float expected_m = 0.1f * dg[i], expected_v = 0.001f * dg[i] * dg[i];
        float expected = gamma[i] - 1.0e-3f * (expected_m / 0.1f) / (std::sqrt(expected_v / 0.001f) + 1.0e-8f);
        gamma_state_error = std::max(gamma_state_error, std::abs(gamma_after[i] - expected));
    }
    float ffn_state_error = 0.0f;
    for (std::size_t i = 0; i < group; ++i) {
        float expected_m = 0.1f * dffn[i], expected_v = 0.001f * dffn[i] * dffn[i];
        float expected = state[i] - 1.0e-3f * (expected_m / 0.1f) / (std::sqrt(expected_v / 0.001f) + 1.0e-8f);
        ffn_state_error = std::max(ffn_state_error, std::abs(ffn_after[i] - expected));
    }
    if (!std::isfinite(before) || !std::isfinite(max_error) || !std::isfinite(gamma_error) ||
        !std::isfinite(gamma_state_error) || !std::isfinite(ffn_state_error) || max_error > 2.0e-3f || gamma_error > 2.0e-3f ||
        gamma_state_error > 2.0e-3f || ffn_state_error > 2.0e-3f)
        { std::cerr << "combined gradient mismatch max=" << max_error << " gamma=" << gamma_error << " gs=" << gamma_state_error << " fs=" << ffn_state_error << "\n"; return 1; }
    // Preserve step-1 weights and moments before the second submission so the
    // CPU reference can apply the true AdamW recurrence rather than comparing
    // against the state produced by the second submission itself.
    std::vector<float> gamma_before2 = gamma_after, gamma_m_before2 = gamma_m, gamma_v_before2 = gamma_v;
    std::vector<float> ffn_before2 = ffn_after;
    for (std::size_t i = 0; i < group; ++i) {
        ffn_m[i] = ffn_after[group + i];
        ffn_v[i] = ffn_after[2 * group + i];
    }
    // Run a second identical step and independently apply the step-2 AdamW
    // recurrence to the gradients exposed by that second submission.
    if (graph.train_positions_adamw(tokens.data(), targets.data(), masks.data(), rows, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 0 ||
        graph.readback_combined_gradients(dg.data(), dffn.data(), dffn.size(), dx.data(), x.data(), scaled.data(), dstate.data(), rows) != 0 ||
        graph.readback_gamma_state(gamma_after.data(), gamma_m.data(), gamma_v.data(), &gamma_step) != 0 ||
        graph.readback_ffn_state(ffn_after.data(), ffn_after.size(), &ffn_step) != 0 || gamma_step != 2 || ffn_step != 2)
        return 1;
    float gamma_state_error2 = 0.0f, ffn_state_error2 = 0.0f;
    for (std::size_t i = 0; i < H; ++i) {
        float m = 0.9f * gamma_m_before2[i] + 0.1f * dg[i];
        float vv = 0.999f * gamma_v_before2[i] + 0.001f * dg[i] * dg[i];
        float expected = (m / (1.0f - std::pow(0.9f, 2.0f))) / (std::sqrt(vv / (1.0f - std::pow(0.999f, 2.0f))) + 1.0e-8f);
        float next = gamma_before2[i] - 1.0e-3f * expected;
        gamma_state_error2 = std::max(gamma_state_error2, std::abs(gamma_after[i] - next));
    }
    for (std::size_t i = 0; i < group; ++i) {
        float m = 0.9f * ffn_m[i] + 0.1f * dffn[i];
        float vv = 0.999f * ffn_v[i] + 0.001f * dffn[i] * dffn[i];
        float expected = (m / (1.0f - std::pow(0.9f, 2.0f))) / (std::sqrt(vv / (1.0f - std::pow(0.999f, 2.0f))) + 1.0e-8f);
        float next = ffn_before2[i] - 1.0e-3f * expected;
        ffn_state_error2 = std::max(ffn_state_error2, std::abs(ffn_after[i] - next));
    }
    if (gamma_state_error2 > 2.0e-3f || ffn_state_error2 > 2.0e-3f)
        return 1;
    if (ForwardResourceGraph::trainable_normalization_supported || ForwardResourceGraph::trainable_ffn_supported ||
        ForwardResourceGraph::trainable_full_base_constructor_supported)
        return 1;
    std::cout << "Combined gradient CPU parity: PASS max_ffn_gradient_error=" << max_error
              << " max_gamma_gradient_error=" << gamma_error << " max_gamma_state_error=" << gamma_state_error
              << " max_ffn_state_error=" << ffn_state_error << " step2_gamma_state_error=" << gamma_state_error2
              << " step2_ffn_state_error=" << ffn_state_error2 << "\n";
    return 0;
}
