#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    constexpr std::uint32_t rows = 4;
    std::vector<float> embeddings(V * H), positions(Tcap * H), query(H * H), key(H * H), value(H * H), output(H * H), lm(H * Vp);
    for (std::size_t i = 0; i < embeddings.size(); ++i) embeddings[i] = 0.01f * std::sin(float(i) * 0.07f);
    for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = 0.02f * std::cos(float(i) * 0.03f);
    for (std::uint32_t i = 0; i < H; ++i) {
        query[i * H + i] = key[i * H + i] = value[i * H + i] = output[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.01f;
    }
    auto context = vulkan_runtime::core::create_context("tiny-rmsnorm-staged-state");
    ForwardResourceGraph graph(context, embeddings.data(), positions.data(), query.data(), key.data(), value.data(), output.data(), lm.data());
    std::vector<std::uint32_t> tokens(Tcap, 0);
    for (std::uint32_t r = 0; r < rows; ++r) tokens[r] = 3 + r;
    std::vector<float> logits(rows * Vp);
    graph.forward(tokens.data(), rows, logits.data(), false);
    if (graph.run_rmsnorm_forward_staged(rows) != 0) return 1;
    std::vector<float> raw(Tcap * H), inv(Tcap);
    if (graph.readback_rmsnorm_state_staged(raw.data(), raw.size(), inv.data(), inv.size()) != 0) return 2;
    float max_raw_error = 0.0f, max_inv_error = 0.0f;
    for (std::uint32_t r = 0; r < rows; ++r) {
        double sum_sq = 0.0;
        for (std::uint32_t c = 0; c < H; ++c) {
            float expected = embeddings[tokens[r] * H + c] + positions[r * H + c];
            max_raw_error = std::max(max_raw_error, std::abs(raw[r * H + c] - expected));
            sum_sq += double(expected) * double(expected);
        }
        float expected_inv = float(1.0 / std::sqrt(sum_sq / double(H) + 1.0e-5));
        max_inv_error = std::max(max_inv_error, std::abs(inv[r] - expected_inv));
    }
    graph.run_rmsnorm_forward_staged(rows, true);
    graph.readback_rmsnorm_state_staged(raw.data(), raw.size(), inv.data(), inv.size());
    constexpr float tolerance = 5.0e-6f;
    bool pass = max_raw_error <= tolerance && max_inv_error <= tolerance;

    std::vector<float> gamma(H), dy(rows * H), dx(Tcap * H), dgamma(H);
    std::vector<std::uint32_t> mask(rows, 1u);
    float dummy = 0.0f;
    std::uint32_t dummy_mask = 0u;
    std::vector<std::uint32_t> hole_mask{1u, 0u, 1u, 0u};
    if (graph.validate_rmsnorm_staged_rows(mask.data(), rows) != 0 ||
        graph.validate_rmsnorm_staged_rows(hole_mask.data(), hole_mask.size()) == 0 ||
        graph.validate_rmsnorm_staged_rows(nullptr, rows) == 0 ||
        graph.validate_rmsnorm_staged_rows(&dummy_mask, 0) == 0 ||
        graph.seed_rmsnorm_staged_gamma(nullptr, H) == 0 || graph.seed_rmsnorm_staged_gamma(&dummy, H - 1) == 0 ||
        graph.seed_rmsnorm_staged_dy(nullptr, H) == 0 || graph.seed_rmsnorm_staged_dy(&dummy, 0) == 0 ||
        graph.seed_rmsnorm_staged_mask_for_rows(nullptr, rows) == 0 || graph.seed_rmsnorm_staged_mask_for_rows(&dummy_mask, 0) == 0) {
        std::cerr << "tiny_rmsnorm_staged_state accepted invalid staged inputs\n";
        return 18;
    }
    for (std::uint32_t c = 0; c < H; ++c) gamma[c] = 1.0f + 0.01f * std::cos(float(c) * 0.11f);
    std::vector<float> partial_states(Tcap * H), all_masked_states(Tcap * H);
    float max_partial_state = 0.0f, max_all_masked_state = 0.0f;
    if (graph.seed_rmsnorm_staged_gamma(gamma.data(), gamma.size()) != 0 ||
        graph.run_rmsnorm_state_only_staged(2) != 0 ||
        graph.readback_rmsnorm_states_staged(partial_states.data(), partial_states.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state partial state-only setup failed\n";
        return 22;
    }
    for (std::uint32_t r = 0; r < 2; ++r) {
        double sum_sq = 0.0;
        for (std::uint32_t c = 0; c < H; ++c) {
            float x = embeddings[tokens[r] * H + c] + positions[r * H + c];
            sum_sq += double(x) * double(x);
        }
        float inv_r = float(1.0 / std::sqrt(sum_sq / double(H) + 1.0e-5));
        for (std::uint32_t c = 0; c < H; ++c) {
            float x = embeddings[tokens[r] * H + c] + positions[r * H + c];
            max_partial_state = std::max(max_partial_state, std::abs(partial_states[r * H + c] - x * inv_r * gamma[c]));
        }
    }
    std::vector<std::uint32_t> all_masked_rows(rows, 0u);
    if (graph.seed_rmsnorm_staged_mask_for_rows(all_masked_rows.data(), rows) != 0 ||
        graph.run_rmsnorm_state_only_staged(rows) != 0 ||
        graph.readback_rmsnorm_states_staged(all_masked_states.data(), all_masked_states.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state all-masked state-only setup failed\n";
        return 23;
    }
    for (std::uint32_t r = 0; r < rows; ++r) {
        double sum_sq = 0.0;
        for (std::uint32_t c = 0; c < H; ++c) {
            float x = embeddings[tokens[r] * H + c] + positions[r * H + c];
            sum_sq += double(x) * double(x);
        }
        float inv_r = float(1.0 / std::sqrt(sum_sq / double(H) + 1.0e-5));
        for (std::uint32_t c = 0; c < H; ++c) {
            float x = embeddings[tokens[r] * H + c] + positions[r * H + c];
            max_all_masked_state = std::max(max_all_masked_state, std::abs(all_masked_states[r * H + c] - x * inv_r * gamma[c]));
        }
    }
    pass = pass && max_partial_state <= 3.0e-5f && max_all_masked_state <= 3.0e-5f;
    if (graph.seed_rmsnorm_staged_mask_for_rows(mask.data(), rows) != 0) return 24;
    for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = 0.03f * std::sin(float(i) * 0.17f);
    if (graph.seed_rmsnorm_staged_mask_for_rows(mask.data(), rows) != 0 ||
        graph.seed_rmsnorm_staged_gamma(gamma.data(), gamma.size()) != 0 ||
        graph.seed_rmsnorm_staged_dy(dy.data(), dy.size()) != 0 ||
        graph.run_rmsnorm_staged_state_chain(rows) != 0 ||
        graph.readback_rmsnorm_staged_diagnostics(dx.data(), dgamma.data(), rows) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state backward setup failed\n";
        return 4;
    }
    float max_dx_error = 0.0f, max_dgamma_error = 0.0f;
    std::vector<float> expected_dx(rows * H), expected_dgamma(H, 0.0f);
    for (std::uint32_t r = 0; r < rows; ++r) {
        double dot = 0.0;
        for (std::uint32_t c = 0; c < H; ++c) {
            float x = raw[r * H + c];
            dot += double(x) * double(gamma[c]) * double(dy[r * H + c]);
        }
        double inv_r = inv[r];
        double inv3 = inv_r * inv_r * inv_r;
        for (std::uint32_t c = 0; c < H; ++c) {
            float x = raw[r * H + c];
            double expected = double(gamma[c]) * inv_r * double(dy[r * H + c]) -
                              double(x) * inv3 * dot / double(H);
            expected_dx[r * H + c] = float(expected);
            expected_dgamma[c] += float(double(x) * inv_r * double(dy[r * H + c]));
            max_dx_error = std::max(max_dx_error, std::abs(dx[r * H + c] - expected_dx[r * H + c]));
        }
    }
    for (std::uint32_t c = 0; c < H; ++c)
        max_dgamma_error = std::max(max_dgamma_error, std::abs(dgamma[c] - expected_dgamma[c]));
    pass = pass && max_dx_error <= 2.0e-5f && max_dgamma_error <= 2.0e-5f;

    // Re-run with a trailing-valid mask and verify masked rows contribute zero.
    std::fill(mask.begin(), mask.end(), 0u);
    for (std::uint32_t r = 0; r < rows / 2; ++r) mask[r] = 1u;
    if (graph.seed_rmsnorm_staged_mask_for_rows(mask.data(), rows) != 0 ||
        graph.seed_rmsnorm_staged_dy(dy.data(), dy.size()) != 0 ||
        graph.run_rmsnorm_staged_state_chain(rows) != 0 ||
        graph.readback_rmsnorm_staged_diagnostics(dx.data(), dgamma.data(), rows) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state masked setup failed\n";
        return 5;
    }
    float max_masked_dx = 0.0f, max_masked_dgamma = 0.0f;
    std::vector<float> expected_masked_dgamma(H, 0.0f);
    for (std::uint32_t r = 0; r < rows; ++r) {
        double dot = 0.0;
        if (mask[r]) for (std::uint32_t c = 0; c < H; ++c)
            dot += double(raw[r * H + c]) * double(gamma[c]) * double(dy[r * H + c]);
        for (std::uint32_t c = 0; c < H; ++c) {
            float expected = 0.0f;
            if (mask[r]) {
                double inv_r = inv[r];
                expected = float(double(gamma[c]) * inv_r * double(dy[r * H + c]) -
                                 double(raw[r * H + c]) * inv_r * inv_r * inv_r * dot / double(H));
                expected_masked_dgamma[c] += float(double(raw[r * H + c]) * inv_r * double(dy[r * H + c]));
            }
            max_masked_dx = std::max(max_masked_dx, std::abs(dx[r * H + c] - expected));
        }
    }
    for (std::uint32_t c = 0; c < H; ++c)
        max_masked_dgamma = std::max(max_masked_dgamma, std::abs(dgamma[c] - expected_masked_dgamma[c]));
    pass = pass && max_masked_dx <= 2.0e-5f && max_masked_dgamma <= 2.0e-5f;
    std::fill(mask.begin(), mask.end(), 0u);
    if (graph.seed_rmsnorm_staged_mask_for_rows(mask.data(), rows) != 0 ||
        graph.seed_rmsnorm_staged_dy(dy.data(), dy.size()) != 0 ||
        graph.run_rmsnorm_staged_state_chain(rows) != 0 ||
        graph.readback_rmsnorm_staged_diagnostics(dx.data(), dgamma.data(), rows) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state all-masked setup failed\n";
        return 6;
    }
    float max_all_masked = 0.0f;
    for (float v : dgamma) max_all_masked = std::max(max_all_masked, std::abs(v));
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < H; ++c) max_all_masked = std::max(max_all_masked, std::abs(dx[r * H + c]));
    pass = pass && max_all_masked <= 2.0e-5f;

    // Exercise the fixed maximum supported window with a fresh input pattern.
    constexpr std::uint32_t max_rows = Tcap;
    std::vector<std::uint32_t> max_tokens(max_rows);
    std::vector<float> max_dy(max_rows * H), max_dx(Tcap * H), max_dgamma(H);
    for (std::uint32_t r = 0; r < max_rows; ++r) {
        max_tokens[r] = 7 + (r % 31);
        for (std::uint32_t c = 0; c < H; ++c) max_dy[r * H + c] = 0.02f * std::cos(float(r + c) * 0.09f);
    }
    std::vector<float> max_logits(max_rows * Vp);
    graph.forward(max_tokens.data(), max_rows, max_logits.data(), false);
    if (graph.seed_rmsnorm_staged_mask_for_rows(std::vector<std::uint32_t>(max_rows, 1u).data(), max_rows) != 0 ||
        graph.seed_rmsnorm_staged_dy(max_dy.data(), max_dy.size()) != 0 ||
        graph.run_rmsnorm_state_only_staged(max_rows) != 0 ||
        graph.readback_rmsnorm_state_staged(raw.data(), raw.size(), inv.data(), inv.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state maximum-window setup failed\n";
        return 7;
    }
    std::vector<float> max_states(Tcap * H);
    if (graph.readback_rmsnorm_states_staged(max_states.data(), max_states.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state state readback failed\n";
        return 8;
    }
    float max_window_raw = 0.0f, max_window_inv = 0.0f, max_window_state = 0.0f;
    for (std::uint32_t r = 0; r < max_rows; ++r) {
        double sum_sq = 0.0;
        for (std::uint32_t c = 0; c < H; ++c) {
            float expected = embeddings[max_tokens[r] * H + c] + positions[r * H + c];
            sum_sq += double(expected) * double(expected);
            max_window_raw = std::max(max_window_raw, std::abs(raw[r * H + c] - expected));
        }
        float expected_inv = float(1.0 / std::sqrt(sum_sq / double(H) + 1.0e-5));
        max_window_inv = std::max(max_window_inv, std::abs(inv[r] - expected_inv));
        for (std::uint32_t c = 0; c < H; ++c) {
            float expected_state = (embeddings[max_tokens[r] * H + c] + positions[r * H + c]) * expected_inv * gamma[c];
            max_window_state = std::max(max_window_state, std::abs(max_states[r * H + c] - expected_state));
        }
    }
    // The maximum-window pass intentionally allows the observed bounded FP32
    // reduction error while retaining a tight CPU-reference check.
    constexpr float max_window_tolerance = 3.0e-5f;
    pass = pass && max_window_raw <= max_window_tolerance && max_window_inv <= max_window_tolerance && max_window_state <= max_window_tolerance;

    // The same maximum window must also support masked backward and dgamma.
    std::vector<float> max_dx_expected(max_rows * H), max_dgamma_expected(H, 0.0f);
    if (graph.seed_rmsnorm_staged_gamma(gamma.data(), gamma.size()) != 0 ||
        graph.run_rmsnorm_backward_staged(max_rows) != 0 ||
        graph.run_rmsnorm_dgamma_staged(max_rows) != 0 ||
        graph.readback_rmsnorm_staged_diagnostics(max_dx.data(), max_dgamma.data(), max_rows) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state maximum-window backward failed\n";
        return 9;
    }
    float max_window_dx = 0.0f, max_window_dgamma = 0.0f;
    for (std::uint32_t r = 0; r < max_rows; ++r) {
        double dot = 0.0;
        for (std::uint32_t c = 0; c < H; ++c)
            dot += double(raw[r * H + c]) * double(gamma[c]) * double(max_dy[r * H + c]);
        double inv_r = inv[r];
        for (std::uint32_t c = 0; c < H; ++c) {
            double x = raw[r * H + c];
            float expected = float(double(gamma[c]) * inv_r * double(max_dy[r * H + c]) -
                                   x * inv_r * inv_r * inv_r * dot / double(H));
            max_window_dx = std::max(max_window_dx, std::abs(max_dx[r * H + c] - expected));
            max_dgamma_expected[c] += float(x * inv_r * double(max_dy[r * H + c]));
        }
    }
    for (std::uint32_t c = 0; c < H; ++c)
        max_window_dgamma = std::max(max_window_dgamma, std::abs(max_dgamma[c] - max_dgamma_expected[c]));
    pass = pass && max_window_dx <= max_window_tolerance && max_window_dgamma <= max_window_tolerance;

    // One-step AdamW parity for gamma, starting from zero moments.
    constexpr float lr = 1.0e-3f, beta1 = 0.9f, beta2 = 0.99f, eps = 1.0e-5f, decay = 0.01f;
    if (graph.run_rmsnorm_gamma_adamw_staged(0.0f, beta1, beta2, eps, decay) == 0 ||
        graph.run_rmsnorm_gamma_adamw_staged(lr, 1.0f, beta2, eps, decay) == 0 ||
        graph.run_rmsnorm_gamma_adamw_staged(lr, beta1, beta2, 0.0f, decay) == 0 ||
        graph.run_rmsnorm_gamma_adamw_staged(lr, beta1, beta2, eps, -1.0f) == 0 ||
        graph.run_rmsnorm_gamma_adamw_staged(std::numeric_limits<float>::quiet_NaN(), beta1, beta2, eps, decay) == 0) {
        std::cerr << "tiny_rmsnorm_staged_state invalid AdamW inputs accepted\n";
        return 15;
    }
    std::vector<float> gamma_before(H), m_before(H), v_before(H);
    std::uint64_t step_before = 0;
    if (graph.readback_rmsnorm_gamma_state_staged(gamma_before.data(), m_before.data(), v_before.data(), &step_before) != 0 ||
        graph.run_rmsnorm_gamma_adamw_staged(lr, beta1, beta2, eps, decay) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state AdamW setup failed\n";
        return 10;
    }
    std::vector<float> gamma_after(H), m_after(H), v_after(H);
    std::uint64_t step_after = 0;
    if (graph.readback_rmsnorm_gamma_state_staged(gamma_after.data(), m_after.data(), v_after.data(), &step_after) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state AdamW readback failed\n";
        return 11;
    }
    float max_adamw_error = 0.0f;
    for (std::uint32_t c = 0; c < H; ++c) {
        float g = max_dgamma_expected[c];
        float m1 = (1.0f - beta1) * g, v1 = (1.0f - beta2) * g * g;
        float expected = (1.0f - lr * decay) * gamma_before[c] - lr * (m1 / (1.0f - beta1)) / (std::sqrt(v1 / (1.0f - beta2)) + eps);
        max_adamw_error = std::max(max_adamw_error, std::abs(gamma_after[c] - expected));
        max_adamw_error = std::max(max_adamw_error, std::abs(m_after[c] - m1));
        max_adamw_error = std::max(max_adamw_error, std::abs(v_after[c] - v1));
    }
    pass = pass && step_after == step_before + 1 && max_adamw_error <= 3.0e-5f;

    // Continue with a nonzero-moment second step to cover checkpoint/resume math.
    if (graph.run_rmsnorm_gamma_adamw_staged(lr, beta1, beta2, eps, decay) != 0 ||
        graph.readback_rmsnorm_gamma_state_staged(gamma_after.data(), m_after.data(), v_after.data(), &step_after) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state AdamW continuation failed\n";
        return 12;
    }
    float max_adamw_two_error = 0.0f;
    for (std::uint32_t c = 0; c < H; ++c) {
        float g = max_dgamma_expected[c];
        float m2 = beta1 * ((1.0f - beta1) * g) + (1.0f - beta1) * g;
        float v2 = beta2 * ((1.0f - beta2) * g * g) + (1.0f - beta2) * g * g;
        float expected = (1.0f - lr * decay) * gamma_after[c];
        // Reconstruct gamma_1 from the first-step result, then apply step 2.
        float m1 = (1.0f - beta1) * g, v1 = (1.0f - beta2) * g * g;
        float gamma1 = gamma_before[c] * (1.0f - lr * decay) - lr * (m1 / (1.0f - beta1)) / (std::sqrt(v1 / (1.0f - beta2)) + eps);
        expected = (1.0f - lr * decay) * gamma1 - lr * (m2 / (1.0f - beta1 * beta1)) / (std::sqrt(v2 / (1.0f - beta2 * beta2)) + eps);
        max_adamw_two_error = std::max(max_adamw_two_error, std::abs(gamma_after[c] - expected));
        max_adamw_two_error = std::max(max_adamw_two_error, std::abs(m_after[c] - m2));
        max_adamw_two_error = std::max(max_adamw_two_error, std::abs(v_after[c] - v2));
    }
    pass = pass && step_after == step_before + 2 && max_adamw_two_error <= 3.0e-5f;
    // Invalid calls above must not consume an optimizer step.
    pass = pass && step_before == 0;

    // Restore the nonzero-moment checkpoint and verify exact state round-trip.
    if (graph.update_gamma_state(gamma_after.data(), m_after.data(), v_after.data(), step_after) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state gamma restore failed\n";
        return 13;
    }
    // Exercise the staged FFN primitive on a deterministic normalized-state input.
    std::vector<float> ffn_input(rows * H), ffn_output(rows * H);
    for (std::size_t i = 0; i < ffn_input.size(); ++i) ffn_input[i] = 0.04f * std::sin(float(i) * 0.13f);
    std::vector<float> staged_states(Tcap * H);
    std::fill(mask.begin(), mask.end(), 1u);
    if (graph.seed_rmsnorm_staged_gamma(gamma_after.data(), gamma_after.size()) != 0 ||
        graph.seed_rmsnorm_staged_mask_for_rows(mask.data(), rows) != 0 ||
        graph.run_rmsnorm_state_only_staged(rows) != 0 ||
        graph.readback_rmsnorm_states_staged(staged_states.data(), staged_states.size()) != 0 ||
        graph.seed_ffn_input_staged(staged_states.data(), rows * H) != 0 ||
        graph.run_ffn_forward(rows) != 0 ||
        graph.readback_ffn_output_staged(ffn_output.data(), ffn_output.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN setup failed\n";
        return 19;
    }
    float max_ffn_error = 0.0f;
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < H; ++c)
            max_ffn_error = std::max(max_ffn_error, std::abs(ffn_output[r * H + c] - staged_states[r * H + c]));
    pass = pass && max_ffn_error <= 2.0e-6f;
    if (graph.seed_ffn_input_staged(ffn_input.data(), ffn_input.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN input restore failed\n";
        return 19;
    }

    // Restore deterministic nonzero FFN parameters and compare the full GELU
    // residual formula against a CPU reference.
    constexpr std::size_t w1n = H * 4 * H, b1n = 4 * H, w2n = 4 * H * H, b2n = H;
    constexpr std::size_t group = w1n + b1n + w2n + b2n;
    std::vector<float> ffn_state(3 * group, 0.0f);
    for (std::size_t i = 0; i < w1n; ++i) ffn_state[i] = 0.002f * std::sin(float(i) * 0.017f);
    for (std::size_t i = 0; i < b1n; ++i) ffn_state[w1n + i] = 0.01f * std::cos(float(i) * 0.07f);
    for (std::size_t i = 0; i < w2n; ++i) ffn_state[w1n + b1n + i] = 0.002f * std::cos(float(i) * 0.013f);
    for (std::size_t i = 0; i < b2n; ++i) ffn_state[w1n + b1n + w2n + i] = 0.01f * std::sin(float(i) * 0.05f);
    if (graph.update_ffn_state(ffn_state.data(), ffn_state.size(), step_after) != 0 ||
        graph.run_ffn_forward(rows) != 0 ||
        graph.readback_ffn_output_staged(ffn_output.data(), ffn_output.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state nonzero FFN setup failed\n";
        return 20;
    }
    auto gelu = [](double z) { return 0.5 * z * (1.0 + std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z))); };
    float max_ffn_nonzero_error = 0.0f;
    for (std::uint32_t r = 0; r < rows; ++r) for (std::uint32_t c = 0; c < H; ++c) {
        double expected = double(ffn_input[r * H + c]) + double(ffn_state[w1n + b1n + w2n + c]);
        for (std::uint32_t j = 0; j < 4 * H; ++j) {
            double z = double(ffn_state[w1n + j]);
            for (std::uint32_t i = 0; i < H; ++i) z += double(ffn_input[r * H + i]) * double(ffn_state[i * (4 * H) + j]);
            expected += gelu(z) * double(ffn_state[w1n + b1n + j * H + c]);
        }
        max_ffn_nonzero_error = std::max(max_ffn_nonzero_error, std::abs(ffn_output[r * H + c] - float(expected)));
    }
    pass = pass && max_ffn_nonzero_error <= 3.0e-5f;
    std::vector<std::uint32_t> ffn_forward_mask(rows, 0u);
    std::vector<std::uint32_t> invalid_ffn_mask{1u, 2u, 0u, 0u};
    std::vector<std::uint32_t> hole_ffn_mask{1u, 0u, 1u, 0u};
    bool invalid_ffn_mask_rejected = graph.seed_rmsnorm_staged_mask_for_rows(invalid_ffn_mask.data(), rows) != 0 &&
                                     graph.seed_rmsnorm_staged_mask_for_rows(hole_ffn_mask.data(), rows) != 0;
    pass = pass && invalid_ffn_mask_rejected;
    for (std::uint32_t r = 0; r < rows / 2; ++r) ffn_forward_mask[r] = 1u;
    if (graph.seed_rmsnorm_staged_mask_for_rows(ffn_forward_mask.data(), rows) != 0 ||
        graph.run_ffn_forward(rows) != 0 || graph.readback_ffn_output_staged(ffn_output.data(), ffn_output.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state masked FFN forward failed\n";
        return 29;
    }
    float max_masked_ffn_forward_error = 0.0f;
    for (std::uint32_t r = 0; r < rows; ++r) for (std::uint32_t c = 0; c < H; ++c) {
        float expected = 0.0f;
        if (r < rows / 2) {
            double value = double(ffn_input[r * H + c]) + double(ffn_state[w1n + b1n + w2n + c]);
            for (std::uint32_t j = 0; j < 4 * H; ++j) {
                double z = double(ffn_state[w1n + j]);
                for (std::uint32_t i = 0; i < H; ++i) z += double(ffn_input[r * H + i]) * double(ffn_state[i * (4 * H) + j]);
                value += gelu(z) * double(ffn_state[w1n + b1n + j * H + c]);
            }
            expected = float(value);
        }
        max_masked_ffn_forward_error = std::max(max_masked_ffn_forward_error, std::abs(ffn_output[r * H + c] - expected));
    }
    pass = pass && max_masked_ffn_forward_error <= 3.0e-5f;
    std::fill(ffn_forward_mask.begin(), ffn_forward_mask.end(), 0u);
    if (graph.seed_rmsnorm_staged_mask_for_rows(ffn_forward_mask.data(), rows) != 0 ||
        graph.run_ffn_forward(rows) != 0 || graph.readback_ffn_output_staged(ffn_output.data(), ffn_output.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state all-masked FFN forward failed\n";
        return 30;
    }
    float max_all_masked_ffn_forward = 0.0f;
    for (float value : ffn_output) max_all_masked_ffn_forward = std::max(max_all_masked_ffn_forward, std::abs(value));
    pass = pass && max_all_masked_ffn_forward <= 1.0e-7f;
    std::fill(ffn_forward_mask.begin(), ffn_forward_mask.end(), 1u);
    if (graph.seed_rmsnorm_staged_mask_for_rows(ffn_forward_mask.data(), rows) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN forward mask restore failed\n";
        return 31;
    }
    std::vector<float> ffn_dy(rows * H), ffn_dx(rows * H);
    for (std::size_t i = 0; i < ffn_dy.size(); ++i) ffn_dy[i] = 0.03f * std::cos(float(i) * 0.19f);
    std::vector<float> composed_dy(rows * H), composed_dx(rows * H), composed_norm_dx(rows * H), composed_raw(Tcap * H), composed_inv(Tcap), composed_states(Tcap * H);
    for (std::size_t i = 0; i < composed_dy.size(); ++i) composed_dy[i] = 0.02f * std::sin(float(i) * 0.23f);
    if (graph.seed_rmsnorm_staged_gamma(gamma.data(), gamma.size()) != 0 ||
        graph.seed_rmsnorm_staged_mask_for_rows(mask.data(), rows) != 0 ||
        graph.run_rmsnorm_state_only_staged(rows) != 0 ||
        graph.readback_rmsnorm_state_staged(composed_raw.data(), composed_raw.size(), composed_inv.data(), composed_inv.size()) != 0 ||
        graph.readback_rmsnorm_states_staged(composed_states.data(), composed_states.size()) != 0 ||
        graph.seed_ffn_input_staged(composed_states.data(), rows * H) != 0 ||
        graph.seed_ffn_output_gradient_staged(composed_dy.data(), composed_dy.size()) != 0 ||
        graph.run_ffn_backward_staged(rows) != 0 ||
        graph.readback_ffn_dx_staged(composed_dx.data(), composed_dx.size()) != 0 ||
        graph.seed_rmsnorm_staged_dy(composed_dx.data(), composed_dx.size()) != 0 ||
        graph.run_rmsnorm_backward_staged(rows) != 0 ||
        graph.readback_rmsnorm_dx_staged(composed_norm_dx.data(), composed_norm_dx.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state composed backward setup failed\n";
        return 25;
    }
    float max_composed_norm_dx = 0.0f;
    for (std::uint32_t r = 0; r < rows; ++r) {
        double dot = 0.0;
        for (std::uint32_t c = 0; c < H; ++c) dot += double(composed_raw[r * H + c]) * double(gamma[c]) * double(composed_dx[r * H + c]);
        for (std::uint32_t c = 0; c < H; ++c) {
            double x = composed_raw[r * H + c], inv_r = composed_inv[r];
            float expected = float(double(gamma[c]) * inv_r * double(composed_dx[r * H + c]) - x * inv_r * inv_r * inv_r * dot / double(H));
            max_composed_norm_dx = std::max(max_composed_norm_dx, std::abs(composed_norm_dx[r * H + c] - expected));
        }
    }
    pass = pass && max_composed_norm_dx <= 3.0e-5f;
    if (graph.seed_rmsnorm_staged_gamma(gamma_after.data(), gamma_after.size()) != 0 ||
        graph.seed_ffn_input_staged(ffn_input.data(), ffn_input.size()) != 0) return 25;
    if (graph.seed_ffn_output_gradient_staged(ffn_dy.data(), ffn_dy.size()) != 0 ||
        graph.run_ffn_backward_staged(rows) != 0 ||
        graph.readback_ffn_dx_staged(ffn_dx.data(), ffn_dx.size()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN backward failed\n";
        return 21;
    }
    float max_ffn_dx_error = 0.0f;
    for (std::uint32_t r = 0; r < rows; ++r) for (std::uint32_t o = 0; o < H; ++o) {
        double expected = double(ffn_dy[r * H + o]);
        for (std::uint32_t j = 0; j < 4 * H; ++j) {
            double z = double(ffn_state[w1n + j]);
            for (std::uint32_t i = 0; i < H; ++i) z += double(ffn_input[r * H + i]) * double(ffn_state[i * (4 * H) + j]);
            double dz = 0.5 * (1.0 + std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)));
            double u = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z);
            double gd = 0.5 * (1.0 + std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z))) +
                        0.5 * z * (1.0 - std::pow(std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)), 2.0)) * u;
            (void)dz;
            double q = 0.0;
            for (std::uint32_t k = 0; k < H; ++k) q += double(ffn_dy[r * H + k]) * double(ffn_state[w1n + b1n + j * H + k]);
            expected += gd * q * double(ffn_state[o * (4 * H) + j]);
        }
        max_ffn_dx_error = std::max(max_ffn_dx_error, std::abs(ffn_dx[r * H + o] - float(expected)));
    }
    pass = pass && max_ffn_dx_error <= 5.0e-5f;

    std::vector<float> fw1, fb1, fw2, fb2;
    fw1.resize(w1n); fb1.resize(b1n); fw2.resize(w2n); fb2.resize(b2n);
    if (graph.run_ffn_parameter_gradients(rows) != 0 ||
        graph.readback_ffn_gradients_staged(fw1.data(), fb1.data(), fw2.data(), fb2.data()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN gradient failed\n";
        return 22;
    }
    float max_ffn_grad_error = 0.0f, max_fw1_error = 0.0f, max_fb1_error = 0.0f, max_fw2_error = 0.0f, max_fb2_error = 0.0f;
    for (std::uint32_t i = 0; i < H; ++i) for (std::uint32_t j = 0; j < 4 * H; ++j) {
        double expected = 0.0;
        for (std::uint32_t r = 0; r < rows; ++r) {
            double z = double(ffn_state[w1n + j]);
            for (std::uint32_t k = 0; k < H; ++k) z += double(ffn_input[r * H + k]) * double(ffn_state[k * (4 * H) + j]);
            double t = std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z));
            double gd = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z);
            double q = 0.0;
            for (std::uint32_t o = 0; o < H; ++o) q += double(ffn_dy[r * H + o]) * double(ffn_state[w1n + b1n + j * H + o]);
            expected += double(ffn_input[r * H + i]) * gd * q;
        }
        max_fw1_error = std::max(max_fw1_error, std::abs(fw1[i * (4 * H) + j] - float(expected)));
    }
    for (std::uint32_t j = 0; j < 4 * H; ++j) {
        double expected_b1 = 0.0;
        for (std::uint32_t r = 0; r < rows; ++r) {
        double z = double(ffn_state[w1n + j]);
        for (std::uint32_t k = 0; k < H; ++k) z += double(ffn_input[r * H + k]) * double(ffn_state[k * (4 * H) + j]);
        double t = std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z));
        double gd = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z);
        double q = 0.0;
        for (std::uint32_t o = 0; o < H; ++o) q += double(ffn_dy[r * H + o]) * double(ffn_state[w1n + b1n + j * H + o]);
        expected_b1 += gd * q;
        }
        max_fb1_error = std::max(max_fb1_error, std::abs(fb1[j] - float(expected_b1)));
    }
    for (std::uint32_t j = 0; j < 4 * H; ++j) for (std::uint32_t o = 0; o < H; ++o) {
        double expected = 0.0;
        for (std::uint32_t r = 0; r < rows; ++r) {
            double z = double(ffn_state[w1n + j]);
            for (std::uint32_t k = 0; k < H; ++k) z += double(ffn_input[r * H + k]) * double(ffn_state[k * (4 * H) + j]);
            expected += gelu(z) * double(ffn_dy[r * H + o]);
        }
        max_fw2_error = std::max(max_fw2_error, std::abs(fw2[j * H + o] - float(expected)));
    }
    for (std::uint32_t o = 0; o < H; ++o) {
        double expected = 0.0; for (std::uint32_t r = 0; r < rows; ++r) expected += ffn_dy[r * H + o];
        max_fb2_error = std::max(max_fb2_error, std::abs(fb2[o] - float(expected)));
    }
    max_ffn_grad_error = std::max(std::max(max_fw1_error, max_fb1_error), std::max(max_fw2_error, max_fb2_error));
    pass = pass && max_ffn_grad_error <= 8.0e-5f;

    // Central finite differences on one parameter from each FFN group verify
    // the staged gradient chain independently of the closed-form reference.
    auto ffn_objective = [&](std::vector<float> const& state) {
        if (graph.update_ffn_state(state.data(), state.size(), step_after) != 0 ||
            graph.run_ffn_forward(rows) != 0 ||
            graph.readback_ffn_output_staged(ffn_output.data(), ffn_output.size()) != 0) return std::numeric_limits<double>::quiet_NaN();
        double value = 0.0;
        for (std::size_t i = 0; i < ffn_output.size(); ++i) value += double(ffn_output[i]) * double(ffn_dy[i]);
        return value;
    };
    float max_ffn_fd_error = 0.0f;
    const float fd_h = 1.0e-3f;
    std::vector<std::size_t> fd_indices{0u, w1n, w1n + b1n, w1n + b1n + w2n};
    std::vector<float> fd_plus = ffn_state, fd_minus = ffn_state;
    for (std::size_t n = 0; n < fd_indices.size(); ++n) {
        fd_plus[fd_indices[n]] += fd_h;
        fd_minus[fd_indices[n]] -= fd_h;
        double numerical = (ffn_objective(fd_plus) - ffn_objective(fd_minus)) / (2.0 * fd_h);
        float analytic = n == 0 ? fw1[0] : n == 1 ? fb1[0] : n == 2 ? fw2[0] : fb2[0];
        max_ffn_fd_error = std::max(max_ffn_fd_error, std::abs(float(numerical) - analytic));
        fd_plus[fd_indices[n]] = ffn_state[fd_indices[n]];
        fd_minus[fd_indices[n]] = ffn_state[fd_indices[n]];
    }
    (void)ffn_objective(ffn_state);
    pass = pass && std::isfinite(max_ffn_fd_error) && max_ffn_fd_error <= 2.0e-4f;

    // Re-run parameter gradients with a trailing-valid mask and verify that
    // masked rows do not contribute to any FFN parameter gradient.
    std::vector<std::uint32_t> ffn_mask(rows, 0u);
    for (std::uint32_t r = 0; r < rows / 2; ++r) ffn_mask[r] = 1u;
    std::vector<float> masked_fw1(w1n), masked_fb1(b1n), masked_fw2(w2n), masked_fb2(b2n);
    if (graph.seed_rmsnorm_staged_mask_for_rows(ffn_mask.data(), rows) != 0 ||
        graph.run_ffn_parameter_gradients(rows) != 0 ||
        graph.readback_ffn_gradients_staged(masked_fw1.data(), masked_fb1.data(), masked_fw2.data(), masked_fb2.data()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state masked FFN gradient failed\n";
        return 23;
    }
    float max_masked_ffn_grad_error = 0.0f;
    for (std::uint32_t i = 0; i < H; ++i) for (std::uint32_t j = 0; j < 4 * H; ++j) {
        double expected = 0.0;
        for (std::uint32_t r = 0; r < rows / 2; ++r) {
            double z = double(ffn_state[w1n + j]);
            for (std::uint32_t k = 0; k < H; ++k) z += double(ffn_input[r * H + k]) * double(ffn_state[k * (4 * H) + j]);
            double t = std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z));
            double gd = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z);
            double q = 0.0; for (std::uint32_t o = 0; o < H; ++o) q += double(ffn_dy[r * H + o]) * double(ffn_state[w1n + b1n + j * H + o]);
            expected += double(ffn_input[r * H + i]) * gd * q;
        }
        max_masked_ffn_grad_error = std::max(max_masked_ffn_grad_error, std::abs(masked_fw1[i * (4 * H) + j] - float(expected)));
    }
    for (std::uint32_t j = 0; j < 4 * H; ++j) {
        double expected = 0.0;
        for (std::uint32_t r = 0; r < rows / 2; ++r) {
            double z = double(ffn_state[w1n + j]); for (std::uint32_t k = 0; k < H; ++k) z += double(ffn_input[r * H + k]) * double(ffn_state[k * (4 * H) + j]);
            double t = std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z));
            double gd = 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z);
            double q = 0.0; for (std::uint32_t o = 0; o < H; ++o) q += double(ffn_dy[r * H + o]) * double(ffn_state[w1n + b1n + j * H + o]);
            expected += gd * q;
        }
        max_masked_ffn_grad_error = std::max(max_masked_ffn_grad_error, std::abs(masked_fb1[j] - float(expected)));
    }
    for (std::uint32_t j = 0; j < 4 * H; ++j) for (std::uint32_t o = 0; o < H; ++o) {
        double expected = 0.0; for (std::uint32_t r = 0; r < rows / 2; ++r) {
            double z = double(ffn_state[w1n + j]); for (std::uint32_t k = 0; k < H; ++k) z += double(ffn_input[r * H + k]) * double(ffn_state[k * (4 * H) + j]);
            expected += gelu(z) * double(ffn_dy[r * H + o]);
        }
        max_masked_ffn_grad_error = std::max(max_masked_ffn_grad_error, std::abs(masked_fw2[j * H + o] - float(expected)));
    }
    for (std::uint32_t o = 0; o < H; ++o) { double expected = 0.0; for (std::uint32_t r = 0; r < rows / 2; ++r) expected += ffn_dy[r * H + o]; max_masked_ffn_grad_error = std::max(max_masked_ffn_grad_error, std::abs(masked_fb2[o] - float(expected))); }
    pass = pass && max_masked_ffn_grad_error <= 8.0e-5f;
    float max_masked_ffn_fd_error = 0.0f;
    auto masked_ffn_objective = [&](std::vector<float> const& state) {
        if (graph.update_ffn_state(state.data(), state.size(), step_after) != 0 ||
            graph.run_ffn_forward(rows) != 0 ||
            graph.readback_ffn_output_staged(ffn_output.data(), ffn_output.size()) != 0) return std::numeric_limits<double>::quiet_NaN();
        double value = 0.0;
        for (std::uint32_t r = 0; r < rows / 2; ++r)
            for (std::uint32_t c = 0; c < H; ++c) value += double(ffn_output[r * H + c]) * double(ffn_dy[r * H + c]);
        return value;
    };
    fd_plus = ffn_state; fd_minus = ffn_state;
    for (std::size_t n = 0; n < fd_indices.size(); ++n) {
        fd_plus[fd_indices[n]] += fd_h;
        fd_minus[fd_indices[n]] -= fd_h;
        double numerical = (masked_ffn_objective(fd_plus) - masked_ffn_objective(fd_minus)) / (2.0 * fd_h);
        float analytic = n == 0 ? masked_fw1[0] : n == 1 ? masked_fb1[0] : n == 2 ? masked_fw2[0] : masked_fb2[0];
        max_masked_ffn_fd_error = std::max(max_masked_ffn_fd_error, std::abs(float(numerical) - analytic));
        fd_plus[fd_indices[n]] = ffn_state[fd_indices[n]];
        fd_minus[fd_indices[n]] = ffn_state[fd_indices[n]];
    }
    (void)masked_ffn_objective(ffn_state);
    pass = pass && std::isfinite(max_masked_ffn_fd_error) && max_masked_ffn_fd_error <= 2.0e-4f;
    std::fill(ffn_mask.begin(), ffn_mask.end(), 0u);
    std::vector<float> all_fw1(w1n), all_fb1(b1n), all_fw2(w2n), all_fb2(b2n);
    if (graph.seed_rmsnorm_staged_mask_for_rows(ffn_mask.data(), rows) != 0 ||
        graph.run_ffn_parameter_gradients(rows) != 0 ||
        graph.readback_ffn_gradients_staged(all_fw1.data(), all_fb1.data(), all_fw2.data(), all_fb2.data()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state all-masked FFN gradient failed\n";
        return 24;
    }
    float max_all_masked_ffn_grad = 0.0f;
    for (float v : all_fw1) max_all_masked_ffn_grad = std::max(max_all_masked_ffn_grad, std::abs(v));
    for (float v : all_fb1) max_all_masked_ffn_grad = std::max(max_all_masked_ffn_grad, std::abs(v));
    for (float v : all_fw2) max_all_masked_ffn_grad = std::max(max_all_masked_ffn_grad, std::abs(v));
    for (float v : all_fb2) max_all_masked_ffn_grad = std::max(max_all_masked_ffn_grad, std::abs(v));
    pass = pass && max_all_masked_ffn_grad <= 1.0e-7f;

    // Restore active mask and verify one all-group AdamW step for FFN state.
    for (std::uint32_t r = 0; r < rows / 2; ++r) ffn_mask[r] = 1u;
    if (graph.seed_rmsnorm_staged_mask_for_rows(ffn_mask.data(), rows) != 0 ||
        graph.run_ffn_parameter_gradients(rows) != 0 ||
        graph.readback_ffn_gradients_staged(fw1.data(), fb1.data(), fw2.data(), fb2.data()) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN mask restore failed\n";
        return 25;
    }
    std::vector<float> ffn_before(3 * group), ffn_after(3 * group);
    std::uint64_t ffn_step_before = 0;
    std::uint64_t ffn_step_after = 0;
    if (graph.readback_ffn_state(ffn_before.data(), ffn_before.size(), &ffn_step_before) != 0 ||
        graph.run_ffn_adamw_staged(lr, beta1, beta2, eps, decay) != 0 ||
        graph.readback_ffn_state(ffn_after.data(), ffn_after.size(), &ffn_step_after) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN AdamW failed\n";
        return 26;
    }
    float max_ffn_adamw_error = 0.0f;
    std::uint64_t expected_ffn_step = ffn_step_before + 1;
    for (std::size_t i = 0; i < group; ++i) {
        std::size_t local = i;
        float g = (local < w1n ? fw1[local] : local < w1n + b1n ? fb1[local - w1n] : local < w1n + b1n + w2n ? fw2[local - w1n - b1n] : fb2[local - w1n - b1n - w2n]);
        float m1 = (1.0f - beta1) * g, v1 = (1.0f - beta2) * g * g;
        float expected = (1.0f - lr * decay) * ffn_before[i] - lr * (m1 / (1.0f - std::pow(beta1, float(expected_ffn_step)))) / (std::sqrt(v1 / (1.0f - std::pow(beta2, float(expected_ffn_step)))) + eps);
        max_ffn_adamw_error = std::max(max_ffn_adamw_error, std::abs(ffn_after[i] - expected));
        max_ffn_adamw_error = std::max(max_ffn_adamw_error, std::abs(ffn_after[group + i] - m1));
        max_ffn_adamw_error = std::max(max_ffn_adamw_error, std::abs(ffn_after[2 * group + i] - v1));
    }
    pass = pass && ffn_step_after == expected_ffn_step && max_ffn_adamw_error <= 8.0e-5f;
    auto ffn_first = ffn_after;
    if (graph.run_ffn_adamw_staged(lr, beta1, beta2, eps, decay) != 0 ||
        graph.readback_ffn_state(ffn_after.data(), ffn_after.size(), &ffn_step_after) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN second AdamW failed\n";
        return 28;
    }
    float max_ffn_adamw_two_error = 0.0f;
    for (std::size_t i = 0; i < group; ++i) {
        std::size_t local = i;
        float g = (local < w1n ? fw1[local] : local < w1n + b1n ? fb1[local - w1n] : local < w1n + b1n + w2n ? fw2[local - w1n - b1n] : fb2[local - w1n - b1n - w2n]);
        float m1 = (1.0f - beta1) * g, v1 = (1.0f - beta2) * g * g;
        float m2 = beta1 * m1 + (1.0f - beta1) * g, v2 = beta2 * v1 + (1.0f - beta2) * g * g;
        float expected = (1.0f - lr * decay) * ffn_first[i] - lr * (m2 / (1.0f - std::pow(beta1, float(expected_ffn_step + 1)))) /
                         (std::sqrt(v2 / (1.0f - std::pow(beta2, float(expected_ffn_step + 1)))) + eps);
        max_ffn_adamw_two_error = std::max(max_ffn_adamw_two_error, std::abs(ffn_after[i] - expected));
        max_ffn_adamw_two_error = std::max(max_ffn_adamw_two_error, std::abs(ffn_after[group + i] - m2));
        max_ffn_adamw_two_error = std::max(max_ffn_adamw_two_error, std::abs(ffn_after[2 * group + i] - v2));
    }
    pass = pass && ffn_step_after == expected_ffn_step + 1 && max_ffn_adamw_two_error <= 8.0e-5f;
    // The graph uses one checkpoint step counter for all staged AdamW groups;
    // FFN's update advances it after the earlier gamma two-step check.
    step_after = ffn_step_after;

    bool invalid_ffn_staged_rejected =
        graph.run_ffn_adamw_staged(0.0f, beta1, beta2, eps, decay) != 0 &&
        graph.run_ffn_adamw_staged(lr, 1.0f, beta2, eps, decay) != 0 &&
        graph.run_ffn_adamw_staged(lr, beta1, beta2, 0.0f, decay) != 0 &&
        graph.run_ffn_adamw_staged(lr, beta1, beta2, eps, -1.0f) != 0 &&
        graph.seed_ffn_input_staged(nullptr, rows * H) != 0 &&
        graph.seed_ffn_input_staged(ffn_input.data(), rows * H - 1) != 0 &&
        graph.seed_ffn_output_gradient_staged(nullptr, rows * H) != 0 &&
        graph.seed_ffn_output_gradient_staged(ffn_dy.data(), rows * H - 1) != 0 &&
        graph.run_ffn_forward(0) != 0 && graph.run_ffn_forward(Tcap + 1) != 0 &&
        graph.run_ffn_backward_staged(0) != 0 && graph.run_ffn_backward_staged(Tcap + 1) != 0 &&
        graph.run_ffn_parameter_gradients(0) != 0 && graph.run_ffn_parameter_gradients(Tcap + 1) != 0 &&
        graph.readback_ffn_output_staged(ffn_output.data(), rows * H - 1) != 0 &&
        graph.readback_ffn_dx_staged(ffn_dx.data(), rows * H - 1) != 0;
    pass = pass && invalid_ffn_staged_rejected;

    std::vector<float> gamma_restored(H), m_restored(H), v_restored(H);
    std::uint64_t restored_step = 0;
    if (graph.readback_rmsnorm_gamma_state_staged(gamma_restored.data(), m_restored.data(), v_restored.data(), &restored_step) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state gamma restore readback failed\n";
        return 14;
    }
    float max_restore_error = 0.0f;
    for (std::uint32_t c = 0; c < H; ++c) {
        max_restore_error = std::max(max_restore_error, std::abs(gamma_restored[c] - gamma_after[c]));
        max_restore_error = std::max(max_restore_error, std::abs(m_restored[c] - m_after[c]));
        max_restore_error = std::max(max_restore_error, std::abs(v_restored[c] - v_after[c]));
    }
    pass = pass && restored_step == step_after && max_restore_error <= 1.0e-7f;

    std::vector<float> ffn_restored(3 * group);
    std::uint64_t ffn_restored_step = 0;
    if (graph.update_ffn_state(ffn_after.data(), ffn_after.size(), ffn_step_after) != 0 ||
        graph.readback_ffn_state(ffn_restored.data(), ffn_restored.size(), &ffn_restored_step) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state FFN restore failed\n";
        return 27;
    }
    float max_ffn_restore_error = 0.0f;
    for (std::size_t i = 0; i < ffn_after.size(); ++i)
        max_ffn_restore_error = std::max(max_ffn_restore_error, std::abs(ffn_restored[i] - ffn_after[i]));
    pass = pass && ffn_restored_step == ffn_step_after && max_ffn_restore_error <= 1.0e-7f;

    // An out-of-range checkpoint step must be rejected without mutating state.
    if (graph.update_gamma_state(gamma_restored.data(), m_restored.data(), v_restored.data(), UINT64_MAX) == 0) {
        std::cerr << "tiny_rmsnorm_staged_state accepted oversized step\n";
        return 16;
    }
    std::vector<float> gamma_after_reject(H), m_after_reject(H), v_after_reject(H);
    std::uint64_t step_after_reject = 0;
    if (graph.readback_rmsnorm_gamma_state_staged(gamma_after_reject.data(), m_after_reject.data(), v_after_reject.data(), &step_after_reject) != 0) {
        std::cerr << "tiny_rmsnorm_staged_state oversized-step readback failed\n";
        return 17;
    }
    float max_reject_error = 0.0f;
    for (std::uint32_t c = 0; c < H; ++c) {
        max_reject_error = std::max(max_reject_error, std::abs(gamma_after_reject[c] - gamma_restored[c]));
        max_reject_error = std::max(max_reject_error, std::abs(m_after_reject[c] - m_restored[c]));
        max_reject_error = std::max(max_reject_error, std::abs(v_after_reject[c] - v_restored[c]));
    }
    std::vector<float> ffn_reject(3 * group);
    std::uint64_t ffn_reject_step = 0;
    bool ffn_bad_step_rejected = graph.update_ffn_state(ffn_restored.data(), ffn_restored.size(), UINT64_MAX) != 0 &&
                                 graph.readback_ffn_state(ffn_reject.data(), ffn_reject.size(), &ffn_reject_step) == 0;
    float max_ffn_reject_error = 0.0f;
    if (ffn_bad_step_rejected) for (std::size_t i = 0; i < ffn_restored.size(); ++i)
        max_ffn_reject_error = std::max(max_ffn_reject_error, std::abs(ffn_reject[i] - ffn_restored[i]));
    pass = pass && step_after_reject == restored_step && max_reject_error <= 1.0e-7f &&
           ffn_bad_step_rejected && ffn_reject_step == ffn_restored_step && max_ffn_reject_error <= 1.0e-7f;
    std::cout << "tiny_rmsnorm_staged_state " << (pass ? "PASS" : "FAIL")
              << " raw=" << max_raw_error << " inv=" << max_inv_error
              << " dx=" << max_dx_error << " dgamma=" << max_dgamma_error
              << " masked_dx=" << max_masked_dx << " masked_dgamma=" << max_masked_dgamma
              << " all_masked=" << max_all_masked << " max_raw=" << max_window_raw
              << " max_inv=" << max_window_inv << " max_state=" << max_window_state
               << " partial_state=" << max_partial_state << " all_masked_state=" << max_all_masked_state
               << " composed_norm_dx=" << max_composed_norm_dx << " ffn_nonzero=" << max_ffn_nonzero_error << " masked_ffn_forward=" << max_masked_ffn_forward_error << " all_masked_ffn_forward=" << max_all_masked_ffn_forward << " ffn_dx=" << max_ffn_dx_error << " ffn_grad=" << max_ffn_grad_error << " ffn_fd=" << max_ffn_fd_error << " masked_ffn_fd=" << max_masked_ffn_fd_error << " masked_ffn_grad=" << max_masked_ffn_grad_error << " all_masked_ffn_grad=" << max_all_masked_ffn_grad << " ffn_adamw=" << max_ffn_adamw_error << " ffn_adamw_two=" << max_ffn_adamw_two_error << " ffn_restore=" << max_ffn_restore_error << " invalid_ffn=" << (invalid_ffn_staged_rejected ? 0 : 1) << " fw1=" << max_fw1_error << " fb1=" << max_fb1_error << " fw2=" << max_fw2_error << " fb2=" << max_fb2_error << " max_adamw=" << max_adamw_error << " max_adamw_two=" << max_adamw_two_error << " ffn=" << max_ffn_error
              << " restore=" << max_restore_error << " reject=" << max_reject_error << " ffn_reject=" << max_ffn_reject_error << " step=" << step_after_reject << "\n";
    return pass ? 0 : 3;
}
