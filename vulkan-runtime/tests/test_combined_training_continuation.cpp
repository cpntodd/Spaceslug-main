#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> embeddings(V * H, 0.002f), positions(Tcap * H, 0.001f);
    std::vector<float> query(H * H, 0.0f), key(H * H, 0.0f), value(H * H, 0.0f), output(H * H, 0.0f), lm(H * Vp, 0.0f);
    for (std::uint32_t i = 0; i < H; ++i) {
        query[i * H + i] = key[i * H + i] = value[i * H + i] = output[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.01f;
    }
    auto context = vulkan_runtime::core::create_context("combined-training-continuation");
    ForwardResourceGraph graph(context, embeddings.data(), positions.data(), query.data(), key.data(), value.data(), output.data(), lm.data());
    constexpr std::uint32_t rows = 2;
    std::uint32_t tokens[rows] = {1, 2}, targets[rows] = {2, 3}, masks[rows] = {1, 1};
    constexpr float lr = 1.0e-3f, beta1 = 0.9f, beta2 = 0.999f, eps = 1.0e-8f, decay = 0.01f;
    std::vector<float> gamma(H), gm(H), gv(H), ffn(3 * (H * 4 * H + 4 * H + 4 * H * H + H));
    std::fill(ffn.begin(), ffn.end(), 0.001f);
    if (graph.update_ffn_state(ffn.data(), ffn.size(), 0) != 0) return 1;
    const std::vector<float> ffn_initial = ffn;
    std::vector<float> gamma_initial(H, 1.0f);
    std::uint64_t gamma_step = 0, ffn_step = 0;
    if (graph.readback_gamma_state(gamma.data(), gm.data(), gv.data(), &gamma_step) != 0 || gamma_step != 0 ||
        graph.readback_ffn_state(ffn.data(), ffn.size(), &ffn_step) != 0 || ffn_step != 0)
        return 1;
    if (std::abs(gamma[0] - 1.0f) > 1.0e-6f || !std::all_of(ffn.begin(), ffn.end(), [](float x) { return std::isfinite(x); }))
        return 1;
    if (graph.train_positions_adamw(tokens, targets, masks, rows, lr, beta1, beta2, eps, decay) != 0 ||
        graph.readback_gamma_state(gamma.data(), gm.data(), gv.data(), &gamma_step) != 0 ||
        graph.readback_ffn_state(ffn.data(), ffn.size(), &ffn_step) != 0 || gamma_step != 1 || ffn_step != 1)
        return 1;
    bool gamma_changed = false, ffn_changed = false;
    for (std::size_t i = 0; i < gamma.size(); ++i) gamma_changed |= std::abs(gamma[i] - gamma_initial[i]) > 1.0e-8f;
    for (std::size_t i = 0; i < ffn_initial.size(); ++i) ffn_changed |= std::abs(ffn[i] - ffn_initial[i]) > 1.0e-8f;
    if (!gamma_changed || !ffn_changed || !std::all_of(gamma.begin(), gamma.end(), [](float x) { return std::isfinite(x); }) ||
        !std::all_of(ffn.begin(), ffn.end(), [](float x) { return std::isfinite(x); }))
        return 1;
    const std::vector<float> gamma_after_first = gamma;
    const std::vector<float> ffn_after_first = ffn;
    if (graph.train_positions_adamw(tokens, targets, masks, rows, lr, beta1, beta2, eps, decay) != 0 ||
        graph.readback_gamma_state(gamma.data(), gm.data(), gv.data(), &gamma_step) != 0 ||
        graph.readback_ffn_state(ffn.data(), ffn.size(), &ffn_step) != 0 || gamma_step != 2 || ffn_step != 2)
        return 1;
    bool gamma_changed_again = false, ffn_changed_again = false;
    for (std::size_t i = 0; i < gamma.size(); ++i) gamma_changed_again |= std::abs(gamma[i] - gamma_after_first[i]) > 1.0e-8f;
    for (std::size_t i = 0; i < ffn.size(); ++i) ffn_changed_again |= std::abs(ffn[i] - ffn_after_first[i]) > 1.0e-8f;
    if (!gamma_changed_again || !ffn_changed_again ||
        !std::all_of(gamma.begin(), gamma.end(), [](float x) { return std::isfinite(x); }) ||
        !std::all_of(ffn.begin(), ffn.end(), [](float x) { return std::isfinite(x); }))
        return 1;
    const auto gamma_before_reject = gamma;
    const auto ffn_before_reject = ffn;
    const auto reject_step = gamma_step;
    if (graph.train_positions_adamw(nullptr, targets, masks, rows, lr, beta1, beta2, eps, decay) == 0 ||
        graph.train_positions_adamw(tokens, targets, masks, rows, -lr, beta1, beta2, eps, decay) == 0 ||
        graph.train_positions_adamw(tokens, targets, masks, rows, lr, beta1, beta2, eps, -decay) == 0 ||
        graph.train_positions_adamw(tokens, targets, masks, rows, NAN, beta1, beta2, eps, decay) == 0)
        return 1;
    std::vector<float> gamma_after_reject(H), ffn_after_reject(ffn.size());
    std::uint64_t reject_gamma_step = 0, reject_ffn_step = 0;
    if (graph.readback_gamma_state(gamma_after_reject.data(), nullptr, nullptr, &reject_gamma_step) == 0 ||
        graph.readback_gamma_state(gamma_after_reject.data(), gm.data(), gv.data(), &reject_gamma_step) != 0 ||
        graph.readback_ffn_state(ffn_after_reject.data(), ffn_after_reject.size(), &reject_ffn_step) != 0 ||
        reject_gamma_step != reject_step || reject_ffn_step != reject_step || gamma_after_reject != gamma_before_reject ||
        ffn_after_reject != ffn_before_reject)
        return 1;
    std::cout << "Combined training continuation: PASS gamma_step=" << gamma_step << " ffn_step=" << ffn_step
              << " gamma_changed=" << gamma_changed << " ffn_changed=" << ffn_changed
              << " gamma_changed_again=" << gamma_changed_again << " ffn_changed_again=" << ffn_changed_again << "\n";
    return 0;
}
