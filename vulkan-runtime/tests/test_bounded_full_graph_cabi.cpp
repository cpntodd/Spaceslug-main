#include "api/tiny_forward_persistent.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> e(V * H, 0.1f), p(Tcap * H, 0.02f), q(H * H, 0.0f), k(H * H, 0.0f), v(H * H, 0.0f), o(H * H, 0.0f), lm(H * Vp, 0.0f);
    for (std::uint32_t i = 0; i < H; ++i) {
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.01f;
    }
    auto* graph = spaceslug_tiny_forward_create_full(e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    if (!graph) return 1;
    if (spaceslug_tiny_forward_train_dataset_batch_full(graph, nullptr, 1.0e-3f, 1.0f) != 1) {
        spaceslug_tiny_forward_destroy(graph);
        return 1;
    }
    bool capability_contract_ok =
        spaceslug_tiny_forward_retained_backward_optimizer_supported() == 0 &&
        spaceslug_tiny_forward_arbitrary_shape_full_base_supported() == 0 &&
        spaceslug_tiny_forward_full_base_training_capability() != nullptr &&
        spaceslug_tiny_forward_bounded_full_graph_training_capability() != nullptr &&
        spaceslug_tiny_forward_retained_backward_optimizer_capability() != nullptr &&
        spaceslug_tiny_forward_arbitrary_shape_full_base_capability() != nullptr &&
        spaceslug_tiny_forward_ffn_capability() != nullptr;
    if (!capability_contract_ok) { spaceslug_tiny_forward_destroy(graph); return 1; }
    std::vector<std::uint32_t> tokens(Tcap, 1), targets(Tcap, 2), masks(Tcap, 0);
    constexpr std::uint32_t rows = 4;
    for (std::uint32_t i = 0; i < rows; ++i) { tokens[i] = i + 1; targets[i] = i + 2; masks[i] = 1; }
    const std::size_t w1_count = H * 4 * H, b1_count = 4 * H, w2_count = 4 * H * H, b2_count = H;
    const std::size_t group = w1_count + b1_count + w2_count + b2_count;
    std::vector<float> ffn(3 * group, 0.001f), gamma(H, 1.0f), zero(H, 0.0f);
    float before = 0.0f, after = 0.0f;
    std::uint32_t count = 0;
    int result = spaceslug_tiny_forward_update_ffn_state(graph, ffn.data(), ffn.size(), 0);
    result |= spaceslug_tiny_forward_update_gamma_state(graph, gamma.data(), zero.data(), zero.data(), 0);
    result |= spaceslug_tiny_forward_loss_fixed_metrics(graph, tokens.data(), targets.data(), masks.data(), &before, &count);
    std::uint64_t step = 99;
    std::vector<float> gamma_state(H), gamma_m(H), gamma_v(H);
    result |= spaceslug_tiny_forward_readback_gamma_state(graph, gamma_state.data(), gamma_m.data(), gamma_v.data(), &step);
    std::vector<float> ffn_readback(ffn.size());
    std::uint64_t ffn_step = 99;
    result |= spaceslug_tiny_forward_readback_ffn_state(graph, ffn_readback.data(), ffn_readback.size(), &ffn_step);
    bool invalid_ffn_readback_rejected = spaceslug_tiny_forward_readback_ffn_state(graph, nullptr, ffn_readback.size(), &ffn_step) != 0 &&
                                         spaceslug_tiny_forward_readback_ffn_state(graph, ffn_readback.data(), ffn_readback.size() - 1, &ffn_step) != 0;
    bool invalid_gamma_readback_rejected = spaceslug_tiny_forward_readback_gamma_state(graph, nullptr, gamma_m.data(), gamma_v.data(), &step) != 0;
    float max_ffn_restore = 0.0f;
    for (std::size_t i = 0; i < ffn.size(); ++i) max_ffn_restore = std::max(max_ffn_restore, std::abs(ffn_readback[i] - ffn[i]));
    bool invalid_ffn_rejected = spaceslug_tiny_forward_update_ffn_state(graph, nullptr, ffn.size(), 0) != 0 &&
                                 spaceslug_tiny_forward_update_ffn_state(graph, ffn.data(), ffn.size() - 1, 0) != 0 &&
                                 spaceslug_tiny_forward_update_ffn_state(graph, ffn.data(), ffn.size(), UINT64_MAX) != 0;
    bool invalid_gamma_rejected = spaceslug_tiny_forward_update_gamma_state(graph, nullptr, zero.data(), zero.data(), 0) != 0 &&
                                  spaceslug_tiny_forward_update_gamma_state(graph, gamma.data(), zero.data(), zero.data(), UINT64_MAX) != 0;
    if (result != 0 || count != rows || step != 0 || ffn_step != 0 || max_ffn_restore > 1.0e-7f ||
        !invalid_ffn_rejected || !invalid_gamma_rejected || !invalid_ffn_readback_rejected || !invalid_gamma_readback_rejected) {
        spaceslug_tiny_forward_destroy(graph); return 1;
    }

    auto* checkpoint = spaceslug_tiny_base_checkpoint_create();
    bool checkpoint_ok = checkpoint &&
                        spaceslug_tiny_forward_readback_base_checkpoint(graph, checkpoint) == 0 &&
                        spaceslug_tiny_base_checkpoint_group_mask(checkpoint) == 127u &&
                        spaceslug_tiny_base_checkpoint_adamw_step(checkpoint) == 0 &&
                        spaceslug_tiny_base_checkpoint_profile_rank(checkpoint) == LoraRank4 &&
                        spaceslug_tiny_base_checkpoint_float_count(checkpoint, SPACESLUG_TINY_BASE_TRAIN_GROUP_NORMALIZATION) == H &&
                        spaceslug_tiny_base_checkpoint_normalization_state(checkpoint, 1) != nullptr &&
                        spaceslug_tiny_base_checkpoint_normalization_state(checkpoint, 2) != nullptr &&
                        spaceslug_tiny_base_checkpoint_state_float_count(checkpoint, SPACESLUG_TINY_BASE_TRAIN_GROUP_FFN) == w1_count + b1_count + w2_count + b2_count &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 0, 0) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 0, 1) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 0, 2) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 1, 0) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 1, 1) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 1, 2) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 2, 0) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 2, 1) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 2, 2) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 3, 0) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 3, 1) != nullptr &&
                        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 3, 2) != nullptr;
    if (!checkpoint_ok) { spaceslug_tiny_forward_destroy(graph); return 1; }
    auto* modified_checkpoint = spaceslug_tiny_base_checkpoint_create();
    bool checkpoint_update_ok = modified_checkpoint && spaceslug_tiny_forward_readback_base_checkpoint(graph, modified_checkpoint) == 0;
    if (checkpoint_update_ok) {
        float* mutable_gamma = const_cast<float*>(spaceslug_tiny_base_checkpoint_normalization_state(modified_checkpoint, 0));
        if (!mutable_gamma) checkpoint_update_ok = false;
        else { mutable_gamma[0] += 0.125f; checkpoint_update_ok = spaceslug_tiny_forward_update_base_checkpoint(graph, modified_checkpoint) == 0; }
    }
    float gamma_after_checkpoint = 0.0f;
    if (checkpoint_update_ok) {
        result |= spaceslug_tiny_forward_readback_gamma_state(graph, gamma_state.data(), gamma_m.data(), gamma_v.data(), &step);
        gamma_after_checkpoint = gamma_state[0];
    }
    if (modified_checkpoint) spaceslug_tiny_base_checkpoint_destroy(modified_checkpoint);
    if (!checkpoint_update_ok || std::abs(gamma_after_checkpoint - 1.125f) > 1.0e-6f) { spaceslug_tiny_forward_destroy(graph); return 1; }
    std::vector<float> gamma_before_invalid = gamma_state;
    std::uint64_t step_before_invalid = step;
    bool checkpoint_null_rejected = spaceslug_tiny_forward_readback_base_checkpoint(graph, nullptr) != 0 &&
                                     spaceslug_tiny_forward_update_base_checkpoint(graph, nullptr) != 0;
    result |= spaceslug_tiny_forward_readback_gamma_state(graph, gamma_state.data(), gamma_m.data(), gamma_v.data(), &step);
    bool checkpoint_invalid_preserved = checkpoint_null_rejected && step == step_before_invalid &&
                                        std::abs(gamma_state[0] - gamma_before_invalid[0]) <= 1.0e-7f;
    if (!checkpoint_invalid_preserved) { spaceslug_tiny_forward_destroy(graph); return 1; }
    const bool invalid_optimizer_rejected =
         spaceslug_tiny_forward_train_bounded_full_graph_adamw(graph, nullptr, targets.data(), masks.data(), rows,
                                                               1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 0 &&
         spaceslug_tiny_forward_train_bounded_full_graph_adamw(graph, tokens.data(), targets.data(), masks.data(), 0,
                                                               1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 0 &&
         spaceslug_tiny_forward_train_bounded_full_graph_adamw(graph, tokens.data(), targets.data(), masks.data(), rows,
                                                               std::numeric_limits<float>::quiet_NaN(), 0.9f, 0.999f, 1.0e-8f, 0.0f) != 0 &&
         spaceslug_tiny_forward_train_bounded_full_graph_adamw(graph, tokens.data(), targets.data(), masks.data(), rows,
                                                               1.0e-3f, 0.9f, 0.999f, 1.0e-8f, -1.0f) != 0;
     if (!invalid_optimizer_rejected) { spaceslug_tiny_forward_destroy(graph); return 1; }
     bool checkpoint_accessors_fail_closed =
        spaceslug_tiny_base_checkpoint_group_mask(nullptr) == 0 &&
        spaceslug_tiny_base_checkpoint_adamw_step(nullptr) == 0 &&
        spaceslug_tiny_base_checkpoint_profile_rank(nullptr) == 0 &&
        spaceslug_tiny_base_checkpoint_float_count(nullptr, SPACESLUG_TINY_BASE_TRAIN_GROUP_FFN) == 0 &&
        spaceslug_tiny_base_checkpoint_weights(nullptr, SPACESLUG_TINY_BASE_TRAIN_GROUP_FFN) == nullptr &&
        spaceslug_tiny_base_checkpoint_normalization_state(nullptr, 0) == nullptr &&
        spaceslug_tiny_base_checkpoint_ffn_state(nullptr, 0, 0) == nullptr &&
        spaceslug_tiny_base_checkpoint_float_count(checkpoint, 99u) == 0 &&
        spaceslug_tiny_base_checkpoint_weights(checkpoint, 99u) == nullptr &&
        spaceslug_tiny_base_checkpoint_normalization_state(checkpoint, 3u) == nullptr &&
        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 4u, 0u) == nullptr &&
        spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 0u, 3u) == nullptr;
    spaceslug_tiny_base_checkpoint_destroy(checkpoint);
    if (!checkpoint_accessors_fail_closed || !invalid_optimizer_rejected) { spaceslug_tiny_forward_destroy(graph); return 1; }

    std::vector<std::uint32_t> middle_masks = masks;
    middle_masks[1] = 0;
    middle_masks[2] = 1;
    if (spaceslug_tiny_forward_train_bounded_full_graph_adamw(graph, tokens.data(), targets.data(), middle_masks.data(), rows,
                                                               1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 1 ||
        spaceslug_tiny_forward_readback_gamma_state(graph, gamma_state.data(), gamma_m.data(), gamma_v.data(), &step) != 0 || step != 0 ||
        spaceslug_tiny_forward_train_bounded_full_graph_adamw(graph, tokens.data(), targets.data(), masks.data(), Tcap + 1,
                                                               1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 1 ||
        spaceslug_tiny_forward_train_bounded_full_graph_adamw(graph, tokens.data(), targets.data(), masks.data(), rows,
                                                               0.0f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 1)
        { spaceslug_tiny_forward_destroy(graph); return 1; }

    result = spaceslug_tiny_forward_train_bounded_full_graph_adamw(graph, tokens.data(), targets.data(), masks.data(), rows,
                                                                    1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f);
    result |= spaceslug_tiny_forward_loss_fixed_metrics(graph, tokens.data(), targets.data(), masks.data(), &after, &count);
    result |= spaceslug_tiny_forward_readback_gamma_state(graph, gamma_state.data(), gamma_m.data(), gamma_v.data(), &step);
    std::vector<std::uint32_t> all_masked(Tcap, 0);
    float masked_loss = 0.0f;
    std::uint32_t masked_count = 99;
    result |= spaceslug_tiny_forward_loss_fixed_metrics(graph, tokens.data(), targets.data(), all_masked.data(), &masked_loss, &masked_count);
    const auto all_masked_result = spaceslug_tiny_forward_train_bounded_full_graph_adamw(
        graph, tokens.data(), targets.data(), all_masked.data(), rows, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f);
    result |= all_masked_result;
    result |= spaceslug_tiny_forward_readback_gamma_state(graph, gamma_state.data(), gamma_m.data(), gamma_v.data(), &step);
    spaceslug_tiny_forward_destroy(graph);
    if (result != 0 || count != rows || step != 2 || masked_count != 0 || !std::isfinite(masked_loss) ||
        !std::isfinite(before) || !std::isfinite(after) || !(after < before)) return 1;
    std::cout << "Bounded full-graph C ABI: PASS before=" << before << " after=" << after << "\n";
    return 0;
}
