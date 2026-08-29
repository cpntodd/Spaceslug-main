#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {
void destroy(spaceslug_tiny_forward_graph* graph) {
    if (graph) spaceslug_tiny_forward_destroy(graph);
}
void destroy(spaceslug_tiny_base_checkpoint* checkpoint) {
    if (checkpoint) spaceslug_tiny_base_checkpoint_destroy(checkpoint);
}
}

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> e(V * H, 0.1f), p(Tcap * H, 0.02f), q(H * H, 0.0f), k(H * H, 0.0f), v(H * H, 0.0f), o(H * H, 0.0f), lm(H * Vp, 0.0f);
    for (std::uint32_t i = 0; i < H; ++i) {
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
        lm[i * Vp + (i % V)] = 0.01f;
    }
    auto* graph = spaceslug_tiny_forward_create_full(e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    if (!graph) return 1;
    auto* checkpoint = spaceslug_tiny_base_checkpoint_create();
    if (!checkpoint || spaceslug_tiny_forward_readback_base_checkpoint(graph, checkpoint) != 0) { destroy(checkpoint); destroy(graph); return 1; }
    const char* capability = spaceslug_tiny_forward_full_base_training_capability();
    if (!capability || std::string(capability).find("unsupported") == std::string::npos) { destroy(checkpoint); destroy(graph); return 1; }
    const std::uint32_t expected_mask = (1u << (SPACESLUG_TINY_BASE_TRAIN_GROUP_LM_HEAD - 1)) | (1u << (SPACESLUG_TINY_BASE_TRAIN_GROUP_OUTPUT - 1)) |
                                        (1u << (SPACESLUG_TINY_BASE_TRAIN_GROUP_QKV - 1)) | (1u << (SPACESLUG_TINY_BASE_TRAIN_GROUP_EMBEDDINGS - 1)) |
                                        (1u << (SPACESLUG_TINY_BASE_TRAIN_GROUP_POSITIONS - 1)) | (1u << (SPACESLUG_TINY_BASE_TRAIN_GROUP_NORMALIZATION - 1)) |
                                        (1u << (SPACESLUG_TINY_BASE_TRAIN_GROUP_FFN - 1));
    const std::uint32_t ffn_count = H * 4 * H + 4 * H + 4 * H * H + H;
    const auto normalization_group = SPACESLUG_TINY_BASE_TRAIN_GROUP_NORMALIZATION;
    const auto ffn_group = SPACESLUG_TINY_BASE_TRAIN_GROUP_FFN;
    bool ok = spaceslug_tiny_base_checkpoint_group_mask(checkpoint) == expected_mask &&
              spaceslug_tiny_base_checkpoint_profile_rank(checkpoint) == LoraRank4 &&
              spaceslug_tiny_base_checkpoint_float_count(checkpoint, normalization_group) == H &&
              spaceslug_tiny_base_checkpoint_float_count(checkpoint, ffn_group) == ffn_count &&
              spaceslug_tiny_base_checkpoint_weights(checkpoint, normalization_group) != nullptr &&
              spaceslug_tiny_base_checkpoint_weights(checkpoint, ffn_group) != nullptr &&
              spaceslug_tiny_base_checkpoint_state_float_count(checkpoint, normalization_group) == H &&
              spaceslug_tiny_base_checkpoint_state_float_count(checkpoint, ffn_group) == ffn_count;
    for (std::uint32_t state = 0; state < 3; ++state)
        ok = ok && spaceslug_tiny_base_checkpoint_normalization_state(checkpoint, state) != nullptr;
    for (std::uint32_t component = 0; component < 4; ++component)
        for (std::uint32_t state = 0; state < 3; ++state)
            ok = ok && spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, component, state) != nullptr;
    ok = ok && spaceslug_tiny_base_checkpoint_float_count(checkpoint, 0) == 0 &&
         spaceslug_tiny_base_checkpoint_float_count(checkpoint, 127) == 0 &&
         spaceslug_tiny_base_checkpoint_state_float_count(checkpoint, 0) == 0 &&
         spaceslug_tiny_base_checkpoint_state_float_count(checkpoint, 127) == 0 &&
         spaceslug_tiny_base_checkpoint_weights(checkpoint, 0) == nullptr &&
         spaceslug_tiny_base_checkpoint_weights(checkpoint, 127) == nullptr &&
         spaceslug_tiny_base_checkpoint_normalization_state(checkpoint, 3) == nullptr &&
         spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 4, 0) == nullptr &&
         spaceslug_tiny_base_checkpoint_ffn_state(checkpoint, 0, 3) == nullptr;
    auto* restored = spaceslug_tiny_forward_create_full(e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    ok = ok && restored && spaceslug_tiny_forward_update_base_checkpoint(restored, checkpoint) == 0;

    // Resume from schema-v4 with nonzero moments and require deterministic
    // continuation of the newly integrated gamma/FFN optimizer state.
    std::vector<std::uint32_t> tokens(Tcap, 1), targets(Tcap, 2), masks(Tcap, 0);
    constexpr std::uint32_t rows = 4;
    for (std::uint32_t i = 0; i < rows; ++i) { tokens[i] = i + 1; targets[i] = i + 2; masks[i] = 1; }
    const auto train = [&](spaceslug_tiny_forward_graph* g) {
        return spaceslug_tiny_forward_train_bounded_full_graph_adamw(g, tokens.data(), targets.data(), masks.data(), rows,
                                                                       1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f);
    };
    ok = ok && restored && train(graph) == 0 && train(restored) == 0;
    auto* continued = spaceslug_tiny_base_checkpoint_create();
    auto* resumed = spaceslug_tiny_base_checkpoint_create();
    ok = ok && continued && resumed &&
         spaceslug_tiny_forward_readback_base_checkpoint(graph, continued) == 0 &&
         spaceslug_tiny_forward_readback_base_checkpoint(restored, resumed) == 0 &&
         spaceslug_tiny_base_checkpoint_adamw_step(continued) == 1 &&
         spaceslug_tiny_base_checkpoint_adamw_step(resumed) == 1;
    for (std::uint32_t state = 0; state < 3; ++state)
        for (std::uint32_t i = 0; i < H; ++i)
            ok = ok && std::isfinite(spaceslug_tiny_base_checkpoint_normalization_state(continued, state)[i]) &&
                 std::isfinite(spaceslug_tiny_base_checkpoint_normalization_state(resumed, state)[i]) &&
                 spaceslug_tiny_base_checkpoint_normalization_state(continued, state)[i] ==
                     spaceslug_tiny_base_checkpoint_normalization_state(resumed, state)[i];
    bool nonzero_moment = false;
    for (std::uint32_t component = 0; component < 4; ++component)
        for (std::uint32_t state = 0; state < 3; ++state) {
            auto* a = spaceslug_tiny_base_checkpoint_ffn_state(continued, component, state);
            auto* b = spaceslug_tiny_base_checkpoint_ffn_state(resumed, component, state);
            const std::uint32_t n = component == 0 ? H * 4 * H : component == 1 ? 4 * H : component == 2 ? 4 * H * H : H;
            for (std::uint32_t i = 0; i < n; ++i) {
                ok = ok && std::isfinite(a[i]) && std::isfinite(b[i]) && a[i] == b[i];
                if ((state == 1 || state == 2) && std::fabs(a[i]) > 1.0e-12f) nonzero_moment = true;
            }
        }
    ok = ok && nonzero_moment;

    // The C ABI handle is opaque by design; exercise the same schema-v4
    // validator through the C++ checkpoint carrier so malformed metadata cannot
    // reach the device update path.
    auto context = vulkan_runtime::core::create_context("schema-v4-malformed-checkpoint");
    ForwardResourceGraph validator_graph(context, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
    BaseCheckpoint malformed;
    bool malformed_rejected = validator_graph.readback_base_checkpoint(malformed) == 0;
    auto before_malformed = malformed.gamma;
    auto before_malformed_ffn = malformed.ffn_w1;
    malformed.version = 3;
    malformed_rejected = malformed_rejected && validator_graph.update_base_checkpoint(malformed) != 0;
    malformed.version = 4;
    malformed.profile.lora_rank = LoraRank8;
    malformed_rejected = malformed_rejected && validator_graph.update_base_checkpoint(malformed) != 0;
    malformed.profile.lora_rank = LoraRank4;
    malformed.group_mask = 0;
    malformed_rejected = malformed_rejected && validator_graph.update_base_checkpoint(malformed) != 0;
    malformed.group_mask = expected_mask;
    malformed.gamma[0] = std::numeric_limits<float>::quiet_NaN();
    malformed_rejected = malformed_rejected && validator_graph.update_base_checkpoint(malformed) != 0;
    malformed.gamma[0] = before_malformed[0];
    malformed.adamw_step = UINT64_MAX;
    malformed_rejected = malformed_rejected && validator_graph.update_base_checkpoint(malformed) != 0;
    malformed.adamw_step = 0;
    malformed.ffn_w1[0] = std::numeric_limits<float>::infinity();
    malformed_rejected = malformed_rejected && validator_graph.update_base_checkpoint(malformed) != 0;
    malformed.ffn_w1[0] = before_malformed_ffn[0];
    ok = ok && malformed_rejected;

    malformed.gamma[0] = -0.0f;
     malformed.ffn_b2[0] = std::numeric_limits<float>::max();
     malformed_rejected = malformed_rejected && validator_graph.update_base_checkpoint(malformed) == 0;
     malformed.gamma[0] = before_malformed[0];
     malformed.ffn_b2[0] = 0.0f;
     malformed_rejected = malformed_rejected && validator_graph.update_base_checkpoint(malformed) == 0;
     ok = ok && malformed_rejected;

     BaseCheckpoint after_malformed;
    bool state_preserved = validator_graph.readback_base_checkpoint(after_malformed) == 0 &&
                           after_malformed.gamma == before_malformed && after_malformed.ffn_w1 == before_malformed_ffn;
    ok = ok && state_preserved;

    spaceslug_tiny_base_checkpoint_destroy(resumed);
    spaceslug_tiny_base_checkpoint_destroy(continued);
    spaceslug_tiny_base_checkpoint_destroy(checkpoint);
    spaceslug_tiny_forward_destroy(restored);
    spaceslug_tiny_forward_destroy(graph);
    if (!ok) return 1;
    std::cout << "Schema-v4 checkpoint C ABI: PASS normalization=" << H << " ffn=" << ffn_count << " resume_step=1 nonzero_moments=1\n";
    return 0;
}
