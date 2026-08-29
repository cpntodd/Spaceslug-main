#include "api/embedding_training_api.h"
#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <vector>

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

bool expect(bool condition, char const* message) {
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

} // namespace

int main() {
    using vulkan_runtime::tiny::ForwardResourceGraph;
    using namespace vulkan_runtime::tiny;
    bool ok = true;

    ok &= expect(std::strcmp(spaceslug_tiny_forward_capability(),
                             "tiny_forward_persistent_bounded_resource_graph_training_lora_rank4_rank8_sgd_adamw") == 0,
                 "forward capability identifies the two Tiny profiles");
    ok &= expect(std::strcmp(spaceslug_tiny_forward_base_train_capability(),
                             "base_train_group_lm_head_output_qkv_embeddings_owned_fp32_fixed_window_sgd_rows_le_128_lm_head_adamw_output_adamw_qkv_adamw_from_gradients_embeddings_sparse_sgd_positions_table_sgd_no_ffn_norm_dataset_retained_no_standalone_bridge") == 0,
                 "base-training capability states ownership and unsupported groups");
    ok &= expect(std::strcmp(spaceslug_tiny_forward_bounded_full_graph_training_capability(),
                             "tiny_fixed_profile_bounded_full_graph_adamw_true_rmsnorm_ffn_position_cabi_vvl_clean") == 0,
                 "bounded full-graph capability is explicit and validation-clean");
    ok &= expect(vulkan_runtime::tiny::bounded_full_graph_max_rows == vulkan_runtime::tiny::Tcap &&
                     vulkan_runtime::tiny::bounded_full_graph_requires_trailing_mask &&
                     !vulkan_runtime::tiny::bounded_full_graph_uses_retained_commands,
                 "bounded full-graph fixed-window contract is explicit");
    ok &= expect(spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_LM_HEAD) == 1,
                 "LM head is graph-integrated");
    ok &= expect(spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_OUTPUT) == 1,
                 "output projection is graph-integrated");
    ok &= expect(spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_QKV) == 1,
                 "QKV projections are graph-integrated");
    ok &= expect(spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_NONE) == 0,
                 "None is not a trainable group");
    ok &= expect(spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_EMBEDDINGS) == 1,
                 "embeddings are graph-integrated");
     ok &= expect(spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_POSITIONS) == 1,
                  "positions are graph-integrated");
     ok &= expect(spaceslug_tiny_forward_base_train_group_supported(6) == 0,
                  "unassigned groups remain unsupported");
    ok &= expect(spaceslug_tiny_forward_base_train_group_supported(127) == 0,
                 "out-of-range group bit remains unsupported");
    ok &= expect(spaceslug_tiny_forward_lora_rank_supported(4) == 1 &&
                     spaceslug_tiny_forward_lora_rank_supported(8) == 1 &&
                     spaceslug_tiny_forward_lora_rank_supported(0) == 0 &&
                     spaceslug_tiny_forward_lora_rank_supported(6) == 0,
                 "only rank4 and rank8 are supported");
    ok &= expect(std::strcmp(spaceslug_tiny_forward_graph_embedding_training_capability(),
                             "tiny_graph_embedding_dstate_gradient_sgd_graph_owned_tokens_fixed_window_rows_le_128_one_submit_cpu_parity") == 0,
                 "graph embedding capability states graph-owned one-submit SGD");
    ok &= expect(spaceslug_tiny_forward_graph_embedding_training_status() == 0,
                 "graph embedding training is supported");
    ok &= expect(std::strcmp(spaceslug_embedding_training_capability(),
                             "standalone_fp32_embedding_training_V259_H64_deterministic_sparse_sgd_no_tiny_graph_integration") == 0,
                 "standalone embedding capability remains a separate API");
    ok &= expect(spaceslug_tiny_forward_graph_embedding_training_capability() !=
                     spaceslug_embedding_training_capability(),
                 "graph and standalone embedding capabilities are distinct");
    // Positions remain a graph-integrated group. Gamma and FFN state are exposed
    // through schema-v4 state APIs and the bounded full-graph ABI, but are not
    // selectable base-group IDs until the complete full-base contract is ready.

    // This test deliberately uses only the metadata C API. In particular, it
    // must remain runnable on hosts without a Vulkan device or ICD.
    ok &= expect(spaceslug_tiny_profile_count() == 2, "profile count is two");

    spaceslug_tiny_profile_descriptor rank4{};
    spaceslug_tiny_profile_descriptor rank8{};
    ok &= expect(spaceslug_tiny_profile_query(0, &rank4) == SPACESLUG_TINY_PROFILE_SUPPORTED,
                 "rank4 profile query succeeds");
    ok &= expect(spaceslug_tiny_profile_query(1, &rank8) == SPACESLUG_TINY_PROFILE_SUPPORTED,
                 "rank8 profile query succeeds");
    ok &= expect(rank4.name != nullptr && rank4.hidden == 64 && rank4.vocab == 259 && rank4.padded_vocab == 320 &&
                     rank4.token_capacity == 128 && rank4.lora_rank == 4,
                 "rank4 descriptor matches the supported shape");
    ok &= expect(rank8.name != nullptr && rank8.hidden == 64 && rank8.vocab == 259 && rank8.padded_vocab == 320 &&
                     rank8.token_capacity == 128 && rank8.lora_rank == 8,
                 "rank8 descriptor matches the supported shape");

    ok &= expect(spaceslug_tiny_profile_validate(64, 259, 320, 128, 4) == SPACESLUG_TINY_PROFILE_SUPPORTED,
                 "rank4 profile validates");
    ok &= expect(spaceslug_tiny_profile_validate(64, 259, 320, 128, 8) == SPACESLUG_TINY_PROFILE_SUPPORTED,
                 "rank8 profile validates");

    ok &= expect(spaceslug_tiny_profile_validate(63, 259, 320, 128, 4) == SPACESLUG_TINY_PROFILE_UNSUPPORTED,
                 "unsupported hidden dimension is rejected");
    ok &= expect(spaceslug_tiny_profile_validate(64, 258, 320, 128, 4) == SPACESLUG_TINY_PROFILE_UNSUPPORTED,
                 "unsupported vocab dimension is rejected");
    ok &= expect(spaceslug_tiny_profile_validate(64, 259, 321, 128, 4) == SPACESLUG_TINY_PROFILE_UNSUPPORTED,
                 "unsupported padded vocab dimension is rejected");
    ok &= expect(spaceslug_tiny_profile_validate(64, 259, 320, 127, 4) == SPACESLUG_TINY_PROFILE_UNSUPPORTED,
                 "unsupported token capacity is rejected");
    ok &= expect(spaceslug_tiny_profile_validate(64, 259, 320, 128, 6) == SPACESLUG_TINY_PROFILE_UNSUPPORTED,
                 "unsupported rank is rejected");

    spaceslug_tiny_profile_descriptor untouched{nullptr, 1, 2, 3, 4, 5};
    ok &= expect(spaceslug_tiny_profile_query(0, nullptr) == SPACESLUG_TINY_PROFILE_INVALID_ARGUMENT,
                 "null query output is rejected");
    ok &= expect(spaceslug_tiny_profile_query(2, &untouched) == SPACESLUG_TINY_PROFILE_UNSUPPORTED,
                 "out-of-range profile query is rejected");
    ok &= expect(untouched.name == nullptr && untouched.hidden == 1 && untouched.vocab == 2 &&
                     untouched.padded_vocab == 3 && untouched.token_capacity == 4 && untouched.lora_rank == 5,
                 "failed query leaves output untouched");

    ok &= expect(spaceslug_tiny_profile_validate(0, 259, 320, 128, 4) == SPACESLUG_TINY_PROFILE_INVALID_ARGUMENT,
                 "null hidden dimension is invalid");
    ok &= expect(spaceslug_tiny_profile_validate(64, 0, 320, 128, 4) == SPACESLUG_TINY_PROFILE_INVALID_ARGUMENT,
                 "null vocab dimension is invalid");
    ok &= expect(spaceslug_tiny_profile_validate(64, 259, 320, 128, 0) == SPACESLUG_TINY_PROFILE_INVALID_ARGUMENT,
                 "null rank is invalid");

    if (!ok)
        return 1;

    auto context = vulkan_runtime::core::create_context("tiny-profile-constructor");
    std::vector<float> e(V * H, 0.0f), p(Tcap * H, 0.0f), lm(H * Vp, 0.0f);
    ok &= expect(ForwardResourceGraph(context, e.data(), p.data(), lm.data(), 4).profile_descriptor().lora_rank == 4,
                 "rank4 constructor selects rank4 profile");
    ok &= expect(ForwardResourceGraph(context, e.data(), p.data(), lm.data(), 8).profile_descriptor().lora_rank == 8,
                 "rank8 constructor selects rank8 profile");
    ok &= expect(spaceslug_tiny_forward_create_rank(nullptr, nullptr, nullptr, 6) == nullptr,
                 "C ABI invalid rank and null inputs fail closed");
    if (!ok)
        return 1;
    std::cout << "tiny profile metadata API passed without Vulkan initialization\n";
    return 0;
}

