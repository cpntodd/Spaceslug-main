#include "api/tiny_forward_persistent.h"
#include <cstring>
#include <iostream>

int main() {
    using namespace vulkan_runtime::tiny;
    if (ForwardResourceGraph::trainable_normalization_supported || ForwardResourceGraph::trainable_ffn_supported ||
        ForwardResourceGraph::trainable_full_base_constructor_supported || ForwardResourceGraph::dataset_training_full_supported ||
        retained_backward_optimizer_supported || arbitrary_shape_full_base_supported)
        return 1;
    if (std::strcmp(full_base_training_capability,
                   "full_base_training_unsupported_gamma_ffn_schema_v4_cabi_dataset_retained_arbitrary_shape_gated") != 0)
        return 1;
    if (std::strcmp(spaceslug_tiny_forward_bounded_full_graph_training_capability(),
                   "tiny_fixed_profile_bounded_full_graph_adamw_true_rmsnorm_ffn_position_cabi_vvl_clean") != 0)
        return 1;
    if (std::strcmp(ForwardResourceGraph::trainable_ffn_capability(),
                   "ffn_graph_state_allocated_forward_backward_w1_adamw_only") != 0)
        return 1;
    if (ForwardResourceGraph::trainable_normalization_unsupported != -5 ||
        ForwardResourceGraph::trainable_ffn_unsupported != -6 ||
        ForwardResourceGraph::dataset_training_full_unsupported != -4 ||
        spaceslug_tiny_forward_retained_backward_optimizer_supported() != 0 ||
        spaceslug_tiny_forward_arbitrary_shape_full_base_supported() != 0)
        return 1;
    std::cout << "Combined training gating: PASS bounded=1 full_base=0\n";
    return 0;
}
