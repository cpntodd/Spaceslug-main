#include "api/tiny_forward_persistent.h"
#include <cstring>
#include <iostream>

int main() {
    if (std::strcmp(spaceslug_tiny_forward_full_base_training_capability(),
                    "full_base_training_unsupported_gamma_ffn_schema_v4_cabi_dataset_retained_arbitrary_shape_gated") != 0 ||
        std::strcmp(spaceslug_tiny_forward_bounded_full_graph_training_capability(),
                    "tiny_fixed_profile_bounded_full_graph_adamw_true_rmsnorm_ffn_position_cabi_vvl_clean") != 0 ||
        spaceslug_tiny_forward_readback_ffn_state(nullptr, nullptr, 0, nullptr) != -1 ||
        spaceslug_tiny_forward_update_ffn_state(nullptr, nullptr, 0, 0) != -1 ||
        spaceslug_tiny_forward_readback_gamma_state(nullptr, nullptr, nullptr, nullptr, nullptr) != -1 ||
        spaceslug_tiny_forward_update_gamma_state(nullptr, nullptr, nullptr, nullptr, 0) != -1 ||
        spaceslug_tiny_forward_readback_base_checkpoint(nullptr, nullptr) != 1 ||
        spaceslug_tiny_forward_update_base_checkpoint(nullptr, nullptr) != 1 ||
        spaceslug_tiny_forward_train_dataset_batch_full(nullptr, nullptr, 1.0e-3f, 1.0f) != 1 ||
        spaceslug_tiny_forward_train_bounded_full_graph_adamw(nullptr, nullptr, nullptr, nullptr, 0, 1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != 1)
        return 1;
    std::cout << "FFN C ABI state boundary: PASS bounded=1 full_base=0\n";
    return 0;
}
