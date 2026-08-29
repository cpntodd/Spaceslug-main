#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    using namespace vulkan_runtime::tiny;
    const std::string capability = full_base_training_capability;
    const std::string bounded = spaceslug_tiny_forward_bounded_full_graph_training_capability();
    const std::string dataset = ForwardResourceGraph::dataset_capability();
    const char* cabi_dataset = spaceslug_tiny_forward_dataset_training_capability();
    const char* retained = spaceslug_tiny_forward_retained_backward_optimizer_capability();
    const char* arbitrary = spaceslug_tiny_forward_arbitrary_shape_full_base_capability();
    if (!cabi_dataset || std::string(cabi_dataset) != dataset || !retained || !arbitrary || std::string(arbitrary).find("unsupported") == std::string::npos ||
        spaceslug_tiny_forward_arbitrary_shape_full_base_supported() != 0 || arbitrary_shape_full_base_supported || std::string(retained).find("unsupported") == std::string::npos ||
        spaceslug_tiny_forward_retained_backward_optimizer_supported() != 0 || retained_backward_optimizer_supported || ForwardResourceGraph::trainable_full_base_constructor_supported ||
        ForwardResourceGraph::trainable_normalization_supported ||
        ForwardResourceGraph::trainable_ffn_supported ||
        ForwardResourceGraph::dataset_training_full_supported ||
        capability.find("unsupported") == std::string::npos ||
         capability.find("gamma_ffn") == std::string::npos ||
         capability.find("schema_v4_cabi") == std::string::npos ||
         capability.find("dataset") == std::string::npos ||
         capability.find("retained") == std::string::npos ||
         capability.find("arbitrary_shape") == std::string::npos ||
        bounded != "tiny_fixed_profile_bounded_full_graph_adamw_true_rmsnorm_ffn_position_cabi_vvl_clean" ||
        dataset.find("production_bounded_persistent_tiny_dataset") == std::string::npos ||
        dataset.find("lm_head_sgd_device_windows") == std::string::npos ||
         std::string(retained) != "retained_forward_loss_only_backward_optimizer_unsupported" ||
         std::string(arbitrary) != "full_base_fixed_profile_only_arbitrary_shape_unsupported")
        return 1;
    auto context = vulkan_runtime::core::create_context("training-gates-unsupported");
    std::vector<float> e(V * H, 0.0f), p(Tcap * H, 0.0f), lm(H * Vp, 0.0f);
    ForwardResourceGraph graph(context, e.data(), p.data(), lm.data());
    std::vector<std::uint32_t> tokens(1, 0), targets(1, 0), masks(1, 1);
    if (graph.train_ffn_forward_loss(nullptr, targets.data(), masks.data(), 1) == 0 ||
        graph.train_ffn_forward_loss(tokens.data(), targets.data(), masks.data(), 0) == 0 ||
        graph.train_gamma_adamw(1.0e-3f, 0.9f, 0.999f, 1.0e-8f, 0.0f) != ForwardResourceGraph::trainable_normalization_unsupported ||
        graph.train_ffn_forward_loss(tokens.data(), targets.data(), masks.data(), 1) != 0) return 1;
    if (ForwardResourceGraph::dataset_training_full_supported ||
        ForwardResourceGraph::dataset_training_full_unsupported != -4 ||
        retained_backward_optimizer_supported ||
        arbitrary_shape_full_base_supported) return 1;
    std::cout << "Full-base training gates: PASS constructor=0 normalization=0 ffn=0 dataset_full=0 retained_backward_optimizer=0 arbitrary_shape_full_base=0\n";
    return 0;
}
