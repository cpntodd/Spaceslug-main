#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <iostream>
#include <string>

int main() {
    using namespace vulkan_runtime::tiny;
    auto context = vulkan_runtime::core::create_context("sync-validation-gate");
    const bool layer_present = context.validationEnabled;
    const std::string capability = full_base_training_capability;
    if (!layer_present || capability.find("gated") == std::string::npos || capability.find("unsupported") == std::string::npos) {
        vulkan_runtime::core::destroy_context(context);
        return 1;
    }
    std::cout << "Synchronization validation gate: PASS validation=1 full_base=gated\n";
    vulkan_runtime::core::destroy_context(context);
    return 0;
}
