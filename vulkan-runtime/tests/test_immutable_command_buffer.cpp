#include "core/vk_setup.h"
#include "exec/engine.h"

#include <iostream>
#include <stdexcept>

int main() {
    try {
        auto ctx = vulkan_runtime::core::create_context("immutable-command-buffer-test");
        {
            vulkan_runtime::exec::ExecEngine engine(ctx, 2, 1);
            engine.recordImmutable([](vk::CommandBuffer) {});
            if (!engine.immutableCommandBufferRetained()) {
                std::cerr << "retained command buffer capability not reported\n";
                return 1;
            }
            auto first = engine.submitImmutable();
            auto second = engine.submitImmutable();
            if (first == 0 || second != first + 1) {
                std::cerr << "immutable resubmit values are not monotonic\n";
                return 1;
            }
            engine.recordImmutable([](vk::CommandBuffer) {}, true);
            if (!engine.immutableSecondaryCommandBufferRetained() || !engine.immutableCommandBuffersRetained()) {
                std::cerr << "secondary retained command buffer capability not reported\n";
                return 1;
            }
            auto third = engine.submitImmutable(0, true);
            if (third != second + 1) {
                std::cerr << "secondary immutable resubmit value is not monotonic\n";
                return 1;
            }
            engine.drain();
        }
        vulkan_runtime::core::destroy_context(ctx);
        std::cout << "immutable command buffer retained and resubmitted\n";
        return 0;
    } catch (std::exception const& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
