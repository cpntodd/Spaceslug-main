#include "api/tiny_immutable_prototype.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>

int main() {
    try {
        auto context = vulkan_runtime::core::create_context("tiny-immutable-prototype-test");
        {
        vulkan_runtime::tiny::ImmutableCommandPrototype prototype(context);
        std::array<std::uint32_t, 8> tokens{1, 2, 3, 4, 5, 6, 7, 8};
        std::array<std::uint32_t, 8> targets{0, 1, 1, 2, 3, 5, 8, 13};
        std::array<std::uint32_t, 8> mask{1, 1, 1, 1, 1, 1, 1, 1};
        std::array<float, 8> doutput{1, 1, 1, 1, 1, 1, 1, 1};
        std::array<float, 2> learning{0.5f, 0.01f};
        std::array<std::uint32_t, 9> control{};
        auto first = prototype.run(tokens, targets, mask, doutput, learning, control);
        tokens[0] = 100;
        targets[7] = 100;
        control[8] = 42;
        auto second = prototype.run(tokens, targets, mask, doutput, learning, control);
        if (prototype.last_submission() < 2 || first == second || second[3] != 42.0f ||
            !std::isfinite(second[0]) || !std::isfinite(second[1]) || !std::isfinite(second[2])) {
            std::cerr << "immutable Tiny prototype did not reuse command with changed inputs\n";
            return 1;
        }
        context.device.waitIdle();
        }
        vulkan_runtime::core::destroy_context(context);
        std::cout << "immutable Tiny prototype reused fixed-shape command with host staging\n";
        return 0;
    } catch (std::exception const& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
