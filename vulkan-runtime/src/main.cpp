// M0 smoke test — proves the scaffold: create a Vulkan instance, pick a
// compute-capable physical device (preferring a discrete GPU), create a
// logical device + compute queue, initialize a VMA allocator, print the
// device name, and tear everything down in reverse order.
//
// Headless: no WSI, no surface, no swapchain. The setup lives in
// src/core/vk_setup.{h,cpp} and is shared with the M1+ compute entry points.

#include "core/vk_setup.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-smoke");

        vk::PhysicalDeviceProperties deviceProps = ctx.physicalDevice.getProperties();
        std::string deviceName = deviceProps.deviceName;

        // --- Report ---------------------------------------------------------
        std::cout << "Device: " << deviceName << "\n";
        std::cout << "Vulkan: "
                  << VK_API_VERSION_MAJOR(deviceProps.apiVersion) << "."
                  << VK_API_VERSION_MINOR(deviceProps.apiVersion) << "."
                  << VK_API_VERSION_PATCH(deviceProps.apiVersion) << "\n";
        std::cout << "Validation layer ("
                  << "VK_LAYER_KHRONOS_validation"
                  << "): "
                  << (ctx.validationEnabled ? "enabled" : "not available") << "\n";
        std::cout << "VMA allocator: created\n";

        // --- Tear down ------------------------------------------------------
        vulkan_runtime::core::destroy_context(ctx);

        std::cout << "Smoke test: PASS\n";
        return EXIT_SUCCESS;
    } catch (vk::SystemError const& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
