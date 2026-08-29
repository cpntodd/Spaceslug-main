// Headless-compute Vulkan setup shared across the runtime: instance, physical
// device, logical device + compute queue, and a VMA allocator.
//
// Plain (non-owning) Vulkan-Hpp handles are used — matching the M0 smoke
// pattern — and torn down explicitly via destroy_context(). Do not copy a
// context (the handles are shared); move is fine.

#pragma once

// VMA must be configured before including vk_mem_alloc.h. These macros must
// stay identical in every translation unit that includes this header.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <vulkan/vulkan.hpp>

#include <cstdint>

namespace vulkan_runtime::core {

// Bundles the Vulkan objects every headless-compute entry point needs.
struct VulkanContext {
    vk::Instance instance{};
    vk::PhysicalDevice physicalDevice{};
    vk::Device device{};
    vk::Queue computeQueue{};
    std::uint32_t computeQueueFamily{0};
    VmaAllocator allocator{nullptr};
    bool validationEnabled{false};

    // Feature flags, set by create_context(). True only when the feature was
    // both reported as supported AND enabled on the logical device.
    bool timelineSemaphoreEnabled{false};  // VkPhysicalDeviceTimelineSemaphoreFeatures
    bool synchronization2Enabled{false};   // VkPhysicalDeviceSynchronization2Features

    // Number of compute queues actually created on the device. create_context
    // requests up to 4 queues from the chosen family (all 4 ACE queues on
    // Polaris; 1 on lavapipe), exposing the extra queues for M5b cross-queue
    // overlap. computeQueue is always queue 0 of computeQueueFamily.
    std::uint32_t computeQueueCount{0};

    VulkanContext() = default;
    VulkanContext(VulkanContext const&) = delete;
    VulkanContext& operator=(VulkanContext const&) = delete;
    VulkanContext(VulkanContext&&) = default;
    VulkanContext& operator=(VulkanContext&&) = default;
};

// Creates an instance (with VK_LAYER_KHRONOS_validation when present), picks a
// compute-capable physical device (preferring a discrete GPU), creates a
// logical device + compute queue, and initializes a VMA allocator.
//
// Throws std::runtime_error / vk::SystemError on failure.
VulkanContext create_context(char const* appName);

// Tears down a context created by create_context(), in reverse creation order.
void destroy_context(VulkanContext& ctx);

} // namespace vulkan_runtime::core
