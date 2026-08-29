#define VMA_IMPLEMENTATION
#include "core/vk_setup.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vulkan_runtime::core {
namespace {

constexpr char const* kValidationLayerName = "VK_LAYER_KHRONOS_validation";

// Returns true if the named instance layer is reported as available.
bool layer_available(std::string const& name) {
    std::vector<vk::LayerProperties> layers = vk::enumerateInstanceLayerProperties();
    for (vk::LayerProperties const& layer : layers) {
        if (name == layer.layerName) {
            return true;
        }
    }
    return false;
}

// Prefer a discrete GPU (the RX580); fall back to the first enumerated device
// if none is discrete (e.g. lavapipe reports CPU).
vk::PhysicalDevice pick_physical_device(vk::Instance const& instance) {
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan physical device found.");
    }

    for (vk::PhysicalDevice const& device : devices) {
        vk::PhysicalDeviceProperties props = device.getProperties();
        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            return device;
        }
    }
    return devices.front();
}

// Find a queue family index that supports compute. Prefers a *pure-compute*
// family (COMPUTE without GRAPHICS) — on gfx803/Polaris RADV exposes its 4 ACE
// engines as a separate compute-only family (index 1) alongside the
// graphics+compute family — and falls back to the first compute-capable family
// otherwise (lavapipe exposes a single graphics+compute+transfer family).
std::uint32_t find_compute_queue_family(vk::PhysicalDevice const& device) {
    std::vector<vk::QueueFamilyProperties> families = device.getQueueFamilyProperties();
    std::uint32_t fallback = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        if (!(families[i].queueFlags & vk::QueueFlagBits::eCompute)) {
            continue;
        }
        if (fallback == std::numeric_limits<std::uint32_t>::max()) {
            fallback = i;
        }
        if (!(families[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
            return i; // pure-compute (ACE) family
        }
    }
    if (fallback != std::numeric_limits<std::uint32_t>::max()) {
        return fallback;
    }
    throw std::runtime_error("No compute-capable queue family found.");
}

} // namespace

VulkanContext create_context(char const* appName) {
    VulkanContext ctx;

    // --- Instance -----------------------------------------------------------
    vk::ApplicationInfo appInfo;
    appInfo.setPApplicationName(appName);
    appInfo.setApplicationVersion(VK_MAKE_VERSION(0, 1, 0));
    appInfo.setPEngineName("vulkan-runtime");
    appInfo.setEngineVersion(VK_MAKE_VERSION(0, 1, 0));
    appInfo.setApiVersion(VK_API_VERSION_1_4);

    vk::InstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.setPApplicationInfo(&appInfo);

    // Enable validation only when it is actually present; degrade to a warning
    // otherwise so the build stays green without the layer.
    if (layer_available(kValidationLayerName)) {
        instanceCreateInfo.setPEnabledLayerNames(kValidationLayerName);
        ctx.validationEnabled = true;
    }

    ctx.instance = vk::createInstance(instanceCreateInfo);

    // --- Physical device ----------------------------------------------------
    ctx.physicalDevice = pick_physical_device(ctx.instance);

    // --- Feature queries ----------------------------------------------------
    // Timeline semaphores (Vulkan 1.2 core) and synchronization2 (1.3 core)
    // are required by the exec engine. Query support up front and enable only
    // what the driver actually reports, so this context still builds on
    // drivers that lack either feature (the M0-M4 entry points use neither).
    vk::PhysicalDeviceFeatures2 features2;
    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineQuery;
    vk::PhysicalDeviceSynchronization2Features sync2Query;
    features2.setPNext(&timelineQuery);
    timelineQuery.setPNext(&sync2Query);
    ctx.physicalDevice.getFeatures2(&features2);

    bool const timelineSupported = timelineQuery.timelineSemaphore == VK_TRUE;
    bool const sync2Supported = sync2Query.synchronization2 == VK_TRUE;

    // --- Logical device + compute queues ------------------------------------
    ctx.computeQueueFamily = find_compute_queue_family(ctx.physicalDevice);

    // Create up to 4 queues from the chosen family: the pure-compute ACE family
    // on Polaris advertises 4; lavapipe's single compute-capable family
    // advertises 1. Queue priorities must be sized to the exact queue count
    // requested, and computeQueueCount records the number actually created so
    // consumers (the exec engine) never over-request via getQueue().
    std::vector<vk::QueueFamilyProperties> familyProps =
        ctx.physicalDevice.getQueueFamilyProperties();
    std::uint32_t const familyQueueCount = familyProps[ctx.computeQueueFamily].queueCount;
    ctx.computeQueueCount =
        std::max<std::uint32_t>(1u, std::min<std::uint32_t>(4u, familyQueueCount));

    std::vector<float> queuePriorities(ctx.computeQueueCount, 1.0f);
    vk::DeviceQueueCreateInfo queueCreateInfo;
    queueCreateInfo.setQueueFamilyIndex(ctx.computeQueueFamily);
    queueCreateInfo.setQueueCount(ctx.computeQueueCount);
    queueCreateInfo.setPQueuePriorities(queuePriorities.data());

    // Build the pNext chain of features to enable at device creation, only
    // for features the device reported as supported. Independent features, so
    // order is irrelevant; sync2 is linked first, timeline at the head.
    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineEnable;
    vk::PhysicalDeviceSynchronization2Features sync2Enable;
    void* pNext = nullptr;
    if (sync2Supported) {
        sync2Enable.setSynchronization2(VK_TRUE);
        sync2Enable.setPNext(pNext);
        pNext = &sync2Enable;
        ctx.synchronization2Enabled = true;
    }
    if (timelineSupported) {
        timelineEnable.setTimelineSemaphore(VK_TRUE);
        timelineEnable.setPNext(pNext);
        pNext = &timelineEnable;
        ctx.timelineSemaphoreEnabled = true;
    }

    vk::DeviceCreateInfo deviceCreateInfo;
    deviceCreateInfo.setQueueCreateInfos(queueCreateInfo);
    deviceCreateInfo.setPNext(pNext);

    ctx.device = ctx.physicalDevice.createDevice(deviceCreateInfo);
    ctx.computeQueue = ctx.device.getQueue(ctx.computeQueueFamily, 0);

    // --- VMA allocator ------------------------------------------------------
    // VMA resolves the rest of the Vulkan API dynamically through these two
    // loader entry points.
    VmaVulkanFunctions vmaVulkanFunctions{};
    vmaVulkanFunctions.vkGetInstanceProcAddr = &::vkGetInstanceProcAddr;
    vmaVulkanFunctions.vkGetDeviceProcAddr = &::vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorCreateInfo{};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_4;
    allocatorCreateInfo.physicalDevice = static_cast<VkPhysicalDevice>(ctx.physicalDevice);
    allocatorCreateInfo.device = static_cast<VkDevice>(ctx.device);
    allocatorCreateInfo.instance = static_cast<VkInstance>(ctx.instance);
    allocatorCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;

    if (vmaCreateAllocator(&allocatorCreateInfo, &ctx.allocator) != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateAllocator failed.");
    }

    return ctx;
}

void destroy_context(VulkanContext& ctx) {
    if (ctx.allocator != nullptr) {
        vmaDestroyAllocator(ctx.allocator);
        ctx.allocator = nullptr;
    }
    if (ctx.device) {
        ctx.device.destroy();
        ctx.device = nullptr;
    }
    if (ctx.instance) {
        ctx.instance.destroy();
        ctx.instance = nullptr;
    }
}

} // namespace vulkan_runtime::core
