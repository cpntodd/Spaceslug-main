#include "core/vk_setup.h"
#include "embedded_shaders.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct B { vk::Buffer buffer{}; VmaAllocation allocation{}; };
B make_buffer(vulkan_runtime::core::VulkanContext const& c, vk::DeviceSize bytes, vk::BufferUsageFlags usage,
              VmaMemoryUsage memory, VmaAllocationCreateFlags flags = 0) {
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = bytes;
    ci.usage = static_cast<VkBufferUsageFlags>(usage);
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo ai{};
    ai.usage = memory;
    ai.flags = flags;
    VkBuffer raw{};
    VmaAllocation allocation{};
    if (vmaCreateBuffer(c.allocator, &ci, &ai, &raw, &allocation, nullptr) != VK_SUCCESS) throw std::runtime_error("buffer");
    return {vk::Buffer(raw), allocation};
}
void destroy(vulkan_runtime::core::VulkanContext const& c, B& b) {
    if (b.allocation) vmaDestroyBuffer(c.allocator, static_cast<VkBuffer>(b.buffer), b.allocation);
    b.allocation = nullptr;
}
}

int main() {
    using namespace vulkan_runtime::core;
    constexpr std::uint32_t H = 64, rows = 6;
    auto context = create_context("position-rmsnorm-dx");
    std::vector<float> dx(rows * H), expected(rows * H), output(rows * H);
    std::vector<std::uint32_t> mask{1, 1, 0, 1, 0, 1};
    for (std::size_t i = 0; i < dx.size(); ++i) dx[i] = 0.03f * std::cos(float(i) * 0.13f);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < H; ++c) expected[r * H + c] = mask[r] ? dx[r * H + c] : 0.0f;
    B bdx = make_buffer(context, dx.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    B bmask = make_buffer(context, mask.size() * sizeof(std::uint32_t), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    B bout = make_buffer(context, output.size() * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    B staging = make_buffer(context, std::max(dx.size() * sizeof(float), mask.size() * sizeof(std::uint32_t)) + output.size() * sizeof(float), vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    void* mapped = nullptr;
    vmaMapMemory(context.allocator, staging.allocation, &mapped);
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    std::memcpy(bytes, dx.data(), dx.size() * sizeof(float));
    std::memcpy(bytes + dx.size() * sizeof(float), mask.data(), mask.size() * sizeof(std::uint32_t));
    vmaFlushAllocation(context.allocator, staging.allocation, 0, dx.size() * sizeof(float) + mask.size() * sizeof(std::uint32_t));
    vmaUnmapMemory(context.allocator, staging.allocation);
    auto upload = [&](B const& dst, vk::DeviceSize source, vk::DeviceSize size) {
        auto pool = context.device.createCommandPool(vk::CommandPoolCreateInfo{}.setQueueFamilyIndex(context.computeQueueFamily));
        auto cmd = context.device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{}.setCommandPool(pool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1)).front();
        cmd.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        cmd.copyBuffer(staging.buffer, dst.buffer, vk::BufferCopy{}.setSrcOffset(source).setSize(size));
        cmd.end(); context.computeQueue.submit(vk::SubmitInfo{}.setCommandBuffers(cmd), {}); context.computeQueue.waitIdle();
        context.device.freeCommandBuffers(pool, cmd); context.device.destroyCommandPool(pool);
    };
    upload(bdx, 0, dx.size() * sizeof(float));
    upload(bmask, dx.size() * sizeof(float), mask.size() * sizeof(std::uint32_t));
    auto code = vulkan_runtime::shaders::get("position_training_grad_rmsnorm.spv");
    auto module = context.device.createShaderModule(vk::ShaderModuleCreateInfo{}.setCodeSize(code.size).setPCode(reinterpret_cast<std::uint32_t const*>(code.data)));
    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
    for (std::uint32_t i = 0; i < 3; ++i) bindings[i].setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
    auto layout = context.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(bindings));
    auto pipeline_layout = context.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{}.setSetLayouts(layout).setPushConstantRanges(vk::PushConstantRange{}.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(8)));
    auto pipeline = context.device.createComputePipeline({}, vk::ComputePipelineCreateInfo{}.setStage(vk::PipelineShaderStageCreateInfo{}.setStage(vk::ShaderStageFlagBits::eCompute).setModule(module).setPName("main")).setLayout(pipeline_layout)).value;
    auto pool = context.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(3)));
    auto set = context.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}.setDescriptorPool(pool).setSetLayouts(layout)).front();
    std::array<vk::DescriptorBufferInfo, 3> infos{{{bdx.buffer, 0, VK_WHOLE_SIZE}, {bmask.buffer, 0, VK_WHOLE_SIZE}, {bout.buffer, 0, VK_WHOLE_SIZE}}};
    std::array<vk::WriteDescriptorSet, 3> writes{};
    for (std::uint32_t i = 0; i < 3; ++i) writes[i].setDstSet(set).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(infos[i]);
    context.device.updateDescriptorSets(writes, {});
    auto cmdpool = context.device.createCommandPool(vk::CommandPoolCreateInfo{}.setQueueFamilyIndex(context.computeQueueFamily));
    auto cmd = context.device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{}.setCommandPool(cmdpool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1)).front();
    cmd.begin(vk::CommandBufferBeginInfo{}.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline); cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, set, {});
    std::array<std::uint32_t, 2> pc{H, rows}; cmd.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), pc.data()); cmd.dispatch((rows * H + 255) / 256, 1, 1);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, vk::BufferMemoryBarrier{}.setBuffer(bout.buffer).setSize(VK_WHOLE_SIZE).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
    cmd.copyBuffer(bout.buffer, staging.buffer, vk::BufferCopy{}.setDstOffset(0).setSize(output.size() * sizeof(float))); cmd.end();
    context.computeQueue.submit(vk::SubmitInfo{}.setCommandBuffers(cmd), {}); context.computeQueue.waitIdle();
    vmaInvalidateAllocation(context.allocator, staging.allocation, 0, output.size() * sizeof(float)); vmaMapMemory(context.allocator, staging.allocation, &mapped); std::memcpy(output.data(), mapped, output.size() * sizeof(float)); vmaUnmapMemory(context.allocator, staging.allocation);
    float error = 0.0f; for (std::size_t i = 0; i < output.size(); ++i) error = std::max(error, std::abs(output[i] - expected[i]));
    context.device.freeCommandBuffers(cmdpool, cmd); context.device.destroyCommandPool(cmdpool); context.device.destroyDescriptorPool(pool); context.device.destroyPipeline(pipeline); context.device.destroyPipelineLayout(pipeline_layout); context.device.destroyDescriptorSetLayout(layout); context.device.destroyShaderModule(module);
    destroy(context, staging); destroy(context, bout); destroy(context, bmask); destroy(context, bdx); destroy_context(context);
    if (error > 1.0e-7f) return 1;
    std::cout << "position_training_rmsnorm_dx PASS max_error=" << error << "\n";
    return 0;
}
