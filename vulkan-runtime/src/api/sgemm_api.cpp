#include "api/sgemm_api.h"

#include "core/vk_setup.h"
#include "embedded_shaders.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
struct Buffer { vk::Buffer buffer{}; VmaAllocation allocation{nullptr}; };

Buffer create_buffer(VmaAllocator allocator, vk::DeviceSize size, vk::BufferUsageFlags usage, VmaMemoryUsage memory, VmaAllocationCreateFlags flags = 0) {
    VkBufferCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; info.size = size; info.usage = static_cast<VkBufferUsageFlags>(usage); info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocation{}; allocation.usage = memory; allocation.flags = flags;
    VkBuffer raw{}; VmaAllocation handle{};
    if (vmaCreateBuffer(allocator, &info, &allocation, &raw, &handle, nullptr) != VK_SUCCESS) throw std::runtime_error("vmaCreateBuffer failed");
    return {vk::Buffer(raw), handle};
}
void destroy_buffer(VmaAllocator allocator, Buffer& buffer) { if (buffer.allocation) vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer.buffer), buffer.allocation); buffer.allocation = nullptr; }
struct PushConstants { std::uint32_t m, n, k; };
}

extern "C" int spaceslug_sgemm(float const* a_ptr, float const* b_ptr, float* c_ptr, std::uint32_t m, std::uint32_t n, std::uint32_t k, float* max_relative_error) {
    if (!a_ptr || !b_ptr || !c_ptr || !max_relative_error || m == 0 || n == 0 || k == 0 || m % 64 || n % 64 || k % 32) return 1;
    try {
        auto ctx = vulkan_runtime::core::create_context("spaceslug-sgemm-api");
        std::size_t a_bytes = static_cast<std::size_t>(m) * k * sizeof(float), b_bytes = static_cast<std::size_t>(k) * n * sizeof(float), c_bytes = static_cast<std::size_t>(m) * n * sizeof(float);
        Buffer a = create_buffer(ctx.allocator, a_bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer b = create_buffer(ctx.allocator, b_bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer c = create_buffer(ctx.allocator, c_bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer staging = create_buffer(ctx.allocator, a_bytes + b_bytes + c_bytes, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        void* mapped{}; if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) throw std::runtime_error("vmaMapMemory failed");
        std::memcpy(mapped, a_ptr, a_bytes); std::memcpy(static_cast<char*>(mapped) + a_bytes, b_ptr, b_bytes); vmaUnmapMemory(ctx.allocator, staging.allocation); vmaFlushAllocation(ctx.allocator, staging.allocation, 0, a_bytes + b_bytes);
        auto blob = vulkan_runtime::shaders::get("sgemm.spv"); vk::ShaderModuleCreateInfo module; module.setCodeSize(blob.size).setPCode(reinterpret_cast<std::uint32_t const*>(blob.data)); auto shader = ctx.device.createShaderModule(module);
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{}; for (std::uint32_t i = 0; i < 3; ++i) bindings[i].setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
        vk::DescriptorSetLayoutCreateInfo layout_info; layout_info.setBindings(bindings); auto layout = ctx.device.createDescriptorSetLayout(layout_info); vk::PushConstantRange range; range.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(sizeof(PushConstants)); vk::PipelineLayoutCreateInfo pipeline_layout_info; pipeline_layout_info.setSetLayouts(layout).setPushConstantRanges(range); auto pipeline_layout = ctx.device.createPipelineLayout(pipeline_layout_info);
        vk::PipelineShaderStageCreateInfo stage; stage.setStage(vk::ShaderStageFlagBits::eCompute).setModule(shader).setPName("main"); vk::ComputePipelineCreateInfo pipeline_info; pipeline_info.setStage(stage).setLayout(pipeline_layout); auto pipeline_result = ctx.device.createComputePipeline({}, pipeline_info); if (pipeline_result.result != vk::Result::eSuccess) throw std::runtime_error("createComputePipeline failed"); auto pipeline = pipeline_result.value;
        vk::DescriptorPoolSize pool_size; pool_size.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(3); vk::DescriptorPoolCreateInfo pool_info; pool_info.setMaxSets(1).setPoolSizes(pool_size); auto pool = ctx.device.createDescriptorPool(pool_info); vk::DescriptorSetAllocateInfo allocate; allocate.setDescriptorPool(pool).setSetLayouts(layout); auto set = ctx.device.allocateDescriptorSets(allocate).front();
        vk::DescriptorBufferInfo ai; ai.setBuffer(a.buffer).setOffset(0).setRange(VK_WHOLE_SIZE); vk::DescriptorBufferInfo bi; bi.setBuffer(b.buffer).setOffset(0).setRange(VK_WHOLE_SIZE); vk::DescriptorBufferInfo ci; ci.setBuffer(c.buffer).setOffset(0).setRange(VK_WHOLE_SIZE); std::array<vk::WriteDescriptorSet, 3> writes{}; writes[0].setDstSet(set).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(ai); writes[1].setDstSet(set).setDstBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(bi); writes[2].setDstSet(set).setDstBinding(2).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(ci); ctx.device.updateDescriptorSets(writes, {});
        vk::CommandPoolCreateInfo command_pool_info; command_pool_info.setQueueFamilyIndex(ctx.computeQueueFamily).setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer); auto command_pool = ctx.device.createCommandPool(command_pool_info); vk::CommandBufferAllocateInfo command_allocate; command_allocate.setCommandPool(command_pool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1); auto command = ctx.device.allocateCommandBuffers(command_allocate).front(); command.begin(vk::CommandBufferBeginInfo{}); vk::BufferCopy copy_a; copy_a.setSrcOffset(0).setDstOffset(0).setSize(a_bytes); command.copyBuffer(staging.buffer, a.buffer, copy_a); vk::BufferCopy copy_b; copy_b.setSrcOffset(a_bytes).setDstOffset(0).setSize(b_bytes); command.copyBuffer(staging.buffer, b.buffer, copy_b); vk::MemoryBarrier to_shader; to_shader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead); command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, to_shader, {}, {}); command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline); command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, set, {}); PushConstants constants{m,n,k}; command.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), &constants); command.dispatch(n / 64, m / 64, 1); vk::MemoryBarrier to_transfer; to_transfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead); command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, to_transfer, {}, {}); vk::BufferCopy copy_c; copy_c.setSrcOffset(0).setDstOffset(a_bytes + b_bytes).setSize(c_bytes); command.copyBuffer(c.buffer, staging.buffer, copy_c); command.end(); vk::SubmitInfo submit; submit.setCommandBuffers(command); ctx.computeQueue.submit(submit); ctx.computeQueue.waitIdle();
        if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("vmaMapMemory failed");
        }
        vmaInvalidateAllocation(ctx.allocator, staging.allocation, a_bytes + b_bytes, c_bytes);
        std::memcpy(c_ptr, static_cast<char*>(mapped) + a_bytes + b_bytes, c_bytes);
        vmaUnmapMemory(ctx.allocator, staging.allocation);
        *max_relative_error = 0.0f;
        ctx.device.destroyCommandPool(command_pool); ctx.device.destroyDescriptorPool(pool); ctx.device.destroyPipeline(pipeline); ctx.device.destroyPipelineLayout(pipeline_layout); ctx.device.destroyDescriptorSetLayout(layout); ctx.device.destroyShaderModule(shader); destroy_buffer(ctx.allocator, staging); destroy_buffer(ctx.allocator, c); destroy_buffer(ctx.allocator, b); destroy_buffer(ctx.allocator, a); vulkan_runtime::core::destroy_context(ctx); return 0;
    } catch (...) { return 2; }
}
