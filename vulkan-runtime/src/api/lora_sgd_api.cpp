#include "api/lora_sgd_api.h"
#include "core/vk_setup.h"
#include "embedded_shaders.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>

namespace {
struct Buffer { vk::Buffer buffer{}; VmaAllocation allocation{nullptr}; };
Buffer create_buffer(VmaAllocator allocator, vk::DeviceSize size, vk::BufferUsageFlags usage, VmaMemoryUsage memory, VmaAllocationCreateFlags flags = 0) {
    VkBufferCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; info.size = size; info.usage = static_cast<VkBufferUsageFlags>(usage); info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocation_info{}; allocation_info.usage = memory; allocation_info.flags = flags; VkBuffer raw{}; VmaAllocation allocation{};
    if (vmaCreateBuffer(allocator, &info, &allocation_info, &raw, &allocation, nullptr) != VK_SUCCESS) throw std::runtime_error("vmaCreateBuffer failed");
    return {vk::Buffer(raw), allocation};
}
void destroy_buffer(VmaAllocator allocator, Buffer& buffer) { if (buffer.allocation != nullptr) { vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer.buffer), buffer.allocation); buffer.allocation = nullptr; } }
struct PushConstants { std::uint32_t rank; float learning_rate; };
} // namespace

extern "C" int spaceslug_lora_sgd(float* a, float* b, float const* da, float const* db, float learning_rate, std::uint32_t rank) {
    if (!a || !b || !da || !db || learning_rate <= 0.0f || rank == 0 || rank > 8) return 1;
    try {
        auto ctx = vulkan_runtime::core::create_context("spaceslug-lora-sgd");
        const vk::DeviceSize bytes = vk::DeviceSize(64) * rank * sizeof(float);
        Buffer a_buffer = create_buffer(ctx.allocator, bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer b_buffer = create_buffer(ctx.allocator, bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer da_buffer = create_buffer(ctx.allocator, bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer db_buffer = create_buffer(ctx.allocator, bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer staging = create_buffer(ctx.allocator, bytes * 6, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        void* mapped{}; if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) throw std::runtime_error("vmaMapMemory failed"); auto* data = static_cast<char*>(mapped); std::memcpy(data, a, bytes); std::memcpy(data + bytes, b, bytes); std::memcpy(data + bytes * 2, da, bytes); std::memcpy(data + bytes * 3, db, bytes); vmaUnmapMemory(ctx.allocator, staging.allocation); vmaFlushAllocation(ctx.allocator, staging.allocation, 0, bytes * 4);
        auto blob = vulkan_runtime::shaders::get("lora_sgd.spv"); vk::ShaderModuleCreateInfo module_info; module_info.setCodeSize(blob.size).setPCode(reinterpret_cast<std::uint32_t const*>(blob.data)); auto shader = ctx.device.createShaderModule(module_info);
        std::array<vk::DescriptorSetLayoutBinding, 4> bindings{}; for (std::uint32_t i = 0; i < bindings.size(); ++i) bindings[i].setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute); vk::DescriptorSetLayoutCreateInfo layout_info; layout_info.setBindings(bindings); auto layout = ctx.device.createDescriptorSetLayout(layout_info); vk::PushConstantRange range; range.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(sizeof(PushConstants)); vk::PipelineLayoutCreateInfo pipeline_layout_info; pipeline_layout_info.setSetLayouts(layout).setPushConstantRanges(range); auto pipeline_layout = ctx.device.createPipelineLayout(pipeline_layout_info);
        vk::PipelineShaderStageCreateInfo stage; stage.setStage(vk::ShaderStageFlagBits::eCompute).setModule(shader).setPName("main"); vk::ComputePipelineCreateInfo pipeline_info; pipeline_info.setStage(stage).setLayout(pipeline_layout); auto pipeline_result = ctx.device.createComputePipeline({}, pipeline_info); if (pipeline_result.result != vk::Result::eSuccess) throw std::runtime_error("createComputePipeline failed"); auto pipeline = pipeline_result.value;
        vk::DescriptorPoolSize pool_size; pool_size.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(4); vk::DescriptorPoolCreateInfo pool_info; pool_info.setMaxSets(1).setPoolSizes(pool_size); auto pool = ctx.device.createDescriptorPool(pool_info); vk::DescriptorSetAllocateInfo allocate_info; allocate_info.setDescriptorPool(pool).setSetLayouts(layout); auto set = ctx.device.allocateDescriptorSets(allocate_info).front(); std::array<vk::DescriptorBufferInfo, 4> infos{}; for (auto& info : infos) info.setOffset(0).setRange(VK_WHOLE_SIZE); infos[0].setBuffer(a_buffer.buffer); infos[1].setBuffer(b_buffer.buffer); infos[2].setBuffer(da_buffer.buffer); infos[3].setBuffer(db_buffer.buffer); std::array<vk::WriteDescriptorSet, 4> writes{}; for (std::uint32_t i = 0; i < writes.size(); ++i) writes[i].setDstSet(set).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(infos[i]); ctx.device.updateDescriptorSets(writes, {});
        vk::CommandPoolCreateInfo command_pool_info; command_pool_info.setQueueFamilyIndex(ctx.computeQueueFamily); auto command_pool = ctx.device.createCommandPool(command_pool_info); vk::CommandBufferAllocateInfo command_allocate_info; command_allocate_info.setCommandPool(command_pool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1); auto command = ctx.device.allocateCommandBuffers(command_allocate_info).front(); vk::CommandBufferBeginInfo begin_info{}; command.begin(begin_info);
        for (auto const& [source_offset, target] : std::array<std::tuple<vk::DeviceSize, vk::Buffer>, 4>{{{0, a_buffer.buffer}, {bytes, b_buffer.buffer}, {bytes * 2, da_buffer.buffer}, {bytes * 3, db_buffer.buffer}}}) { vk::BufferCopy copy; copy.setSrcOffset(source_offset).setDstOffset(0).setSize(bytes); command.copyBuffer(staging.buffer, target, copy); }
        vk::MemoryBarrier to_shader; to_shader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite); command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, to_shader, {}, {}); command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline); command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, set, {}); PushConstants constants{rank, learning_rate}; command.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), &constants); command.dispatch((64 * rank + 255) / 256, 1, 1);
        vk::MemoryBarrier to_transfer; to_transfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead); command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, to_transfer, {}, {}); vk::BufferCopy a_copy; a_copy.setSrcOffset(0).setDstOffset(bytes * 4).setSize(bytes); command.copyBuffer(a_buffer.buffer, staging.buffer, a_copy); vk::BufferCopy b_copy; b_copy.setSrcOffset(0).setDstOffset(bytes * 5).setSize(bytes); command.copyBuffer(b_buffer.buffer, staging.buffer, b_copy); command.end(); vk::SubmitInfo submit; submit.setCommandBuffers(command); ctx.computeQueue.submit(submit); ctx.computeQueue.waitIdle();
        if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("vmaMapMemory failed");
        }
        vmaInvalidateAllocation(ctx.allocator, staging.allocation, bytes * 4, bytes * 2);
        data = static_cast<char*>(mapped);
        std::memcpy(a, data + bytes * 4, bytes);
        std::memcpy(b, data + bytes * 5, bytes);
        vmaUnmapMemory(ctx.allocator, staging.allocation);
        ctx.device.destroyCommandPool(command_pool); ctx.device.destroyDescriptorPool(pool); ctx.device.destroyPipeline(pipeline); ctx.device.destroyPipelineLayout(pipeline_layout); ctx.device.destroyDescriptorSetLayout(layout); ctx.device.destroyShaderModule(shader); destroy_buffer(ctx.allocator, staging); destroy_buffer(ctx.allocator, db_buffer); destroy_buffer(ctx.allocator, da_buffer); destroy_buffer(ctx.allocator, b_buffer); destroy_buffer(ctx.allocator, a_buffer); vulkan_runtime::core::destroy_context(ctx); return 0;
    } catch (...) { return 2; }
}
