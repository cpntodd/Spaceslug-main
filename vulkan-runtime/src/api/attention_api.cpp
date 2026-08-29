#include "api/attention_api.h"

#include "core/vk_setup.h"
#include "embedded_shaders.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {
struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
};

Buffer create_buffer(VmaAllocator allocator,
                     vk::DeviceSize size,
                     vk::BufferUsageFlags usage,
                     VmaMemoryUsage memory,
                     VmaAllocationCreateFlags flags = 0) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = static_cast<VkBufferUsageFlags>(usage);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocation{};
    allocation.usage = memory;
    allocation.flags = flags;
    VkBuffer raw{};
    VmaAllocation handle{};
    if (vmaCreateBuffer(allocator, &info, &allocation, &raw, &handle, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateBuffer failed");
    }
    return {vk::Buffer(raw), handle};
}

void destroy_buffer(VmaAllocator allocator, Buffer& buffer) {
    if (buffer.allocation != nullptr) {
        vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer.buffer), buffer.allocation);
        buffer.allocation = nullptr;
    }
}

struct PushConstants {
    std::uint32_t t;
    std::uint32_t d;
};
} // namespace

extern "C" int spaceslug_attention(float const* q, float const* k, float const* v, float* output) {
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr)
        return 1;

    try {
        constexpr std::size_t bytes = static_cast<std::size_t>(SPACESLUG_ATTENTION_FLOATS) * sizeof(float);
        vulkan_runtime::core::VulkanContext ctx = vulkan_runtime::core::create_context("spaceslug-attention-api");

        Buffer devQ = create_buffer(ctx.allocator,
                                    bytes,
                                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer devK = create_buffer(ctx.allocator,
                                    bytes,
                                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer devV = create_buffer(ctx.allocator,
                                    bytes,
                                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer devO = create_buffer(ctx.allocator,
                                    bytes,
                                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer staging = create_buffer(ctx.allocator,
                                       bytes * 4,
                                       vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        void* mapped = nullptr;
        if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS)
            throw std::runtime_error("vmaMapMemory failed");
        auto* stagingBytes = static_cast<char*>(mapped);
        std::memcpy(stagingBytes, q, bytes);
        std::memcpy(stagingBytes + bytes, k, bytes);
        std::memcpy(stagingBytes + bytes * 2, v, bytes);
        vmaUnmapMemory(ctx.allocator, staging.allocation);
        vmaFlushAllocation(ctx.allocator, staging.allocation, 0, bytes * 3);

        auto blob = vulkan_runtime::shaders::get("attention.spv");
        if (blob.data == nullptr || blob.size == 0)
            throw std::runtime_error("attention.spv not embedded");
        vk::ShaderModuleCreateInfo module;
        module.setCodeSize(blob.size).setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
        vk::ShaderModule shader = ctx.device.createShaderModule(module);

        std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
        for (std::uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[i]
                .setBinding(i)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        }
        vk::DescriptorSetLayoutCreateInfo layoutInfo;
        layoutInfo.setBindings(bindings);
        vk::DescriptorSetLayout layout = ctx.device.createDescriptorSetLayout(layoutInfo);
        vk::PushConstantRange range;
        range.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(sizeof(PushConstants));
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setSetLayouts(layout).setPushConstantRanges(range);
        vk::PipelineLayout pipelineLayout = ctx.device.createPipelineLayout(pipelineLayoutInfo);
        vk::PipelineShaderStageCreateInfo stage;
        stage.setStage(vk::ShaderStageFlagBits::eCompute).setModule(shader).setPName("main");
        vk::ComputePipelineCreateInfo pipelineInfo;
        pipelineInfo.setStage(stage).setLayout(pipelineLayout);
        auto pipelineResult = ctx.device.createComputePipeline({}, pipelineInfo);
        if (pipelineResult.result != vk::Result::eSuccess)
            throw std::runtime_error("createComputePipeline failed");
        vk::Pipeline pipeline = pipelineResult.value;

        vk::DescriptorPoolSize poolSize;
        poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(4);
        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setMaxSets(1).setPoolSizes(poolSize);
        vk::DescriptorPool pool = ctx.device.createDescriptorPool(poolInfo);
        vk::DescriptorSetAllocateInfo allocInfo;
        allocInfo.setDescriptorPool(pool).setSetLayouts(layout);
        vk::DescriptorSet set = ctx.device.allocateDescriptorSets(allocInfo).front();
        std::array<vk::DescriptorBufferInfo, 4> infos{};
        infos[0].setBuffer(devQ.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        infos[1].setBuffer(devK.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        infos[2].setBuffer(devV.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        infos[3].setBuffer(devO.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        std::array<vk::WriteDescriptorSet, 4> writes{};
        for (std::uint32_t i = 0; i < writes.size(); ++i) {
            writes[i]
                .setDstSet(set)
                .setDstBinding(i)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setBufferInfo(infos[i]);
        }
        ctx.device.updateDescriptorSets(writes, {});

        vk::CommandPoolCreateInfo cmdPoolInfo;
        cmdPoolInfo.setQueueFamilyIndex(ctx.computeQueueFamily)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        vk::CommandPool cmdPool = ctx.device.createCommandPool(cmdPoolInfo);
        vk::CommandBufferAllocateInfo cmdAllocInfo;
        cmdAllocInfo.setCommandPool(cmdPool).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1);
        vk::CommandBuffer cmd = ctx.device.allocateCommandBuffers(cmdAllocInfo).front();
        cmd.begin(vk::CommandBufferBeginInfo{});
        for (std::uint32_t i = 0; i < 3; ++i) {
            vk::BufferCopy copy;
            copy.setSrcOffset(static_cast<vk::DeviceSize>(i) * bytes).setDstOffset(0).setSize(bytes);
            cmd.copyBuffer(staging.buffer, std::array{devQ.buffer, devK.buffer, devV.buffer}[i], copy);
        }
        vk::MemoryBarrier toShader;
        toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, toShader, {}, {});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, set, {});
        PushConstants constants{SPACESLUG_ATTENTION_T, SPACESLUG_ATTENTION_D};
        cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), &constants);
        cmd.dispatch(SPACESLUG_ATTENTION_T / 16, 1, 1);
        vk::MemoryBarrier toTransfer;
        toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, toTransfer, {}, {});
        vk::BufferCopy copyOut;
        copyOut.setSrcOffset(0).setDstOffset(bytes * 3).setSize(bytes);
        cmd.copyBuffer(devO.buffer, staging.buffer, copyOut);
        cmd.end();
        vk::SubmitInfo submit;
        submit.setCommandBuffers(cmd);
        ctx.computeQueue.submit(submit);
        ctx.computeQueue.waitIdle();
        if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS)
            throw std::runtime_error("vmaMapMemory failed");
        vmaInvalidateAllocation(ctx.allocator, staging.allocation, bytes * 3, bytes);
        std::memcpy(output, static_cast<char*>(mapped) + bytes * 3, bytes);
        vmaUnmapMemory(ctx.allocator, staging.allocation);

        ctx.device.destroyCommandPool(cmdPool);
        ctx.device.destroyDescriptorPool(pool);
        ctx.device.destroyPipeline(pipeline);
        ctx.device.destroyPipelineLayout(pipelineLayout);
        ctx.device.destroyDescriptorSetLayout(layout);
        ctx.device.destroyShaderModule(shader);
        destroy_buffer(ctx.allocator, staging);
        destroy_buffer(ctx.allocator, devO);
        destroy_buffer(ctx.allocator, devV);
        destroy_buffer(ctx.allocator, devK);
        destroy_buffer(ctx.allocator, devQ);
        vulkan_runtime::core::destroy_context(ctx);
        return 0;
    } catch (...) {
        return 2;
    }
}
