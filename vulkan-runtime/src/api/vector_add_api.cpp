// M1 hello-compute: vector-add on the GPU, verified byte-for-byte against a
// CPU reference on both RADV (gfx803) and lavapipe.
//
// Flow (headless, no WSI):
//   context -> device buffers (in_a, in_b, out) + host staging -> fill staging
//   -> descriptor set layout / pipeline layout / compute pipeline -> descriptor
//   set -> one command buffer (copy-in + barrier + dispatch + barrier +
//   copy-out) -> submit -> wait idle -> map staging -> compare vs CPU.

#include "core/vk_setup.h"

#include "embedded_shaders.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t kWorkgroupSize = 256;   // 4 wave64 on gfx803
constexpr std::size_t kN = 1u << 20;           // 1,048,576 floats (4 MiB/buffer)

// A VMA-managed buffer: a Vulkan buffer + its allocation.
struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
    vk::DeviceSize size{0};
};

Buffer create_buffer(VmaAllocator allocator, vk::DeviceSize size,
                     vk::BufferUsageFlags usage, VmaMemoryUsage memUsage,
                     VmaAllocationCreateFlags flags = 0) {
    VkBufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size = size;
    createInfo.usage = static_cast<VkBufferUsageFlags>(usage);
    createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = memUsage;
    allocCreateInfo.flags = flags;

    VkBuffer vkBuffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    if (vmaCreateBuffer(allocator, &createInfo, &allocCreateInfo, &vkBuffer,
                        &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateBuffer failed.");
    }

    return Buffer{vk::Buffer(vkBuffer), allocation, size};
}

void destroy_buffer(VmaAllocator allocator, Buffer& b) {
    if (b.allocation != nullptr) {
        vmaDestroyBuffer(allocator, static_cast<VkBuffer>(b.buffer), b.allocation);
        b.allocation = nullptr;
    }
}

} // namespace

extern "C" int spaceslug_vector_add(float const* in_a_ptr, float const* in_b_ptr, float* out_ptr, std::size_t n) {
    if (in_a_ptr == nullptr || in_b_ptr == nullptr || out_ptr == nullptr || n != kN) return EXIT_FAILURE;
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-vector-add");
        std::vector<float> in_a(in_a_ptr, in_a_ptr + n);
        std::vector<float> in_b(in_b_ptr, in_b_ptr + n);
        std::vector<float> expected(n);
        for (std::size_t i = 0; i < n; ++i) expected[i] = in_a[i] + in_b[i];

        // --- Buffers (device-local + host staging) --------------------------
        vk::DeviceSize elemSize = sizeof(float);
        vk::DeviceSize nBytes = kN * elemSize;

        Buffer dev_a = create_buffer(
            ctx.allocator, nBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_b = create_buffer(
            ctx.allocator, nBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_out = create_buffer(
            ctx.allocator, nBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // One staging buffer: [ in_a | in_b | out ] regions. Needs the host
        // access flag so VMA allows vmaMapMemory on the AUTO_PREFER_HOST block.
        Buffer staging = create_buffer(
            ctx.allocator, 3 * nBytes,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        // Fill the input regions of the staging buffer from the CPU inputs.
        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            std::memcpy(mapped, in_a.data(), nBytes);
            std::memcpy(static_cast<char*>(mapped) + nBytes, in_b.data(), nBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            // No-op for HOST_COHERENT memory; required for non-coherent.
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, 3 * nBytes);
        }

        // --- Pipeline -------------------------------------------------------
        vk::ShaderModuleCreateInfo moduleInfo;
        vulkan_runtime::shaders::ShaderBlob blob =
            vulkan_runtime::shaders::get("vector_add.spv");
        if (blob.data == nullptr || blob.size == 0) {
            throw std::runtime_error("vector_add.spv not embedded.");
        }
        moduleInfo.setCodeSize(blob.size)
            .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
        vk::ShaderModule shaderModule = ctx.device.createShaderModule(moduleInfo);

        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
        bindings[0].setBinding(0)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        bindings[1].setBinding(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        bindings[2].setBinding(2)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);

        vk::DescriptorSetLayoutCreateInfo setLayoutInfo;
        setLayoutInfo.setBindings(bindings);
        vk::DescriptorSetLayout setLayout = ctx.device.createDescriptorSetLayout(setLayoutInfo);

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setSetLayouts(setLayout);
        vk::PipelineLayout pipelineLayout = ctx.device.createPipelineLayout(pipelineLayoutInfo);

        vk::PipelineShaderStageCreateInfo stageInfo;
        stageInfo.setStage(vk::ShaderStageFlagBits::eCompute)
            .setModule(shaderModule)
            .setPName("main");

        vk::ComputePipelineCreateInfo pipelineInfo;
        pipelineInfo.setStage(stageInfo).setLayout(pipelineLayout);
        // createComputePipeline returns a ResultValue<Pipeline> (no throwing
        // overload exists); check the result code explicitly.
        auto pipelineResult = ctx.device.createComputePipeline({}, pipelineInfo);
        if (pipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("createComputePipeline failed.");
        }
        vk::Pipeline pipeline = pipelineResult.value;

        // --- Descriptor set ------------------------------------------------
        vk::DescriptorPoolSize poolSize;
        poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(3);
        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setMaxSets(1).setPoolSizes(poolSize);
        vk::DescriptorPool descriptorPool = ctx.device.createDescriptorPool(poolInfo);

        vk::DescriptorSetAllocateInfo setAllocInfo;
        setAllocInfo.setDescriptorPool(descriptorPool).setSetLayouts(setLayout);
        std::vector<vk::DescriptorSet> sets = ctx.device.allocateDescriptorSets(setAllocInfo);
        vk::DescriptorSet descriptorSet = sets.front();

        vk::DescriptorBufferInfo aInfo;
        aInfo.setBuffer(dev_a.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo bInfo;
        bInfo.setBuffer(dev_b.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo outInfo;
        outInfo.setBuffer(dev_out.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);

        std::array<vk::WriteDescriptorSet, 3> writes{};
        writes[0].setDstSet(descriptorSet)
            .setDstBinding(0)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(aInfo);
        writes[1].setDstSet(descriptorSet)
            .setDstBinding(1)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(bInfo);
        writes[2].setDstSet(descriptorSet)
            .setDstBinding(2)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(outInfo);
        ctx.device.updateDescriptorSets(writes, {});

        // --- Command buffer: copy in -> dispatch -> copy out ---------------
        vk::CommandPoolCreateInfo cmdPoolInfo;
        cmdPoolInfo.setQueueFamilyIndex(ctx.computeQueueFamily)
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        vk::CommandPool commandPool = ctx.device.createCommandPool(cmdPoolInfo);

        vk::CommandBufferAllocateInfo cmdAllocInfo;
        cmdAllocInfo.setCommandPool(commandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1);
        vk::CommandBuffer cmd = ctx.device.allocateCommandBuffers(cmdAllocInfo).front();

        cmd.begin(vk::CommandBufferBeginInfo{});

        vk::BufferCopy copyA;
        copyA.setSrcOffset(0).setDstOffset(0).setSize(nBytes);
        cmd.copyBuffer(staging.buffer, dev_a.buffer, copyA);

        vk::BufferCopy copyB;
        copyB.setSrcOffset(nBytes).setDstOffset(0).setSize(nBytes);
        cmd.copyBuffer(staging.buffer, dev_b.buffer, copyB);

        vk::MemoryBarrier toShader;
        toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {},
                            toShader, {}, {});

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0,
                               descriptorSet, {});
        cmd.dispatch(static_cast<std::uint32_t>(kN / kWorkgroupSize), 1, 1);

        vk::MemoryBarrier toTransfer;
        toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {},
                            toTransfer, {}, {});

        vk::BufferCopy copyOut;
        copyOut.setSrcOffset(0).setDstOffset(2 * nBytes).setSize(nBytes);
        cmd.copyBuffer(dev_out.buffer, staging.buffer, copyOut);

        cmd.end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(cmd);
        ctx.computeQueue.submit(submitInfo);
        ctx.computeQueue.waitIdle();

        // --- Read back + compare -------------------------------------------
        std::size_t firstBad = 0;
        bool ok = true;
        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            // No-op for HOST_COHERENT memory; required for non-coherent.
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, 2 * nBytes, nBytes);

            float const* gpuOut =
                reinterpret_cast<float const*>(static_cast<char const*>(mapped) + 2 * nBytes);
            for (std::size_t i = 0; i < kN; ++i) {
                if (gpuOut[i] != expected[i]) {
                    firstBad = i;
                    ok = false;
                    break;
                }
            }
            std::memcpy(out_ptr, gpuOut, nBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
        }

        if (!ok) {
            std::cerr << "vector_add: MISMATCH at index " << firstBad
                      << " (expected " << expected[firstBad] << ", got GPU value)\n";
        }

        // --- Tear down (reverse order) --------------------------------------
        ctx.device.destroyCommandPool(commandPool);
        ctx.device.destroyDescriptorPool(descriptorPool);
        ctx.device.destroyPipeline(pipeline);
        ctx.device.destroyPipelineLayout(pipelineLayout);
        ctx.device.destroyDescriptorSetLayout(setLayout);
        ctx.device.destroyShaderModule(shaderModule);
        destroy_buffer(ctx.allocator, staging);
        destroy_buffer(ctx.allocator, dev_out);
        destroy_buffer(ctx.allocator, dev_b);
        destroy_buffer(ctx.allocator, dev_a);
        vulkan_runtime::core::destroy_context(ctx);

        if (ok) {
            std::cout << "vector_add: N=" << kN << " PASS\n";
            return EXIT_SUCCESS;
        }
        return EXIT_FAILURE;
    } catch (vk::SystemError const& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
