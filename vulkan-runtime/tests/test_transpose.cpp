// M4a kernel library: fp32 matrix transpose (M x N -> N x M) on the GPU,
// verified BITWISE-EXACTLY against a CPU reference on both RADV (gfx803) and
// lavapipe. A transpose is a pure permutation (each output element is a verbatim
// copy of one input float), so exact equality is expected — no tolerance.
//
// Flow (headless, no WSI) — mirrors test_sgemm.cpp:
//   context -> device buffers (A, B) + host staging -> fill staging
//   -> descriptor set layout / pipeline layout (2 SSBOs + push constants)
//   -> compute pipeline -> descriptor set -> one command buffer (copy-in +
//   barrier + dispatch + barrier + copy-out) -> submit -> wait idle ->
//   map staging -> bitwise compare vs CPU transpose.

#include "core/vk_setup.h"

#include "embedded_shaders.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t kM = 512;
constexpr std::uint32_t kN = 512;

// Shader contract (transpose.comp): 64x64 tile.
constexpr std::uint32_t kTile = 64;

// Push-constant block matching the shader's `layout(push_constant) uniform PC`.
struct PushConstants {
    std::uint32_t M;
    std::uint32_t N;
};

// Deterministic LCG (Numerical Recipes) -> floats in [0, 1).
std::vector<float> generate_inputs(std::size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }
    return v;
}

// CPU reference transpose (row-major A[M x N] -> B[N x M]), pure permutation.
std::vector<float> cpu_reference(std::vector<float> const& a,
                                 std::uint32_t M, std::uint32_t N) {
    std::vector<float> b(static_cast<std::size_t>(M) * N);
    for (std::uint32_t m = 0; m < M; ++m) {
        for (std::uint32_t n = 0; n < N; ++n) {
            b[static_cast<std::size_t>(n) * M + m] = a[static_cast<std::size_t>(m) * N + n];
        }
    }
    return b;
}

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

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-transpose");

        // --- CPU inputs / reference -----------------------------------------
        std::vector<float> in_a = generate_inputs(static_cast<std::size_t>(kM) * kN, 0x12345678u);
        std::vector<float> expected = cpu_reference(in_a, kM, kN);

        // --- Buffers --------------------------------------------------------
        vk::DeviceSize aBytes = static_cast<vk::DeviceSize>(kM) * kN * sizeof(float);
        vk::DeviceSize bBytes = static_cast<vk::DeviceSize>(kN) * kM * sizeof(float);

        Buffer dev_a = create_buffer(
            ctx.allocator, aBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_b = create_buffer(
            ctx.allocator, bBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize stagingBytes = aBytes + bBytes;
        Buffer staging = create_buffer(
            ctx.allocator, stagingBytes,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            std::memcpy(mapped, in_a.data(), aBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, aBytes);
        }

        // --- Pipeline -------------------------------------------------------
        vk::ShaderModuleCreateInfo moduleInfo;
        vulkan_runtime::shaders::ShaderBlob blob = vulkan_runtime::shaders::get("transpose.spv");
        if (blob.data == nullptr || blob.size == 0) {
            throw std::runtime_error("transpose.spv not embedded.");
        }
        moduleInfo.setCodeSize(blob.size)
            .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
        vk::ShaderModule shaderModule = ctx.device.createShaderModule(moduleInfo);

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            bindings[i].setBinding(i)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        }

        vk::DescriptorSetLayoutCreateInfo setLayoutInfo;
        setLayoutInfo.setBindings(bindings);
        vk::DescriptorSetLayout setLayout = ctx.device.createDescriptorSetLayout(setLayoutInfo);

        vk::PushConstantRange pcRange;
        pcRange.setStageFlags(vk::ShaderStageFlagBits::eCompute)
            .setOffset(0)
            .setSize(sizeof(PushConstants));

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setSetLayouts(setLayout).setPushConstantRanges(pcRange);
        vk::PipelineLayout pipelineLayout = ctx.device.createPipelineLayout(pipelineLayoutInfo);

        vk::PipelineShaderStageCreateInfo stageInfo;
        stageInfo.setStage(vk::ShaderStageFlagBits::eCompute)
            .setModule(shaderModule)
            .setPName("main");

        vk::ComputePipelineCreateInfo pipelineInfo;
        pipelineInfo.setStage(stageInfo).setLayout(pipelineLayout);
        auto pipelineResult = ctx.device.createComputePipeline({}, pipelineInfo);
        if (pipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("createComputePipeline failed.");
        }
        vk::Pipeline pipeline = pipelineResult.value;

        // --- Descriptor set ------------------------------------------------
        vk::DescriptorPoolSize poolSize;
        poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(2);
        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setMaxSets(1).setPoolSizes(poolSize);
        vk::DescriptorPool descriptorPool = ctx.device.createDescriptorPool(poolInfo);

        vk::DescriptorSetAllocateInfo setAllocInfo;
        setAllocInfo.setDescriptorPool(descriptorPool).setSetLayouts(setLayout);
        vk::DescriptorSet descriptorSet =
            ctx.device.allocateDescriptorSets(setAllocInfo).front();

        vk::DescriptorBufferInfo aInfo;
        aInfo.setBuffer(dev_a.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo bInfo;
        bInfo.setBuffer(dev_b.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);

        std::array<vk::WriteDescriptorSet, 2> writes{};
        writes[0].setDstSet(descriptorSet)
            .setDstBinding(0).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(aInfo);
        writes[1].setDstSet(descriptorSet)
            .setDstBinding(1).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(bInfo);
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
        copyA.setSrcOffset(0).setDstOffset(0).setSize(aBytes);
        cmd.copyBuffer(staging.buffer, dev_a.buffer, copyA);

        vk::MemoryBarrier toShader;
        toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {},
                            toShader, {}, {});

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0,
                               descriptorSet, {});

        PushConstants pc{kM, kN};
        cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(PushConstants), &pc);
        // Grid: (N/tile) x (M/tile) workgroups. B rows come from A cols.
        cmd.dispatch((kN + kTile - 1) / kTile, (kM + kTile - 1) / kTile, 1);

        vk::MemoryBarrier toTransfer;
        toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {},
                            toTransfer, {}, {});

        vk::BufferCopy copyB;
        copyB.setSrcOffset(0).setDstOffset(aBytes).setSize(bBytes);
        cmd.copyBuffer(dev_b.buffer, staging.buffer, copyB);

        cmd.end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(cmd);
        ctx.computeQueue.submit(submitInfo);
        ctx.computeQueue.waitIdle();

        // --- Read back + bitwise compare ------------------------------------
        bool ok = true;
        std::size_t firstBad = 0;
        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, aBytes, bBytes);

            float const* gpuB =
                reinterpret_cast<float const*>(static_cast<char const*>(mapped) + aBytes);
            std::size_t n = static_cast<std::size_t>(kN) * kM;
            for (std::size_t i = 0; i < n; ++i) {
                if (gpuB[i] != expected[i]) {
                    firstBad = i;
                    ok = false;
                    break;
                }
            }
            vmaUnmapMemory(ctx.allocator, staging.allocation);
        }

        // --- Tear down (reverse order) --------------------------------------
        ctx.device.destroyCommandPool(commandPool);
        ctx.device.destroyDescriptorPool(descriptorPool);
        ctx.device.destroyPipeline(pipeline);
        ctx.device.destroyPipelineLayout(pipelineLayout);
        ctx.device.destroyDescriptorSetLayout(setLayout);
        ctx.device.destroyShaderModule(shaderModule);
        destroy_buffer(ctx.allocator, staging);
        destroy_buffer(ctx.allocator, dev_b);
        destroy_buffer(ctx.allocator, dev_a);
        vulkan_runtime::core::destroy_context(ctx);

        if (ok) {
            std::cout << "transpose: M=" << kM << " N=" << kN << " PASS\n";
            return EXIT_SUCCESS;
        }
        std::cerr << "transpose: MISMATCH at index " << firstBad
                  << " (expected " << expected[firstBad] << ", got GPU value)\n";
        return EXIT_FAILURE;
    } catch (vk::SystemError const& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
