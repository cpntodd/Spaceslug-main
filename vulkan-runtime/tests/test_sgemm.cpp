// M2 first GEMM: fp32 SGEMM C = A*B on the GPU, verified against a
// double-precision CPU reference on both RADV (gfx803) and lavapipe.
//
// Flow (headless, no WSI) — mirrors test_vector_add.cpp:
//   context -> device buffers (A, B, C) + host staging -> fill staging
//   -> descriptor set layout / pipeline layout (3 SSBOs + push constants)
//   -> compute pipeline -> descriptor set -> one command buffer (copy-in +
//   barrier + dispatch + barrier + copy-out) -> submit -> wait idle ->
//   map staging -> compare vs CPU (double).
//
// Tolerance: fp32 accumulation order differs from the double reference, so
// exact equality is NOT expected. We require |gpu - ref| <= 1e-3 * max(1,|ref|)
// per element and print the actual max absolute / relative error.

#include "core/vk_setup.h"

#include "embedded_shaders.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t kM = 256;
constexpr std::uint32_t kN = 256;
constexpr std::uint32_t kK = 256;

// Shader contract (sgemm.comp): M % 64 == 0, N % 64 == 0, K % 32 == 0.
constexpr std::uint32_t kTileM = 64;
constexpr std::uint32_t kTileN = 64;
constexpr std::uint32_t kChunkK = 32;

// Push-constant block matching the shader's `layout(push_constant) uniform PC`.
struct PushConstants {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

// Deterministic LCG (Numerical Recipes) -> floats in [0,1), scaled to [-1,1].
std::vector<float> generate_inputs(std::size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        v[i] = u * 2.0f - 1.0f;
    }
    return v;
}

// CPU reference in double precision (row-major A[M x K], B[K x N] -> C[M x N]).
std::vector<double> cpu_reference(std::vector<float> const& a,
                                  std::vector<float> const& b,
                                  std::uint32_t M, std::uint32_t N, std::uint32_t K) {
    std::vector<double> c(M * N, 0.0);
    for (std::uint32_t m = 0; m < M; ++m) {
        for (std::uint32_t k = 0; k < K; ++k) {
            double av = static_cast<double>(a[m * K + k]);
            // Accumulate over N for this (m, k).
            for (std::uint32_t n = 0; n < N; ++n) {
                c[m * N + n] += av * static_cast<double>(b[k * N + n]);
            }
        }
    }
    return c;
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
        static_assert(kM % kTileM == 0 && kN % kTileN == 0 && kK % kChunkK == 0,
                      "sgemm sizes must be tile-multiples (M%64==0, N%64==0, K%32==0)");

        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-sgemm");

        // --- CPU inputs / reference -----------------------------------------
        std::vector<float> in_a = generate_inputs(kM * kK, 0x12345678u);
        std::vector<float> in_b = generate_inputs(kK * kN, 0x9abcdef0u);
        std::vector<double> c_ref = cpu_reference(in_a, in_b, kM, kN, kK);

        // --- Buffers --------------------------------------------------------
        vk::DeviceSize aBytes = static_cast<vk::DeviceSize>(kM) * kK * sizeof(float);
        vk::DeviceSize bBytes = static_cast<vk::DeviceSize>(kK) * kN * sizeof(float);
        vk::DeviceSize cBytes = static_cast<vk::DeviceSize>(kM) * kN * sizeof(float);

        Buffer dev_a = create_buffer(
            ctx.allocator, aBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_b = create_buffer(
            ctx.allocator, bBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_c = create_buffer(
            ctx.allocator, cBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize stagingBytes = aBytes + bBytes + cBytes;
        Buffer staging = create_buffer(
            ctx.allocator, stagingBytes,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        // Fill A and B regions of the staging buffer from the CPU inputs.
        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            std::memcpy(mapped, in_a.data(), aBytes);
            std::memcpy(static_cast<char*>(mapped) + aBytes, in_b.data(), bBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, aBytes + bBytes);
        }

        // --- Pipeline -------------------------------------------------------
        vk::ShaderModuleCreateInfo moduleInfo;
        vulkan_runtime::shaders::ShaderBlob blob = vulkan_runtime::shaders::get("sgemm.spv");
        if (blob.data == nullptr || blob.size == 0) {
            throw std::runtime_error("sgemm.spv not embedded.");
        }
        moduleInfo.setCodeSize(blob.size)
            .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
        vk::ShaderModule shaderModule = ctx.device.createShaderModule(moduleInfo);

        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
        for (std::uint32_t i = 0; i < 3; ++i) {
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
        poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(3);
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
        vk::DescriptorBufferInfo cInfo;
        cInfo.setBuffer(dev_c.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);

        std::array<vk::WriteDescriptorSet, 3> writes{};
        writes[0].setDstSet(descriptorSet)
            .setDstBinding(0).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(aInfo);
        writes[1].setDstSet(descriptorSet)
            .setDstBinding(1).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(bInfo);
        writes[2].setDstSet(descriptorSet)
            .setDstBinding(2).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(cInfo);
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

        vk::BufferCopy copyB;
        copyB.setSrcOffset(aBytes).setDstOffset(0).setSize(bBytes);
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

        PushConstants pc{kM, kN, kK};
        cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(PushConstants), &pc);
        cmd.dispatch(kN / kTileN, kM / kTileM, 1);

        vk::MemoryBarrier toTransfer;
        toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {},
                            toTransfer, {}, {});

        vk::BufferCopy copyC;
        copyC.setSrcOffset(0).setDstOffset(aBytes + bBytes).setSize(cBytes);
        cmd.copyBuffer(dev_c.buffer, staging.buffer, copyC);

        cmd.end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(cmd);
        ctx.computeQueue.submit(submitInfo);
        ctx.computeQueue.waitIdle();

        // --- Read back + compare vs double reference ------------------------
        bool ok = true;
        std::size_t firstBad = 0;
        double refAtBad = 0.0;
        float gpuAtBad = 0.0f;
        double maxAbs = 0.0;
        double maxRel = 0.0;
        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, aBytes + bBytes, cBytes);

            float const* gpuC = reinterpret_cast<float const*>(
                static_cast<char const*>(mapped) + aBytes + bBytes);
            std::size_t n = static_cast<std::size_t>(kM) * kN;
            constexpr double kRelTol = 1e-3;
            for (std::size_t i = 0; i < n; ++i) {
                double ref = c_ref[i];
                double err = std::fabs(static_cast<double>(gpuC[i]) - ref);
                maxAbs = std::max(maxAbs, err);
                double denom = std::max(1.0, std::fabs(ref));
                double rel = err / denom;
                maxRel = std::max(maxRel, rel);
                if (rel > kRelTol) {
                    firstBad = i;
                    refAtBad = ref;
                    gpuAtBad = gpuC[i];
                    ok = false;
                    break;
                }
            }
            vmaUnmapMemory(ctx.allocator, staging.allocation);
        }

        // --- Tear down ------------------------------------------------------
        ctx.device.destroyCommandPool(commandPool);
        ctx.device.destroyDescriptorPool(descriptorPool);
        ctx.device.destroyPipeline(pipeline);
        ctx.device.destroyPipelineLayout(pipelineLayout);
        ctx.device.destroyDescriptorSetLayout(setLayout);
        ctx.device.destroyShaderModule(shaderModule);
        destroy_buffer(ctx.allocator, staging);
        destroy_buffer(ctx.allocator, dev_c);
        destroy_buffer(ctx.allocator, dev_b);
        destroy_buffer(ctx.allocator, dev_a);
        vulkan_runtime::core::destroy_context(ctx);

        if (ok) {
            std::cout << "sgemm: M=" << kM << " N=" << kN << " K=" << kK
                      << " PASS (max_rel_err=" << maxRel
                      << ", max_abs_err=" << maxAbs << ")\n";
            return EXIT_SUCCESS;
        }
        std::cerr << "sgemm: MISMATCH at index " << firstBad
                  << " (ref=" << refAtBad
                  << ", gpu=" << static_cast<double>(gpuAtBad) << ")\n";
        return EXIT_FAILURE;
    } catch (vk::SystemError const& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
