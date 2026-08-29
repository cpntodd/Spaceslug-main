// M4a kernel library: fp32 RMSNorm on the GPU, verified against a
// double-precision CPU reference on both RADV (gfx803) and lavapipe.
//
// Per row r:  y[r][i] = x[r][i] * (w[i] / sqrt(mean_i(x[r][i]^2) + eps)).
//
// Tolerance: the GPU accumulates the sum of squares in fp32 (tree reduction)
// and normalizes in fp32, while the reference does everything in double, so a
// small legitimate difference is expected. We pass an element iff
// |gpu - ref| <= max(1e-4, 1e-5 * |ref|) — a 1e-5 relative tolerance with a
// 1e-4 absolute floor (the floor guards near-zero outputs where relative error
// is meaningless). We print the actual max absolute / relative error on pass.
//
// Flow mirrors test_sgemm.cpp (headless, no WSI).

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

constexpr std::uint32_t kR = 128;
constexpr std::uint32_t kN = 4096;
constexpr double kEps = 1e-5;

constexpr double kAbsTol = 1e-4;
constexpr double kRelTol = 1e-5;

// Push-constant block matching the shader's `layout(push_constant) uniform PC`.
struct PushConstants {
    std::uint32_t R;
    std::uint32_t N;
};

// Deterministic LCG (Numerical Recipes) -> floats in [0, 1), scaled to [lo, hi).
std::vector<float> generate_inputs(std::size_t n, std::uint32_t seed,
                                   float lo, float hi) {
    std::vector<float> v(n);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        v[i] = lo + u * (hi - lo);
    }
    return v;
}

// CPU reference in double precision.
std::vector<double> cpu_reference(std::vector<float> const& x,
                                  std::vector<float> const& w,
                                  std::uint32_t R, std::uint32_t N) {
    std::vector<double> y(static_cast<std::size_t>(R) * N);
    for (std::uint32_t r = 0; r < R; ++r) {
        double sum = 0.0;
        for (std::uint32_t i = 0; i < N; ++i) {
            double xv = static_cast<double>(x[static_cast<std::size_t>(r) * N + i]);
            sum += xv * xv;
        }
        double inv = 1.0 / std::sqrt(sum / static_cast<double>(N) + kEps);
        for (std::uint32_t i = 0; i < N; ++i) {
            std::size_t idx = static_cast<std::size_t>(r) * N + i;
            y[idx] = static_cast<double>(x[idx]) * static_cast<double>(w[i]) * inv;
        }
    }
    return y;
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
            vulkan_runtime::core::create_context("vulkan-runtime-rmsnorm");

        // --- CPU inputs / reference -----------------------------------------
        std::vector<float> in_x = generate_inputs(static_cast<std::size_t>(kR) * kN,
                                                  0x12345678u, -1.0f, 1.0f);
        std::vector<float> in_w = generate_inputs(kN, 0x9abcdef0u, 0.5f, 1.5f);
        std::vector<double> y_ref = cpu_reference(in_x, in_w, kR, kN);

        // --- Buffers --------------------------------------------------------
        vk::DeviceSize xBytes = static_cast<vk::DeviceSize>(kR) * kN * sizeof(float);
        vk::DeviceSize wBytes = static_cast<vk::DeviceSize>(kN) * sizeof(float);
        vk::DeviceSize yBytes = xBytes;

        Buffer dev_x = create_buffer(
            ctx.allocator, xBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_w = create_buffer(
            ctx.allocator, wBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_y = create_buffer(
            ctx.allocator, yBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize stagingBytes = xBytes + wBytes + yBytes;
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
            std::memcpy(mapped, in_x.data(), xBytes);
            std::memcpy(static_cast<char*>(mapped) + xBytes, in_w.data(), wBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, xBytes + wBytes);
        }

        // --- Pipeline -------------------------------------------------------
        vk::ShaderModuleCreateInfo moduleInfo;
        vulkan_runtime::shaders::ShaderBlob blob = vulkan_runtime::shaders::get("rmsnorm.spv");
        if (blob.data == nullptr || blob.size == 0) {
            throw std::runtime_error("rmsnorm.spv not embedded.");
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

        vk::DescriptorBufferInfo xInfo;
        xInfo.setBuffer(dev_x.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo wInfo;
        wInfo.setBuffer(dev_w.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo yInfo;
        yInfo.setBuffer(dev_y.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);

        std::array<vk::WriteDescriptorSet, 3> writes{};
        writes[0].setDstSet(descriptorSet)
            .setDstBinding(0).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(xInfo);
        writes[1].setDstSet(descriptorSet)
            .setDstBinding(1).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(wInfo);
        writes[2].setDstSet(descriptorSet)
            .setDstBinding(2).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(yInfo);
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

        vk::BufferCopy copyX;
        copyX.setSrcOffset(0).setDstOffset(0).setSize(xBytes);
        cmd.copyBuffer(staging.buffer, dev_x.buffer, copyX);

        vk::BufferCopy copyW;
        copyW.setSrcOffset(xBytes).setDstOffset(0).setSize(wBytes);
        cmd.copyBuffer(staging.buffer, dev_w.buffer, copyW);

        vk::MemoryBarrier toShader;
        toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {},
                            toShader, {}, {});

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0,
                               descriptorSet, {});

        PushConstants pc{kR, kN};
        cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(PushConstants), &pc);
        cmd.dispatch(kR, 1, 1); // one workgroup per row

        vk::MemoryBarrier toTransfer;
        toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {},
                            toTransfer, {}, {});

        vk::BufferCopy copyY;
        copyY.setSrcOffset(0).setDstOffset(xBytes + wBytes).setSize(yBytes);
        cmd.copyBuffer(dev_y.buffer, staging.buffer, copyY);

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
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, xBytes + wBytes, yBytes);

            float const* gpuY = reinterpret_cast<float const*>(
                static_cast<char const*>(mapped) + xBytes + wBytes);
            std::size_t n = static_cast<std::size_t>(kR) * kN;
            for (std::size_t i = 0; i < n; ++i) {
                double ref = y_ref[i];
                double err = std::fabs(static_cast<double>(gpuY[i]) - ref);
                maxAbs = std::max(maxAbs, err);
                double rel = err / std::max(1.0, std::fabs(ref));
                maxRel = std::max(maxRel, rel);
                // Combined tolerance: fail only if the error exceeds BOTH the
                // relative and the absolute floor.
                if (err > kAbsTol && err > kRelTol * std::fabs(ref)) {
                    firstBad = i;
                    refAtBad = ref;
                    gpuAtBad = gpuY[i];
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
        destroy_buffer(ctx.allocator, dev_y);
        destroy_buffer(ctx.allocator, dev_w);
        destroy_buffer(ctx.allocator, dev_x);
        vulkan_runtime::core::destroy_context(ctx);

        if (ok) {
            std::cout << "rmsnorm: R=" << kR << " N=" << kN << " PASS"
                      << " (max_rel_err=" << maxRel
                      << ", max_abs_err=" << maxAbs << ")\n";
            return EXIT_SUCCESS;
        }
        std::cerr << "rmsnorm: MISMATCH at index " << firstBad
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
