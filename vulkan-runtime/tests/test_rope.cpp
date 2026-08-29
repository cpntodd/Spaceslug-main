// M4a kernel library: fp32 rotary position embeddings (RoPE, GPT-NeoX
// non-interleaved layout) on the GPU, verified against a double-precision CPU
// reference on both RADV (gfx803) and lavapipe.
//
// For token p and channel pair i:  theta = p / 10000^(2i/D);
//   y[2i]   = x[2i]  *cos(theta) - x[2i+1]*sin(theta)
//   y[2i+1] = x[2i+1]*cos(theta) + x[2i]  *sin(theta)
//
// The cos/sin tables are precomputed on the host in double and rounded to fp32
// (T * D/2 floats each), so the kernel is pure FMA. This is a deliberate design
// choice: GCN's hardware V_SIN/V_COS argument reduction degrades for the large
// thetas at i=0 (theta up to ~127 rad here), which would force a much looser
// tolerance. With precomputed tables the GPU result tracks the double reference
// to within fp32 rounding (~1e-7).
//
// Tolerance: pass an element iff |gpu - ref| <= max(1e-4, 1e-5 * |ref|).
// Print the actual max absolute / relative error on pass.
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

constexpr std::uint32_t kT = 128;
constexpr std::uint32_t kD = 64;

constexpr double kAbsTol = 1e-4;
constexpr double kRelTol = 1e-5;

// Push-constant block matching the shader's `layout(push_constant) uniform PC`.
struct PushConstants {
    std::uint32_t T;
    std::uint32_t D;
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

// Precompute cos/sin tables (T x D/2) in double, rounded to fp32, and produce
// the double-precision CPU reference in one pass.
struct RoPEReference {
    std::vector<float> cosTable;
    std::vector<float> sinTable;
    std::vector<double> y;
};

RoPEReference cpu_reference(std::vector<float> const& x,
                            std::uint32_t T, std::uint32_t D) {
    std::uint32_t halfD = D / 2;
    RoPEReference ref;
    ref.cosTable.resize(static_cast<std::size_t>(T) * halfD);
    ref.sinTable.resize(static_cast<std::size_t>(T) * halfD);
    ref.y.resize(static_cast<std::size_t>(T) * D);

    for (std::uint32_t t = 0; t < T; ++t) {
        for (std::uint32_t i = 0; i < halfD; ++i) {
            double theta = static_cast<double>(t) *
                           std::pow(10000.0, -2.0 * static_cast<double>(i) / static_cast<double>(D));
            double c = std::cos(theta);
            double s = std::sin(theta);
            std::size_t ti = static_cast<std::size_t>(t) * halfD + i;
            ref.cosTable[ti] = static_cast<float>(c);
            ref.sinTable[ti] = static_cast<float>(s);

            double x0 = static_cast<double>(x[static_cast<std::size_t>(t) * D + 2 * i]);
            double x1 = static_cast<double>(x[static_cast<std::size_t>(t) * D + 2 * i + 1]);
            ref.y[static_cast<std::size_t>(t) * D + 2 * i] = x0 * c - x1 * s;
            ref.y[static_cast<std::size_t>(t) * D + 2 * i + 1] = x1 * c + x0 * s;
        }
    }
    return ref;
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
        static_assert(kD % 2 == 0, "RoPE requires an even head dim");

        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-rope");

        // --- CPU inputs / reference -----------------------------------------
        std::vector<float> in_x = generate_inputs(static_cast<std::size_t>(kT) * kD,
                                                  0x12345678u, -1.0f, 1.0f);
        RoPEReference ref = cpu_reference(in_x, kT, kD);

        // --- Buffers --------------------------------------------------------
        vk::DeviceSize xBytes = static_cast<vk::DeviceSize>(kT) * kD * sizeof(float);
        vk::DeviceSize halfD = kD / 2;
        vk::DeviceSize tableBytes = static_cast<vk::DeviceSize>(kT) * halfD * sizeof(float);
        vk::DeviceSize yBytes = xBytes;

        Buffer dev_x = create_buffer(
            ctx.allocator, xBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_cos = create_buffer(
            ctx.allocator, tableBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_sin = create_buffer(
            ctx.allocator, tableBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_y = create_buffer(
            ctx.allocator, yBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize stagingBytes = xBytes + tableBytes + tableBytes + yBytes;
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
            std::memcpy(static_cast<char*>(mapped) + xBytes, ref.cosTable.data(), tableBytes);
            std::memcpy(static_cast<char*>(mapped) + xBytes + tableBytes,
                        ref.sinTable.data(), tableBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0,
                               xBytes + tableBytes + tableBytes);
        }

        // --- Pipeline -------------------------------------------------------
        vk::ShaderModuleCreateInfo moduleInfo;
        vulkan_runtime::shaders::ShaderBlob blob = vulkan_runtime::shaders::get("rope.spv");
        if (blob.data == nullptr || blob.size == 0) {
            throw std::runtime_error("rope.spv not embedded.");
        }
        moduleInfo.setCodeSize(blob.size)
            .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
        vk::ShaderModule shaderModule = ctx.device.createShaderModule(moduleInfo);

        std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
        for (std::uint32_t i = 0; i < 4; ++i) {
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
        poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(4);
        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setMaxSets(1).setPoolSizes(poolSize);
        vk::DescriptorPool descriptorPool = ctx.device.createDescriptorPool(poolInfo);

        vk::DescriptorSetAllocateInfo setAllocInfo;
        setAllocInfo.setDescriptorPool(descriptorPool).setSetLayouts(setLayout);
        vk::DescriptorSet descriptorSet =
            ctx.device.allocateDescriptorSets(setAllocInfo).front();

        vk::DescriptorBufferInfo xInfo;
        xInfo.setBuffer(dev_x.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo cosInfo;
        cosInfo.setBuffer(dev_cos.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo sinInfo;
        sinInfo.setBuffer(dev_sin.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo yInfo;
        yInfo.setBuffer(dev_y.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);

        std::array<vk::WriteDescriptorSet, 4> writes{};
        writes[0].setDstSet(descriptorSet)
            .setDstBinding(0).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(xInfo);
        writes[1].setDstSet(descriptorSet)
            .setDstBinding(1).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(cosInfo);
        writes[2].setDstSet(descriptorSet)
            .setDstBinding(2).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(sinInfo);
        writes[3].setDstSet(descriptorSet)
            .setDstBinding(3).setDescriptorCount(1)
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

        vk::BufferCopy copyCos;
        copyCos.setSrcOffset(xBytes).setDstOffset(0).setSize(tableBytes);
        cmd.copyBuffer(staging.buffer, dev_cos.buffer, copyCos);

        vk::BufferCopy copySin;
        copySin.setSrcOffset(xBytes + tableBytes).setDstOffset(0).setSize(tableBytes);
        cmd.copyBuffer(staging.buffer, dev_sin.buffer, copySin);

        vk::MemoryBarrier toShader;
        toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {},
                            toShader, {}, {});

        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0,
                               descriptorSet, {});

        PushConstants pc{kT, kD};
        cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(PushConstants), &pc);
        std::uint32_t totalPairs = kT * (kD / 2);
        cmd.dispatch((totalPairs + 255u) / 256u, 1, 1);

        vk::MemoryBarrier toTransfer;
        toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {},
                            toTransfer, {}, {});

        vk::BufferCopy copyY;
        copyY.setSrcOffset(0).setDstOffset(xBytes + tableBytes + tableBytes).setSize(yBytes);
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
            vmaInvalidateAllocation(ctx.allocator, staging.allocation,
                                    xBytes + tableBytes + tableBytes, yBytes);

            float const* gpuY = reinterpret_cast<float const*>(
                static_cast<char const*>(mapped) + xBytes + tableBytes + tableBytes);
            std::size_t n = static_cast<std::size_t>(kT) * kD;
            for (std::size_t i = 0; i < n; ++i) {
                double refVal = ref.y[i];
                double err = std::fabs(static_cast<double>(gpuY[i]) - refVal);
                maxAbs = std::max(maxAbs, err);
                double rel = err / std::max(1.0, std::fabs(refVal));
                maxRel = std::max(maxRel, rel);
                if (err > kAbsTol && err > kRelTol * std::fabs(refVal)) {
                    firstBad = i;
                    refAtBad = refVal;
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
        destroy_buffer(ctx.allocator, dev_sin);
        destroy_buffer(ctx.allocator, dev_cos);
        destroy_buffer(ctx.allocator, dev_x);
        vulkan_runtime::core::destroy_context(ctx);

        if (ok) {
            std::cout << "rope: T=" << kT << " D=" << kD << " PASS"
                      << " (max_rel_err=" << maxRel
                      << ", max_abs_err=" << maxAbs << ")\n";
            return EXIT_SUCCESS;
        }
        std::cerr << "rope: MISMATCH at index " << firstBad
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
