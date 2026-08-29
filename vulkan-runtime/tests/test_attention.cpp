// M4b: single-head flash-style attention (online softmax) on the GPU, verified
// against a double-precision full-softmax CPU reference on both RADV (gfx803)
// and lavapipe.
//
// O[t][d] = softmax(Q[t] . K / sqrt(D)) . V, single head, fp32, no causal mask
// (full KV). Q/K/V each T x D, O T x D, scale 1/sqrt(D).
//
// Tolerance: the GPU computes the online softmax in fp32 (running max +
// rescale, streamed over KV blocks), while the reference materializes the full
// score row in double, so a small legitimate difference is expected. We pass an
// element iff |gpu - ref| <= max(1e-5, 1e-4 * |ref|): a 1e-4 relative bound
// with a 1e-5 absolute floor for near-zero outputs. We print the actual max
// absolute / relative error on pass.
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
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t kT = 128;
constexpr std::uint32_t kD = 64;

constexpr double kAbsTol = 1e-5;
constexpr double kRelTol = 1e-4;

struct PushConstants {
    std::uint32_t T;
    std::uint32_t D;
};

// Deterministic LCG (Numerical Recipes) -> floats in [-1, 1].
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

// CPU reference in double precision: full softmax over the T keys, then
// weighted sum over V. No causal mask (full KV).
std::vector<double> cpu_reference(std::vector<float> const& q,
                                  std::vector<float> const& k,
                                  std::vector<float> const& v,
                                  std::uint32_t T, std::uint32_t D) {
    double scale = 1.0 / std::sqrt(static_cast<double>(D));
    std::vector<double> o(static_cast<std::size_t>(T) * D, 0.0);
    std::vector<double> s(T);
    for (std::uint32_t t = 0; t < T; ++t) {
        double maxS = -std::numeric_limits<double>::infinity();
        for (std::uint32_t kk = 0; kk < T; ++kk) {
            double dot = 0.0;
            for (std::uint32_t d = 0; d < D; ++d) {
                dot += static_cast<double>(q[static_cast<std::size_t>(t) * D + d]) *
                       static_cast<double>(k[static_cast<std::size_t>(kk) * D + d]);
            }
            s[kk] = dot * scale;
            maxS = std::max(maxS, s[kk]);
        }
        double sum = 0.0;
        for (std::uint32_t kk = 0; kk < T; ++kk) {
            sum += std::exp(s[kk] - maxS);
        }
        for (std::uint32_t d = 0; d < D; ++d) {
            double acc = 0.0;
            for (std::uint32_t kk = 0; kk < T; ++kk) {
                acc += std::exp(s[kk] - maxS) *
                       static_cast<double>(v[static_cast<std::size_t>(kk) * D + d]);
            }
            o[static_cast<std::size_t>(t) * D + d] = acc / sum;
        }
    }
    return o;
}

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
            vulkan_runtime::core::create_context("vulkan-runtime-attention");

        // --- CPU inputs / reference -----------------------------------------
        std::vector<float> in_q = generate_inputs(static_cast<std::size_t>(kT) * kD, 0x11111111u);
        std::vector<float> in_k = generate_inputs(static_cast<std::size_t>(kT) * kD, 0x22222222u);
        std::vector<float> in_v = generate_inputs(static_cast<std::size_t>(kT) * kD, 0x33333333u);
        std::vector<double> o_ref = cpu_reference(in_q, in_k, in_v, kT, kD);

        // --- Buffers --------------------------------------------------------
        vk::DeviceSize tdBytes = static_cast<vk::DeviceSize>(kT) * kD * sizeof(float);

        Buffer dev_q = create_buffer(
            ctx.allocator, tdBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_k = create_buffer(
            ctx.allocator, tdBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_v = create_buffer(
            ctx.allocator, tdBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_o = create_buffer(
            ctx.allocator, tdBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize stagingBytes = tdBytes * 4;
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
            char* dst = static_cast<char*>(mapped);
            std::memcpy(dst, in_q.data(), tdBytes);
            std::memcpy(dst + tdBytes, in_k.data(), tdBytes);
            std::memcpy(dst + tdBytes * 2, in_v.data(), tdBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, tdBytes * 3);
        }

        // --- Pipeline -------------------------------------------------------
        vk::ShaderModuleCreateInfo moduleInfo;
        vulkan_runtime::shaders::ShaderBlob blob = vulkan_runtime::shaders::get("attention.spv");
        if (blob.data == nullptr || blob.size == 0) {
            throw std::runtime_error("attention.spv not embedded.");
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

        vk::DescriptorBufferInfo qInfo;
        qInfo.setBuffer(dev_q.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo kInfo;
        kInfo.setBuffer(dev_k.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo vInfo;
        vInfo.setBuffer(dev_v.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo oInfo;
        oInfo.setBuffer(dev_o.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);

        std::array<vk::WriteDescriptorSet, 4> writes{};
        writes[0].setDstSet(descriptorSet).setDstBinding(0).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(qInfo);
        writes[1].setDstSet(descriptorSet).setDstBinding(1).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(kInfo);
        writes[2].setDstSet(descriptorSet).setDstBinding(2).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(vInfo);
        writes[3].setDstSet(descriptorSet).setDstBinding(3).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(oInfo);
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

        vk::BufferCopy copyQ;
        copyQ.setSrcOffset(0).setDstOffset(0).setSize(tdBytes);
        cmd.copyBuffer(staging.buffer, dev_q.buffer, copyQ);

        vk::BufferCopy copyK;
        copyK.setSrcOffset(tdBytes).setDstOffset(0).setSize(tdBytes);
        cmd.copyBuffer(staging.buffer, dev_k.buffer, copyK);

        vk::BufferCopy copyV;
        copyV.setSrcOffset(tdBytes * 2).setDstOffset(0).setSize(tdBytes);
        cmd.copyBuffer(staging.buffer, dev_v.buffer, copyV);

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
        cmd.dispatch(kT / 16, 1, 1); // one workgroup per 16 query rows

        vk::MemoryBarrier toTransfer;
        toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer, {},
                            toTransfer, {}, {});

        vk::BufferCopy copyO;
        copyO.setSrcOffset(0).setDstOffset(tdBytes * 3).setSize(tdBytes);
        cmd.copyBuffer(dev_o.buffer, staging.buffer, copyO);

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
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, tdBytes * 3, tdBytes);

            float const* gpuO = reinterpret_cast<float const*>(
                static_cast<char const*>(mapped) + tdBytes * 3);
            std::size_t n = static_cast<std::size_t>(kT) * kD;
            for (std::size_t i = 0; i < n; ++i) {
                double ref = o_ref[i];
                double err = std::fabs(static_cast<double>(gpuO[i]) - ref);
                maxAbs = std::max(maxAbs, err);
                double rel = err / std::max(1.0, std::fabs(ref));
                maxRel = std::max(maxRel, rel);
                if (err > kAbsTol && err > kRelTol * std::fabs(ref)) {
                    firstBad = i;
                    refAtBad = ref;
                    gpuAtBad = gpuO[i];
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
        destroy_buffer(ctx.allocator, dev_o);
        destroy_buffer(ctx.allocator, dev_v);
        destroy_buffer(ctx.allocator, dev_k);
        destroy_buffer(ctx.allocator, dev_q);
        vulkan_runtime::core::destroy_context(ctx);

        if (ok) {
            std::cout << "attention: T=" << kT << " D=" << kD << " PASS"
                      << " (max_rel_err=" << maxRel
                      << ", max_abs_err=" << maxAbs << ")\n";
            return EXIT_SUCCESS;
        }
        std::cerr << "attention: MISMATCH at index " << firstBad
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
