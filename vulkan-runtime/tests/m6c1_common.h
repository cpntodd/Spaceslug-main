// M6c-1: shared utilities for the cactus GPU kernel tests.
//
// Keeps the 9 self-contained tests free of copy-pasted boilerplate for the
// bits that are identical everywhere: the deterministic LCG, fp16 (uint16_t
// bits) <-> uint32 (one-per-word) widening, the VMA Buffer wrapper, the
// repeated compute-pipeline/dispatch plumbing, and the compare/report helpers.
//
// Each test still owns its own push-constant struct, buffer layout, dispatch
// grid, and CPU-reference call — only the mechanical scaffolding is shared.

#pragma once

#include "core/vk_setup.h"

#include "cactus/cactus_x86_bridge.h"

#include "embedded_shaders.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace m6c1 {

// ---------------------------------------------------------------------------
// Deterministic LCG (Numerical Recipes) -> floats in [lo, hi).
// ---------------------------------------------------------------------------
inline std::vector<float> gen_f32(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        v[i] = lo + u * (hi - lo);
    }
    return v;
}

// fp16 bits -> one fp16 per uint32 (low 16 bits) for GPU buffers.
inline std::vector<std::uint32_t> widen16(std::vector<std::uint16_t> const& h) {
    std::vector<std::uint32_t> w(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) w[i] = h[i];
    return w;
}

// one fp16 per uint32 (GPU readback) -> fp16 bits.
inline std::vector<std::uint16_t> narrow16(std::vector<std::uint32_t> const& w) {
    std::vector<std::uint16_t> h(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) h[i] = static_cast<std::uint16_t>(w[i] & 0xFFFFu);
    return h;
}

// --- M7a packed fp16 (2 per uint32) ---------------------------------------
// Convention: element index i -> word i>>1, lane i&1 (0 = low 16 bits, 1 =
// high 16 bits). This is byte-identical to fp16pack mode 1 (u32->u16) and
// packHalf2x16(vec2(lo, hi)), so the backend's fp16pack consumes/produces it
// directly.

// fp16 bits (elements) -> packed uint32 (2 per word: even=low, odd=high).
inline std::vector<std::uint32_t> pack16(std::vector<std::uint16_t> const& h) {
    std::vector<std::uint32_t> w((h.size() + 1u) / 2u, 0u);
    for (std::size_t i = 0; i < h.size(); ++i) {
        w[i >> 1u] |= static_cast<std::uint32_t>(h[i]) << ((i & 1u) * 16u);
    }
    return w;
}

// packed uint32 (GPU readback) -> fp16 bits (first `count` elements).
inline std::vector<std::uint16_t> unpack16(std::vector<std::uint32_t> const& w,
                                           std::size_t count) {
    std::vector<std::uint16_t> h(count);
    for (std::size_t i = 0; i < count; ++i) {
        h[i] = static_cast<std::uint16_t>((w[i >> 1u] >> ((i & 1u) * 16u)) & 0xFFFFu);
    }
    return h;
}

// ---------------------------------------------------------------------------
// VMA-managed buffer wrapper (identical across the existing tests).
// ---------------------------------------------------------------------------
struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
    vk::DeviceSize size{0};
};

inline Buffer create_buffer(VmaAllocator allocator, vk::DeviceSize size,
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

inline void destroy_buffer(VmaAllocator allocator, Buffer& b) {
    if (b.allocation != nullptr) {
        vmaDestroyBuffer(allocator, static_cast<VkBuffer>(b.buffer), b.allocation);
        b.allocation = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Compute-pipeline + dispatch plumbing. Builds a pipeline over N storage-buffer
// bindings, records copy-in -> barrier -> dispatch -> barrier -> copy-out, and
// submits + waits. The caller fills `staging` (flush) before, and reads it
// (invalidate) after.
//
//   dev_buffers : device buffers, bound at binding 0..N-1 in order
//   in_copies   : (staging_offset, size) pairs copied staging -> device (in order)
//   out_copies  : (staging_offset, size) pairs copied device -> staging (in order)
// ---------------------------------------------------------------------------
struct Copy {
    vk::DeviceSize stagingOffset;
    vk::DeviceSize size;
};

inline void run_kernel(vulkan_runtime::core::VulkanContext& ctx, char const* shaderName,
                       std::vector<Buffer> const& dev_buffers, Buffer const& staging,
                       std::vector<Copy> const& in_copies, std::vector<Copy> const& out_copies,
                       void const* pushData, std::uint32_t pushSize,
                       std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
    std::uint32_t const n = static_cast<std::uint32_t>(dev_buffers.size());

    vulkan_runtime::shaders::ShaderBlob blob = vulkan_runtime::shaders::get(shaderName);
    if (blob.data == nullptr || blob.size == 0) {
        throw std::runtime_error(std::string(shaderName) + " not embedded.");
    }

    vk::ShaderModuleCreateInfo moduleInfo;
    moduleInfo.setCodeSize(blob.size)
        .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
    vk::ShaderModule shaderModule = ctx.device.createShaderModule(moduleInfo);

    std::vector<vk::DescriptorSetLayoutBinding> bindings(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        bindings[i].setBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    }

    vk::DescriptorSetLayoutCreateInfo setLayoutInfo;
    setLayoutInfo.setBindings(bindings);
    vk::DescriptorSetLayout setLayout = ctx.device.createDescriptorSetLayout(setLayoutInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(setLayout);
    if (pushSize > 0) {
        vk::PushConstantRange pcRange;
        pcRange.setStageFlags(vk::ShaderStageFlagBits::eCompute)
            .setOffset(0)
            .setSize(pushSize);
        pipelineLayoutInfo.setPushConstantRanges(pcRange);
    }
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

    vk::DescriptorPoolSize poolSize;
    poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(n);
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(1).setPoolSizes(poolSize);
    vk::DescriptorPool descriptorPool = ctx.device.createDescriptorPool(poolInfo);

    vk::DescriptorSetAllocateInfo setAllocInfo;
    setAllocInfo.setDescriptorPool(descriptorPool).setSetLayouts(setLayout);
    vk::DescriptorSet descriptorSet = ctx.device.allocateDescriptorSets(setAllocInfo).front();

    std::vector<vk::DescriptorBufferInfo> bufInfos(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        bufInfos[i].setBuffer(dev_buffers[i].buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
    }
    std::vector<vk::WriteDescriptorSet> writes(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        writes[i].setDstSet(descriptorSet)
            .setDstBinding(i).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(bufInfos[i]);
    }
    ctx.device.updateDescriptorSets(writes, {});

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

    std::uint32_t bi = 0;
    for (Copy const& c : in_copies) {
        vk::BufferCopy bc;
        bc.setSrcOffset(c.stagingOffset).setDstOffset(0).setSize(c.size);
        cmd.copyBuffer(staging.buffer, dev_buffers[bi].buffer, bc);
        ++bi;
    }

    vk::MemoryBarrier toShader;
    toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
        .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {},
                        toShader, {}, {});

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0,
                           descriptorSet, {});
    if (pushSize > 0) {
        cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                          pushSize, pushData);
    }
    cmd.dispatch(gx, gy, gz);

    vk::MemoryBarrier toTransfer;
    toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
        .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eTransfer, {},
                        toTransfer, {}, {});

    std::uint32_t bo = 0;
    for (Copy const& c : out_copies) {
        vk::BufferCopy bc;
        bc.setSrcOffset(0).setDstOffset(c.stagingOffset).setSize(c.size);
        cmd.copyBuffer(dev_buffers[n - out_copies.size() + bo].buffer, staging.buffer, bc);
        ++bo;
    }

    cmd.end();

    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(cmd);
    ctx.computeQueue.submit(submitInfo);
    ctx.computeQueue.waitIdle();

    ctx.device.destroyCommandPool(commandPool);
    ctx.device.destroyDescriptorPool(descriptorPool);
    ctx.device.destroyPipeline(pipeline);
    ctx.device.destroyPipelineLayout(pipelineLayout);
    ctx.device.destroyDescriptorSetLayout(setLayout);
    ctx.device.destroyShaderModule(shaderModule);
}

// ---------------------------------------------------------------------------
// Compare + report helpers. Each prints "test: <name> PASS|FAIL (...)" and
// returns ok. rel/abs floors match the fp16 I/O tolerance contract.
// ---------------------------------------------------------------------------

// fp16 buffer compare: convert both sides to fp32 (exact for binary16) and
// measure elementwise error.
inline bool compare_fp16(char const* name, std::vector<std::uint16_t> const& got,
                         std::vector<std::uint16_t> const& ref, double abs_tol, double rel_tol) {
    std::size_t n = std::min(got.size(), ref.size());
    std::vector<float> gf(n), rf(n);
    cactus_bridge_fp16_to_fp32(got.data(), gf.data(), n);
    cactus_bridge_fp16_to_fp32(ref.data(), rf.data(), n);
    double max_rel = 0.0, max_abs = 0.0;
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) {
        double g = gf[i], r = rf[i];
        double err = std::fabs(g - r);
        max_abs = std::max(max_abs, err);
        max_rel = std::max(max_rel, err / std::max(1.0, std::fabs(r)));
        if (err > abs_tol && err > rel_tol * std::fabs(r)) ok = false;
    }
    std::cout << "test: " << name << (ok ? " PASS" : " FAIL")
              << " (max_rel=" << max_rel << ", max_abs=" << max_abs << ")\n";
    return ok;
}

// fp32 buffer compare.
inline bool compare_f32(char const* name, std::vector<float> const& got,
                        std::vector<float> const& ref, double abs_tol, double rel_tol) {
    std::size_t n = std::min(got.size(), ref.size());
    double max_rel = 0.0, max_abs = 0.0;
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) {
        double g = got[i], r = ref[i];
        double err = std::fabs(g - r);
        max_abs = std::max(max_abs, err);
        max_rel = std::max(max_rel, err / std::max(1.0, std::fabs(r)));
        if (err > abs_tol && err > rel_tol * std::fabs(r)) ok = false;
    }
    std::cout << "test: " << name << (ok ? " PASS" : " FAIL")
              << " (max_rel=" << max_rel << ", max_abs=" << max_abs << ")\n";
    return ok;
}

// Bitwise-exact fp16 compare.
inline bool compare_fp16_exact(char const* name, std::vector<std::uint16_t> const& got,
                               std::vector<std::uint16_t> const& ref) {
    bool ok = got.size() == ref.size() &&
              std::memcmp(got.data(), ref.data(), got.size() * sizeof(std::uint16_t)) == 0;
    std::cout << "test: " << name << (ok ? " PASS" : " FAIL")
              << " (max_rel=" << (ok ? 0.0 : 1.0) << ", max_abs=" << (ok ? 0.0 : 1.0) << ")\n";
    return ok;
}

// Exact int8 compare (GPU int8 as int32 words vs CPU int8_t).
inline bool compare_int8_exact(char const* name, std::vector<std::int32_t> const& got,
                               std::vector<std::int8_t> const& ref) {
    bool ok = got.size() == ref.size();
    if (ok) {
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (got[i] != ref[i]) { ok = false; break; }
        }
    }
    std::cout << "test: " << name << (ok ? " PASS" : " FAIL")
              << " (max_rel=" << (ok ? 0.0 : 1.0) << ", max_abs=" << (ok ? 0.0 : 1.0) << ")\n";
    return ok;
}

// Exact float compare (scales).
inline bool compare_f32_exact(char const* name, std::vector<float> const& got,
                              std::vector<float> const& ref) {
    bool ok = got.size() == ref.size();
    if (ok) {
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (got[i] != ref[i]) { ok = false; break; }
        }
    }
    std::cout << "test: " << name << (ok ? " PASS" : " FAIL")
              << " (max_rel=" << (ok ? 0.0 : 1.0) << ", max_abs=" << (ok ? 0.0 : 1.0) << ")\n";
    return ok;
}

} // namespace m6c1
