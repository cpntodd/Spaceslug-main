// M4b: int4-dequant GEMM (Q4_0-style) on the GPU, verified against a
// double-precision CPU reference on both RADV (gfx803) and lavapipe.
//
// C[M x N] = A[M x K] * dequant(B), where A is fp32 (row-major) and B is
// K x N int4 stored as packed bytes + a per-32-K-group fp32 scale.
//
// Weight layout (MUST match shaders/q4gemm.comp):
//   - int4 value v in [-8, 7], stored as a 4-bit two's-complement nibble
//     (v & 0xF).
//   - LSB-FIRST packing: within a packed byte, the low nibble (bits 0..3)
//     holds the int4 for even k, the high nibble (bits 4..7) for odd k.
//   - packed-byte index for (k, n): pb = n * (K/2) + (k >> 1)  (column-major:
//     K inner, N outer).
//   - bytes stored 4-per-uint32 in the packed SSBO, little-endian lane order
//     (byte pb -> word pb>>2, lane pb&3, bits lane*8).
//   - scale for (k, n): scales[n * (K/32) + (k/32)].
//   - dequant: b = int4_value(k, n) * scale(n, k/32).
//
// Tolerance: the GPU dequantizes in fp32 and accumulates the GEMM in fp32,
// while the reference dequantizes in double and accumulates in double, so a
// small legitimate difference is expected (same 1e-3 relative bound as sgemm).
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

constexpr std::uint32_t kM = 256;
constexpr std::uint32_t kN = 256;
constexpr std::uint32_t kK = 256;

// Shader contract (q4gemm.comp): M % 64 == 0, N % 64 == 0, K % 32 == 0.
constexpr std::uint32_t kTileM = 64;
constexpr std::uint32_t kTileN = 64;
constexpr std::uint32_t kChunkK = 32;

constexpr double kRelTol = 1e-3;

struct PushConstants {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

// Deterministic LCG (Numerical Recipes) -> floats in [lo, hi).
std::vector<float> generate_floats(std::size_t n, std::uint32_t seed,
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

// Deterministic LCG -> int4 values in [-8, 7].
std::vector<std::int8_t> generate_int4(std::size_t n, std::uint32_t seed) {
    std::vector<std::int8_t> v(n);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        std::uint32_t r = s >> 8;
        v[i] = static_cast<std::int8_t>(static_cast<int>(r % 16u) - 8);
    }
    return v;
}

// Packs B (K x N int4, values in [-8,7]) into uint32 words, 2 int4 per byte,
// 4 bytes per word, LSB-first. See the layout note at the top of this file.
std::vector<std::uint32_t> pack_q4(std::vector<std::int8_t> const& b,
                                   std::uint32_t K, std::uint32_t N) {
    std::uint32_t kHalf = K / 2u;
    std::vector<std::uint32_t> packed(static_cast<std::size_t>(kHalf) * N / 4u, 0u);
    for (std::uint32_t n = 0; n < N; ++n) {
        for (std::uint32_t k = 0; k < K; ++k) {
            std::uint32_t nib = static_cast<std::uint32_t>(b[static_cast<std::size_t>(k) * N + n]) & 0xFu;
            std::uint32_t shift = (k & 1u) * 4u; // even k -> low nibble, odd k -> high nibble
            std::uint32_t pb = n * kHalf + (k >> 1u);
            std::uint32_t word = pb >> 2u;
            std::uint32_t lane = pb & 3u;
            packed[word] |= nib << (shift + lane * 8u);
        }
    }
    return packed;
}

// CPU reference in double: dequant int4 * scale, then GEMM A * dequant(B).
std::vector<double> cpu_reference(std::vector<float> const& a,
                                  std::vector<std::int8_t> const& b,
                                  std::vector<float> const& scales,
                                  std::uint32_t M, std::uint32_t N, std::uint32_t K) {
    std::uint32_t numGroupsK = K / 32u;
    std::vector<double> c(static_cast<std::size_t>(M) * N, 0.0);
    for (std::uint32_t m = 0; m < M; ++m) {
        for (std::uint32_t k = 0; k < K; ++k) {
            double av = static_cast<double>(a[static_cast<std::size_t>(m) * K + k]);
            for (std::uint32_t n = 0; n < N; ++n) {
                double bv = static_cast<double>(b[static_cast<std::size_t>(k) * N + n]) *
                            static_cast<double>(scales[static_cast<std::size_t>(n) * numGroupsK + (k / 32u)]);
                c[static_cast<std::size_t>(m) * N + n] += av * bv;
            }
        }
    }
    return c;
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
        static_assert(kM % kTileM == 0 && kN % kTileN == 0 && kK % kChunkK == 0,
                      "q4gemm sizes must be tile-multiples (M%64==0, N%64==0, K%32==0)");

        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-q4gemm");

        // --- CPU inputs / reference -----------------------------------------
        std::vector<float> in_a = generate_floats(static_cast<std::size_t>(kM) * kK,
                                                  0x12345678u, -1.0f, 1.0f);
        std::vector<std::int8_t> in_b_int = generate_int4(static_cast<std::size_t>(kK) * kN,
                                                          0x9abcdef0u);
        std::uint32_t numGroupsK = kK / kChunkK;
        std::vector<float> in_scales =
            generate_floats(static_cast<std::size_t>(kN) * numGroupsK, 0x13579bdfu, 0.5f, 1.5f);
        std::vector<std::uint32_t> packed_words = pack_q4(in_b_int, kK, kN);
        std::vector<double> c_ref = cpu_reference(in_a, in_b_int, in_scales, kM, kN, kK);

        // --- Buffers --------------------------------------------------------
        vk::DeviceSize aBytes = static_cast<vk::DeviceSize>(kM) * kK * sizeof(float);
        vk::DeviceSize packedBytes = packed_words.size() * sizeof(std::uint32_t);
        vk::DeviceSize scalesBytes = in_scales.size() * sizeof(float);
        vk::DeviceSize cBytes = static_cast<vk::DeviceSize>(kM) * kN * sizeof(float);

        Buffer dev_a = create_buffer(
            ctx.allocator, aBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_packed = create_buffer(
            ctx.allocator, packedBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_scales = create_buffer(
            ctx.allocator, scalesBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_c = create_buffer(
            ctx.allocator, cBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize stagingBytes = aBytes + packedBytes + scalesBytes + cBytes;
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
            std::memcpy(dst, in_a.data(), aBytes);
            std::memcpy(dst + aBytes, packed_words.data(), packedBytes);
            std::memcpy(dst + aBytes + packedBytes, in_scales.data(), scalesBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0,
                               aBytes + packedBytes + scalesBytes);
        }

        // --- Pipeline -------------------------------------------------------
        vk::ShaderModuleCreateInfo moduleInfo;
        vulkan_runtime::shaders::ShaderBlob blob = vulkan_runtime::shaders::get("q4gemm.spv");
        if (blob.data == nullptr || blob.size == 0) {
            throw std::runtime_error("q4gemm.spv not embedded.");
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

        vk::DescriptorBufferInfo aInfo;
        aInfo.setBuffer(dev_a.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo packedInfo;
        packedInfo.setBuffer(dev_packed.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo scalesInfo;
        scalesInfo.setBuffer(dev_scales.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo cInfo;
        cInfo.setBuffer(dev_c.buffer).setOffset(0).setRange(VK_WHOLE_SIZE);

        std::array<vk::WriteDescriptorSet, 4> writes{};
        writes[0].setDstSet(descriptorSet).setDstBinding(0).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(aInfo);
        writes[1].setDstSet(descriptorSet).setDstBinding(1).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(packedInfo);
        writes[2].setDstSet(descriptorSet).setDstBinding(2).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(scalesInfo);
        writes[3].setDstSet(descriptorSet).setDstBinding(3).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(cInfo);
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

        vk::BufferCopy copyPacked;
        copyPacked.setSrcOffset(aBytes).setDstOffset(0).setSize(packedBytes);
        cmd.copyBuffer(staging.buffer, dev_packed.buffer, copyPacked);

        vk::BufferCopy copyScales;
        copyScales.setSrcOffset(aBytes + packedBytes).setDstOffset(0).setSize(scalesBytes);
        cmd.copyBuffer(staging.buffer, dev_scales.buffer, copyScales);

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
        copyC.setSrcOffset(0).setDstOffset(aBytes + packedBytes + scalesBytes).setSize(cBytes);
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
            vmaInvalidateAllocation(ctx.allocator, staging.allocation,
                                    aBytes + packedBytes + scalesBytes, cBytes);

            float const* gpuC = reinterpret_cast<float const*>(
                static_cast<char const*>(mapped) + aBytes + packedBytes + scalesBytes);
            std::size_t n = static_cast<std::size_t>(kM) * kN;
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
        destroy_buffer(ctx.allocator, dev_scales);
        destroy_buffer(ctx.allocator, dev_packed);
        destroy_buffer(ctx.allocator, dev_a);
        vulkan_runtime::core::destroy_context(ctx);

        if (ok) {
            std::cout << "q4gemm: M=" << kM << " N=" << kN << " K=" << kK
                      << " PASS (max_rel_err=" << maxRel
                      << ", max_abs_err=" << maxAbs << ")\n";
            return EXIT_SUCCESS;
        }
        std::cerr << "q4gemm: MISMATCH at index " << firstBad
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
