// M4b benchmark: int4-dequant GEMM (Q4_0-style) at M=N=K=1024.
//
// Measures C = A * dequant(B) where A is fp32 (4 MB) and B is K x N int4 packed
// 2-per-byte (512 KB) + a per-32-K-group fp32 scale (128 KB). Uses the same
// 64x64-tile / 256-thread / BK=32 kernel structure as sgemm.comp, with the B
// operand dequantized on the fly from packed int4 -> fp32 during the LDS load
// phase.
//
// Perf protocol matches bench_sgemm.cpp: 128 warmup dispatches (RX580 clock
// ramp) then 15 timestamp-bracketed timed dispatches, median reported (never
// the mean). Effective FLOPs = 2*M*N*K (the FMA count the inner loop performs;
// the dequant is extra ALU on top). Correctness is test_q4gemm's job; this
// measures time only.
//
// The dequant overhead is small (~1.5% extra ALU vs the FMA count) but the
// packed-byte + scale loads are less coalesced than sgemm's clean fp32 B load,
// so q4gemm is expected to land below the sgemm baseline (3.21 TFLOPS @ 1024^3,
// M4a). We report the number honestly and flag the gap.

#include "bench/bench_common.h"

#include "core/vk_setup.h"

#include "embedded_shaders.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t kM = 1024;
constexpr std::uint32_t kN = 1024;
constexpr std::uint32_t kK = 1024;

constexpr std::uint32_t kTileM = 64;
constexpr std::uint32_t kTileN = 64;
constexpr std::uint32_t kChunkK = 32;

constexpr std::uint32_t kWarmupRuns = 128;
constexpr std::uint32_t kTimedRuns = 15;

constexpr double kSgemmBaselineTflops = 3.21; // sgemm baseline @ 1024^3 (M4a)

struct PushConstants {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

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

// Deterministic LCG -> floats in [lo, hi).
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

// Packs B (K x N int4) into uint32 words, 2 int4 per byte, 4 bytes per word,
// LSB-first (matches shaders/q4gemm.comp and tests/test_q4gemm.cpp).
std::vector<std::uint32_t> pack_q4(std::vector<std::int8_t> const& b,
                                   std::uint32_t K, std::uint32_t N) {
    std::uint32_t kHalf = K / 2u;
    std::vector<std::uint32_t> packed(static_cast<std::size_t>(kHalf) * N / 4u, 0u);
    for (std::uint32_t n = 0; n < N; ++n) {
        for (std::uint32_t k = 0; k < K; ++k) {
            std::uint32_t nib = static_cast<std::uint32_t>(b[static_cast<std::size_t>(k) * N + n]) & 0xFu;
            std::uint32_t shift = (k & 1u) * 4u;
            std::uint32_t pb = n * kHalf + (k >> 1u);
            std::uint32_t word = pb >> 2u;
            std::uint32_t lane = pb & 3u;
            packed[word] |= nib << (shift + lane * 8u);
        }
    }
    return packed;
}

double tflops(std::uint32_t n, double medianMs) {
    double flops = 2.0 * static_cast<double>(n) * n * n;
    return flops / (medianMs * 1e-3) / 1e12;
}

} // namespace

int main() {
    try {
        using namespace vulkan_runtime;

        core::VulkanContext ctx = core::create_context("vulkan-runtime-q4gemm-bench");

        // Perf numbers are only meaningful on a discrete GPU. Skip fast on
        // lavapipe/llvmpipe so `ctest -L bench` stays cheap in CI.
        if (!bench::is_discrete_gpu(ctx)) {
            std::cout << "bench skipped: not a discrete GPU\n";
            core::destroy_context(ctx);
            return EXIT_SUCCESS;
        }

        vk::PhysicalDeviceProperties props = ctx.physicalDevice.getProperties();
        double period = bench::timestamp_period(ctx.physicalDevice);
        std::uint32_t validBits = bench::timestamp_valid_bits(ctx);

        std::cout << "device: " << props.deviceName << " (timestampPeriod="
                  << std::fixed << std::setprecision(3) << period << " ns, validBits="
                  << validBits << ")\n";
        std::cout << "q4gemm: M=N=K=1024 | warmup=" << kWarmupRuns
                  << " timed=" << kTimedRuns << " (median)\n\n";

        // --- CPU operand packing ---------------------------------------------
        std::vector<float> a = generate_floats(static_cast<std::size_t>(kM) * kK,
                                               0x12345678u, -1.0f, 1.0f);
        std::vector<std::int8_t> b_int = generate_int4(static_cast<std::size_t>(kK) * kN,
                                                       0x9abcdef0u);
        std::uint32_t numGroupsK = kK / kChunkK;
        std::vector<float> scales = generate_floats(static_cast<std::size_t>(kN) * numGroupsK,
                                                    0x13579bdfu, 0.5f, 1.5f);
        std::vector<std::uint32_t> packed = pack_q4(b_int, kK, kN);

        vk::DeviceSize aBytes = static_cast<vk::DeviceSize>(kM) * kK * sizeof(float);
        vk::DeviceSize packedBytes = packed.size() * sizeof(std::uint32_t);
        vk::DeviceSize scalesBytes = scales.size() * sizeof(float);
        vk::DeviceSize cBytes = static_cast<vk::DeviceSize>(kM) * kN * sizeof(float);

        // --- Buffers ---------------------------------------------------------
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
            vk::BufferUsageFlagBits::eStorageBuffer,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize stagingBytes = aBytes + packedBytes + scalesBytes;
        Buffer staging = create_buffer(
            ctx.allocator, stagingBytes,
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            char* dst = static_cast<char*>(mapped);
            std::memcpy(dst, a.data(), aBytes);
            std::memcpy(dst + aBytes, packed.data(), packedBytes);
            std::memcpy(dst + aBytes + packedBytes, scales.data(), scalesBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, stagingBytes);
        }

        // --- Upload operands once (transfer -> compute barrier) ---------------
        {
            vk::CommandPoolCreateInfo poolInfo;
            poolInfo.setQueueFamilyIndex(ctx.computeQueueFamily)
                .setFlags(vk::CommandPoolCreateFlagBits::eTransient);
            vk::CommandPool pool = ctx.device.createCommandPool(poolInfo);

            vk::CommandBufferAllocateInfo allocInfo;
            allocInfo.setCommandPool(pool)
                .setLevel(vk::CommandBufferLevel::ePrimary)
                .setCommandBufferCount(1);
            vk::CommandBuffer cb = ctx.device.allocateCommandBuffers(allocInfo).front();

            cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

            vk::BufferCopy copyA;
            copyA.setSrcOffset(0).setDstOffset(0).setSize(aBytes);
            cb.copyBuffer(staging.buffer, dev_a.buffer, copyA);

            vk::BufferCopy copyPacked;
            copyPacked.setSrcOffset(aBytes).setDstOffset(0).setSize(packedBytes);
            cb.copyBuffer(staging.buffer, dev_packed.buffer, copyPacked);

            vk::BufferCopy copyScales;
            copyScales.setSrcOffset(aBytes + packedBytes).setDstOffset(0).setSize(scalesBytes);
            cb.copyBuffer(staging.buffer, dev_scales.buffer, copyScales);

            vk::MemoryBarrier toShader;
            toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
            cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eComputeShader, {},
                               toShader, {}, {});

            cb.end();

            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBuffers(cb);
            ctx.computeQueue.submit(submitInfo);
            ctx.computeQueue.waitIdle();

            ctx.device.destroyCommandPool(pool);
        }

        // --- Pipeline ---------------------------------------------------------
        vk::ShaderModuleCreateInfo moduleInfo;
        shaders::ShaderBlob blob = shaders::get("q4gemm.spv");
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

        vk::PipelineLayoutCreateInfo layoutInfo;
        layoutInfo.setSetLayouts(setLayout).setPushConstantRanges(pcRange);
        vk::PipelineLayout pipelineLayout = ctx.device.createPipelineLayout(layoutInfo);

        vk::PipelineShaderStageCreateInfo stageInfo;
        stageInfo.setStage(vk::ShaderStageFlagBits::eCompute)
            .setModule(shaderModule)
            .setPName("main");

        vk::ComputePipelineCreateInfo pipelineInfo;
        pipelineInfo.setStage(stageInfo).setLayout(pipelineLayout);
        auto pr = ctx.device.createComputePipeline({}, pipelineInfo);
        if (pr.result != vk::Result::eSuccess) {
            throw std::runtime_error("createComputePipeline failed.");
        }
        vk::Pipeline pipeline = pr.value;

        // --- Descriptor set ---------------------------------------------------
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

        // --- Time the dispatch -------------------------------------------------
        auto recordSetup = [&](vk::CommandBuffer const& cb) {
            cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0,
                                  descriptorSet, {});
            PushConstants pc{kM, kN, kK};
            cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                             sizeof(PushConstants), &pc);
        };
        auto recordDispatch = [&](vk::CommandBuffer const& cb) {
            cb.dispatch(kN / kTileN, kM / kTileM, 1);
        };

        bench::TimedRun run = bench::time_dispatches(ctx, recordSetup, recordDispatch,
                                                     kWarmupRuns, kTimedRuns);

        // --- Report -------------------------------------------------------------
        double tfl = tflops(kN, run.medianMs);
        double packedMb = static_cast<double>(packedBytes) / 1e6;
        double scalesMb = static_cast<double>(scalesBytes) / 1e6;
        double aMb = static_cast<double>(aBytes) / 1e6;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "q4gemm 1024^3: median " << run.medianMs << " ms ("
                  << run.minMs << " .. " << run.maxMs << " ms)\n";
        std::cout << std::setprecision(2);
        std::cout << "q4gemm 1024^3: " << tfl << " TFLOPS (2*N^3 effective FMAs)\n";
        std::cout << "weights: A fp32 " << std::setprecision(2) << aMb
                  << " MB | B packed int4 " << packedMb
                  << " MB + scales " << scalesMb << " MB\n";
        std::cout << std::setprecision(2);
        std::cout << "vs sgemm baseline " << kSgemmBaselineTflops << " TFLOPS: "
                  << (tfl / kSgemmBaselineTflops * 100.0) << "% ("
                  << (tfl - kSgemmBaselineTflops) << " TFLOPS delta)\n";
        std::cout << "note: dequant adds ~1.5% ALU + less-coalesced packed/scale loads; "
                     "a gap vs sgemm is expected (int4 halves B memory traffic, not ALU)\n";

        // --- Tear down ----------------------------------------------------------
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
        core::destroy_context(ctx);

        return EXIT_SUCCESS;
    } catch (vk::SystemError const& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
