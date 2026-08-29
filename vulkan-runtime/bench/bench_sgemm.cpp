// M3 tuning harness driver — fp32 SGEMM size sweep + variant ranking.
//
// Ranks the M2 baseline (v0, sgemm.comp) against three tuning variants:
//   v1 (sgemm_bk48.comp)  — BK=48, fewer barriers per K-loop
//   v2 (sgemm_vec4a.comp) — vec4 A-operand LDS loads (ds_read_b128)
//   v3 (sgemm_8x2.comp)   — 8x2 micro-tile (different FMA ILP / reg pressure)
//
// For every (variant, size) it runs 128 warmup dispatches (RX580 clock ramp)
// then 15 timestamp-bracketed timed dispatches and reports the median (never
// the mean — scheduler noise), plus min/max as the reproducibility evidence.
//
// Correctness is NOT checked here (that is test_sgemm's job); this measures
// time only. FLOPs = 2*M*N*K. TFLOPS = FLOPs / median time.
//
// Exit policy: non-zero only if the best 1024^3 variant falls below 1.0 TFLOPS
// (the M2 gate — a signal that something is genuinely broken, e.g. missing
// glslc -O). A best below the M2 measured baseline (3.19 TFLOPS) is printed as
// a loud REGRESSION warning but still exits 0, so clock variance never flakes
// ctest. Both numbers are always reported.

#include "bench/bench_common.h"

#include "core/vk_setup.h"

#include "embedded_shaders.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kWarmupRuns = 128;
constexpr std::uint32_t kTimedRuns = 15; // >= 9; median of 15 for tighter medians

constexpr std::uint32_t kTileM = 64;
constexpr std::uint32_t kTileN = 64;

constexpr std::size_t kAnchorIdx = 2; // kSizes[2] == 1024

constexpr double kM2BaselineTflops = 3.19; // M2 measured @ 1024^3 (reference)
constexpr double kGateTflops = 1.0;        // M2 milestone gate (hard floor)
constexpr double kStretchTflops = 2.54;    // vkblas reference on gfx803
constexpr double kPeakTflops = 6.0;        // ~5.8-6.2 TFLOPS fp32 peak

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

// One ranked variant: a display name and its embedded shader registry key.
struct Variant {
    char const* name;
    char const* shader; // "<name>.spv"
};

std::array<Variant, 4> const kVariants{{
    {"v0_bk32_4x4", "sgemm.spv"},
    {"v1_bk48_4x4", "sgemm_bk48.spv"},
    {"v2_bk32_vec4a", "sgemm_vec4a.spv"},
    {"v3_bk32_8x2", "sgemm_8x2.spv"},
}};

std::array<std::uint32_t, 5> const kSizes{{256, 512, 1024, 1536, 2048}};

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

// Deterministic fill of a staging buffer region (values in [-1,1]).
void fill(std::vector<float>& dst, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (float& x : dst) {
        s = s * 1664525u + 1013904223u;
        float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        x = u * 2.0f - 1.0f;
    }
}

// Copies A and B from the staging buffer into the device-local buffers and
// inserts the transfer->compute barrier. Runs once per size (the operand data
// is identical across variants, so it is uploaded before the variant loop).
void upload_operands(vulkan_runtime::core::VulkanContext const& ctx,
                     Buffer const& staging, Buffer const& dev_a, Buffer const& dev_b,
                     vk::DeviceSize aBytes, vk::DeviceSize bBytes) {
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

    vk::BufferCopy copyB;
    copyB.setSrcOffset(aBytes).setDstOffset(0).setSize(bBytes);
    cb.copyBuffer(staging.buffer, dev_b.buffer, copyB);

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

double tflops(std::uint32_t n, double medianMs) {
    double flops = 2.0 * static_cast<double>(n) * n * n;
    return flops / (medianMs * 1e-3) / 1e12;
}

} // namespace

int main() {
    try {
        using namespace vulkan_runtime;

        core::VulkanContext ctx = core::create_context("vulkan-runtime-sgemm-bench");

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
        std::cout << "sweep: M=N=K in {256,512,1024,1536,2048} | warmup=" << kWarmupRuns
                  << " timed=" << kTimedRuns << " (median)\n\n";

        // --- Shared pipeline layout (all variants use 3 SSBOs + push consts) --
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

        vk::PipelineLayoutCreateInfo layoutInfo;
        layoutInfo.setSetLayouts(setLayout).setPushConstantRanges(pcRange);
        vk::PipelineLayout pipelineLayout = ctx.device.createPipelineLayout(layoutInfo);

        // --- Per-variant pipelines -------------------------------------------
        struct PipelinedVariant {
            Variant meta;
            vk::ShaderModule module{};
            vk::Pipeline pipeline{};
        };
        std::vector<PipelinedVariant> variants;
        for (Variant const& v : kVariants) {
            shaders::ShaderBlob blob = shaders::get(v.shader);
            if (blob.data == nullptr || blob.size == 0) {
                throw std::runtime_error(std::string(v.shader) + " not embedded.");
            }
            vk::ShaderModuleCreateInfo modInfo;
            modInfo.setCodeSize(blob.size)
                .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
            vk::ShaderModule module = ctx.device.createShaderModule(modInfo);

            vk::PipelineShaderStageCreateInfo stageInfo;
            stageInfo.setStage(vk::ShaderStageFlagBits::eCompute)
                .setModule(module)
                .setPName("main");

            vk::ComputePipelineCreateInfo pipeInfo;
            pipeInfo.setStage(stageInfo).setLayout(pipelineLayout);
            auto pr = ctx.device.createComputePipeline({}, pipeInfo);
            if (pr.result != vk::Result::eSuccess) {
                throw std::runtime_error("createComputePipeline failed for " +
                                         std::string(v.name));
            }
            variants.push_back({v, module, pr.value});
        }

        // --- Results matrix: [variant][size] ---------------------------------
        std::vector<std::vector<bench::TimedRun>> results(
            variants.size(), std::vector<bench::TimedRun>(kSizes.size()));

        for (std::size_t si = 0; si < kSizes.size(); ++si) {
            std::uint32_t S = kSizes[si];
            vk::DeviceSize aBytes = static_cast<vk::DeviceSize>(S) * S * sizeof(float);
            vk::DeviceSize bBytes = aBytes;
            vk::DeviceSize cBytes = aBytes;
            vk::DeviceSize stagingBytes = aBytes + bBytes;

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
            Buffer staging = create_buffer(
                ctx.allocator, stagingBytes,
                vk::BufferUsageFlagBits::eTransferSrc,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

            // Fill staging with the operands once per size (shared by variants).
            {
                std::vector<float> a(static_cast<std::size_t>(S) * S);
                std::vector<float> b(static_cast<std::size_t>(S) * S);
                fill(a, 0x12345678u);
                fill(b, 0x9abcdef0u);

                void* mapped = nullptr;
                if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                    throw std::runtime_error("vmaMapMemory failed.");
                }
                std::memcpy(mapped, a.data(), aBytes);
                std::memcpy(static_cast<char*>(mapped) + aBytes, b.data(), bBytes);
                vmaUnmapMemory(ctx.allocator, staging.allocation);
                vmaFlushAllocation(ctx.allocator, staging.allocation, 0, stagingBytes);
            }

            upload_operands(ctx, staging, dev_a, dev_b, aBytes, bBytes);

            // Descriptor set binding this size's A/B/C (works with every variant's
            // pipeline — they all share the same set layout).
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
            writes[0].setDstSet(descriptorSet).setDstBinding(0).setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(aInfo);
            writes[1].setDstSet(descriptorSet).setDstBinding(1).setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(bInfo);
            writes[2].setDstSet(descriptorSet).setDstBinding(2).setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(cInfo);
            ctx.device.updateDescriptorSets(writes, {});

            for (std::size_t vi = 0; vi < variants.size(); ++vi) {
                PipelinedVariant const& pv = variants[vi];

                auto recordSetup = [&](vk::CommandBuffer const& cb) {
                    cb.bindPipeline(vk::PipelineBindPoint::eCompute, pv.pipeline);
                    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0,
                                          descriptorSet, {});
                    PushConstants pc{S, S, S};
                    cb.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                     sizeof(PushConstants), &pc);
                };
                auto recordDispatch = [&](vk::CommandBuffer const& cb) {
                    cb.dispatch(S / kTileN, S / kTileM, 1);
                };

                results[vi][si] =
                    bench::time_dispatches(ctx, recordSetup, recordDispatch,
                                           kWarmupRuns, kTimedRuns);
            }

            ctx.device.destroyDescriptorPool(descriptorPool);
            destroy_buffer(ctx.allocator, staging);
            destroy_buffer(ctx.allocator, dev_c);
            destroy_buffer(ctx.allocator, dev_b);
            destroy_buffer(ctx.allocator, dev_a);
        }

        // --- Report: TFLOPS table (rows = size, cols = variant) ---------------
        std::cout << "TFLOPS (median of " << kTimedRuns << " timed runs)\n";
        std::cout << "  size |";
        for (Variant const& v : kVariants) {
            std::cout << std::setw(14) << v.name;
        }
        std::cout << " | best\n";
        std::cout << std::string(6, '-') << '+';
        for (std::size_t vi = 0; vi < kVariants.size(); ++vi) {
            std::cout << std::string(14, '-');
        }
        std::cout << "+" << std::string(16, '-') << "\n";

        std::vector<std::size_t> bestPerSize(kSizes.size());
        for (std::size_t si = 0; si < kSizes.size(); ++si) {
            std::size_t best = 0;
            for (std::size_t vi = 1; vi < kVariants.size(); ++vi) {
                if (results[vi][si].medianMs < results[best][si].medianMs) {
                    best = vi;
                }
            }
            bestPerSize[si] = best;

            std::cout << std::setw(6) << kSizes[si] << " |";
            std::cout << std::fixed << std::setprecision(2);
            for (std::size_t vi = 0; vi < kVariants.size(); ++vi) {
                std::cout << std::setw(14) << tflops(kSizes[si], results[vi][si].medianMs);
            }
            std::cout << " | " << std::setw(14) << kVariants[best].name << "\n";
        }
        std::cout << "\n";

        // --- Per-size ranking (best first) ------------------------------------
        for (std::size_t si = 0; si < kSizes.size(); ++si) {
            std::vector<std::size_t> order(kVariants.size());
            for (std::size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return results[a][si].medianMs < results[b][si].medianMs;
            });
            std::cout << "ranking @ " << kSizes[si] << "\xC2\xB3: ";
            std::cout << std::fixed << std::setprecision(2);
            for (std::size_t r = 0; r < order.size(); ++r) {
                std::size_t vi = order[r];
                if (r > 0) {
                    std::cout << ", ";
                }
                std::cout << kVariants[vi].name << " "
                          << tflops(kSizes[si], results[vi][si].medianMs);
            }
            std::cout << " TFLOPS\n";
        }
        std::cout << "\n";

        // --- Anchor: acceptance + reproducibility + where-time-goes -----------
        bench::TimedRun const& v0 = results[0][kAnchorIdx];
        double v0Tflops = tflops(kSizes[kAnchorIdx], v0.medianMs);

        std::cout << std::setprecision(3);
        std::cout << "sgemm 1024\xC2\xB3 (default " << kVariants[0].name << "): median "
                  << v0.medianMs << " ms -> " << std::setprecision(2) << v0Tflops
                  << " TFLOPS (gate " << kGateTflops << ", stretch " << kStretchTflops
                  << ")\n";
        std::cout << std::setprecision(3);
        std::cout << "reproducibility @ 1024\xC2\xB3 (" << kVariants[0].name << "): "
                  << v0.minMs << " .. " << v0.medianMs << " .. " << v0.maxMs
                  << " ms (spread " << std::setprecision(2)
                  << (v0.maxMs - v0.minMs) / v0.medianMs * 100.0 << "%)\n";

        double n = static_cast<double>(kSizes[kAnchorIdx]);
        double aluFloorMs = 2.0 * n * n * n / kPeakTflops / 1e12 * 1e3;
        std::cout << std::setprecision(3);
        std::cout << "where time goes @ 1024\xC2\xB3: ALU floor " << aluFloorMs
                  << " ms (2*N\xC2\xB3 / " << kPeakTflops
                  << " TFLOPS) vs measured " << v0.medianMs << " ms ("
                  << std::setprecision(1) << (v0Tflops / kPeakTflops * 100.0)
                  << "% of peak)\n\n";

        // --- Verdict: best 1024^3 variant vs M2 baseline ----------------------
        std::size_t bestV = bestPerSize[kAnchorIdx];
        double bestTflops = tflops(kSizes[kAnchorIdx], results[bestV][kAnchorIdx].medianMs);
        std::cout << std::setprecision(2);
        std::cout << "best 1024\xC2\xB3 variant: " << kVariants[bestV].name << " @ "
                  << bestTflops << " TFLOPS";
        if (bestV == 0) {
            std::cout << " (no variant beat the M2 baseline " << kM2BaselineTflops << ")";
        } else {
            double delta = (bestTflops - kM2BaselineTflops) / kM2BaselineTflops * 100.0;
            std::cout << " (+" << delta << "% vs M2 baseline " << kM2BaselineTflops << ")";
        }
        std::cout << "\n";

        int exitCode = EXIT_SUCCESS;
        if (bestTflops < kM2BaselineTflops) {
            std::cout << "*** REGRESSION vs M2 baseline: best 1024\xC2\xB3 variant ("
                      << bestTflops << " TFLOPS) < " << kM2BaselineTflops
                      << " TFLOPS — ranking found no improvement; investigate -O / "
                         "barriers / alignment ***\n";
        }
        if (bestTflops < kGateTflops) {
            std::cout << "*** FAIL: best 1024\xC2\xB3 variant (" << bestTflops
                      << " TFLOPS) is below the M2 gate (" << kGateTflops
                      << " TFLOPS) — build is broken (check glslc -O) ***\n";
            exitCode = EXIT_FAILURE;
        }

        // --- Tear down (reverse creation order) -------------------------------
        for (PipelinedVariant& pv : variants) {
            ctx.device.destroyPipeline(pv.pipeline);
            ctx.device.destroyShaderModule(pv.module);
        }
        ctx.device.destroyPipelineLayout(pipelineLayout);
        ctx.device.destroyDescriptorSetLayout(setLayout);
        core::destroy_context(ctx);

        return exitCode;
    } catch (vk::SystemError const& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
