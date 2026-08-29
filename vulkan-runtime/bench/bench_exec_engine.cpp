// M5b: cross-queue ACE overlap benchmark.
//
// Demonstrates that the copy (host -> device staging) and the fp32 sgemm
// dispatch run on *different* hardware queues and overlap. On gfx803/Polaris,
// RADV exposes the 4 ACE engines as a separate pure-compute queue family
// (compute + transfer, no graphics): queue 1 records vkCmdCopyBuffer for
// batch i while queue 0 runs the 1024^3 sgemm for batch i-1.
//
// Pipeline (32 batches, double-buffered host staging):
//   host:    memcpy batch i's A,B into staging pair (i % 2)   [host work]
//   queue 1: vkCmdCopyBuffer staging -> device A_i, B_i        [ACE transfer]
//   queue 0: sgemm dispatch C_i = A_i * B_i, waiting device-side (timeline
//            semaphore) on queue 1's copy-signaled value       [compute]
//
// Correctness: after drain, all 32 C_i are read back and compared against a
// double-precision CPU reference. A racy cross-queue pipeline (a missing or
// wrong wait) would let the dispatch read a partially-copied A_i/B_i and
// corrupt the outputs — this is the M5b proof that the cross-queue timeline
// sync is actually correct, not merely fast.
//
// Measurement (steady_clock wall time, after a 128-dispatch warmup so the RX580
// clock is at boost):
//   serial    = per batch: copy, drain, dispatch, drain   (no overlap)
//   pipelined = per batch: copy on q1, dispatch on q0 (waits on the copy),
//               all 32 submitted, then drain once
//   overlap   = serial / pipelined
//
// Guards: exits 0 on lavapipe/llvmpipe ("not a discrete GPU"), and exits 0
// with "fewer than 2 compute queues" when the device exposes <2 compute queues
// (no cross-queue overlap possible), so ctest stays fast on software.

#include "bench/bench_common.h"

#include "core/vk_setup.h"
#include "exec/engine.h"

#include "embedded_shaders.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint32_t kM = 1024;
constexpr std::uint32_t kN = 1024;
constexpr std::uint32_t kK = 1024;
constexpr std::uint32_t kTileM = 64;
constexpr std::uint32_t kTileN = 64;
constexpr std::uint32_t kBatches = 32;
constexpr std::uint32_t kStagingPairs = 2; // double-buffered host staging
constexpr std::uint32_t kWarmup = 128;     // RX580 clock ramp -> boost
constexpr std::uint32_t kRingSlots = 8;    // in-flight pipeline depth
constexpr std::uint32_t kSeedA = 0x12345678u;
constexpr std::uint32_t kSeedB = 0x9abcdef0u;
constexpr double kRelTol = 1e-3; // as in test_sgemm

// Push-constant block matching shaders/sgemm.comp.
struct PushConstants {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

// A VMA-managed buffer: a Vulkan buffer + its allocation.
struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
    vk::DeviceSize size{0};
};

Buffer create_buffer(VmaAllocator allocator,
                     vk::DeviceSize size,
                     vk::BufferUsageFlags usage,
                     VmaMemoryUsage memUsage,
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
    if (vmaCreateBuffer(allocator, &createInfo, &allocCreateInfo, &vkBuffer, &allocation, nullptr) != VK_SUCCESS) {
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

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Deterministic LCG (Numerical Recipes) -> floats in [0,1), scaled to [-1,1].
// Written directly into `dst` (used for both staging fill and verification).
void fill_floats(float* dst, std::size_t n, std::uint32_t seed) {
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        dst[i] = u * 2.0f - 1.0f;
    }
}

std::vector<float> generate_inputs(std::size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    fill_floats(v.data(), n, seed);
    return v;
}

// CPU reference in double precision (row-major A[M x K], B[K x N] -> C[M x N]).
std::vector<double> cpu_reference(std::vector<float> const& a,
                                  std::vector<float> const& b,
                                  std::uint32_t M,
                                  std::uint32_t N,
                                  std::uint32_t K) {
    std::vector<double> c(M * N, 0.0);
    for (std::uint32_t m = 0; m < M; ++m) {
        for (std::uint32_t k = 0; k < K; ++k) {
            double av = static_cast<double>(a[m * K + k]);
            for (std::uint32_t n = 0; n < N; ++n) {
                c[m * N + n] += av * static_cast<double>(b[k * N + n]);
            }
        }
    }
    return c;
}

// Per-batch descriptor sets: set i binds A@i*aBytes, B@i*bBytes, C@i*cBytes.
// Static (never re-bound) sets let every dispatch target a distinct region with
// no runtime updates racing in-flight work.
std::vector<vk::DescriptorSet> write_sets(vk::Device device,
                                          vk::DescriptorSetLayout layout,
                                          vk::DescriptorPool pool,
                                          std::uint32_t count,
                                          vk::Buffer a,
                                          vk::Buffer b,
                                          vk::Buffer c,
                                          vk::DeviceSize aBytes,
                                          vk::DeviceSize bBytes,
                                          vk::DeviceSize cBytes) {
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.setDescriptorPool(pool).setSetLayouts(layouts);
    std::vector<vk::DescriptorSet> sets = device.allocateDescriptorSets(allocInfo);

    for (std::uint32_t i = 0; i < count; ++i) {
        vk::DescriptorBufferInfo aInfo;
        aInfo.setBuffer(a).setOffset(i * aBytes).setRange(aBytes);
        vk::DescriptorBufferInfo bInfo;
        bInfo.setBuffer(b).setOffset(i * bBytes).setRange(bBytes);
        vk::DescriptorBufferInfo cInfo;
        cInfo.setBuffer(c).setOffset(i * cBytes).setRange(cBytes);

        std::array<vk::WriteDescriptorSet, 3> writes{};
        writes[0]
            .setDstSet(sets[i])
            .setDstBinding(0)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(aInfo);
        writes[1]
            .setDstSet(sets[i])
            .setDstBinding(1)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(bInfo);
        writes[2]
            .setDstSet(sets[i])
            .setDstBinding(2)
            .setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setBufferInfo(cInfo);
        device.updateDescriptorSets(writes, {});
    }
    return sets;
}

} // namespace

int main() {
    try {
        using namespace vulkan_runtime;

        core::VulkanContext ctx = core::create_context("vulkan-runtime-exec-engine-bench");

        // Perf is only meaningful on a discrete GPU; skip fast on lavapipe so
        // `ctest -L bench` stays cheap in CI.
        if (!bench::is_discrete_gpu(ctx)) {
            std::cout << "bench skipped: not a discrete GPU\n";
            core::destroy_context(ctx);
            return EXIT_SUCCESS;
        }
        // Cross-queue overlap needs >= 2 compute queues (the ACE family on
        // Polaris; some discrete GPUs may expose only one).
        if (ctx.computeQueueCount < 2) {
            std::cout << "bench_exec_engine: fewer than 2 compute queues (" << ctx.computeQueueCount
                      << "), overlap demo skipped\n";
            core::destroy_context(ctx);
            return EXIT_SUCCESS;
        }

        vk::PhysicalDeviceProperties props = ctx.physicalDevice.getProperties();
        std::cout << "device: " << props.deviceName << " (computeQueueCount=" << ctx.computeQueueCount
                  << ", family=" << ctx.computeQueueFamily << ")\n";

        vk::DeviceSize const aBytes = static_cast<vk::DeviceSize>(kM) * kK * sizeof(float);
        vk::DeviceSize const bBytes = static_cast<vk::DeviceSize>(kK) * kN * sizeof(float);
        vk::DeviceSize const cBytes = static_cast<vk::DeviceSize>(kM) * kN * sizeof(float);
        vk::DeviceSize const pairBytes = aBytes + bBytes;
        std::size_t const aElems = static_cast<std::size_t>(kM) * kK;
        std::size_t const bElems = static_cast<std::size_t>(kK) * kN;

        // --- CPU inputs (pre-generated once; the timed loops only memcpy) ----
        std::vector<float> all_inputs(kBatches * (aElems + bElems));
        for (std::uint32_t i = 0; i < kBatches; ++i) {
            float* base = all_inputs.data() + static_cast<std::size_t>(i) * (aElems + bElems);
            fill_floats(base, aElems, kSeedA + i);
            fill_floats(base + aElems, bElems, kSeedB + i);
        }

        // --- Device buffers (one big buffer per operand, 32 regions each) ----
        Buffer dev_a = create_buffer(ctx.allocator,
                                     kBatches * aBytes,
                                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_b = create_buffer(ctx.allocator,
                                     kBatches * bBytes,
                                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer dev_c = create_buffer(ctx.allocator,
                                     kBatches * cBytes,
                                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        // Double-buffered host staging (copy source) + a readback buffer.
        Buffer stage = create_buffer(ctx.allocator,
                                     kStagingPairs * pairBytes,
                                     vk::BufferUsageFlagBits::eTransferSrc,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                     VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        Buffer readback = create_buffer(ctx.allocator,
                                        kBatches * cBytes,
                                        vk::BufferUsageFlagBits::eTransferDst,
                                        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        // --- Pipeline (sgemm.spv: 3 SSBOs + push constants) ------------------
        shaders::ShaderBlob blob = shaders::get("sgemm.spv");
        if (blob.data == nullptr || blob.size == 0) {
            throw std::runtime_error("sgemm.spv not embedded.");
        }
        vk::ShaderModuleCreateInfo moduleInfo;
        moduleInfo.setCodeSize(blob.size).setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
        vk::ShaderModule shaderModule = ctx.device.createShaderModule(moduleInfo);

        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            bindings[i]
                .setBinding(i)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        }
        vk::DescriptorSetLayoutCreateInfo setLayoutInfo;
        setLayoutInfo.setBindings(bindings);
        vk::DescriptorSetLayout setLayout = ctx.device.createDescriptorSetLayout(setLayoutInfo);

        vk::PushConstantRange pcRange;
        pcRange.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(sizeof(PushConstants));
        vk::PipelineLayoutCreateInfo layoutInfo;
        layoutInfo.setSetLayouts(setLayout).setPushConstantRanges(pcRange);
        vk::PipelineLayout pipelineLayout = ctx.device.createPipelineLayout(layoutInfo);

        vk::PipelineShaderStageCreateInfo stageInfo;
        stageInfo.setStage(vk::ShaderStageFlagBits::eCompute).setModule(shaderModule).setPName("main");
        vk::ComputePipelineCreateInfo pipelineInfo;
        pipelineInfo.setStage(stageInfo).setLayout(pipelineLayout);
        auto pipelineResult = ctx.device.createComputePipeline({}, pipelineInfo);
        if (pipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("createComputePipeline failed.");
        }
        vk::Pipeline pipeline = pipelineResult.value;

        vk::DescriptorPoolSize poolSize;
        poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(3 * kBatches);
        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setMaxSets(kBatches).setPoolSizes(poolSize);
        vk::DescriptorPool descriptorPool = ctx.device.createDescriptorPool(poolInfo);

        std::vector<vk::DescriptorSet> sets = write_sets(ctx.device,
                                                         setLayout,
                                                         descriptorPool,
                                                         kBatches,
                                                         dev_a.buffer,
                                                         dev_b.buffer,
                                                         dev_c.buffer,
                                                         aBytes,
                                                         bBytes,
                                                         cBytes);

        // --- Engine: 8-slot ring across the compute queues -------------------
        // Scoped so the engine (and its command pool + per-queue timeline
        // semaphores) is destroyed *before* destroy_context(ctx) below: the
        // device must outlive the engine, otherwise teardown trips VVL
        // (VUID-vkDestroyDevice-05137 / "Invalid device" on leaked handles).
        // serialMs/pipelinedMs are hoisted out for the report below.
        double serialMs = 0.0;
        double pipelinedMs = 0.0;
        {
            exec::ExecEngine engine(ctx, kRingSlots, ctx.computeQueueCount);

            auto recordCopy = [&](std::uint32_t i, vk::CommandBuffer cmd) {
                std::uint32_t const s = i % kStagingPairs;
                vk::BufferCopy cA;
                cA.setSrcOffset(s * pairBytes).setDstOffset(i * aBytes).setSize(aBytes);
                cmd.copyBuffer(stage.buffer, dev_a.buffer, cA);
                vk::BufferCopy cB;
                cB.setSrcOffset(s * pairBytes + aBytes).setDstOffset(i * bBytes).setSize(bBytes);
                cmd.copyBuffer(stage.buffer, dev_b.buffer, cB);
            };
            auto recordGemm = [&](std::uint32_t i, vk::CommandBuffer cmd) {
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, sets[i], {});
                PushConstants pc{kM, kN, kK};
                cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants), &pc);
                cmd.dispatch(kN / kTileN, kM / kTileM, 1);
            };

            // Host-writes batch `i` into staging pair (i % kStagingPairs).
            auto stage_batch = [&](std::uint32_t i) {
                std::uint32_t const s = i % kStagingPairs;
                std::size_t const srcOff = static_cast<std::size_t>(i) * (aElems + bElems);
                void* mapped = nullptr;
                if (vmaMapMemory(ctx.allocator, stage.allocation, &mapped) != VK_SUCCESS) {
                    throw std::runtime_error("vmaMapMemory failed.");
                }
                std::memcpy(static_cast<char*>(mapped) + s * pairBytes, all_inputs.data() + srcOff, pairBytes);
                vmaUnmapMemory(ctx.allocator, stage.allocation);
                vmaFlushAllocation(ctx.allocator, stage.allocation, s * pairBytes, pairBytes);
            };

            // --- Warm up: stage+copy batch 0, then 128 gemm dispatches -----------
            // (the RX580 idles at 300-600 MHz and only boosts after sustained load).
            stage_batch(0);
            engine.submit([&](vk::CommandBuffer cmd) { recordCopy(0, cmd); }, 1);
            engine.drain();
            for (std::uint32_t w = 0; w < kWarmup; ++w) {
                engine.submit([&](vk::CommandBuffer cmd) { recordGemm(0, cmd); }, 0);
            }
            engine.drain();

            // --- Serial baseline: copy -> drain -> dispatch -> drain, per batch --
            Clock::time_point t0 = Clock::now();
            for (std::uint32_t i = 0; i < kBatches; ++i) {
                stage_batch(i);
                engine.submit([&, i](vk::CommandBuffer cmd) { recordCopy(i, cmd); }, 1);
                engine.drain();
                engine.submit([&, i](vk::CommandBuffer cmd) { recordGemm(i, cmd); }, 0);
                engine.drain();
            }
            serialMs = elapsed_ms(t0, Clock::now());

            // --- Pipelined: copy on q1, dispatch on q0 (waits on the copy) -------
            std::vector<std::uint64_t> copyValues(kBatches);
            std::uint64_t lastGemmValue = 0;
            t0 = Clock::now();
            for (std::uint32_t i = 0; i < kBatches; ++i) {
                // Staging pair reuse: the pair (i % 2) was last read by copy (i-2);
                // host-wait on that copy's timeline value before overwriting.
                if (i >= kStagingPairs) {
                    engine.wait(copyValues[i - kStagingPairs]);
                }
                stage_batch(i);
                copyValues[i] = engine.submit([&, i](vk::CommandBuffer cmd) { recordCopy(i, cmd); }, 1);
                lastGemmValue =
                    engine.submit([&, i](vk::CommandBuffer cmd) { recordGemm(i, cmd); }, 0, {{copyValues[i], 1u}});
            }
            engine.drain();
            pipelinedMs = elapsed_ms(t0, Clock::now());

            // --- Read back all 32 outputs (wait on the last gemm's value so the
            // shader writes are visible to this transfer read) -------------------
            engine.submit(
                [&](vk::CommandBuffer cmd) {
                    vk::BufferCopy c;
                    c.setSrcOffset(0).setDstOffset(0).setSize(kBatches * cBytes);
                    cmd.copyBuffer(dev_c.buffer, readback.buffer, c);
                },
                0,
                {{lastGemmValue, 0u}});
            engine.drain();
        } // end engine scope — engine (and its timeline semaphores) destroyed here

        // --- Verify all 32 outputs against the double CPU reference ----------
        bool ok = true;
        std::uint32_t badBatch = 0;
        std::size_t badIndex = 0;
        double maxRel = 0.0;
        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, readback.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            vmaInvalidateAllocation(ctx.allocator, readback.allocation, 0, kBatches * cBytes);

            float const* gpuC = static_cast<float const*>(mapped);
            std::size_t const cElems = static_cast<std::size_t>(kM) * kN;
            for (std::uint32_t i = 0; i < kBatches && ok; ++i) {
                std::vector<float> in_a = generate_inputs(aElems, kSeedA + i);
                std::vector<float> in_b = generate_inputs(bElems, kSeedB + i);
                std::vector<double> ref = cpu_reference(in_a, in_b, kM, kN, kK);
                float const* out = gpuC + static_cast<std::size_t>(i) * cElems;
                for (std::size_t j = 0; j < cElems; ++j) {
                    double err = std::fabs(static_cast<double>(out[j]) - ref[j]);
                    double rel = err / std::max(1.0, std::fabs(ref[j]));
                    maxRel = std::max(maxRel, rel);
                    if (rel > kRelTol) {
                        ok = false;
                        badBatch = i;
                        badIndex = j;
                        break;
                    }
                }
            }
            vmaUnmapMemory(ctx.allocator, readback.allocation);
        }

        // --- Report ----------------------------------------------------------
        double const overlap = (pipelinedMs > 0.0) ? serialMs / pipelinedMs : 0.0;
        double const perBatchSerial = serialMs / kBatches;
        double const perBatchPipelined = pipelinedMs / kBatches;
        double const copyPerBatchMs = static_cast<double>(pairBytes) / 12.0e9 * 1e3; // 8 MB / 12 GB/s
        double const gemmPerBatchMs = 2.0 * kM * kN * kK / 6.0e12 * 1e3;             // 2*N^3 / 6 TFLOPS

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "bench_exec_engine: " << kBatches << "x" << kM << "\xC2\xB3"
                  << " serial=" << serialMs << "ms pipelined=" << pipelinedMs << "ms overlap=" << std::setprecision(2)
                  << overlap << "x correctness " << (ok ? "PASS" : "FAIL") << "\n";
        std::cout << std::setprecision(3);
        std::cout << "  per-batch: serial=" << perBatchSerial << "ms pipelined=" << perBatchPipelined << "ms\n";
        std::cout << "  where time goes / batch: copy " << copyPerBatchMs << "ms (8 MB @ ~12 GB/s PCIe) + gemm "
                  << gemmPerBatchMs << "ms (ALU floor @ 6 TFLOPS) + host staging memcpy "
                  << "(the copy and host staging hide behind the ALU-bound gemm "
                     "on the ACE queue; serial sums them, pipelined overlaps them)\n";

        if (!ok) {
            std::cerr << "bench_exec_engine: MISMATCH batch " << badBatch << " index " << badIndex
                      << " (max_rel_err=" << maxRel << ")\n";
        }

        // --- Tear down (reverse creation order) -------------------------------
        ctx.device.destroyDescriptorPool(descriptorPool);
        ctx.device.destroyPipeline(pipeline);
        ctx.device.destroyPipelineLayout(pipelineLayout);
        ctx.device.destroyDescriptorSetLayout(setLayout);
        ctx.device.destroyShaderModule(shaderModule);
        destroy_buffer(ctx.allocator, readback);
        destroy_buffer(ctx.allocator, stage);
        destroy_buffer(ctx.allocator, dev_c);
        destroy_buffer(ctx.allocator, dev_b);
        destroy_buffer(ctx.allocator, dev_a);
        core::destroy_context(ctx);

        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (vk::SystemError const& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
