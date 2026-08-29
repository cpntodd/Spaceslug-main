// M5a exec-engine test: N-slot ring + timeline semaphores.
//
// Two phases, both headless and self-contained (mirroring the M1/M2 test
// patterns — each test duplicates its own buffer/descriptor/pipeline helpers):
//
//   1. Ring-reuse correctness: 256 vector-add dispatches (N=1<<20) through a
//      3-slot ExecEngine. 256 > 3 forces slot reuse on every ring command
//      buffer. Every dispatch reads *distinct* deterministic LCG inputs
//      (staged through a small rotating host pool) and writes a *distinct*
//      output region; after drain() every one of the 256 outputs is compared
//      bit-exactly against a CPU reference. A broken timeline throttle would
//      re-record a command buffer still in flight and corrupt these outputs
//      (or trip VVL) — this catches it without relying on validation layers.
//
//   2. Pipelining metric: 24 x sgemm 1024^3 dispatches through the 3-slot ring
//      vs the same 24 submitted serially (submit -> drain -> next). Wall-clock
//      (steady_clock) comparison; overlap = serial / pipelined. The assert
//      "pipelined < serial" runs only on discrete GPUs — lavapipe is a
//      software rasterizer with noisy, scheduler-dependent timing, so there we
//      print the numbers and skip the assert (documented below).

#include "core/vk_setup.h"
#include "exec/engine.h"

#include "embedded_shaders.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// --- shared helpers ----------------------------------------------------------

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

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

// lavapipe reports deviceType == CPU and deviceName "llvmpipe ..."; used to
// skip the overlap assert on software where timing is not meaningful.
bool is_software_device(vk::PhysicalDevice device) {
    vk::PhysicalDeviceProperties props = device.getProperties();
    if (props.deviceType == vk::PhysicalDeviceType::eCpu) {
        return true;
    }
    std::string name(props.deviceName.data());
    return name.find("llvmpipe") != std::string::npos ||
           name.find("lavapipe") != std::string::npos;
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

// A compute pipeline plus its descriptor-set layout / pool. Both kernels here
// (vector-add, sgemm) use exactly three SSBO bindings {a, b, out}; sgemm adds
// a small push-constant block.
struct PushConstants {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

struct ComputePipeline {
    vk::ShaderModule module{};
    vk::DescriptorSetLayout setLayout{};
    vk::PipelineLayout layout{};
    vk::Pipeline pipeline{};
    vk::DescriptorPool pool{};
};

ComputePipeline build_compute_pipeline(vk::Device device, char const* shaderName,
                                       std::uint32_t maxSets,
                                       std::uint32_t pushConstantSize) {
    ComputePipeline out;

    vulkan_runtime::shaders::ShaderBlob blob = vulkan_runtime::shaders::get(shaderName);
    if (blob.data == nullptr || blob.size == 0) {
        throw std::runtime_error(std::string(shaderName) + " not embedded.");
    }
    vk::ShaderModuleCreateInfo moduleInfo;
    moduleInfo.setCodeSize(blob.size)
        .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
    out.module = device.createShaderModule(moduleInfo);

    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        bindings[i].setBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    }

    vk::DescriptorSetLayoutCreateInfo setLayoutInfo;
    setLayoutInfo.setBindings(bindings);
    out.setLayout = device.createDescriptorSetLayout(setLayoutInfo);

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.setSetLayouts(out.setLayout);
    vk::PushConstantRange pcRange{};
    if (pushConstantSize > 0) {
        pcRange.setStageFlags(vk::ShaderStageFlagBits::eCompute)
            .setOffset(0)
            .setSize(pushConstantSize);
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pcRange;
    }
    out.layout = device.createPipelineLayout(layoutInfo);

    vk::PipelineShaderStageCreateInfo stageInfo;
    stageInfo.setStage(vk::ShaderStageFlagBits::eCompute)
        .setModule(out.module)
        .setPName("main");

    vk::ComputePipelineCreateInfo pipelineInfo;
    pipelineInfo.setStage(stageInfo).setLayout(out.layout);
    auto result = device.createComputePipeline({}, pipelineInfo);
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("createComputePipeline failed.");
    }
    out.pipeline = result.value;

    vk::DescriptorPoolSize poolSize;
    poolSize.setType(vk::DescriptorType::eStorageBuffer)
        .setDescriptorCount(3 * maxSets);
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(maxSets).setPoolSizes(poolSize);
    out.pool = device.createDescriptorPool(poolInfo);

    return out;
}

void destroy_pipeline(vk::Device device, ComputePipeline& p) {
    device.destroyDescriptorPool(p.pool);
    device.destroyPipeline(p.pipeline);
    device.destroyPipelineLayout(p.layout);
    device.destroyDescriptorSetLayout(p.setLayout);
    device.destroyShaderModule(p.module);
}

// Allocates `count` descriptor sets (all sharing one layout) binding:
//   0 -> a (whole), 1 -> b (whole), 2 -> out @ i*outRegionBytes (region).
// This lets each dispatch target a distinct output region with no runtime
// descriptor updates (static sets never race in-flight work).
std::vector<vk::DescriptorSet> write_sets(vk::Device device, ComputePipeline const& p,
                                          std::uint32_t count, vk::Buffer a,
                                          vk::Buffer b, vk::Buffer out,
                                          vk::DeviceSize outRegionBytes) {
    std::vector<vk::DescriptorSetLayout> layouts(count, p.setLayout);
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.setDescriptorPool(p.pool).setSetLayouts(layouts);
    std::vector<vk::DescriptorSet> sets = device.allocateDescriptorSets(allocInfo);

    for (std::uint32_t i = 0; i < count; ++i) {
        vk::DescriptorBufferInfo aInfo;
        aInfo.setBuffer(a).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo bInfo;
        bInfo.setBuffer(b).setOffset(0).setRange(VK_WHOLE_SIZE);
        vk::DescriptorBufferInfo oInfo;
        oInfo.setBuffer(out).setOffset(i * outRegionBytes).setRange(outRegionBytes);

        std::array<vk::WriteDescriptorSet, 3> writes{};
        writes[0].setDstSet(sets[i]).setDstBinding(0).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(aInfo);
        writes[1].setDstSet(sets[i]).setDstBinding(1).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(bInfo);
        writes[2].setDstSet(sets[i]).setDstBinding(2).setDescriptorCount(1)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(oInfo);
        device.updateDescriptorSets(writes, {});
    }
    return sets;
}

// --- phase 1: ring-reuse correctness ----------------------------------------

bool run_correctness(vulkan_runtime::core::VulkanContext const& ctx) {
    constexpr std::size_t kN = 1u << 20;   // 1,048,576 floats (4 MiB/buffer)
    constexpr std::uint32_t kWg = 256;      // 4 wave64 on gfx803
    constexpr std::uint32_t kChain = 256;   // > ring slots => forced slot reuse
    constexpr std::uint32_t kStagePool = 8; // rotating host staging slots
    constexpr std::uint32_t kSeedA = 0x12345678u;
    constexpr std::uint32_t kSeedB = 0x9abcdef0u;

    vk::DeviceSize const nBytes = kN * sizeof(float);
    vk::DeviceSize const stageSlotBytes = 2 * nBytes; // [a_i | b_i]

    // Device buffers: shared a/b (rewritten each dispatch) + one big distinct
    // output buffer (256 regions, one per dispatch).
    Buffer dev_a = create_buffer(
        ctx.allocator, nBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    Buffer dev_b = create_buffer(
        ctx.allocator, nBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    Buffer out_all = create_buffer(
        ctx.allocator, kChain * nBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Host staging: rotating input slots + a readback buffer for all outputs.
    Buffer stage_in = create_buffer(
        ctx.allocator, kStagePool * stageSlotBytes,
        vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    Buffer stage_out = create_buffer(
        ctx.allocator, kChain * nBytes,
        vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    ComputePipeline p = build_compute_pipeline(ctx.device, "vector_add.spv", kChain, 0);
    std::vector<vk::DescriptorSet> sets =
        write_sets(ctx.device, p, kChain, dev_a.buffer, dev_b.buffer, out_all.buffer, nBytes);

    vulkan_runtime::exec::ExecEngine engine(ctx, 3, 1);

    // 256 distinct dispatches. Each records into one of the engine's 3 ring
    // command buffers; every slot is reused ~85 times. The host stages input i
    // into staging slot (i % kStagePool): because kStagePool > ringSlots, the
    // previous user of that staging slot (dispatch i-kStagePool) has already
    // retired by the time submit(i-1) throttled — so the host write can never
    // race an in-flight copy-in.
    for (std::uint32_t i = 0; i < kChain; ++i) {
        std::uint32_t const s = i % kStagePool;

        std::vector<float> a = generate_inputs(kN, kSeedA + i);
        std::vector<float> b = generate_inputs(kN, kSeedB + i);
        {
            void* mapped = nullptr;
            if (vmaMapMemory(ctx.allocator, stage_in.allocation, &mapped) != VK_SUCCESS) {
                throw std::runtime_error("vmaMapMemory failed.");
            }
            std::memcpy(static_cast<char*>(mapped) + s * stageSlotBytes, a.data(), nBytes);
            std::memcpy(static_cast<char*>(mapped) + s * stageSlotBytes + nBytes, b.data(), nBytes);
            vmaUnmapMemory(ctx.allocator, stage_in.allocation);
            vmaFlushAllocation(ctx.allocator, stage_in.allocation, s * stageSlotBytes, stageSlotBytes);
        }

        engine.submit([&, s](vk::CommandBuffer cmd) {
            vk::BufferCopy cA;
            cA.setSrcOffset(s * stageSlotBytes).setDstOffset(0).setSize(nBytes);
            cmd.copyBuffer(stage_in.buffer, dev_a.buffer, cA);

            vk::BufferCopy cB;
            cB.setSrcOffset(s * stageSlotBytes + nBytes).setDstOffset(0).setSize(nBytes);
            cmd.copyBuffer(stage_in.buffer, dev_b.buffer, cB);

            vk::MemoryBarrier toShader;
            toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader, {},
                                toShader, {}, {});

            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.layout, 0,
                                   sets[i], {});
            cmd.dispatch(static_cast<std::uint32_t>(kN / kWg), 1, 1);

            // Release the shader's reads (a, b) and write (out) for the next
            // dispatch's transfer writes and the eventual readback.
            vk::MemoryBarrier toTransfer;
            toTransfer.setSrcAccessMask(vk::AccessFlagBits::eShaderRead |
                                        vk::AccessFlagBits::eShaderWrite)
                .setDstAccessMask(vk::AccessFlagBits::eTransferRead |
                                  vk::AccessFlagBits::eTransferWrite);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer, {},
                                toTransfer, {}, {});
        });
    }
    engine.drain();

    // Copy all 256 outputs back in one transfer, then drain.
    engine.submit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy c;
        c.setSrcOffset(0).setDstOffset(0).setSize(kChain * nBytes);
        cmd.copyBuffer(out_all.buffer, stage_out.buffer, c);
    });
    engine.drain();

    // Verify every one of the 256 outputs bit-exactly (fp32 vector-add is a
    // single add, so GPU == CPU reference exactly).
    bool ok = true;
    std::uint32_t badDispatch = 0;
    std::size_t badIndex = 0;
    {
        void* mapped = nullptr;
        if (vmaMapMemory(ctx.allocator, stage_out.allocation, &mapped) != VK_SUCCESS) {
            throw std::runtime_error("vmaMapMemory failed.");
        }
        vmaInvalidateAllocation(ctx.allocator, stage_out.allocation, 0, kChain * nBytes);

        float const* gpu = static_cast<float const*>(mapped);
        for (std::uint32_t i = 0; i < kChain && ok; ++i) {
            std::vector<float> a = generate_inputs(kN, kSeedA + i);
            std::vector<float> b = generate_inputs(kN, kSeedB + i);
            float const* out = gpu + static_cast<std::size_t>(i) * kN;
            for (std::size_t j = 0; j < kN; ++j) {
                if (out[j] != a[j] + b[j]) {
                    ok = false;
                    badDispatch = i;
                    badIndex = j;
                    break;
                }
            }
        }
        vmaUnmapMemory(ctx.allocator, stage_out.allocation);
    }

    if (!ok) {
        std::cerr << "exec_engine: MISMATCH dispatch " << badDispatch << " index "
                  << badIndex << "\n";
    }

    // Tear down (reverse order).
    destroy_pipeline(ctx.device, p);
    destroy_buffer(ctx.allocator, stage_out);
    destroy_buffer(ctx.allocator, stage_in);
    destroy_buffer(ctx.allocator, out_all);
    destroy_buffer(ctx.allocator, dev_b);
    destroy_buffer(ctx.allocator, dev_a);

    if (ok) {
        std::cout << "exec_engine: 256-chain correctness PASS\n";
    }
    return ok;
}

// --- phase 2: pipelining metric ---------------------------------------------

bool run_pipelining(vulkan_runtime::core::VulkanContext const& ctx, bool software) {
    constexpr std::uint32_t kM = 1024;
    constexpr std::uint32_t kN = 1024;
    constexpr std::uint32_t kK = 1024;
    constexpr std::uint32_t kTileM = 64;
    constexpr std::uint32_t kTileN = 64;
    constexpr std::uint32_t kDispatches = 24;

    vk::DeviceSize const aBytes = static_cast<vk::DeviceSize>(kM) * kK * sizeof(float);
    vk::DeviceSize const bBytes = static_cast<vk::DeviceSize>(kK) * kN * sizeof(float);
    vk::DeviceSize const cBytes = static_cast<vk::DeviceSize>(kM) * kN * sizeof(float);

    std::vector<float> in_a = generate_inputs(static_cast<std::size_t>(kM) * kK, 0x11111111u);
    std::vector<float> in_b = generate_inputs(static_cast<std::size_t>(kK) * kN, 0x22222222u);

    Buffer dev_a = create_buffer(
        ctx.allocator, aBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    Buffer dev_b = create_buffer(
        ctx.allocator, bBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    // Distinct output region per dispatch avoids any write-after-write hazard
    // between ring slots (VVL-clean; we only time, never read back).
    Buffer c_all = create_buffer(
        ctx.allocator, kDispatches * cBytes,
        vk::BufferUsageFlagBits::eStorageBuffer,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    Buffer staging = create_buffer(
        ctx.allocator, aBytes + bBytes,
        vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
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

    ComputePipeline p = build_compute_pipeline(ctx.device, "sgemm.spv", kDispatches,
                                               sizeof(PushConstants));
    std::vector<vk::DescriptorSet> sets =
        write_sets(ctx.device, p, kDispatches, dev_a.buffer, dev_b.buffer, c_all.buffer, cBytes);

    vulkan_runtime::exec::ExecEngine engine(ctx, 3, 1);

    // Upload A and B once (single submission + drain), so the timed section is
    // pure GEMM dispatch.
    engine.submit([&](vk::CommandBuffer cmd) {
        vk::BufferCopy cA;
        cA.setSrcOffset(0).setDstOffset(0).setSize(aBytes);
        cmd.copyBuffer(staging.buffer, dev_a.buffer, cA);
        vk::BufferCopy cB;
        cB.setSrcOffset(aBytes).setDstOffset(0).setSize(bBytes);
        cmd.copyBuffer(staging.buffer, dev_b.buffer, cB);
        vk::MemoryBarrier toShader;
        toShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader, {},
                            toShader, {}, {});
    });
    engine.drain();

    auto record = [&](std::uint32_t i, vk::CommandBuffer cmd) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.layout, 0,
                               sets[i % kDispatches], {});
        PushConstants pc{kM, kN, kK};
        cmd.pushConstants(p.layout, vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(PushConstants), &pc);
        cmd.dispatch(kN / kTileN, kM / kTileM, 1);
    };

    // Warm up: the RX580 idles at 300-600 MHz and only boosts after sustained
    // load; without this the first timed dispatches run at a fraction of peak
    // and skew the overlap ratio. lavapipe has no clock ramp, so a couple of
    // dispatches suffice there (keeps software CI fast).
    std::uint32_t const warmup = software ? 4 : 128;
    for (std::uint32_t i = 0; i < warmup; ++i) {
        engine.submit([&, i](vk::CommandBuffer cmd) { record(i, cmd); });
    }
    engine.drain();

    // Single-dispatch latency (submit + drain).
    Clock::time_point t0 = Clock::now();
    engine.submit([&](vk::CommandBuffer cmd) { record(0, cmd); });
    engine.drain();
    double const singleMs = elapsed_ms(t0, Clock::now());

    // Serial: submit -> drain -> next (fully serialized, no overlap).
    t0 = Clock::now();
    for (std::uint32_t i = 0; i < kDispatches; ++i) {
        engine.submit([&, i](vk::CommandBuffer cmd) { record(i, cmd); });
        engine.drain();
    }
    double const serialMs = elapsed_ms(t0, Clock::now());

    // Pipelined: submit all 24 through the 3-slot ring, then drain once.
    t0 = Clock::now();
    for (std::uint32_t i = 0; i < kDispatches; ++i) {
        engine.submit([&, i](vk::CommandBuffer cmd) { record(i, cmd); });
    }
    engine.drain();
    double const pipelinedMs = elapsed_ms(t0, Clock::now());

    double overlap = (pipelinedMs > 0.0) ? serialMs / pipelinedMs : 0.0;
    bool const overlapped = pipelinedMs < serialMs;

    std::cout << "exec_engine: single-dispatch 1024^3 = " << singleMs << " ms\n";

    // Tear down before any return (a leaked VMA allocation trips a
    // VmaDeviceMemoryBlock assertion at vmaDestroyAllocator).
    destroy_pipeline(ctx.device, p);
    destroy_buffer(ctx.allocator, staging);
    destroy_buffer(ctx.allocator, c_all);
    destroy_buffer(ctx.allocator, dev_b);
    destroy_buffer(ctx.allocator, dev_a);

    // On discrete hardware the overlap is real; assert it. On lavapipe the
    // llvmpipe thread scheduler dominates and per-submit jitter dwarfs any
    // pipelining effect, so we print the numbers but skip the assert.
    if (software) {
        std::cout << "exec_engine: pipelining 24x1024³: serial=" << serialMs
                  << "ms pipelined=" << pipelinedMs << "ms overlap=" << overlap
                  << "x (software, assert skipped)\n";
        return true;
    }

    if (!overlapped) {
        std::cout << "exec_engine: pipelining 24x1024³: serial=" << serialMs
                  << "ms pipelined=" << pipelinedMs << "ms overlap=" << overlap
                  << "x FAIL\n";
        return false;
    }

    std::cout << "exec_engine: pipelining 24x1024³: serial=" << serialMs
              << "ms pipelined=" << pipelinedMs << "ms overlap=" << overlap
              << "x PASS\n";
    return true;
}

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-exec-engine");

        bool const software = is_software_device(ctx.physicalDevice);
        bool const correct = run_correctness(ctx);
        bool const pipelined = run_pipelining(ctx, software);

        vulkan_runtime::core::destroy_context(ctx);

        if (correct && pipelined) {
            return EXIT_SUCCESS;
        }
        return EXIT_FAILURE;
    } catch (vk::SystemError const& e) {
        std::cerr << "Vulkan error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
