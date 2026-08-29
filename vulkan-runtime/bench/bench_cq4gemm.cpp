// M7a benchmark: the full CQ4 chain end-to-end — transform_act -> dequant_w ->
// sgemm_f16_am — the same 3-dispatch (all-packed, no cast) sequence the cactus
// gfx803 backend executes for a quant_matmul. This drives the shaders directly
// via a host loop, NOT through the cactus backend (that is M7b).
//
// Sizes: 1024^3 (anchor, like bench_sgemm) plus two needle-realistic shapes —
// M=1,N=512,K=512 (single-token decode) and M=1024,N=512,K=512 (prefill).
// Effective FLOPs = 2*M*N*K (the GEMM FMA count; transform + dequant are extra
// ALU on top and are NOT counted). For M=1 the chain is launch-bound (three
// dispatches, ~zero arithmetic), and we report it honestly as such.
//
// Perf protocol matches bench_sgemm.cpp: 128 warmup dispatches (RX580 clock
// ramp) then 15 timestamp-bracketed timed runs, median reported (never the
// mean). Correctness is test_cq4's job; this measures time only.
//
// CQ4 fixture (mirrors test_cq4's hadamard path): bits=4, group_size=128,
// non-interleaved, no signs/permutation, input_scale_recip. All inputs are
// generated deterministically; values are irrelevant to timing.

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

constexpr std::uint32_t kWarmupRuns = 128;
constexpr std::uint32_t kTimedRuns = 15; // >= 9; median of 15

constexpr double kSgemmBaselineTflops = 3.21; // sgemm fp32 baseline @ 1024^3 (M4a)

constexpr std::uint32_t kBits = 4;
constexpr std::uint32_t kGroupSize = 128;

struct Size {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
    char const* name;
};

std::array<Size, 3> const kSizes{{
    {1024, 1024, 1024, "1024x1024x1024"},
    {1, 512, 512, "1x512x512"},
    {1024, 512, 512, "1024x512x512"},
}};

struct TAPush {
    std::uint32_t mode, M, K, group_size, num_groups, has_left, has_right, has_perm;
};
struct DWPush {
    std::uint32_t mode, bits, K, N, group_size, num_groups;
};
struct GEMMPush {
    std::uint32_t M, N, K;
};

struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
    vk::DeviceSize size{0};
};

struct Kernel {
    vk::ShaderModule module{};
    vk::Pipeline pipeline{};
    vk::DescriptorSetLayout setLayout{};
    vk::PipelineLayout pipelineLayout{};
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

// Deterministic LCG -> fp32 in [lo, hi).
std::vector<float> gen_f32(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        float u = static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        v[i] = lo + u * (hi - lo);
    }
    return v;
}

// Deterministic LCG -> normal finite fp16 bits in +-[0.5, 1.0) (sane, no denormals/NaN).
std::vector<std::uint16_t> gen_fp16(std::size_t n, std::uint32_t seed) {
    std::vector<std::uint16_t> v(n);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        std::uint32_t r = s >> 8;
        std::uint16_t sign = (r & 0x10000u) ? 0x8000u : 0x0000u;
        v[i] = static_cast<std::uint16_t>(sign | 0x3800u | (r & 0x03FFu));
    }
    return v;
}

// fp16 bits -> one per uint32 (word-per-value, transform/dequant INPUTS).
std::vector<std::uint32_t> widen16(std::vector<std::uint16_t> const& h) {
    std::vector<std::uint32_t> w(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) w[i] = h[i];
    return w;
}

// random bytes -> one byte per uint32 (CQ4 packed_indices storage).
std::vector<std::uint32_t> gen_packed_indices(std::size_t nbytes, std::uint32_t seed) {
    std::vector<std::uint32_t> v(nbytes);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < nbytes; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = (s >> 8) & 0xFFu;
    }
    return v;
}

double tflops(std::uint32_t M, std::uint32_t N, std::uint32_t K, double medianMs) {
    double flops = 2.0 * static_cast<double>(M) * N * K;
    return flops / (medianMs * 1e-3) / 1e12;
}

} // namespace

int main() {
    try {
        using namespace vulkan_runtime;

        core::VulkanContext ctx = core::create_context("vulkan-runtime-cq4gemm-bench");

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
        std::cout << "cq4gemm chain: transform_act -> dequant_w -> sgemm_f16_am (packed fp16)"
                  << " | bits=" << kBits << " gs=" << kGroupSize
                  << " | warmup=" << kWarmupRuns << " timed=" << kTimedRuns << " (median)\n\n";

        // --- Pipeline layout helpers ------------------------------------------
        auto makeSetLayout = [&](std::uint32_t n) {
            std::vector<vk::DescriptorSetLayoutBinding> bindings(n);
            for (std::uint32_t i = 0; i < n; ++i) {
                bindings[i].setBinding(i)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setDescriptorCount(1)
                    .setStageFlags(vk::ShaderStageFlagBits::eCompute);
            }
            vk::DescriptorSetLayoutCreateInfo info;
            info.setBindings(bindings);
            return ctx.device.createDescriptorSetLayout(info);
        };

        auto makeKernel = [&](char const* shader, vk::DescriptorSetLayout setLayout,
                              vk::PipelineLayout pipelineLayout) {
            Kernel k;
            shaders::ShaderBlob blob = shaders::get(shader);
            if (blob.data == nullptr || blob.size == 0) {
                throw std::runtime_error(std::string(shader) + " not embedded.");
            }
            vk::ShaderModuleCreateInfo modInfo;
            modInfo.setCodeSize(blob.size)
                .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
            k.module = ctx.device.createShaderModule(modInfo);

            vk::PipelineShaderStageCreateInfo stageInfo;
            stageInfo.setStage(vk::ShaderStageFlagBits::eCompute)
                .setModule(k.module)
                .setPName("main");

            vk::ComputePipelineCreateInfo pipeInfo;
            pipeInfo.setStage(stageInfo).setLayout(pipelineLayout);
            auto pr = ctx.device.createComputePipeline({}, pipeInfo);
            if (pr.result != vk::Result::eSuccess) {
                throw std::runtime_error(std::string("createComputePipeline failed for ") + shader);
            }
            k.pipeline = pr.value;
            k.setLayout = setLayout;
            k.pipelineLayout = pipelineLayout;
            return k;
        };

        // Transform: 7 SSBOs + 8-word push. Dequant: 4 SSBOs + 6-word push.
        // GEMM: 3 SSBOs + 3-word push.
        vk::DescriptorSetLayout taSetLayout = makeSetLayout(7);
        vk::DescriptorSetLayout dwSetLayout = makeSetLayout(4);
        vk::DescriptorSetLayout gmSetLayout = makeSetLayout(3);

        vk::PushConstantRange taPc;
        taPc.setStageFlags(vk::ShaderStageFlagBits::eCompute)
            .setOffset(0)
            .setSize(static_cast<std::uint32_t>(sizeof(TAPush)));
        vk::PushConstantRange dwPc;
        dwPc.setStageFlags(vk::ShaderStageFlagBits::eCompute)
            .setOffset(0)
            .setSize(static_cast<std::uint32_t>(sizeof(DWPush)));
        vk::PushConstantRange gmPc;
        gmPc.setStageFlags(vk::ShaderStageFlagBits::eCompute)
            .setOffset(0)
            .setSize(static_cast<std::uint32_t>(sizeof(GEMMPush)));

        vk::PipelineLayoutCreateInfo taLayoutInfo;
        taLayoutInfo.setSetLayouts(taSetLayout).setPushConstantRanges(taPc);
        vk::PipelineLayout taPipelineLayout = ctx.device.createPipelineLayout(taLayoutInfo);
        vk::PipelineLayoutCreateInfo dwLayoutInfo;
        dwLayoutInfo.setSetLayouts(dwSetLayout).setPushConstantRanges(dwPc);
        vk::PipelineLayout dwPipelineLayout = ctx.device.createPipelineLayout(dwLayoutInfo);
        vk::PipelineLayoutCreateInfo gmLayLayoutInfo;
        gmLayLayoutInfo.setSetLayouts(gmSetLayout).setPushConstantRanges(gmPc);
        vk::PipelineLayout gmPipelineLayout = ctx.device.createPipelineLayout(gmLayLayoutInfo);

        Kernel ta = makeKernel("transform_act.spv", taSetLayout, taPipelineLayout);
        Kernel dw = makeKernel("dequant_w.spv", dwSetLayout, dwPipelineLayout);
        Kernel gm = makeKernel("sgemm_f16_am.spv", gmSetLayout, gmPipelineLayout);

        // --- Per-size run -----------------------------------------------------
        for (Size const& sz : kSizes) {
            std::uint32_t const M = sz.M, N = sz.N, K = sz.K;
            std::uint32_t const numGroups = K / kGroupSize;
            std::uint32_t const pgb = (kGroupSize * kBits + 7u) / 8u; // 64 bytes/group

            // Buffer sizes (uint32 words).
            vk::DeviceSize aWords = static_cast<vk::DeviceSize>(M) * K;      // fp16 word-per-value
            vk::DeviceSize recipWords = K;                                   // fp32
            vk::DeviceSize aPrimeWords = static_cast<vk::DeviceSize>(M) * K / 2u; // packed fp16
            vk::DeviceSize packedWords = static_cast<vk::DeviceSize>(N) * numGroups * pgb; // bytes as words
            vk::DeviceSize cbWords = 16u;                                    // fp16 word-per-value
            vk::DeviceSize normsWords = static_cast<vk::DeviceSize>(N) * numGroups; // fp16
            vk::DeviceSize wPrimeWords = static_cast<vk::DeviceSize>(N) * K / 2u; // packed fp16
            vk::DeviceSize cWords = (static_cast<vk::DeviceSize>(M) * N + 1u) / 2u; // packed fp16

            auto wBytes = [](vk::DeviceSize words) { return words * sizeof(std::uint32_t); };

            // --- CPU fixture generation --------------------------------------
            std::vector<std::uint32_t> a = widen16(gen_fp16(aWords, 0xC0100001u));
            std::vector<float> recip_f = gen_f32(recipWords, 0xC0100002u, 0.5f, 1.5f);
            std::vector<std::uint32_t> recip(recipWords);
            std::memcpy(recip.data(), recip_f.data(), wBytes(recipWords));
            std::vector<std::uint32_t> packed = gen_packed_indices(packedWords, 0xC0100003u);
            std::vector<std::uint32_t> codebook = widen16(gen_fp16(16, 0xC0100004u));
            std::vector<std::uint32_t> norms = widen16(gen_fp16(normsWords, 0xC0100005u));
            std::vector<std::uint32_t> one(1, 0u); // dummy ls/rs/perm/rot

            // --- Device buffers -------------------------------------------------
            Buffer dev_a = create_buffer(ctx.allocator, wBytes(aWords),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_recip = create_buffer(ctx.allocator, wBytes(recipWords),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_ls = create_buffer(ctx.allocator, wBytes(1),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_rs = create_buffer(ctx.allocator, wBytes(1),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_perm = create_buffer(ctx.allocator, wBytes(1),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_rot = create_buffer(ctx.allocator, wBytes(1),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_aPrime = create_buffer(ctx.allocator, wBytes(aPrimeWords),
                vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_packed = create_buffer(ctx.allocator, wBytes(packedWords),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_cb = create_buffer(ctx.allocator, wBytes(cbWords),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_norms = create_buffer(ctx.allocator, wBytes(normsWords),
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_wPrime = create_buffer(ctx.allocator, wBytes(wPrimeWords),
                vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            Buffer dev_c = create_buffer(ctx.allocator, wBytes(cWords),
                vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

            // --- Staging (all inputs contiguous) --------------------------------
            vk::DeviceSize inTotal = wBytes(aWords) + wBytes(recipWords) + wBytes(1) +
                                     wBytes(1) + wBytes(1) + wBytes(1) +
                                     wBytes(packedWords) + wBytes(cbWords) + wBytes(normsWords);
            Buffer staging = create_buffer(ctx.allocator, inTotal,
                vk::BufferUsageFlagBits::eTransferSrc,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

            {
                void* mapped = nullptr;
                if (vmaMapMemory(ctx.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
                    throw std::runtime_error("vmaMapMemory failed.");
                }
                char* dst = static_cast<char*>(mapped);
                vk::DeviceSize off = 0;
                auto put = [&](void const* src, vk::DeviceSize n) {
                    std::memcpy(dst + off, src, n);
                    off += n;
                };
                put(a.data(), wBytes(aWords));
                put(recip.data(), wBytes(recipWords));
                put(one.data(), wBytes(1));   // ls
                put(one.data(), wBytes(1));   // rs
                put(one.data(), wBytes(1));   // perm
                put(one.data(), wBytes(1));   // rot
                put(packed.data(), wBytes(packedWords));
                put(codebook.data(), wBytes(cbWords));
                put(norms.data(), wBytes(normsWords));
                vmaUnmapMemory(ctx.allocator, staging.allocation);
                vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inTotal);
            }

            // --- Upload operands once (transfer -> compute barrier) -------------
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

                vk::DeviceSize off = 0;
                auto copy = [&](Buffer const& dst, vk::DeviceSize n) {
                    vk::BufferCopy bc;
                    bc.setSrcOffset(off).setDstOffset(0).setSize(n);
                    cb.copyBuffer(staging.buffer, dst.buffer, bc);
                    off += n;
                };
                copy(dev_a, wBytes(aWords));
                copy(dev_recip, wBytes(recipWords));
                copy(dev_ls, wBytes(1));
                copy(dev_rs, wBytes(1));
                copy(dev_perm, wBytes(1));
                copy(dev_rot, wBytes(1));
                copy(dev_packed, wBytes(packedWords));
                copy(dev_cb, wBytes(cbWords));
                copy(dev_norms, wBytes(normsWords));

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

            // --- Descriptor sets --------------------------------------------------
            auto makeDescSet = [&](vk::DescriptorSetLayout layout, std::uint32_t n,
                                   std::vector<Buffer const*> const& bufs) {
                vk::DescriptorPoolSize poolSize;
                poolSize.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(n);
                vk::DescriptorPoolCreateInfo poolInfo;
                poolInfo.setMaxSets(1).setPoolSizes(poolSize);
                vk::DescriptorPool pool = ctx.device.createDescriptorPool(poolInfo);

                vk::DescriptorSetAllocateInfo setAllocInfo;
                setAllocInfo.setDescriptorPool(pool).setSetLayouts(layout);
                vk::DescriptorSet set = ctx.device.allocateDescriptorSets(setAllocInfo).front();

                std::vector<vk::DescriptorBufferInfo> infos(n);
                for (std::uint32_t i = 0; i < n; ++i) {
                    infos[i].setBuffer(bufs[i]->buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
                }
                std::vector<vk::WriteDescriptorSet> writes(n);
                for (std::uint32_t i = 0; i < n; ++i) {
                    writes[i].setDstSet(set).setDstBinding(i).setDescriptorCount(1)
                        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setBufferInfo(infos[i]);
                }
                ctx.device.updateDescriptorSets(writes, {});
                return std::make_pair(set, pool);
            };

            auto [taSet, taPool] = makeDescSet(taSetLayout, 7,
                {&dev_a, &dev_recip, &dev_ls, &dev_rs, &dev_perm, &dev_rot, &dev_aPrime});
            auto [dwSet, dwPool] = makeDescSet(dwSetLayout, 4,
                {&dev_packed, &dev_cb, &dev_norms, &dev_wPrime});
            auto [gmSet, gmPool] = makeDescSet(gmSetLayout, 3,
                {&dev_aPrime, &dev_wPrime, &dev_c});

            // --- Dispatch grids --------------------------------------------------
            std::uint32_t taTotal = M * numGroups;
            std::uint32_t taGx = (taTotal + 63u) / 64u;
            std::uint32_t dwTotal = N * (K / 2u);
            std::uint32_t dwGx = (dwTotal + 63u) / 64u;
            std::uint32_t gmGx = (N + 63u) / 64u;
            std::uint32_t gmGy = (M + 63u) / 64u;

            TAPush taPc{0u, M, K, kGroupSize, numGroups, 0u, 0u, 0u};
            DWPush dwPc{0u, kBits, K, N, kGroupSize, numGroups};
            GEMMPush gmPc{M, N, K};

            // The 3-dispatch chain with shader->shader memory barriers. A' and W'
            // are written by kernels 1/2 and read by kernel 3, so a barrier
            // between each pair guarantees visibility.
            auto recordDispatch = [&](vk::CommandBuffer const& cb) {
                cb.bindPipeline(vk::PipelineBindPoint::eCompute, ta.pipeline);
                cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ta.pipelineLayout, 0,
                                      taSet, {});
                cb.pushConstants(ta.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(TAPush), &taPc);
                cb.dispatch(taGx, 1, 1);

                vk::MemoryBarrier b1;
                b1.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                    .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
                cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader, {}, b1, {}, {});

                cb.bindPipeline(vk::PipelineBindPoint::eCompute, dw.pipeline);
                cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, dw.pipelineLayout, 0,
                                      dwSet, {});
                cb.pushConstants(dw.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(DWPush), &dwPc);
                cb.dispatch(dwGx, 1, 1);

                vk::MemoryBarrier b2;
                b2.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                    .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
                cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                   vk::PipelineStageFlagBits::eComputeShader, {}, b2, {}, {});

                cb.bindPipeline(vk::PipelineBindPoint::eCompute, gm.pipeline);
                cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, gm.pipelineLayout, 0,
                                      gmSet, {});
                cb.pushConstants(gm.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(GEMMPush), &gmPc);
                cb.dispatch(gmGx, gmGy, 1);
            };

            bench::RecordFn recordSetup; // empty: full chain is recorded per run
            bench::TimedRun run = bench::time_dispatches(ctx, recordSetup, recordDispatch,
                                                         kWarmupRuns, kTimedRuns);

            double tfl = tflops(M, N, K, run.medianMs);
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "cq4gemm " << sz.name << ": median " << run.medianMs << " ms ("
                      << run.minMs << " .. " << run.maxMs << " ms)\n";
            std::cout << std::setprecision(2);
            std::cout << "cq4gemm " << sz.name << ": " << tfl
                      << " TFLOPS (vs sgemm fp32 " << kSgemmBaselineTflops << ")\n";
            if (M == 1) {
                std::cout << "  note: M=1 is launch-bound (3 dispatches, ~zero arithmetic); "
                             "effective TFLOPS is honest but not throughput-bound\n";
            }
            std::cout << "\n";

            // --- Tear down this size --------------------------------------------
            ctx.device.destroyDescriptorPool(taPool);
            ctx.device.destroyDescriptorPool(dwPool);
            ctx.device.destroyDescriptorPool(gmPool);
            destroy_buffer(ctx.allocator, staging);
            destroy_buffer(ctx.allocator, dev_c);
            destroy_buffer(ctx.allocator, dev_wPrime);
            destroy_buffer(ctx.allocator, dev_norms);
            destroy_buffer(ctx.allocator, dev_cb);
            destroy_buffer(ctx.allocator, dev_packed);
            destroy_buffer(ctx.allocator, dev_aPrime);
            destroy_buffer(ctx.allocator, dev_rot);
            destroy_buffer(ctx.allocator, dev_perm);
            destroy_buffer(ctx.allocator, dev_rs);
            destroy_buffer(ctx.allocator, dev_ls);
            destroy_buffer(ctx.allocator, dev_recip);
            destroy_buffer(ctx.allocator, dev_a);
        }

        // --- Global teardown (reverse creation order) ---------------------------
        ctx.device.destroyPipeline(gm.pipeline);
        ctx.device.destroyPipeline(dw.pipeline);
        ctx.device.destroyPipeline(ta.pipeline);
        ctx.device.destroyPipelineLayout(gmPipelineLayout);
        ctx.device.destroyPipelineLayout(dwPipelineLayout);
        ctx.device.destroyPipelineLayout(taPipelineLayout);
        ctx.device.destroyShaderModule(gm.module);
        ctx.device.destroyShaderModule(dw.module);
        ctx.device.destroyShaderModule(ta.module);
        ctx.device.destroyDescriptorSetLayout(gmSetLayout);
        ctx.device.destroyDescriptorSetLayout(dwSetLayout);
        ctx.device.destroyDescriptorSetLayout(taSetLayout);
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
