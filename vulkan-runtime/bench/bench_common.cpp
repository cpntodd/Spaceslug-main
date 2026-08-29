#include "bench/bench_common.h"

#include <algorithm>
#include <stdexcept>

namespace vulkan_runtime::bench {

bool is_discrete_gpu(core::VulkanContext const& ctx) {
    return ctx.physicalDevice.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
}

double timestamp_period(vk::PhysicalDevice const& device) {
    return static_cast<double>(device.getProperties().limits.timestampPeriod);
}

std::uint32_t timestamp_valid_bits(core::VulkanContext const& ctx) {
    std::vector<vk::QueueFamilyProperties> families =
        ctx.physicalDevice.getQueueFamilyProperties();
    if (ctx.computeQueueFamily < families.size()) {
        return families[ctx.computeQueueFamily].timestampValidBits;
    }
    return 0;
}

TimedRun time_dispatches(core::VulkanContext const& ctx,
                         RecordFn const& record_setup,
                         RecordFn const& record_dispatch,
                         std::uint32_t warmupRuns,
                         std::uint32_t timedRuns) {
    if (timedRuns == 0) {
        throw std::runtime_error("time_dispatches: timedRuns must be >= 1");
    }

    std::uint32_t queryCount = timedRuns * 2;
    double period = timestamp_period(ctx.physicalDevice);
    std::uint32_t validBits = timestamp_valid_bits(ctx);

    vk::QueryPoolCreateInfo qpInfo;
    qpInfo.setQueryType(vk::QueryType::eTimestamp).setQueryCount(queryCount);
    vk::QueryPool queryPool = ctx.device.createQueryPool(qpInfo);

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
    cmd.resetQueryPool(queryPool, 0, queryCount);

    if (record_setup) {
        record_setup(cmd);
    }
    for (std::uint32_t w = 0; w < warmupRuns; ++w) {
        record_dispatch(cmd);
    }
    for (std::uint32_t r = 0; r < timedRuns; ++r) {
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, queryPool, r * 2);
        record_dispatch(cmd);
        cmd.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, queryPool, r * 2 + 1);
    }
    cmd.end();

    vk::SubmitInfo submitInfo;
    submitInfo.setCommandBuffers(cmd);
    ctx.computeQueue.submit(submitInfo);
    ctx.computeQueue.waitIdle();

    std::vector<std::uint64_t> timestamps(queryCount);
    vk::Result qr = ctx.device.getQueryPoolResults(
        queryPool, 0, queryCount, timestamps.size() * sizeof(std::uint64_t),
        timestamps.data(), sizeof(std::uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (qr != vk::Result::eSuccess) {
        ctx.device.destroyCommandPool(commandPool);
        ctx.device.destroyQueryPool(queryPool);
        throw std::runtime_error("getQueryPoolResults failed.");
    }

    std::uint64_t mask = (validBits >= 64) ? ~0ull : ((1ull << validBits) - 1ull);
    TimedRun result;
    result.runsMs.resize(timedRuns);
    for (std::uint32_t r = 0; r < timedRuns; ++r) {
        std::uint64_t start = timestamps[r * 2] & mask;
        std::uint64_t end = timestamps[r * 2 + 1] & mask;
        double ns = static_cast<double>(end - start) * period;
        result.runsMs[r] = ns / 1e6;
    }
    std::vector<double> sorted = result.runsMs;
    std::sort(sorted.begin(), sorted.end());
    result.medianMs = sorted[timedRuns / 2];
    result.minMs = sorted.front();
    result.maxMs = sorted.back();

    ctx.device.destroyCommandPool(commandPool);
    ctx.device.destroyQueryPool(queryPool);
    return result;
}

} // namespace vulkan_runtime::bench
