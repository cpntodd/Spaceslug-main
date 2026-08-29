// Reusable benchmark harness (M3) — timestamp-query timing with median-of-runs
// statistics, shared by the M3 SGEMM driver and (from M4 on) every kernel
// benchmark.
//
// The one non-obvious piece is `time_dispatches`: it records the warmup and
// timed dispatches into a *single* command buffer and submits once, so the GPU
// stays continuously busy across the whole measurement. Per-run fresh submits
// would let the RX580 clock ramp back down between runs (the M2 lesson: idle
// clock is 300–600 MHz, boost 1366 MHz; ~128 back-to-back dispatches are
// needed for a stable median).
//
// Perf numbers are only meaningful on RADV (discrete GPU). The driver guards
// on `is_discrete_gpu()` and skips the sweep on lavapipe/llvmpipe, where a full
// sweep would otherwise take minutes.

#pragma once

#include "core/vk_setup.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace vulkan_runtime::bench {

// One timed measurement: the median of the raw runs (ms), min, max, and the
// raw run list in dispatch order.
struct TimedRun {
    double medianMs{0.0};
    double minMs{0.0};
    double maxMs{0.0};
    std::vector<double> runsMs{};
};

// Callback that records GPU work into a command buffer. `record_setup` runs
// once (bind pipeline / descriptors / push constants); `record_dispatch` runs
// once per dispatch (a bare vkCmdDispatch). Both are optional.
using RecordFn = std::function<void(vk::CommandBuffer const&)>;

// Times a compute dispatch using timestamp queries.
//
// Allocates a timestamp query pool (2 queries/timed run) and a fresh command
// buffer; records `warmupRuns` untimed dispatches (clock/JIT ramp), then
// `timedRuns` dispatches each bracketed by writeTimestamp before/after;
// submits once, waits idle, reads back the timestamps, scales by
// timestampPeriod, and returns median/min/max + raw runs.
//
// Throws std::runtime_error if timedRuns == 0 or the query read-back fails.
TimedRun time_dispatches(core::VulkanContext const& ctx,
                         RecordFn const& record_setup,
                         RecordFn const& record_dispatch,
                         std::uint32_t warmupRuns,
                         std::uint32_t timedRuns);

// True when the selected physical device is a discrete GPU. Perf numbers are
// only meaningful there; lavapipe/llvmpipe (CPU) is skipped by the driver.
bool is_discrete_gpu(core::VulkanContext const& ctx);

// timestampPeriod (ns/tick) of the physical device.
double timestamp_period(vk::PhysicalDevice const& device);

// timestampValidBits of the compute queue family (for wrapping 64-bit reads).
std::uint32_t timestamp_valid_bits(core::VulkanContext const& ctx);

} // namespace vulkan_runtime::bench
