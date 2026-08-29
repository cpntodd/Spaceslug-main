// N-slot ring execution engine — M5a core, extended in M5b for cross-queue.
//
// An N-slot ring of command buffers driven by a single strictly-increasing
// "monotonic counter" of submission values (the single source of truth for
// ordering). Each ring slot remembers the last value it signaled. Reusing a
// slot first *host-waits* (vk::SemaphoreWaitInfo) on that slot's own prior
// value — never on a queue-idle — which (a) throttles the host to at most
// `ringSlots` in-flight submissions and (b) guarantees the slot's command
// buffer is no longer in flight, so it is safe to reset and re-record.
//
// --- M5b cross-queue invariant ----------------------------------------------
//
// A single timeline semaphore CANNOT be signaled by multiple queues: queues
// retire out of order, so a slow queue's "lower" signal would arrive after a
// fast queue's "higher" signal and violate the monotonic-signal requirement
// (VUID-vkQueueSubmit-pSignalSemaphores-03242), corrupting the counter and
// deadlocking drain(). The fix is **one timeline semaphore per queue**, while
// the submission *value* namespace stays a single shared counter:
//
//     every submit is assigned a strictly increasing value V = nextValue++, and
//     signals *its own queue's* timeline semaphore with V. Within a queue the
//     signals are therefore strictly increasing (submissions retire in submit
//     order on a single queue), so no cross-queue signal conflict is possible.
//     A value -> queue map records which queue signaled each V, so any V can be
//     host-waited on via the correct queue's semaphore.
//
// Consequences:
//   * Slot reuse always host-waits on the slot's own last value, on whichever
//     queue it was signaled — independent of any cross-queue waitValues.
//   * `waitValues` become device-side waits on the *producing* queue's
//     timeline (one value per queue, max(v) per queue); they never disturb the
//     slot throttle or the signal counter.
//   * drain() host-waits on *every* queue's last signal, since no single
//     queue's signal implies the others' completion.
//
// The engine never calls vkQueueWaitIdle / vkDeviceWaitIdle. It is
// single-threaded: submit()/wait()/drain() must not be called concurrently.

#pragma once

#include "core/vk_setup.h"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace vulkan_runtime::exec {

class ExecEngine {
public:
    // Creates a command pool with one command buffer per ring slot, one
    // timeline semaphore per queue (initial value 0), per-slot signal tracking,
    // and `queueCount` compute queues (clamped to what the device actually
    // provides). Throws std::runtime_error if the context lacks timeline
    // semaphore support.
    ExecEngine(vulkan_runtime::core::VulkanContext const& ctx,
               std::uint32_t ringSlots = 3,
               std::uint32_t queueCount = 1);
    ~ExecEngine();

    ExecEngine(ExecEngine const&) = delete;
    ExecEngine& operator=(ExecEngine const&) = delete;
    ExecEngine(ExecEngine&&) = delete;
    ExecEngine& operator=(ExecEngine&&) = delete;

    // Acquires the next free ring slot (throttling via a host wait on that
    // slot's own previous timeline value — on whichever queue it was signaled),
    // records `record` into it, submits it to queues[queueIndex % queueCount]
    // signaling *that queue's* timeline semaphore with a fresh monotonic value,
    // and returns that value.
    //
    // `waitValues` adds device-side cross-queue dependencies: each entry is
    // {timelineValue, queueIndex}, and the submission must not execute until
    // the *producing* queue's timeline semaphore reaches `timelineValue`. The
    // queueIndex is informational — the value uniquely identifies the queue
    // that signaled it (via the value -> queue map), so the correct semaphore
    // is derived, never trusted from the caller. A single submit may wait on a
    // given semaphore with only one value, so waits are grouped per queue and
    // the effective wait on each queue is max(value) — which, by per-queue
    // monotonicity, implies every smaller value on that queue has been reached.
    // Empty (the default) preserves the M5a single-queue behavior: no
    // device-side wait, only the host slot throttle.
    std::uint64_t submit(std::function<void(vk::CommandBuffer)> record,
                         std::uint32_t queueIndex = 0,
                         std::vector<std::pair<std::uint64_t, std::uint32_t>> const& waitValues = {});

    // Fixed-shape prototype: retain up to two fully recorded command buffers
    // and resubmit either one without resetting or re-recording. The callback
    // must only encode immutable commands (including fixed descriptor bindings
    // and push constants); callers must wait for the returned value before
    // changing any resources referenced by the command buffer. This is
    // intentionally separate from submit(), whose callback is recorded on
    // every invocation. Both retained buffers share the engine timeline and
    // are serialized by the retained slot's last signal.
    void recordImmutable(std::function<void(vk::CommandBuffer)> record, bool secondary = false);
    std::uint64_t submitImmutable(std::uint32_t queueIndex = 0, bool secondary = false);
    bool immutableCommandBufferRetained() const noexcept { return immutableRecorded_; }
    bool immutableSecondaryCommandBufferRetained() const noexcept { return immutableSecondaryRecorded_; }
    bool immutableCommandBuffersRetained() const noexcept { return immutableRecorded_ && immutableSecondaryRecorded_; }
    static constexpr char const* immutableReuseCapability() noexcept {
        return "fixed_shape_retained_command_buffer_resubmit";
    }

    // Host-waits until the queue that signaled `value` reaches (>=) `value`.
    void wait(std::uint64_t value);

    // Blocks the host until a timeline semaphore owned by this context reaches
    // value. Used for external synchronization domains that share buffers.
    void waitExternal(vk::Semaphore semaphore, std::uint64_t value) const;

    // Host-waits until every queue has signaled its most recent submission.
    void drain();

    // Returns the timeline semaphore that signals a submission value. The
    // producing queue is derived from the engine's value namespace; callers
    // use this only to build explicit cross-engine waits for shared buffers.
    vk::Semaphore semaphoreForValue(std::uint64_t value) const;

    // Introspection.
    std::uint64_t lastValue() const noexcept { return nextValue_ - 1; }
    std::uint64_t nextValue() const noexcept { return nextValue_; }
    std::uint32_t inFlight() const; // submissions not yet retired (any queue)
    std::uint32_t ringSlots() const noexcept { return ringSlots_; }
    std::uint32_t queueCount() const noexcept {
        return static_cast<std::uint32_t>(queues_.size());
    }

private:
    // Host-waits on `queueTimelines_[queueIndex]` until it reaches `value`.
    void waitOnQueue(std::uint32_t queueIndex, std::uint64_t value);

    vk::Device device_{}; // non-owning handle; the context owns the device
    std::uint32_t queueFamily_{0};
    vk::CommandPool commandPool_{};              // owning
    std::vector<vk::Semaphore> queueTimelines_;  // owning; one timeline per queue
    std::vector<vk::Queue> queues_;
    std::vector<vk::CommandBuffer> cmdBuffers_;
    std::vector<std::uint64_t> slotLastSignal_;  // last global value per slot
    std::vector<std::uint64_t> queueLastSignal_; // last signaled value per queue
    std::vector<std::uint32_t> valueQueue_;      // value -> queue map (dense; grows with submissions)
    vk::CommandBuffer immutableCmd_{};
    vk::CommandBuffer immutableCmdSecondary_{};
    bool immutableRecorded_{false};
    bool immutableSecondaryRecorded_{false};
    std::uint32_t ringSlots_{0};
    std::uint64_t nextValue_{1}; // next monotonic timeline value to signal
};

} // namespace vulkan_runtime::exec
