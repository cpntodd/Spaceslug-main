#include "exec/engine.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace vulkan_runtime::exec {
namespace {

// The compute family's advertised queue count (the number of queues the
// device *supports*, not the number currently created). Used to clamp the
// requested queue count so we never ask for queues the family cannot give.
std::uint32_t family_queue_count(vk::PhysicalDevice const& physicalDevice,
                                 std::uint32_t familyIndex) {
    std::vector<vk::QueueFamilyProperties> families =
        physicalDevice.getQueueFamilyProperties();
    if (familyIndex >= families.size()) {
        throw std::runtime_error("compute queue family index out of range.");
    }
    return families[familyIndex].queueCount;
}

} // namespace

ExecEngine::ExecEngine(vulkan_runtime::core::VulkanContext const& ctx,
                       std::uint32_t ringSlots,
                       std::uint32_t queueCount)
    : device_(ctx.device),
      queueFamily_(ctx.computeQueueFamily),
      ringSlots_(ringSlots == 0 ? 1 : ringSlots) {
    if (!ctx.timelineSemaphoreEnabled) {
        throw std::runtime_error(
            "ExecEngine requires timeline semaphores, but the device did not "
            "enable VkPhysicalDeviceTimelineSemaphoreFeatures.");
    }

    // --- Queues (clamp to what the family supports *and* what the device
    // actually created). queueCount is requested; the context's
    // computeQueueCount is the number really created at device-creation time
    // (1 in M5a — getQueue() beyond it would fail, so never over-request).
    std::uint32_t const supported = family_queue_count(ctx.physicalDevice, queueFamily_);
    std::uint32_t const created =
        ctx.computeQueueCount == 0 ? 1 : ctx.computeQueueCount;
    std::uint32_t numQueues = std::min(queueCount, std::min(supported, created));
    if (numQueues == 0) {
        numQueues = 1;
    }
    queues_.reserve(numQueues);
    for (std::uint32_t i = 0; i < numQueues; ++i) {
        queues_.push_back(device_.getQueue(queueFamily_, i));
    }

    // --- One timeline semaphore *per queue*, initial value 0 ----------------
    // A single shared timeline semaphore cannot be signaled by multiple queues:
    // queues retire out of order, so a slow queue's lower signal would arrive
    // after a fast queue's higher signal (VUID-vkQueueSubmit-pSignalSemaphores
    // -03242) and corrupt the counter. Each queue instead advances its own
    // semaphore with the globally-monotonic submission value, which is strictly
    // increasing *within* that queue (submissions retire in submit order on a
    // single queue), so no cross-queue signal conflict is possible.
    queueTimelines_.reserve(numQueues);
    for (std::uint32_t i = 0; i < numQueues; ++i) {
        vk::SemaphoreTypeCreateInfo typeInfo;
        typeInfo.setSemaphoreType(vk::SemaphoreType::eTimeline).setInitialValue(0);

        vk::SemaphoreCreateInfo semaphoreInfo;
        semaphoreInfo.setPNext(&typeInfo);
        queueTimelines_.push_back(device_.createSemaphore(semaphoreInfo));
    }
    queueLastSignal_.assign(numQueues, 0);

    // --- Command pool + one command buffer per ring slot --------------------
    vk::CommandPoolCreateInfo poolInfo;
    poolInfo.setQueueFamilyIndex(queueFamily_)
        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPool_ = device_.createCommandPool(poolInfo);

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool_)
        .setLevel(vk::CommandBufferLevel::ePrimary)
        .setCommandBufferCount(ringSlots_ + 2);
    cmdBuffers_ = device_.allocateCommandBuffers(allocInfo);

    slotLastSignal_.assign(ringSlots_, 0);
    immutableCmd_ = cmdBuffers_[ringSlots_];
    immutableCmdSecondary_ = cmdBuffers_.back();

    // value -> queue map: value 0 is the never-submitted sentinel.
    valueQueue_.assign(1, 0);
}

ExecEngine::~ExecEngine() {
    if (!device_) {
        return;
    }
    // Reverse creation order. The device itself is owned by the context and
    // must outlive the engine (declared/created after it in the caller).
    for (vk::Semaphore& sem : queueTimelines_) {
        device_.destroySemaphore(sem);
        sem = nullptr;
    }
    device_.destroyCommandPool(commandPool_);
    commandPool_ = nullptr;
}

void ExecEngine::waitOnQueue(std::uint32_t queueIndex, std::uint64_t value) {
    vk::Semaphore sem = queueTimelines_[queueIndex];

    vk::SemaphoreWaitInfo waitInfo;
    waitInfo.setFlags({});
    waitInfo.setSemaphoreCount(1);
    waitInfo.setPSemaphores(&sem);
    waitInfo.setPValues(&value);

    // UINT64_MAX = wait indefinitely. waitSemaphores returns eSuccess once the
    // counter reaches `value`; a timeout is impossible here.
    vk::Result result =
        device_.waitSemaphores(waitInfo, std::numeric_limits<std::uint64_t>::max());
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("ExecEngine::waitOnQueue: waitSemaphores failed.");
    }
}

std::uint64_t ExecEngine::submit(std::function<void(vk::CommandBuffer)> record,
                                 std::uint32_t queueIndex,
                                 std::vector<std::pair<std::uint64_t, std::uint32_t>> const& waitValues) {
    if (!record) {
        throw std::runtime_error("ExecEngine::submit: null record callback.");
    }

    std::uint32_t const numQueues = static_cast<std::uint32_t>(queues_.size());
    std::uint32_t const q = queueIndex % numQueues;

    std::uint64_t const value = nextValue_++;
    std::uint32_t const slot = static_cast<std::uint32_t>((value - 1) % ringSlots_);

    // Throttle: wait for *this slot's* previous submission to retire before
    // touching its command buffer — on whichever queue it was signaled. This is
    // a host wait on that queue's timeline semaphore — NOT a queue idle — so
    // independent slots stay in flight while one slot blocks, and it guarantees
    // the slot's command buffer is no longer pending before reset/begin.
    if (slotLastSignal_[slot] != 0) {
        wait(slotLastSignal_[slot]);
    }

    vk::CommandBuffer cmd = cmdBuffers_[slot];
    cmd.reset({});
    cmd.begin(vk::CommandBufferBeginInfo{});
    record(cmd);
    cmd.end();

    // Cross-queue device-side dependency: group the requested wait values by
    // the queue that signaled them (derived from the value -> queue map; the
    // queueIndex in each pair is informational). A single submit may wait on a
    // given timeline semaphore with only one value, so take max(v) per queue —
    // per-queue monotonicity makes max(v) imply every smaller value on that
    // queue. These waits never touch the slot throttle or the signal counter.
    std::vector<std::uint64_t> waitByQueue(numQueues, 0);
    for (auto const& [v, qq] : waitValues) {
        (void)qq;
        if (v == 0 || v >= valueQueue_.size()) {
            continue; // never-submitted / not-yet-assigned value: nothing to wait on
        }
        std::uint32_t const qi = valueQueue_[v];
        waitByQueue[qi] = std::max(waitByQueue[qi], v);
    }

    std::vector<vk::Semaphore> waitSems;
    std::vector<std::uint64_t> waitVals;
    std::vector<vk::PipelineStageFlags> waitStages;
    waitSems.reserve(numQueues);
    waitVals.reserve(numQueues);
    waitStages.reserve(numQueues);
    for (std::uint32_t qi = 0; qi < numQueues; ++qi) {
        if (waitByQueue[qi] != 0) {
            waitSems.push_back(queueTimelines_[qi]);
            waitVals.push_back(waitByQueue[qi]);
            waitStages.push_back(vk::PipelineStageFlagBits::eAllCommands);
        }
    }

    std::uint64_t signalValue = value;
    vk::TimelineSemaphoreSubmitInfo timelineInfo;
    timelineInfo.setSignalSemaphoreValueCount(1);
    timelineInfo.setPSignalSemaphoreValues(&signalValue);
    timelineInfo.setWaitSemaphoreValueCount(static_cast<std::uint32_t>(waitVals.size()));
    timelineInfo.setPWaitSemaphoreValues(waitVals.empty() ? nullptr : waitVals.data());

    vk::Semaphore const signalSem = queueTimelines_[q];

    vk::SubmitInfo submitInfo;
    submitInfo.setPNext(&timelineInfo);
    submitInfo.setCommandBufferCount(1);
    submitInfo.setPCommandBuffers(&cmd);
    submitInfo.setWaitSemaphoreCount(static_cast<std::uint32_t>(waitSems.size()));
    submitInfo.setPWaitSemaphores(waitSems.empty() ? nullptr : waitSems.data());
    submitInfo.setPWaitDstStageMask(waitStages.empty() ? nullptr : waitStages.data());
    submitInfo.setSignalSemaphoreCount(1);
    submitInfo.setPSignalSemaphores(&signalSem);

    queues_[q].submit(submitInfo);

    slotLastSignal_[slot] = value;
    queueLastSignal_[q] = value;
    valueQueue_.push_back(q); // value -> queue map now covers `value`
    return value;
}

void ExecEngine::recordImmutable(std::function<void(vk::CommandBuffer)> record, bool secondary) {
    if (!record) {
        throw std::runtime_error("ExecEngine::recordImmutable: null record callback.");
    }
    bool& recorded = secondary ? immutableSecondaryRecorded_ : immutableRecorded_;
    vk::CommandBuffer cmd = secondary ? immutableCmdSecondary_ : immutableCmd_;
    if (recorded) {
        throw std::runtime_error("ExecEngine::recordImmutable: command buffer already recorded.");
    }
    cmd.reset({});
    cmd.begin(vk::CommandBufferBeginInfo{});
    record(cmd);
    cmd.end();
    recorded = true;
}

std::uint64_t ExecEngine::submitImmutable(std::uint32_t queueIndex, bool secondary) {
    bool const recorded = secondary ? immutableSecondaryRecorded_ : immutableRecorded_;
    vk::CommandBuffer const cmd = secondary ? immutableCmdSecondary_ : immutableCmd_;
    if (!recorded) {
        throw std::runtime_error("ExecEngine::submitImmutable: no retained command buffer.");
    }
    std::uint32_t const q = queueIndex % static_cast<std::uint32_t>(queues_.size());
    std::uint64_t const value = nextValue_++;
    std::uint64_t const previous = slotLastSignal_.front();
    if (previous != 0) {
        wait(previous);
    }
    vk::Semaphore const signalSem = queueTimelines_[q];
    vk::TimelineSemaphoreSubmitInfo timelineInfo;
    timelineInfo.setSignalSemaphoreValueCount(1).setPSignalSemaphoreValues(&value);
    vk::SubmitInfo submitInfo;
    submitInfo.setPNext(&timelineInfo).setCommandBufferCount(1).setPCommandBuffers(&cmd)
        .setSignalSemaphoreCount(1).setPSignalSemaphores(&signalSem);
    queues_[q].submit(submitInfo);
    slotLastSignal_.front() = value;
    queueLastSignal_[q] = value;
    valueQueue_.push_back(q);
    return value;
}

vk::Semaphore ExecEngine::semaphoreForValue(std::uint64_t value) const {
    if (value == 0 || value >= valueQueue_.size()) {
        throw std::invalid_argument("ExecEngine::semaphoreForValue: value not submitted.");
    }
    return queueTimelines_[valueQueue_[value]];
}

void ExecEngine::wait(std::uint64_t value) {
    if (value == 0) {
        return; // never-submitted sentinel
    }
    if (value >= valueQueue_.size()) {
        throw std::runtime_error("ExecEngine::wait: value not yet submitted.");
    }
    waitOnQueue(valueQueue_[value], value);
}

void ExecEngine::waitExternal(vk::Semaphore semaphore, std::uint64_t value) const {
    if (!semaphore || value == 0) {
        return;
    }
    vk::SemaphoreWaitInfo waitInfo;
    waitInfo.setSemaphoreCount(1).setPSemaphores(&semaphore).setPValues(&value);
    if (device_.waitSemaphores(waitInfo, std::numeric_limits<std::uint64_t>::max()) != vk::Result::eSuccess) {
        throw std::runtime_error("ExecEngine::waitExternal: waitSemaphores failed.");
    }
}

void ExecEngine::drain() {
    for (std::size_t q = 0; q < queues_.size(); ++q) {
        if (queueLastSignal_[q] != 0) {
            waitOnQueue(static_cast<std::uint32_t>(q), queueLastSignal_[q]);
        }
    }
}

std::uint32_t ExecEngine::inFlight() const {
    std::uint64_t const submitted = nextValue_ - 1;
    if (submitted == 0) {
        return 0;
    }
    std::vector<std::uint64_t> counters(queues_.size(), 0);
    for (std::size_t q = 0; q < queues_.size(); ++q) {
        counters[q] = device_.getSemaphoreCounterValue(queueTimelines_[q]);
    }
    std::uint64_t completed = 0;
    for (std::uint64_t v = 1; v <= submitted; ++v) {
        if (counters[valueQueue_[v]] >= v) {
            ++completed;
        }
    }
    return static_cast<std::uint32_t>(submitted - completed);
}

} // namespace vulkan_runtime::exec
