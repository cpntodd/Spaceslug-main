#pragma once

#include "core/vk_setup.h"
#include "exec/engine.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace vulkan_runtime::dataset {

// Device-resident fixed-window dataset storage. The graph integration consumes
// these buffers directly; process() remains the standalone validation path.
class BatchBuffer {
  public:
    BatchBuffer(core::VulkanContext const& context, std::uint32_t windowCount, std::uint32_t windowTokens);
    ~BatchBuffer();
    BatchBuffer(BatchBuffer const&) = delete;
    BatchBuffer& operator=(BatchBuffer const&) = delete;

    struct DeviceView {
        vk::Buffer tokens{};
        vk::Buffer targets{};
        vk::Buffer masks{};
        vk::Buffer controls{};
        vk::Buffer results{};
        std::uint32_t window_count{};
        std::uint32_t window_tokens{};
    };

    // Non-owning view for one normal submission. The BatchBuffer must outlive
    // that submission; no allocation handles escape this API.
    DeviceView device_view() const noexcept;

    // Upload exactly windowCount * windowTokens tokens/targets/masks and one
    // control per window into persistent device-local buffers. No graph work is
    // implied: this is a bounded dataset staging operation.
    void upload(std::vector<std::uint32_t> const& tokens,
                std::vector<std::uint32_t> const& targets,
                std::vector<std::uint32_t> const& masks,
                std::vector<std::uint32_t> const& controls);

    // Process the retained windows on device and read back two floats per
    // window: masked squared error and the retained device control value.
    std::vector<float> process_readback();

    // Read-only device validation metrics: masked squared error and control
    // value per window. This does not execute Tiny forward or mutate weights.
    std::vector<float> metrics_readback();

    // Convenience upload followed by process_readback().
    std::vector<float> process(std::vector<std::uint32_t> const& tokens,
                               std::vector<std::uint32_t> const& targets,
                               std::vector<std::uint32_t> const& masks,
                               std::vector<std::uint32_t> const& controls);

    std::uint32_t window_count() const noexcept { return windowCount_; }
    std::uint32_t window_tokens() const noexcept { return windowTokens_; }
    std::uint64_t last_submission() const noexcept { return lastSubmission_; }
    static constexpr char const* capability() noexcept {
        return "production_bounded_persistent_tiny_dataset_lm_head_sgd_device_windows_scalar_readback_full_graph_pending";
    }

  public:
    struct Buffer;

  private:
    core::VulkanContext const& context_;
    exec::ExecEngine engine_;
    std::uint32_t windowCount_;
    std::uint32_t windowTokens_;
    std::unique_ptr<Buffer> tokens_, targets_, masks_, controls_, results_, staging_;
    vk::ShaderModule shader_{};
    vk::DescriptorSetLayout descriptorLayout_{};
    vk::PipelineLayout pipelineLayout_{};
    vk::Pipeline pipeline_{};
    vk::DescriptorPool descriptorPool_{};
    vk::DescriptorSet descriptorSet_{};
    std::uint64_t lastSubmission_{0};
};

} // namespace vulkan_runtime::dataset

extern "C" const char* vulkan_runtime_dataset_batch_capability();
extern "C" void* vulkan_runtime_dataset_batch_create(vulkan_runtime::core::VulkanContext const*,
                                                      std::uint32_t, std::uint32_t);
extern "C" void vulkan_runtime_dataset_batch_destroy(void*);
extern "C" int vulkan_runtime_dataset_batch_metrics(void*, float*);
extern "C" int vulkan_runtime_dataset_batch_process(void*, std::uint32_t const*, std::uint32_t const*,
                                                     std::uint32_t const*, std::uint32_t const*, float*);
