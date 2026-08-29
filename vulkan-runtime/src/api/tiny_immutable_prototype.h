#pragma once

#include "core/vk_setup.h"
#include "exec/engine.h"
#include <array>
#include <cstdint>
#include <memory>

namespace vulkan_runtime::tiny {

// Bounded diagnostic-only fixed-shape prototype (not production). It is
// intentionally separate from ForwardResourceGraph: all command parameters are
// device-side buffers, so the same retained command buffer can be submitted
// after host staging changes.
class ImmutableCommandPrototype {
  public:
    explicit ImmutableCommandPrototype(core::VulkanContext const& context);
    ~ImmutableCommandPrototype();
    ImmutableCommandPrototype(ImmutableCommandPrototype const&) = delete;
    ImmutableCommandPrototype& operator=(ImmutableCommandPrototype const&) = delete;

    // Copies mutable inputs into the persistent host staging allocation, then
    // resubmits the command recorded by the constructor. The shape is always 8.
    std::array<float, 4> run(std::array<std::uint32_t, 8> const& tokens,
                             std::array<std::uint32_t, 8> const& targets,
                             std::array<std::uint32_t, 8> const& mask,
                             std::array<float, 8> const& doutput,
                             std::array<float, 2> const& learning,
                             std::array<std::uint32_t, 9> const& control);
    std::uint64_t last_submission() const noexcept { return lastSubmission_; }
    static constexpr char const* capability_name() noexcept {
        return "bounded_diagnostic_fixed_shape_forward_loss_backward_lora_immutable_host_staging";
    }

  private:
    struct Buffer;
    core::VulkanContext const& context_;
    exec::ExecEngine engine_;
    std::unique_ptr<Buffer> tokens_, targets_, mask_, doutput_, learning_, output_, control_, staging_;
    vk::ShaderModule shader_{};
    vk::DescriptorSetLayout descriptorLayout_{};
    vk::PipelineLayout pipelineLayout_{};
    vk::Pipeline pipeline_{};
    vk::DescriptorPool descriptorPool_{};
    vk::DescriptorSet descriptorSet_{};
    std::uint64_t lastSubmission_{0};
};

} // namespace vulkan_runtime::tiny
