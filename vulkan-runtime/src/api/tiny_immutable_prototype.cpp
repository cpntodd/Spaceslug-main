#include "api/tiny_immutable_prototype.h"
#include "embedded_shaders.hpp"
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace vulkan_runtime::tiny {
namespace {
struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
};
Buffer make_buffer(core::VulkanContext const& c, vk::DeviceSize bytes, vk::BufferUsageFlags usage, VmaMemoryUsage memory,
                   VmaAllocationCreateFlags flags = {}) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = static_cast<VkBufferUsageFlags>(usage);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo ai{};
    ai.usage = memory;
    ai.flags = flags;
    VkBuffer raw{};
    VmaAllocation allocation{};
    if (vmaCreateBuffer(c.allocator, &info, &ai, &raw, &allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("immutable prototype buffer allocation failed");
    return {vk::Buffer(raw), allocation};
}
template <typename B>
void destroy(core::VulkanContext const& c, B& b) {
    if (b.allocation) {
        vmaDestroyBuffer(c.allocator, b.buffer, b.allocation);
        b.allocation = nullptr;
    }
}
constexpr vk::DeviceSize TOKENS = 0;
constexpr vk::DeviceSize TARGETS = TOKENS + 8 * sizeof(std::uint32_t);
constexpr vk::DeviceSize MASK = TARGETS + 8 * sizeof(std::uint32_t);
constexpr vk::DeviceSize DOUTPUT = MASK + 8 * sizeof(std::uint32_t);
constexpr vk::DeviceSize LEARNING = DOUTPUT + 8 * sizeof(float);
constexpr vk::DeviceSize CONTROL = LEARNING + 2 * sizeof(float);
constexpr vk::DeviceSize OUTPUT = CONTROL + 9 * sizeof(std::uint32_t);
constexpr vk::DeviceSize STAGING_BYTES = OUTPUT + 4 * sizeof(float);
}
struct ImmutableCommandPrototype::Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
};
ImmutableCommandPrototype::ImmutableCommandPrototype(core::VulkanContext const& c)
    : context_(c), engine_(c, 1, 1) {
    auto device_buffer = [&](vk::DeviceSize bytes, vk::BufferUsageFlags usage) {
        auto b = std::make_unique<Buffer>();
        auto created = make_buffer(context_, bytes,
                                    usage | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        b->buffer = created.buffer;
        b->allocation = created.allocation;
        return b;
    };
    tokens_ = device_buffer(8 * sizeof(std::uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    targets_ = device_buffer(8 * sizeof(std::uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    mask_ = device_buffer(8 * sizeof(std::uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    doutput_ = device_buffer(8 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer);
    learning_ = device_buffer(2 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer);
    output_ = device_buffer(4 * sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer);
    control_ = device_buffer(9 * sizeof(std::uint32_t), vk::BufferUsageFlagBits::eStorageBuffer);
    staging_ = std::make_unique<Buffer>();
    auto staging = make_buffer(context_, STAGING_BYTES,
                               vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                               VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    staging_->buffer = staging.buffer;
    staging_->allocation = staging.allocation;

    auto blob = shaders::get("tiny_immutable_prototype.spv");
    if (!blob.data) throw std::runtime_error("immutable prototype shader missing");
    shader_ = context_.device.createShaderModule(
        vk::ShaderModuleCreateInfo{}.setCodeSize(blob.size).setPCode(reinterpret_cast<std::uint32_t const*>(blob.data)));
    std::array<vk::DescriptorSetLayoutBinding, 7> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i)
        bindings[i] = vk::DescriptorSetLayoutBinding{}
                          .setBinding(i)
                          .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                          .setDescriptorCount(1)
                          .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    descriptorLayout_ = context_.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(bindings));
    pipelineLayout_ = context_.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{}.setSetLayouts(descriptorLayout_));
    auto pipeline = context_.device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{}
                 .setStage(vk::PipelineShaderStageCreateInfo{}
                               .setStage(vk::ShaderStageFlagBits::eCompute)
                               .setModule(shader_)
                               .setPName("main"))
                 .setLayout(pipelineLayout_));
    if (pipeline.result != vk::Result::eSuccess) throw std::runtime_error("immutable prototype pipeline failed");
    pipeline_ = pipeline.value;
    descriptorPool_ = context_.device.createDescriptorPool(
        vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(
            vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(7)));
    descriptorSet_ = context_.device.allocateDescriptorSets(
        vk::DescriptorSetAllocateInfo{}.setDescriptorPool(descriptorPool_).setSetLayouts(descriptorLayout_)).front();
    std::array<vk::Buffer, 7> buffers{tokens_->buffer, targets_->buffer, mask_->buffer, doutput_->buffer,
                                       learning_->buffer, output_->buffer, control_->buffer};
    std::array<vk::DescriptorBufferInfo, 7> infos{};
    std::array<vk::WriteDescriptorSet, 7> writes{};
    for (std::uint32_t i = 0; i < buffers.size(); ++i) {
        infos[i] = vk::DescriptorBufferInfo{}.setBuffer(buffers[i]).setRange(VK_WHOLE_SIZE);
        writes[i] = vk::WriteDescriptorSet{}
                        .setDstSet(descriptorSet_).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setDescriptorCount(1).setBufferInfo(infos[i]);
    }
    context_.device.updateDescriptorSets(writes, {});
    engine_.recordImmutable([this](vk::CommandBuffer cmd) {
        auto copy = [&](vk::Buffer dst, vk::DeviceSize srcOffset, vk::DeviceSize bytes) {
            cmd.copyBuffer(staging_->buffer, dst, vk::BufferCopy{}.setSrcOffset(srcOffset).setSize(bytes));
        };
        copy(tokens_->buffer, TOKENS, 8 * sizeof(std::uint32_t));
        copy(targets_->buffer, TARGETS, 8 * sizeof(std::uint32_t));
        copy(mask_->buffer, MASK, 8 * sizeof(std::uint32_t));
        copy(doutput_->buffer, DOUTPUT, 8 * sizeof(float));
        copy(learning_->buffer, LEARNING, 2 * sizeof(float));
        copy(control_->buffer, CONTROL, 9 * sizeof(std::uint32_t));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        cmd.dispatch(1, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead), {}, {});
        cmd.copyBuffer(output_->buffer, staging_->buffer,
                       vk::BufferCopy{}.setDstOffset(OUTPUT).setSize(4 * sizeof(float)));
    });
}
ImmutableCommandPrototype::~ImmutableCommandPrototype() {
    if (!context_.device) return;
    engine_.drain();
    destroy(context_, *staging_); destroy(context_, *control_); destroy(context_, *output_);
    destroy(context_, *learning_); destroy(context_, *doutput_); destroy(context_, *mask_);
    destroy(context_, *targets_); destroy(context_, *tokens_);
    context_.device.destroyDescriptorPool(descriptorPool_); context_.device.destroyPipeline(pipeline_);
    context_.device.destroyPipelineLayout(pipelineLayout_); context_.device.destroyDescriptorSetLayout(descriptorLayout_);
    context_.device.destroyShaderModule(shader_);
}
std::array<float, 4> ImmutableCommandPrototype::run(std::array<std::uint32_t, 8> const& tokens,
                                                     std::array<std::uint32_t, 8> const& targets,
                                                     std::array<std::uint32_t, 8> const& mask,
                                                     std::array<float, 8> const& doutput,
                                                     std::array<float, 2> const& learning,
                                                     std::array<std::uint32_t, 9> const& control) {
    struct Inputs { std::array<std::uint32_t, 8> t, y, m; std::array<float, 8> d; std::array<float, 2> lr; std::array<std::uint32_t, 9> c; };
    Inputs input{tokens, targets, mask, doutput, learning, control};
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, staging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("immutable prototype staging map failed");
    std::memcpy(mapped, &input, OUTPUT);
    vmaFlushAllocation(context_.allocator, staging_->allocation, 0, OUTPUT);
    vmaUnmapMemory(context_.allocator, staging_->allocation);
    lastSubmission_ = engine_.submitImmutable();
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, staging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("immutable prototype readback map failed");
    vmaInvalidateAllocation(context_.allocator, staging_->allocation, OUTPUT, 4 * sizeof(float));
    std::array<float, 4> result{};
    std::memcpy(result.data(), static_cast<char*>(mapped) + OUTPUT, sizeof(result));
    vmaUnmapMemory(context_.allocator, staging_->allocation);
    return result;
}
} // namespace vulkan_runtime::tiny
