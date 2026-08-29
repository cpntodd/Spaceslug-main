#include "api/dataset_batch_buffer.h"
#include "embedded_shaders.hpp"

#include <array>
#include <cstring>
#include <stdexcept>

namespace vulkan_runtime::dataset {
namespace {
struct RawBuffer { vk::Buffer buffer{}; VmaAllocation allocation{nullptr}; };
RawBuffer make_buffer(core::VulkanContext const& c, vk::DeviceSize bytes, vk::BufferUsageFlags usage,
                      VmaMemoryUsage memory, VmaAllocationCreateFlags flags = {}) {
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
        throw std::runtime_error("dataset batch buffer allocation failed");
    return {vk::Buffer(raw), allocation};
}
} // namespace

struct BatchBuffer::Buffer { vk::Buffer buffer{}; VmaAllocation allocation{nullptr}; };

namespace {
void destroy(core::VulkanContext const& c, BatchBuffer::Buffer& b) {
    if (b.allocation) { vmaDestroyBuffer(c.allocator, b.buffer, b.allocation); b.allocation = nullptr; }
}
} // namespace

BatchBuffer::BatchBuffer(core::VulkanContext const& c, std::uint32_t windows, std::uint32_t tokens)
    : context_(c), engine_(c, 1, 1), windowCount_(windows), windowTokens_(tokens) {
    if (windows == 0 || tokens == 0) throw std::invalid_argument("dataset batch dimensions must be non-zero");
    vk::DeviceSize const tokenBytes = vk::DeviceSize(windows) * tokens * sizeof(std::uint32_t);
    vk::DeviceSize const controlBytes = vk::DeviceSize(windows) * sizeof(std::uint32_t);
    vk::DeviceSize const resultBytes = vk::DeviceSize(windows) * 2 * sizeof(float);
    vk::DeviceSize const stagingBytes = tokenBytes * 3 + controlBytes;
    auto device_buffer = [&](vk::DeviceSize bytes) {
        auto out = std::make_unique<Buffer>();
        auto raw = make_buffer(context_, bytes, vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        out->buffer = raw.buffer; out->allocation = raw.allocation; return out;
    };
    tokens_ = device_buffer(tokenBytes); targets_ = device_buffer(tokenBytes); masks_ = device_buffer(tokenBytes);
    controls_ = device_buffer(controlBytes); results_ = device_buffer(resultBytes);
    staging_ = std::make_unique<Buffer>();
    auto staging = make_buffer(context_, stagingBytes + resultBytes,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    staging_->buffer = staging.buffer; staging_->allocation = staging.allocation;

    auto blob = shaders::get("dataset_batch_buffer.spv");
    if (!blob.data) throw std::runtime_error("dataset batch shader missing");
    shader_ = context_.device.createShaderModule(vk::ShaderModuleCreateInfo{}.setCodeSize(blob.size)
        .setPCode(reinterpret_cast<std::uint32_t const*>(blob.data)));
    std::array<vk::DescriptorSetLayoutBinding, 5> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i)
        bindings[i] = vk::DescriptorSetLayoutBinding{}.setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
    descriptorLayout_ = context_.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(bindings));
    vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute, 0, 2 * sizeof(std::uint32_t)};
    pipelineLayout_ = context_.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{}.setSetLayouts(descriptorLayout_).setPushConstantRanges(push));
    auto pipe = context_.device.createComputePipeline({}, vk::ComputePipelineCreateInfo{}.setStage(
        vk::PipelineShaderStageCreateInfo{}.setStage(vk::ShaderStageFlagBits::eCompute).setModule(shader_).setPName("main"))
        .setLayout(pipelineLayout_));
    if (pipe.result != vk::Result::eSuccess) throw std::runtime_error("dataset batch pipeline failed");
    pipeline_ = pipe.value;
    descriptorPool_ = context_.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(
        vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(5)));
    descriptorSet_ = context_.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}.setDescriptorPool(descriptorPool_)
        .setSetLayouts(descriptorLayout_)).front();
    std::array<vk::Buffer, 5> buffers{tokens_->buffer, targets_->buffer, masks_->buffer, controls_->buffer, results_->buffer};
    std::array<vk::DescriptorBufferInfo, 5> infos{}; std::array<vk::WriteDescriptorSet, 5> writes{};
    for (std::uint32_t i = 0; i < buffers.size(); ++i) {
        infos[i] = vk::DescriptorBufferInfo{}.setBuffer(buffers[i]).setRange(VK_WHOLE_SIZE);
        writes[i] = vk::WriteDescriptorSet{}.setDstSet(descriptorSet_).setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(infos[i]);
    }
    context_.device.updateDescriptorSets(writes, {});
}

BatchBuffer::DeviceView BatchBuffer::device_view() const noexcept {
    return {tokens_->buffer, targets_->buffer, masks_->buffer, controls_->buffer, results_->buffer,
            windowCount_, windowTokens_};
}

BatchBuffer::~BatchBuffer() {
    if (!context_.device) return;
    engine_.drain();
    destroy(context_, *staging_); destroy(context_, *results_); destroy(context_, *controls_);
    destroy(context_, *masks_); destroy(context_, *targets_); destroy(context_, *tokens_);
    context_.device.destroyDescriptorPool(descriptorPool_); context_.device.destroyPipeline(pipeline_);
    context_.device.destroyPipelineLayout(pipelineLayout_); context_.device.destroyDescriptorSetLayout(descriptorLayout_);
    context_.device.destroyShaderModule(shader_);
}

void BatchBuffer::upload(std::vector<std::uint32_t> const& tokens, std::vector<std::uint32_t> const& targets,
                         std::vector<std::uint32_t> const& masks, std::vector<std::uint32_t> const& controls) {
    std::size_t const elements = std::size_t(windowCount_) * windowTokens_;
    if (tokens.size() != elements || targets.size() != elements || masks.size() != elements || controls.size() != windowCount_)
        throw std::invalid_argument("dataset batch input shape mismatch");
    vk::DeviceSize const tokenBytes = vk::DeviceSize(elements) * sizeof(std::uint32_t);
    vk::DeviceSize const controlBytes = vk::DeviceSize(windowCount_) * sizeof(std::uint32_t);
    vk::DeviceSize const targetOffset = tokenBytes, maskOffset = tokenBytes * 2, controlOffset = tokenBytes * 3;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, staging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("dataset staging map failed");
    auto* bytes = static_cast<std::byte*>(mapped);
    std::memcpy(bytes, tokens.data(), tokenBytes);
    std::memcpy(bytes + targetOffset, targets.data(), tokenBytes);
    std::memcpy(bytes + maskOffset, masks.data(), tokenBytes);
    std::memcpy(bytes + controlOffset, controls.data(), controlBytes);
    vmaFlushAllocation(context_.allocator, staging_->allocation, 0, controlOffset + controlBytes);
    vmaUnmapMemory(context_.allocator, staging_->allocation);
    lastSubmission_ = engine_.submit([this, tokenBytes, controlBytes, targetOffset, maskOffset, controlOffset](vk::CommandBuffer cmd) {
        auto copy = [&](vk::DeviceSize src, vk::Buffer dst, vk::DeviceSize size) {
            cmd.copyBuffer(staging_->buffer, dst, vk::BufferCopy{}.setSrcOffset(src).setSize(size));
        };
        copy(0, tokens_->buffer, tokenBytes);
        copy(targetOffset, targets_->buffer, tokenBytes);
        copy(maskOffset, masks_->buffer, tokenBytes);
        copy(controlOffset, controls_->buffer, controlBytes);
    });
    engine_.wait(lastSubmission_);
}

std::vector<float> BatchBuffer::metrics_readback() {
    // The standalone dataset shader already computes these exact read-only
    // values; keep this named API separate from Tiny model evaluation.
    return process_readback();
}

std::vector<float> BatchBuffer::process_readback() {
    vk::DeviceSize const tokenBytes = vk::DeviceSize(windowCount_) * windowTokens_ * sizeof(std::uint32_t);
    vk::DeviceSize const controlBytes = vk::DeviceSize(windowCount_) * sizeof(std::uint32_t);
    vk::DeviceSize const resultBytes = vk::DeviceSize(windowCount_) * 2 * sizeof(float);
    vk::DeviceSize const resultOffset = tokenBytes * 3 + controlBytes;
    lastSubmission_ = engine_.submit([this, resultBytes, resultOffset](vk::CommandBuffer cmd) {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
            vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        std::array<std::uint32_t, 2> push{windowTokens_, windowCount_};
        cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
        cmd.dispatch(windowCount_, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {},
            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead), {}, {});
        cmd.copyBuffer(results_->buffer, staging_->buffer, vk::BufferCopy{}.setDstOffset(resultOffset).setSize(resultBytes));
    });
    engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, staging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("dataset readback map failed");
    vmaInvalidateAllocation(context_.allocator, staging_->allocation, resultOffset, resultBytes);
    std::vector<float> output(windowCount_ * 2);
    std::memcpy(output.data(), static_cast<std::byte*>(mapped) + resultOffset, resultBytes);
    vmaUnmapMemory(context_.allocator, staging_->allocation);
    return output;
}

std::vector<float> BatchBuffer::process(std::vector<std::uint32_t> const& tokens, std::vector<std::uint32_t> const& targets,
                                        std::vector<std::uint32_t> const& masks, std::vector<std::uint32_t> const& controls) {
    std::size_t const elements = std::size_t(windowCount_) * windowTokens_;
    if (tokens.size() != elements || targets.size() != elements || masks.size() != elements || controls.size() != windowCount_)
        throw std::invalid_argument("dataset batch input shape mismatch");
    vk::DeviceSize const tokenBytes = vk::DeviceSize(elements) * sizeof(std::uint32_t);
    vk::DeviceSize const controlBytes = vk::DeviceSize(windowCount_) * sizeof(std::uint32_t);
    vk::DeviceSize const resultBytes = vk::DeviceSize(windowCount_) * 2 * sizeof(float);
    vk::DeviceSize const targetOffset = tokenBytes, maskOffset = tokenBytes * 2, controlOffset = tokenBytes * 3;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, staging_->allocation, &mapped) != VK_SUCCESS) throw std::runtime_error("dataset staging map failed");
    auto* bytes = static_cast<std::byte*>(mapped);
    std::memcpy(bytes, tokens.data(), tokenBytes); std::memcpy(bytes + targetOffset, targets.data(), tokenBytes);
    std::memcpy(bytes + maskOffset, masks.data(), tokenBytes); std::memcpy(bytes + controlOffset, controls.data(), controlBytes);
    vmaFlushAllocation(context_.allocator, staging_->allocation, 0, controlOffset + controlBytes);
    vmaUnmapMemory(context_.allocator, staging_->allocation);
    lastSubmission_ = engine_.submit([this, tokenBytes, controlBytes, resultBytes, targetOffset, maskOffset, controlOffset](vk::CommandBuffer cmd) {
        auto copy = [&](vk::DeviceSize src, vk::Buffer dst, vk::DeviceSize size) { cmd.copyBuffer(staging_->buffer, dst, vk::BufferCopy{}.setSrcOffset(src).setSize(size)); };
        copy(0, tokens_->buffer, tokenBytes); copy(targetOffset, targets_->buffer, tokenBytes); copy(maskOffset, masks_->buffer, tokenBytes); copy(controlOffset, controls_->buffer, controlBytes);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
            vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_); cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        std::array<std::uint32_t, 2> push{windowTokens_, windowCount_};
        cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
        cmd.dispatch(windowCount_, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {},
            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead), {}, {});
        cmd.copyBuffer(results_->buffer, staging_->buffer, vk::BufferCopy{}.setDstOffset(tokenBytes * 3 + controlBytes).setSize(resultBytes));
    });
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, staging_->allocation, &mapped) != VK_SUCCESS) throw std::runtime_error("dataset readback map failed");
    auto* read = static_cast<std::byte*>(mapped) + tokenBytes * 3 + controlBytes;
    vmaInvalidateAllocation(context_.allocator, staging_->allocation, tokenBytes * 3 + controlBytes, resultBytes);
    std::vector<float> output(windowCount_ * 2); std::memcpy(output.data(), read, resultBytes); vmaUnmapMemory(context_.allocator, staging_->allocation);
    return output;
}

} // namespace vulkan_runtime::dataset

extern "C" const char* vulkan_runtime_dataset_batch_capability() { return vulkan_runtime::dataset::BatchBuffer::capability(); }
extern "C" void* vulkan_runtime_dataset_batch_create(vulkan_runtime::core::VulkanContext const* c, std::uint32_t w, std::uint32_t t) {
    try { return new vulkan_runtime::dataset::BatchBuffer(*c, w, t); } catch (...) { return nullptr; }
}
extern "C" void vulkan_runtime_dataset_batch_destroy(void* p) { delete static_cast<vulkan_runtime::dataset::BatchBuffer*>(p); }
extern "C" int vulkan_runtime_dataset_batch_metrics(void* p, float* out) {
    if (!p || !out) return -1;
    try {
        auto* b = static_cast<vulkan_runtime::dataset::BatchBuffer*>(p);
        auto r = b->metrics_readback();
        std::memcpy(out, r.data(), r.size() * sizeof(float));
        return 0;
    } catch (...) { return -1; }
}
extern "C" int vulkan_runtime_dataset_batch_process(void* p, std::uint32_t const* t, std::uint32_t const* y, std::uint32_t const* m, std::uint32_t const* c, float* out) {
    if (!p || !t || !y || !m || !c || !out) return -1;
    try { auto* b = static_cast<vulkan_runtime::dataset::BatchBuffer*>(p); std::size_t n = std::size_t(b->window_count()) * b->window_tokens(); auto r = b->process({t, t+n}, {y, y+n}, {m, m+n}, {c, c+b->window_count()}); std::memcpy(out, r.data(), r.size()*sizeof(float)); return 0; } catch (...) { return -1; }
}
