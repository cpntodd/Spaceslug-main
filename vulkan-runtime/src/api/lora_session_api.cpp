#include "api/lora_session_api.h"

#include "core/vk_setup.h"
#include "embedded_shaders.hpp"
#include "exec/engine.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
};

Buffer make_buffer(VmaAllocator allocator,
                   vk::DeviceSize size,
                   vk::BufferUsageFlags usage,
                   VmaMemoryUsage memory,
                   VmaAllocationCreateFlags flags = {}) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = static_cast<VkBufferUsageFlags>(usage);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = memory;
    alloc_info.flags = flags;
    VkBuffer raw{};
    VmaAllocation allocation{};
    if (vmaCreateBuffer(allocator, &info, &alloc_info, &raw, &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateBuffer failed");
    }
    return {vk::Buffer(raw), allocation};
}

void drop_buffer(VmaAllocator allocator, Buffer& buffer) {
    if (buffer.allocation != nullptr) {
        vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer.buffer), buffer.allocation);
        buffer.buffer = nullptr;
        buffer.allocation = nullptr;
    }
}

struct PushDelta {
    uint32_t m;
    uint32_t rank;
};
struct PushSgd {
    uint32_t rank;
    float learning_rate;
};

struct Pipeline {
    vk::ShaderModule shader{};
    vk::DescriptorSetLayout layout{};
    vk::PipelineLayout pipeline_layout{};
    vk::Pipeline pipeline{};
    vk::DescriptorPool pool{};
    vk::DescriptorSet set{};
};

} // namespace

struct spaceslug_lora_session {
    vulkan_runtime::core::VulkanContext context;
    vulkan_runtime::exec::ExecEngine engine;
    uint32_t m;
    uint32_t rank;
    float learning_rate;
    vk::DeviceSize x_bytes;
    vk::DeviceSize adapter_bytes;
    Buffer x;
    Buffer dy;
    Buffer a;
    Buffer b;
    Buffer da;
    Buffer db;
    Buffer y;
    Buffer staging;
    void* staging_mapped{nullptr};
    Pipeline delta;
    Pipeline gradients;
    Pipeline sgd;

    spaceslug_lora_session(vulkan_runtime::core::VulkanContext&& ctx, uint32_t rows, uint32_t r, float lr)
        : context(std::move(ctx)), engine(context, 3, 1), m(rows), rank(r), learning_rate(lr),
          x_bytes(vk::DeviceSize(rows) * 64 * sizeof(float)), adapter_bytes(vk::DeviceSize(r) * 64 * sizeof(float)) {}
};

namespace {

Pipeline make_pipeline(spaceslug_lora_session& s,
                       char const* shader_name,
                       uint32_t binding_count,
                       vk::ArrayProxyNoTemporaries<vk::DescriptorBufferInfo const> infos,
                       uint32_t push_size) {
    auto blob = vulkan_runtime::shaders::get(shader_name);
    vk::ShaderModuleCreateInfo module_info;
    module_info.setCodeSize(blob.size).setPCode(reinterpret_cast<uint32_t const*>(blob.data));
    Pipeline p;
    p.shader = s.context.device.createShaderModule(module_info);

    std::vector<vk::DescriptorSetLayoutBinding> bindings(binding_count);
    for (uint32_t i = 0; i < binding_count; ++i) {
        bindings[i]
            .setBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    }
    vk::DescriptorSetLayoutCreateInfo layout_info;
    layout_info.setBindings(bindings);
    p.layout = s.context.device.createDescriptorSetLayout(layout_info);

    vk::PushConstantRange range;
    range.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(push_size);
    vk::PipelineLayoutCreateInfo pipeline_layout_info;
    pipeline_layout_info.setSetLayouts(p.layout).setPushConstantRanges(range);
    p.pipeline_layout = s.context.device.createPipelineLayout(pipeline_layout_info);

    vk::PipelineShaderStageCreateInfo stage;
    stage.setStage(vk::ShaderStageFlagBits::eCompute).setModule(p.shader).setPName("main");
    vk::ComputePipelineCreateInfo pipeline_info;
    pipeline_info.setStage(stage).setLayout(p.pipeline_layout);
    auto result = s.context.device.createComputePipeline({}, pipeline_info);
    if (result.result != vk::Result::eSuccess)
        throw std::runtime_error("createComputePipeline failed");
    p.pipeline = result.value;

    vk::DescriptorPoolSize pool_size;
    pool_size.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(binding_count);
    vk::DescriptorPoolCreateInfo pool_info;
    pool_info.setMaxSets(1).setPoolSizes(pool_size);
    p.pool = s.context.device.createDescriptorPool(pool_info);
    vk::DescriptorSetAllocateInfo alloc_info;
    alloc_info.setDescriptorPool(p.pool).setSetLayouts(p.layout);
    p.set = s.context.device.allocateDescriptorSets(alloc_info).front();

    std::vector<vk::WriteDescriptorSet> writes(binding_count);
    for (uint32_t i = 0; i < binding_count; ++i) {
        writes[i]
            .setDstSet(p.set)
            .setDstBinding(i)
            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
            .setDescriptorCount(1)
            .setBufferInfo(infos.data()[i]);
    }
    s.context.device.updateDescriptorSets(writes, {});
    return p;
}

void destroy_pipeline(vk::Device device, Pipeline& p) {
    if (p.pool)
        device.destroyDescriptorPool(p.pool);
    if (p.pipeline)
        device.destroyPipeline(p.pipeline);
    if (p.pipeline_layout)
        device.destroyPipelineLayout(p.pipeline_layout);
    if (p.layout)
        device.destroyDescriptorSetLayout(p.layout);
    if (p.shader)
        device.destroyShaderModule(p.shader);
    p = {};
}

void copy_to_staging(spaceslug_lora_session& s, float const* a, float const* b) {
    auto* bytes = static_cast<char*>(s.staging_mapped);
    std::memcpy(bytes, a, s.adapter_bytes);
    std::memcpy(bytes + s.adapter_bytes, b, s.adapter_bytes);
    vmaFlushAllocation(s.context.allocator, s.staging.allocation, 0, 2 * s.adapter_bytes);
}

void record_copy(vk::CommandBuffer cmd,
                 vk::Buffer src,
                 vk::Buffer dst,
                 vk::DeviceSize src_offset,
                 vk::DeviceSize dst_offset,
                 vk::DeviceSize size) {
    vk::BufferCopy copy;
    copy.setSrcOffset(src_offset).setDstOffset(dst_offset).setSize(size);
    cmd.copyBuffer(src, dst, copy);
}

} // namespace

extern "C" int spaceslug_lora_session_create(uint32_t m,
                                             uint32_t rank,
                                             float learning_rate,
                                             float const* a,
                                             float const* b,
                                             spaceslug_lora_session** out_session) {
    if (!a || !b || !out_session || m == 0 || m > 128 || rank == 0 || rank > 8 || !std::isfinite(learning_rate) ||
        learning_rate <= 0.0f)
        return 1;
    *out_session = nullptr;
    try {
        auto context = vulkan_runtime::core::create_context("spaceslug-lora-session");
        auto session = std::make_unique<spaceslug_lora_session>(std::move(context), m, rank, learning_rate);
        auto& c = session->context;
        auto storage = vk::BufferUsageFlagBits::eStorageBuffer;
        session->x = make_buffer(c.allocator,
                                 session->x_bytes,
                                 storage | vk::BufferUsageFlagBits::eTransferDst,
                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        session->dy = make_buffer(c.allocator,
                                  session->x_bytes,
                                  storage | vk::BufferUsageFlagBits::eTransferDst,
                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        session->a =
            make_buffer(c.allocator,
                        session->adapter_bytes,
                        storage | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        session->b =
            make_buffer(c.allocator,
                        session->adapter_bytes,
                        storage | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        session->da = make_buffer(c.allocator, session->adapter_bytes, storage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        session->db = make_buffer(c.allocator, session->adapter_bytes, storage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        session->y = make_buffer(c.allocator,
                                 session->x_bytes,
                                 storage | vk::BufferUsageFlagBits::eTransferSrc,
                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        session->staging = make_buffer(c.allocator,
                                       4 * session->adapter_bytes + 2 * session->x_bytes,
                                       vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                       VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        if (vmaMapMemory(c.allocator, session->staging.allocation, &session->staging_mapped) != VK_SUCCESS)
            throw std::runtime_error("vmaMapMemory failed");
        auto range = [](vk::Buffer buffer) {
            vk::DescriptorBufferInfo i;
            i.setBuffer(buffer).setOffset(0).setRange(VK_WHOLE_SIZE);
            return i;
        };
        std::array<vk::DescriptorBufferInfo, 4> delta_infos{
            range(session->x.buffer), range(session->a.buffer), range(session->b.buffer), range(session->y.buffer)};
        std::array<vk::DescriptorBufferInfo, 6> gradient_infos{range(session->x.buffer),
                                                               range(session->dy.buffer),
                                                               range(session->a.buffer),
                                                               range(session->b.buffer),
                                                               range(session->da.buffer),
                                                               range(session->db.buffer)};
        std::array<vk::DescriptorBufferInfo, 4> sgd_infos{
            range(session->a.buffer), range(session->b.buffer), range(session->da.buffer), range(session->db.buffer)};
        session->delta = make_pipeline(*session, "lora_delta.spv", 4, delta_infos, sizeof(PushDelta));
        session->gradients = make_pipeline(*session, "lora_gradients.spv", 6, gradient_infos, sizeof(PushDelta));
        session->sgd = make_pipeline(*session, "lora_sgd.spv", 4, sgd_infos, sizeof(PushSgd));
        copy_to_staging(*session, a, b);
        auto upload = session->engine.submit([&](vk::CommandBuffer cmd) {
            record_copy(cmd, session->staging.buffer, session->a.buffer, 0, 0, session->adapter_bytes);
            record_copy(
                cmd, session->staging.buffer, session->b.buffer, session->adapter_bytes, 0, session->adapter_bytes);
            vk::MemoryBarrier barrier;
            barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, barrier, {}, {});
        });
        session->engine.wait(upload);
        *out_session = session.release();
        return 0;
    } catch (...) {
        return 2;
    }
}

extern "C" int spaceslug_lora_session_step(spaceslug_lora_session* s, float const* x, float const* dy, float* y_out) {
    if (!s || !x || !dy)
        return 1;
    try {
        auto* bytes = static_cast<char*>(s->staging_mapped);
        std::memcpy(bytes, x, s->x_bytes);
        std::memcpy(bytes + s->x_bytes, dy, s->x_bytes);
        vmaFlushAllocation(s->context.allocator, s->staging.allocation, 0, 2 * s->x_bytes);
        auto value = s->engine.submit([&](vk::CommandBuffer cmd) {
            record_copy(cmd, s->staging.buffer, s->x.buffer, 0, 0, s->x_bytes);
            record_copy(cmd, s->staging.buffer, s->dy.buffer, s->x_bytes, 0, s->x_bytes);
            vk::MemoryBarrier upload_barrier;
            upload_barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eShaderRead);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                upload_barrier,
                                {},
                                {});
            PushDelta delta_pc{s->m, s->rank};
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, s->delta.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, s->delta.pipeline_layout, 0, s->delta.set, {});
            cmd.pushConstants(
                s->delta.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(delta_pc), &delta_pc);
            cmd.dispatch((s->m * 64 + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, s->gradients.pipeline);
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, s->gradients.pipeline_layout, 0, s->gradients.set, {});
            cmd.pushConstants(
                s->gradients.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(delta_pc), &delta_pc);
            cmd.dispatch((64 * s->rank + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            PushSgd sgd_pc{s->rank, s->learning_rate};
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, s->sgd.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, s->sgd.pipeline_layout, 0, s->sgd.set, {});
            cmd.pushConstants(s->sgd.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(sgd_pc), &sgd_pc);
            cmd.dispatch((64 * s->rank + 255) / 256, 1, 1);
            vk::MemoryBarrier output_barrier;
            output_barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                output_barrier,
                                {},
                                {});
            if (y_out)
                record_copy(cmd, s->y.buffer, s->staging.buffer, 0, 2 * s->x_bytes, s->x_bytes);
        });
        s->engine.wait(value);
        if (y_out) {
            vmaInvalidateAllocation(s->context.allocator, s->staging.allocation, 2 * s->x_bytes, s->x_bytes);
            std::memcpy(y_out, bytes + 2 * s->x_bytes, s->x_bytes);
        }
        return 0;
    } catch (...) {
        return 2;
    }
}

extern "C" int spaceslug_lora_session_token_step(spaceslug_lora_session* session,
                                                 uint32_t const* tokens,
                                                 uint32_t const* targets,
                                                 uint32_t const* mask,
                                                 uint32_t rows,
                                                 float* loss) {
    (void)session;
    (void)tokens;
    (void)targets;
    (void)mask;
    (void)rows;
    (void)loss;
    return 3;
}

extern "C" int spaceslug_lora_session_readback(spaceslug_lora_session* s, float* a_out, float* b_out) {
    if (!s || !a_out || !b_out)
        return 1;
    try {
        vk::DeviceSize const a_offset = 2 * s->adapter_bytes + 2 * s->x_bytes;
        auto value = s->engine.submit([&](vk::CommandBuffer cmd) {
            vk::MemoryBarrier barrier;
            barrier.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, barrier, {}, {});
            record_copy(cmd, s->a.buffer, s->staging.buffer, 0, a_offset, s->adapter_bytes);
            record_copy(cmd, s->b.buffer, s->staging.buffer, 0, a_offset + s->adapter_bytes, s->adapter_bytes);
        });
        s->engine.wait(value);
        vmaInvalidateAllocation(s->context.allocator, s->staging.allocation, a_offset, 2 * s->adapter_bytes);
        auto* bytes = static_cast<char*>(s->staging_mapped);
        std::memcpy(a_out, bytes + a_offset, s->adapter_bytes);
        std::memcpy(b_out, bytes + a_offset + s->adapter_bytes, s->adapter_bytes);
        return 0;
    } catch (...) {
        return 2;
    }
}

extern "C" int spaceslug_lora_session_destroy(spaceslug_lora_session* s) {
    if (!s)
        return 0;
    try {
        s->engine.drain();
        if (s->staging_mapped)
            vmaUnmapMemory(s->context.allocator, s->staging.allocation);
        destroy_pipeline(s->context.device, s->sgd);
        destroy_pipeline(s->context.device, s->gradients);
        destroy_pipeline(s->context.device, s->delta);
        drop_buffer(s->context.allocator, s->staging);
        drop_buffer(s->context.allocator, s->y);
        drop_buffer(s->context.allocator, s->db);
        drop_buffer(s->context.allocator, s->da);
        drop_buffer(s->context.allocator, s->b);
        drop_buffer(s->context.allocator, s->a);
        drop_buffer(s->context.allocator, s->dy);
        drop_buffer(s->context.allocator, s->x);
        delete s;
        return 0;
    } catch (...) {
        return 2;
    }
}
