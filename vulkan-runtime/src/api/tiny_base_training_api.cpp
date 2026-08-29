#include "api/tiny_base_training_api.h"
#include "core/vk_setup.h"
#include "embedded_shaders.hpp"
#include "exec/engine.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace vulkan_runtime::tiny_base {
namespace {
struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
};
Buffer make_buffer(VmaAllocator allocator, vk::DeviceSize size, vk::BufferUsageFlags usage,
                   VmaMemoryUsage memory, VmaAllocationCreateFlags flags = 0) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size;
    info.usage = static_cast<VkBufferUsageFlags>(usage);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo ai{};
    ai.usage = memory;
    ai.flags = flags;
    VkBuffer raw{};
    VmaAllocation allocation{};
    if (vmaCreateBuffer(allocator, &info, &ai, &raw, &allocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("vmaCreateBuffer failed");
    return {vk::Buffer(raw), allocation};
}
void drop(VmaAllocator allocator, Buffer& b) {
    if (b.allocation) {
        vmaDestroyBuffer(allocator, static_cast<VkBuffer>(b.buffer), b.allocation);
        b = {};
    }
}
struct GradPC { std::uint32_t rows, tcap, vocab, padded_vocab, hidden; };
struct SgdPC { std::uint32_t count; float learning_rate; };
struct AdamwPC { std::uint32_t count, step; float learning_rate, beta1, beta2, epsilon, weight_decay; };

vk::Pipeline make_pipeline(vk::Device device, char const* name, vk::ShaderModule& shader,
                           vk::DescriptorSetLayout layout, vk::PipelineLayout& pipeline_layout,
                           std::size_t pc_size) {
    auto blob = shaders::get(name);
    vk::ShaderModuleCreateInfo mi;
    mi.setCodeSize(blob.size).setPCode(reinterpret_cast<std::uint32_t const*>(blob.data));
    shader = device.createShaderModule(mi);
    vk::PushConstantRange range;
    range.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(static_cast<std::uint32_t>(pc_size));
    vk::PipelineLayoutCreateInfo pli;
    pli.setSetLayouts(layout).setPushConstantRanges(range);
    pipeline_layout = device.createPipelineLayout(pli);
    vk::PipelineShaderStageCreateInfo stage;
    stage.setStage(vk::ShaderStageFlagBits::eCompute).setModule(shader).setPName("main");
    vk::ComputePipelineCreateInfo ci;
    ci.setStage(stage).setLayout(pipeline_layout);
    auto result = device.createComputePipeline({}, ci);
    if (result.result != vk::Result::eSuccess) throw std::runtime_error("createComputePipeline failed");
    return result.value;
}
} // namespace

struct Graph {
    core::VulkanContext context;
    std::unique_ptr<exec::ExecEngine> engine;
    std::uint32_t hidden, vocab, padded_vocab, tcap;
    vk::DeviceSize weight_bytes, activation_bytes, logits_bytes;
    Buffer weight, gradient, activation, dlogits, m, v, staging;
    vk::ShaderModule grad_shader{}, sgd_shader{}, adamw_shader{};
    vk::DescriptorSetLayout grad_layout{}, sgd_layout{}, adamw_layout{};
    vk::PipelineLayout grad_pipeline_layout{}, sgd_pipeline_layout{}, adamw_pipeline_layout{};
    vk::Pipeline grad_pipeline{}, sgd_pipeline{}, adamw_pipeline{};
    vk::DescriptorPool descriptor_pool{};
    vk::DescriptorSet grad_set{}, sgd_set{}, adamw_set{};
    std::uint64_t adamw_step{0};

    Graph(float const* initial, std::uint32_t h, std::uint32_t v, std::uint32_t vp, std::uint32_t tc)
        : context(core::create_context("spaceslug-tiny-base-training")), engine(std::make_unique<exec::ExecEngine>(context, 3, 1)),
          hidden(h), vocab(v), padded_vocab(vp), tcap(tc),
          weight_bytes(vk::DeviceSize(h) * vp * sizeof(float)),
          activation_bytes(vk::DeviceSize(tc) * h * sizeof(float)),
          logits_bytes(vk::DeviceSize(tc) * vp * sizeof(float)),
          weight(make_buffer(context.allocator, weight_bytes, vk::BufferUsageFlagBits::eStorageBuffer |
                               vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                             VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),
          gradient(make_buffer(context.allocator, weight_bytes, vk::BufferUsageFlagBits::eStorageBuffer |
                                 vk::BufferUsageFlagBits::eTransferSrc,
                               VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),
          activation(make_buffer(context.allocator, activation_bytes, vk::BufferUsageFlagBits::eStorageBuffer |
                                   vk::BufferUsageFlagBits::eTransferDst,
                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),
          dlogits(make_buffer(context.allocator, logits_bytes, vk::BufferUsageFlagBits::eStorageBuffer |
                                vk::BufferUsageFlagBits::eTransferDst,
                              VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),
          m(make_buffer(context.allocator, weight_bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),
           v(make_buffer(context.allocator, weight_bytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),
           staging(make_buffer(context.allocator, activation_bytes + logits_bytes + weight_bytes * 5 + sizeof(std::uint64_t),
                              vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                              VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)) {
        if (!initial) throw std::runtime_error("null initial weight");
        void* mapped{};
        if (vmaMapMemory(context.allocator, staging.allocation, &mapped) != VK_SUCCESS) throw std::runtime_error("map");
        auto* initial_dst = reinterpret_cast<float*>(static_cast<char*>(mapped) + activation_bytes + logits_bytes);
        for (std::uint32_t hrow = 0; hrow < hidden; ++hrow) {
            std::memcpy(initial_dst + hrow * padded_vocab, initial + hrow * padded_vocab, vocab * sizeof(float));
            std::memset(initial_dst + hrow * padded_vocab + vocab, 0, (padded_vocab - vocab) * sizeof(float));
        }
        vmaFlushAllocation(context.allocator, staging.allocation, activation_bytes + logits_bytes, weight_bytes);
        vmaUnmapMemory(context.allocator, staging.allocation);

        std::array<vk::DescriptorSetLayoutBinding, 3> gb{};
        for (std::uint32_t i = 0; i < 3; ++i)
            gb[i].setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        vk::DescriptorSetLayoutCreateInfo gli; gli.setBindings(gb); grad_layout = context.device.createDescriptorSetLayout(gli);
        std::array<vk::DescriptorSetLayoutBinding, 2> sb{};
        for (std::uint32_t i = 0; i < 2; ++i)
            sb[i].setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        vk::DescriptorSetLayoutCreateInfo sli; sli.setBindings(sb); sgd_layout = context.device.createDescriptorSetLayout(sli);
        std::array<vk::DescriptorSetLayoutBinding, 4> ab{};
        for (std::uint32_t i = 0; i < 4; ++i) ab[i].setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
        vk::DescriptorSetLayoutCreateInfo ali; ali.setBindings(ab); adamw_layout = context.device.createDescriptorSetLayout(ali);
        grad_pipeline = make_pipeline(context.device, "lm_head_weight_grad.spv", grad_shader, grad_layout, grad_pipeline_layout, 24);
        sgd_pipeline = make_pipeline(context.device, "lm_head_sgd.spv", sgd_shader, sgd_layout, sgd_pipeline_layout, 8);
        adamw_pipeline = make_pipeline(context.device, "lm_head_adamw.spv", adamw_shader, adamw_layout, adamw_pipeline_layout, 28);
        std::array<vk::DescriptorPoolSize, 1> sizes{}; sizes[0].setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(9);
        vk::DescriptorPoolCreateInfo pi; pi.setMaxSets(3).setPoolSizes(sizes); descriptor_pool = context.device.createDescriptorPool(pi);
        std::array<vk::DescriptorSetLayout, 3> layouts{grad_layout, sgd_layout, adamw_layout};
        vk::DescriptorSetAllocateInfo ai; ai.setDescriptorPool(descriptor_pool).setSetLayouts(layouts);
        auto sets = context.device.allocateDescriptorSets(ai); grad_set = sets[0]; sgd_set = sets[1]; adamw_set = sets[2];
        std::array<vk::DescriptorBufferInfo, 4> gi{{{activation.buffer, 0, VK_WHOLE_SIZE}, {dlogits.buffer, 0, VK_WHOLE_SIZE}, {gradient.buffer, 0, VK_WHOLE_SIZE}, {m.buffer, 0, VK_WHOLE_SIZE}}};
        std::array<vk::DescriptorBufferInfo, 2> si{{{weight.buffer, 0, VK_WHOLE_SIZE}, {gradient.buffer, 0, VK_WHOLE_SIZE}}};
        std::array<vk::DescriptorBufferInfo, 4> ai_buf{{{weight.buffer, 0, VK_WHOLE_SIZE}, {gradient.buffer, 0, VK_WHOLE_SIZE}, {m.buffer, 0, VK_WHOLE_SIZE}, {this->v.buffer, 0, VK_WHOLE_SIZE}}};
        std::array<vk::WriteDescriptorSet, 9> writes{};
        for (std::uint32_t i = 0; i < 3; ++i) writes[i].setDstSet(grad_set).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(gi[i]);
        for (std::uint32_t i = 0; i < 2; ++i) writes[i + 3].setDstSet(sgd_set).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(si[i]);
        for (std::uint32_t i = 0; i < 4; ++i) writes[i + 5].setDstSet(adamw_set).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(ai_buf[i]);
        context.device.updateDescriptorSets(writes, {});
        auto zero = engine->submit([&](vk::CommandBuffer cmd) { vk::BufferMemoryBarrier b; b.setSrcAccessMask({}).setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite).setBuffer(m.buffer).setOffset(0).setSize(weight_bytes); cmd.fillBuffer(m.buffer, 0, VK_WHOLE_SIZE, 0); cmd.fillBuffer(this->v.buffer, 0, VK_WHOLE_SIZE, 0); }); engine->wait(zero);
        submit_copy_initial();
    }
    ~Graph() {
        engine->drain();
        context.device.destroyDescriptorPool(descriptor_pool);
        context.device.destroyPipeline(adamw_pipeline);
        context.device.destroyPipeline(sgd_pipeline); context.device.destroyPipeline(grad_pipeline);
        context.device.destroyPipelineLayout(adamw_pipeline_layout); context.device.destroyPipelineLayout(sgd_pipeline_layout); context.device.destroyPipelineLayout(grad_pipeline_layout);
        context.device.destroyDescriptorSetLayout(adamw_layout); context.device.destroyDescriptorSetLayout(sgd_layout); context.device.destroyDescriptorSetLayout(grad_layout);
        context.device.destroyShaderModule(adamw_shader); context.device.destroyShaderModule(sgd_shader); context.device.destroyShaderModule(grad_shader);
        engine.reset();
        drop(context.allocator, staging); drop(context.allocator, dlogits); drop(context.allocator, activation);
        drop(context.allocator, v); drop(context.allocator, m); drop(context.allocator, gradient); drop(context.allocator, weight);
        core::destroy_context(context);
    }
    void submit_copy_initial() {
        auto value = engine->submit([&](vk::CommandBuffer cmd) {
            vk::BufferCopy c; c.setSrcOffset(activation_bytes + logits_bytes).setDstOffset(0).setSize(weight_bytes);
            cmd.copyBuffer(staging.buffer, weight.buffer, c);
        }); engine->wait(value);
    }
    int step(float const* a, float const* dl, std::uint32_t rows, float lr) {
        void* mapped{};
        if (vmaMapMemory(context.allocator, staging.allocation, &mapped) != VK_SUCCESS) return 2;
        auto* bytes = static_cast<char*>(mapped);
        std::memcpy(bytes, a, activation_bytes);
        std::memcpy(bytes + activation_bytes, dl, logits_bytes);
        vmaFlushAllocation(context.allocator, staging.allocation, 0, activation_bytes + logits_bytes);
        vmaUnmapMemory(context.allocator, staging.allocation);
        auto value = engine->submit([&](vk::CommandBuffer cmd) {
            vk::BufferCopy ca; ca.setSrcOffset(0).setDstOffset(0).setSize(activation_bytes); cmd.copyBuffer(staging.buffer, activation.buffer, ca);
            vk::BufferCopy cd; cd.setSrcOffset(activation_bytes).setDstOffset(0).setSize(logits_bytes); cmd.copyBuffer(staging.buffer, dlogits.buffer, cd);
            vk::MemoryBarrier in; in.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead); cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, in, {}, {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, grad_pipeline); cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, grad_pipeline_layout, 0, grad_set, {});
            GradPC pc{rows, tcap, vocab, padded_vocab, hidden}; cmd.pushConstants(grad_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc); cmd.dispatch((hidden * padded_vocab + 255) / 256, 1, 1);
            vk::MemoryBarrier grad_bar; grad_bar.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead); cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, grad_bar, {}, {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, sgd_pipeline); cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, sgd_pipeline_layout, 0, sgd_set, {});
            SgdPC sp{hidden * padded_vocab, lr}; cmd.pushConstants(sgd_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(sp), &sp); cmd.dispatch((hidden * padded_vocab + 255) / 256, 1, 1);
            vk::MemoryBarrier out; out.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead); cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, out, {}, {});
            vk::BufferCopy cw; cw.setSrcOffset(0).setDstOffset(activation_bytes + logits_bytes + weight_bytes).setSize(weight_bytes); cmd.copyBuffer(weight.buffer, staging.buffer, cw);
            vk::BufferCopy cg; cg.setSrcOffset(0).setDstOffset(activation_bytes + logits_bytes + weight_bytes * 2).setSize(weight_bytes); cmd.copyBuffer(gradient.buffer, staging.buffer, cg);
        });
        engine->wait(value);
        return 0;
    }
    int adamw_step_run(float const* a, float const* dl, std::uint32_t rows, float lr, float b1, float b2, float eps, float decay) {
         // Standalone C ABI validation is performed by the wrapper; this method
         // retains the same checks for callers inside the translation unit.
         if (!std::isfinite(lr) || !std::isfinite(b1) || !std::isfinite(b2) || !std::isfinite(eps) || !std::isfinite(decay) ||
             lr <= 0.0f || !(b1 >= 0.0f && b1 < 1.0f) || !(b2 >= 0.0f && b2 < 1.0f) || eps <= 0.0f || decay < 0.0f || adamw_step == UINT64_MAX)
             return 1;
         const auto next_adamw_step = adamw_step + 1;
         void* mapped{}; if (vmaMapMemory(context.allocator, staging.allocation, &mapped) != VK_SUCCESS) return 2;
         ++adamw_step;
         auto* bytes = static_cast<char*>(mapped); std::memcpy(bytes, a, activation_bytes); std::memcpy(bytes + activation_bytes, dl, logits_bytes); vmaFlushAllocation(context.allocator, staging.allocation, 0, activation_bytes + logits_bytes); vmaUnmapMemory(context.allocator, staging.allocation);
         auto value = engine->submit([&](vk::CommandBuffer cmd) { vk::BufferCopy ca{0,0,activation_bytes}, cd{activation_bytes,0,logits_bytes}; cmd.copyBuffer(staging.buffer, activation.buffer, ca); cmd.copyBuffer(staging.buffer, dlogits.buffer, cd); vk::MemoryBarrier in; in.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead); cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, in, {}, {}); cmd.bindPipeline(vk::PipelineBindPoint::eCompute, grad_pipeline); cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, grad_pipeline_layout, 0, grad_set, {}); GradPC gp{rows,tcap,vocab,padded_vocab,hidden}; cmd.pushConstants(grad_pipeline_layout,vk::ShaderStageFlagBits::eCompute,0,sizeof(gp),&gp); cmd.dispatch((hidden*padded_vocab+255)/256,1,1); vk::MemoryBarrier bar; bar.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead); cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader,{},bar,{},{}); cmd.bindPipeline(vk::PipelineBindPoint::eCompute, adamw_pipeline); cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, adamw_pipeline_layout, 0, adamw_set, {}); AdamwPC ap{hidden*padded_vocab,static_cast<uint32_t>(next_adamw_step),lr,b1,b2,eps,decay}; cmd.pushConstants(adamw_pipeline_layout,vk::ShaderStageFlagBits::eCompute,0,sizeof(ap),&ap); cmd.dispatch((hidden*padded_vocab+255)/256,1,1); vk::MemoryBarrier out; out.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead); cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, out, {}, {}); vk::BufferCopy cw{0, activation_bytes + logits_bytes + weight_bytes, weight_bytes}, cg{0, activation_bytes + logits_bytes + weight_bytes * 2, weight_bytes}; cmd.copyBuffer(weight.buffer, staging.buffer, cw); cmd.copyBuffer(gradient.buffer, staging.buffer, cg); }); engine->wait(value); return 0;
     }
     int readback(float* w, float* g) {
        void* mapped{}; if (vmaMapMemory(context.allocator, staging.allocation, &mapped) != VK_SUCCESS) return 2;
        vmaInvalidateAllocation(context.allocator, staging.allocation, activation_bytes + logits_bytes + weight_bytes, weight_bytes * 2);
        auto* bytes = static_cast<char*>(mapped); std::memcpy(w, bytes + activation_bytes + logits_bytes + weight_bytes, weight_bytes); std::memcpy(g, bytes + activation_bytes + logits_bytes + weight_bytes * 2, weight_bytes); vmaUnmapMemory(context.allocator, staging.allocation); return 0;
    }
    int update(float const* w) {
        void* mapped{}; if (vmaMapMemory(context.allocator, staging.allocation, &mapped) != VK_SUCCESS) return 2;
        auto* bytes = static_cast<char*>(mapped);
        auto* dst = reinterpret_cast<float*>(bytes + activation_bytes + logits_bytes);
        for (std::uint32_t hrow = 0; hrow < hidden; ++hrow) {
            std::memcpy(dst + hrow * padded_vocab, w + hrow * padded_vocab, vocab * sizeof(float));
            std::memset(dst + hrow * padded_vocab + vocab, 0, (padded_vocab - vocab) * sizeof(float));
        }
        vmaFlushAllocation(context.allocator, staging.allocation, activation_bytes + logits_bytes, weight_bytes); vmaUnmapMemory(context.allocator, staging.allocation);
        auto value = engine->submit([&](vk::CommandBuffer cmd) { vk::BufferCopy c; c.setSrcOffset(activation_bytes + logits_bytes).setDstOffset(0).setSize(weight_bytes); cmd.copyBuffer(staging.buffer, weight.buffer, c); }); engine->wait(value); return 0;
    }
};
} // namespace vulkan_runtime::tiny_base

struct spaceslug_tiny_base_training { vulkan_runtime::tiny_base::Graph* graph; };
extern "C" const char* spaceslug_tiny_base_training_capability() { return "fp32_lm_head_only_sgd_adamw_standalone_caller_supplied_activations_persistent_m_v_step_no_tiny_forward_integration_no_retained_training_no_batch_buffer"; }
extern "C" spaceslug_tiny_base_training* spaceslug_tiny_base_training_create(float const* w, std::uint32_t h, std::uint32_t v, std::uint32_t vp, std::uint32_t tc) {
    if (!w || h == 0 || v == 0 || vp < v || tc == 0 || h > 4096 || vp > 65536 || tc > 4096) return nullptr;
    try { return new spaceslug_tiny_base_training{new vulkan_runtime::tiny_base::Graph(w, h, v, vp, tc)}; } catch (...) { return nullptr; }
}
extern "C" void spaceslug_tiny_base_training_destroy(spaceslug_tiny_base_training* h) { if (h) { delete h->graph; delete h; } }
extern "C" int spaceslug_tiny_base_training_step(spaceslug_tiny_base_training* h, float const* a, float const* dl, std::uint32_t rows, float lr) { if (!h || !h->graph || !a || !dl || rows > h->graph->tcap || !(lr > 0.0f)) return 1; return h->graph->step(a, dl, rows, lr); }
extern "C" int spaceslug_tiny_base_training_adamw_step(spaceslug_tiny_base_training* h, float const* a, float const* dl, std::uint32_t rows, float lr, float b1, float b2, float eps, float decay) { if (!h || !h->graph || !a || !dl || rows == 0 || rows > h->graph->tcap || !std::isfinite(lr) || !std::isfinite(b1) || !std::isfinite(b2) || !std::isfinite(eps) || !std::isfinite(decay) || lr <= 0.0f || b1 < 0.0f || b1 >= 1.0f || b2 < 0.0f || b2 >= 1.0f || eps <= 0.0f || decay < 0.0f) return 1; return h->graph->adamw_step_run(a, dl, rows, lr, b1, b2, eps, decay); }
extern "C" int spaceslug_tiny_base_training_readback_adamw_state(spaceslug_tiny_base_training* h, float* w, float* g, float* m, float* v, std::uint64_t* step) { if (!h || !h->graph || !w || !g || !m || !v || !step) return 1; if (h->graph->readback(w, g) != 0) return 2; void* mapped{}; if (vmaMapMemory(h->graph->context.allocator, h->graph->staging.allocation, &mapped) != VK_SUCCESS) return 2; auto value = h->graph->engine->submit([&](vk::CommandBuffer cmd) { vk::MemoryBarrier b; b.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead); cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, b, {}, {}); vk::BufferCopy cm{0, h->graph->activation_bytes + h->graph->logits_bytes + h->graph->weight_bytes * 3, h->graph->weight_bytes}, cv{0, h->graph->activation_bytes + h->graph->logits_bytes + h->graph->weight_bytes * 4, h->graph->weight_bytes}; cmd.copyBuffer(h->graph->m.buffer, h->graph->staging.buffer, cm); cmd.copyBuffer(h->graph->v.buffer, h->graph->staging.buffer, cv); }); h->graph->engine->wait(value); auto base = h->graph->activation_bytes + h->graph->logits_bytes + h->graph->weight_bytes * 3; vmaInvalidateAllocation(h->graph->context.allocator, h->graph->staging.allocation, base, h->graph->weight_bytes * 2); auto* bytes = static_cast<char*>(mapped); std::memcpy(m, bytes + base, h->graph->weight_bytes); std::memcpy(v, bytes + base + h->graph->weight_bytes, h->graph->weight_bytes); vmaUnmapMemory(h->graph->context.allocator, h->graph->staging.allocation); *step = h->graph->adamw_step; return 0; }
extern "C" int spaceslug_tiny_base_training_update_adamw_state(spaceslug_tiny_base_training* h, float const* w, float const* m, float const* v, std::uint64_t step) { if (!h || !h->graph || !w || !m || !v || step > UINT32_MAX) return 1; auto* g=h->graph; for (std::uint64_t i = 0; i < std::uint64_t(g->hidden) * g->padded_vocab; ++i) if (!std::isfinite(w[i]) || !std::isfinite(m[i]) || !std::isfinite(v[i])) return 1; void* mapped{}; if (vmaMapMemory(g->context.allocator,g->staging.allocation,&mapped)!=VK_SUCCESS) return 2; auto base=g->activation_bytes+g->logits_bytes; auto* bytes=static_cast<char*>(mapped); std::memcpy(bytes+base,w,g->weight_bytes); std::memcpy(bytes+base+g->weight_bytes*3,m,g->weight_bytes); std::memcpy(bytes+base+g->weight_bytes*4,v,g->weight_bytes); vmaFlushAllocation(g->context.allocator,g->staging.allocation,base,g->weight_bytes*5); vmaUnmapMemory(g->context.allocator,g->staging.allocation); auto value=g->engine->submit([&](vk::CommandBuffer cmd){vk::BufferCopy cw{base,0,g->weight_bytes}, cm{base+g->weight_bytes*3,0,g->weight_bytes}, cv{base+g->weight_bytes*4,0,g->weight_bytes}; cmd.copyBuffer(g->staging.buffer,g->weight.buffer,cw); cmd.copyBuffer(g->staging.buffer,g->m.buffer,cm); cmd.copyBuffer(g->staging.buffer,g->v.buffer,cv);}); g->engine->wait(value); g->adamw_step=step; return 0; }
extern "C" int spaceslug_tiny_base_training_readback(spaceslug_tiny_base_training* h, float* w, float* g) { if (!h || !h->graph || !w || !g) return 1; return h->graph->readback(w, g); }
extern "C" int spaceslug_tiny_base_training_update(spaceslug_tiny_base_training* h, float const* w) { if (!h || !h->graph || !w) return 1; return h->graph->update(w); }
