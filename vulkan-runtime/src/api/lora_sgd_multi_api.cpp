#include "api/lora_sgd_multi_api.h"
#include "core/vk_setup.h"
#include "embedded_shaders.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace {
struct Buffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
};
Buffer
make(VmaAllocator a, vk::DeviceSize s, vk::BufferUsageFlags u, VmaMemoryUsage m, VmaAllocationCreateFlags f = 0) {
    VkBufferCreateInfo i{};
    i.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    i.size = s;
    i.usage = static_cast<VkBufferUsageFlags>(u);
    i.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo x{};
    x.usage = m;
    x.flags = f;
    VkBuffer b{};
    VmaAllocation h{};
    if (vmaCreateBuffer(a, &i, &x, &b, &h, nullptr) != VK_SUCCESS)
        throw std::runtime_error("buffer");
    return {vk::Buffer(b), h};
}
void drop(VmaAllocator a, Buffer& b) {
    if (b.allocation) {
        vmaDestroyBuffer(a, static_cast<VkBuffer>(b.buffer), b.allocation);
        b.buffer = nullptr;
        b.allocation = nullptr;
    }
}
struct PC {
    std::uint32_t rank;
    float learning_rate;
};
} // namespace
extern "C" int
spaceslug_lora_sgd_multi(float* a, float* b, float const* da, float const* db, float lr, std::uint32_t rank) {
    if (!a || !b || !da || !db || lr <= 0 || !rank || rank > 8)
        return 1;
    try {
        auto c = vulkan_runtime::core::create_context("lora-sgd-multi");
        vk::DeviceSize ab = vk::DeviceSize(4) * 64 * rank * sizeof(float),
                       bb = vk::DeviceSize(4) * rank * 64 * sizeof(float);
        Buffer abuf = make(c.allocator,
                           ab,
                           vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                               vk::BufferUsageFlagBits::eTransferSrc,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               bbuf = make(c.allocator,
                           bb,
                           vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                               vk::BufferUsageFlagBits::eTransferSrc,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               dabuf = make(c.allocator,
                            ab,
                            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               dbbuf = make(c.allocator,
                            bb,
                            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               st = make(c.allocator,
                         2 * ab + 2 * bb,
                         vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        void* m{};
        if (vmaMapMemory(c.allocator, st.allocation, &m) != VK_SUCCESS)
            throw std::runtime_error("map");
        auto* data = static_cast<char*>(m);
        std::memcpy(data, a, ab);
        std::memcpy(data + ab, b, bb);
        std::memcpy(data + ab + bb, da, ab);
        std::memcpy(data + 2 * ab + bb, db, bb);
        vmaFlushAllocation(c.allocator, st.allocation, 0, 2 * ab + 2 * bb);
        vmaUnmapMemory(c.allocator, st.allocation);
        auto z = vulkan_runtime::shaders::get("lora_sgd_multi.spv");
        vk::ShaderModuleCreateInfo mi;
        mi.setCodeSize(z.size).setPCode(reinterpret_cast<std::uint32_t const*>(z.data));
        auto sm = c.device.createShaderModule(mi);
        std::array<vk::DescriptorSetLayoutBinding, 4> bs{};
        for (std::uint32_t i = 0; i < 4; ++i)
            bs[i]
                .setBinding(i)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        vk::DescriptorSetLayoutCreateInfo li;
        li.setBindings(bs);
        auto sl = c.device.createDescriptorSetLayout(li);
        vk::PushConstantRange p;
        p.setStageFlags(vk::ShaderStageFlagBits::eCompute).setOffset(0).setSize(sizeof(PC));
        vk::PipelineLayoutCreateInfo pli;
        pli.setSetLayouts(sl).setPushConstantRanges(p);
        auto pl = c.device.createPipelineLayout(pli);
        vk::PipelineShaderStageCreateInfo ss;
        ss.setStage(vk::ShaderStageFlagBits::eCompute).setModule(sm).setPName("main");
        vk::ComputePipelineCreateInfo ci;
        ci.setStage(ss).setLayout(pl);
        auto cr = c.device.createComputePipeline({}, ci);
        if (cr.result != vk::Result::eSuccess)
            throw std::runtime_error("pipeline");
        auto pipe = cr.value;
        vk::DescriptorPoolSize ps;
        ps.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(4);
        vk::DescriptorPoolCreateInfo dpi;
        dpi.setMaxSets(1).setPoolSizes(ps);
        auto pool = c.device.createDescriptorPool(dpi);
        vk::DescriptorSetAllocateInfo dai;
        dai.setDescriptorPool(pool).setSetLayouts(sl);
        auto set = c.device.allocateDescriptorSets(dai).front();
        std::array<vk::DescriptorBufferInfo, 4> bi{};
        for (auto& i : bi)
            i.setRange(VK_WHOLE_SIZE);
        bi[0].setBuffer(abuf.buffer);
        bi[1].setBuffer(bbuf.buffer);
        bi[2].setBuffer(dabuf.buffer);
        bi[3].setBuffer(dbbuf.buffer);
        std::array<vk::WriteDescriptorSet, 4> ws{};
        for (std::uint32_t i = 0; i < 4; ++i)
            ws[i]
                .setDstSet(set)
                .setDstBinding(i)
                .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                .setDescriptorCount(1)
                .setBufferInfo(bi[i]);
        c.device.updateDescriptorSets(ws, {});
        vk::CommandPoolCreateInfo cpi;
        cpi.setQueueFamilyIndex(c.computeQueueFamily);
        auto cp = c.device.createCommandPool(cpi);
        vk::CommandBufferAllocateInfo cai;
        cai.setCommandPool(cp).setLevel(vk::CommandBufferLevel::ePrimary).setCommandBufferCount(1);
        auto cmd = c.device.allocateCommandBuffers(cai).front();
        vk::CommandBufferBeginInfo begin{};
        cmd.begin(begin);
        for (auto const& it : std::array<std::pair<vk::DeviceSize, vk::Buffer>, 4>{
                 {{0, abuf.buffer}, {ab, bbuf.buffer}, {ab + bb, dabuf.buffer}, {2 * ab + bb, dbbuf.buffer}}}) {
            vk::BufferCopy bc;
            bc.setSrcOffset(it.first).setSize(it.first == 0 || it.first == ab + bb ? ab : bb);
            cmd.copyBuffer(st.buffer, it.second, bc);
        }
        vk::MemoryBarrier ib;
        ib.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
            .setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, ib, {}, {});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pl, 0, set, {});
        PC pc{rank, lr};
        cmd.pushConstants(pl, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        cmd.dispatch((4 * 64 * rank + 255) / 256, 1, 1);
        vk::MemoryBarrier ob;
        ob.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, ob, {}, {});
        vk::BufferCopy ca;
        ca.setDstOffset(0).setSize(ab);
        cmd.copyBuffer(abuf.buffer, st.buffer, ca);
        vk::BufferCopy cb;
        cb.setDstOffset(ab).setSize(bb);
        cmd.copyBuffer(bbuf.buffer, st.buffer, cb);
        cmd.end();
        vk::SubmitInfo si;
        si.setCommandBuffers(cmd);
        c.computeQueue.submit(si);
        c.computeQueue.waitIdle();
        if (vmaMapMemory(c.allocator, st.allocation, &m) != VK_SUCCESS)
            throw std::runtime_error("map output");
        vmaInvalidateAllocation(c.allocator, st.allocation, 0, ab + bb);
        data = static_cast<char*>(m);
        std::memcpy(a, data, ab);
        std::memcpy(b, data + ab, bb);
        vmaUnmapMemory(c.allocator, st.allocation);
        c.device.destroyCommandPool(cp);
        c.device.destroyDescriptorPool(pool);
        c.device.destroyPipeline(pipe);
        c.device.destroyPipelineLayout(pl);
        c.device.destroyDescriptorSetLayout(sl);
        c.device.destroyShaderModule(sm);
        drop(c.allocator, st);
        drop(c.allocator, dbbuf);
        drop(c.allocator, dabuf);
        drop(c.allocator, bbuf);
        drop(c.allocator, abuf);
        vulkan_runtime::core::destroy_context(c);
        return 0;
    } catch (...) {
        return 2;
    }
}
