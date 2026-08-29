#include "api/lora_gradients_multi_api.h"
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
    std::uint32_t rows, hidden, rank, target;
};
} // namespace
extern "C" int spaceslug_lora_gradients_multi(float const* x,
                                              float const* dy,
                                              float const* a,
                                              float const* b,
                                              float* da,
                                              float* db,
                                              std::uint32_t rows,
                                              std::uint32_t hidden,
                                              std::uint32_t rank,
                                              std::uint32_t target) {
    if (!x || !dy || !a || !b || !da || !db || !rows || rows > 128 || hidden != 64 || !rank || rank > 8 || target > 3)
        return 1;
    try {
        auto c = vulkan_runtime::core::create_context("lora-gradients-multi");
        vk::DeviceSize xb = vk::DeviceSize(rows) * hidden * sizeof(float),
                       ab = vk::DeviceSize(4) * hidden * rank * sizeof(float),
                       bb = vk::DeviceSize(4) * rank * hidden * sizeof(float);
        Buffer xbuf = make(c.allocator,
                           xb,
                           vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               dybuf = make(c.allocator,
                            xb,
                            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               abuf = make(c.allocator,
                           ab,
                           vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               bbuf = make(c.allocator,
                           bb,
                           vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               dabuf = make(c.allocator,
                            ab,
                            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               dbbuf = make(c.allocator,
                            bb,
                            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               st = make(c.allocator,
                         xb * 2 + 2 * ab + 2 * bb,
                         vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        void* m{};
        if (vmaMapMemory(c.allocator, st.allocation, &m) != VK_SUCCESS)
            throw std::runtime_error("map");
        auto* data = static_cast<char*>(m);
        std::memcpy(data, x, xb);
        std::memcpy(data + xb, dy, xb);
        std::memcpy(data + 2 * xb, a, ab);
        std::memcpy(data + 2 * xb + ab, b, bb);
        vmaFlushAllocation(c.allocator, st.allocation, 0, 2 * xb + ab + bb);
        vmaUnmapMemory(c.allocator, st.allocation);
        auto z = vulkan_runtime::shaders::get("lora_gradients_multi.spv");
        vk::ShaderModuleCreateInfo mi;
        mi.setCodeSize(z.size).setPCode(reinterpret_cast<std::uint32_t const*>(z.data));
        auto sm = c.device.createShaderModule(mi);
        std::array<vk::DescriptorSetLayoutBinding, 6> bs{};
        for (std::uint32_t i = 0; i < 6; ++i)
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
        ps.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(6);
        vk::DescriptorPoolCreateInfo dpi;
        dpi.setMaxSets(1).setPoolSizes(ps);
        auto pool = c.device.createDescriptorPool(dpi);
        vk::DescriptorSetAllocateInfo dai;
        dai.setDescriptorPool(pool).setSetLayouts(sl);
        auto set = c.device.allocateDescriptorSets(dai).front();
        std::array<vk::DescriptorBufferInfo, 6> bi{};
        for (auto& i : bi)
            i.setRange(VK_WHOLE_SIZE);
        bi[0].setBuffer(xbuf.buffer);
        bi[1].setBuffer(dybuf.buffer);
        bi[2].setBuffer(abuf.buffer);
        bi[3].setBuffer(bbuf.buffer);
        bi[4].setBuffer(dabuf.buffer);
        bi[5].setBuffer(dbbuf.buffer);
        std::array<vk::WriteDescriptorSet, 6> ws{};
        for (std::uint32_t i = 0; i < 6; ++i)
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
                 {{0, xbuf.buffer}, {xb, dybuf.buffer}, {2 * xb, abuf.buffer}, {2 * xb + ab, bbuf.buffer}}}) {
            vk::BufferCopy bc;
            bc.setSrcOffset(it.first).setSize(it.first == 2 * xb ? ab : it.first == 2 * xb + ab ? bb : xb);
            cmd.copyBuffer(st.buffer, it.second, bc);
        }
        vk::MemoryBarrier ib;
        ib.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, ib, {}, {});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pl, 0, set, {});
        PC pc{rows, hidden, rank, target};
        cmd.pushConstants(pl, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        cmd.dispatch((hidden * rank + 255) / 256, 1, 1);
        vk::MemoryBarrier ob;
        ob.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, ob, {}, {});
        vk::BufferCopy ca;
        ca.setDstOffset(2 * xb + ab + bb).setSize(ab);
        cmd.copyBuffer(dabuf.buffer, st.buffer, ca);
        vk::BufferCopy cb;
        cb.setDstOffset(2 * xb + 2 * ab + bb).setSize(bb);
        cmd.copyBuffer(dbbuf.buffer, st.buffer, cb);
        cmd.end();
        vk::SubmitInfo si;
        si.setCommandBuffers(cmd);
        c.computeQueue.submit(si);
        c.computeQueue.waitIdle();
        if (vmaMapMemory(c.allocator, st.allocation, &m) != VK_SUCCESS)
            throw std::runtime_error("map output");
        vmaInvalidateAllocation(c.allocator, st.allocation, 2 * xb + ab + bb, ab + bb);
        data = static_cast<char*>(m);
        std::memcpy(da, data + 2 * xb + ab + bb, ab);
        std::memcpy(db, data + 2 * xb + 2 * ab + bb, bb);
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
        drop(c.allocator, dybuf);
        drop(c.allocator, xbuf);
        vulkan_runtime::core::destroy_context(c);
        return 0;
    } catch (...) {
        return 2;
    }
}
