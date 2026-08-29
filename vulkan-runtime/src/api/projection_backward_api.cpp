#include "api/projection_backward_api.h"
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
    std::uint32_t rows, input_size, output_size, accumulate;
};
} // namespace
extern "C" int spaceslug_projection_backward(float const* dy,
                                             float const* w,
                                             float* dx,
                                             std::uint32_t rows,
                                             std::uint32_t input_size,
                                             std::uint32_t output_size) {
    if (!dy || !w || !dx || !rows || rows > 128 || !input_size || input_size > 64 || !output_size || output_size > 64)
        return 1;
    try {
        auto c = vulkan_runtime::core::create_context("projection-backward");
        vk::DeviceSize dyb = vk::DeviceSize(rows) * output_size * sizeof(float),
                       wb = vk::DeviceSize(input_size) * output_size * sizeof(float),
                       dxb = vk::DeviceSize(rows) * input_size * sizeof(float);
        Buffer a = make(c.allocator,
                        dyb,
                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               b = make(c.allocator,
                        wb,
                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               o = make(c.allocator,
                        dxb,
                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE),
               st = make(c.allocator,
                         dyb + wb + dxb,
                         vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        void* m{};
        if (vmaMapMemory(c.allocator, st.allocation, &m) != VK_SUCCESS)
            throw std::runtime_error("map");
        auto* x = static_cast<char*>(m);
        std::memcpy(x, dy, dyb);
        std::memcpy(x + dyb, w, wb);
        vmaFlushAllocation(c.allocator, st.allocation, 0, dyb + wb);
        vmaUnmapMemory(c.allocator, st.allocation);
        auto z = vulkan_runtime::shaders::get("projection_backward.spv");
        vk::ShaderModuleCreateInfo mi;
        mi.setCodeSize(z.size).setPCode(reinterpret_cast<std::uint32_t const*>(z.data));
        auto sm = c.device.createShaderModule(mi);
        std::array<vk::DescriptorSetLayoutBinding, 3> bs{};
        for (std::uint32_t i = 0; i < 3; ++i)
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
        ps.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(3);
        vk::DescriptorPoolCreateInfo dpi;
        dpi.setMaxSets(1).setPoolSizes(ps);
        auto pool = c.device.createDescriptorPool(dpi);
        vk::DescriptorSetAllocateInfo dai;
        dai.setDescriptorPool(pool).setSetLayouts(sl);
        auto set = c.device.allocateDescriptorSets(dai).front();
        std::array<vk::DescriptorBufferInfo, 3> bi{};
        for (auto& i : bi)
            i.setRange(VK_WHOLE_SIZE);
        bi[0].setBuffer(a.buffer);
        bi[1].setBuffer(b.buffer);
        bi[2].setBuffer(o.buffer);
        std::array<vk::WriteDescriptorSet, 3> ws{};
        for (std::uint32_t i = 0; i < 3; ++i)
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
        vk::BufferCopy c0;
        c0.setSize(dyb);
        cmd.copyBuffer(st.buffer, a.buffer, c0);
        vk::BufferCopy c1;
        c1.setSrcOffset(dyb).setSize(wb);
        cmd.copyBuffer(st.buffer, b.buffer, c1);
        vk::MemoryBarrier ib;
        ib.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, ib, {}, {});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pl, 0, set, {});
        PC pc{rows, input_size, output_size, 0};
        cmd.pushConstants(pl, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        cmd.dispatch((rows * input_size + 255) / 256, 1, 1);
        vk::MemoryBarrier ob;
        ob.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, ob, {}, {});
        vk::BufferCopy back;
        back.setDstOffset(dyb + wb).setSize(dxb);
        cmd.copyBuffer(o.buffer, st.buffer, back);
        cmd.end();
        vk::SubmitInfo si;
        si.setCommandBuffers(cmd);
        c.computeQueue.submit(si);
        c.computeQueue.waitIdle();
        if (vmaMapMemory(c.allocator, st.allocation, &m) != VK_SUCCESS)
            throw std::runtime_error("map output");
        vmaInvalidateAllocation(c.allocator, st.allocation, dyb + wb, dxb);
        std::memcpy(dx, static_cast<char*>(m) + dyb + wb, dxb);
        vmaUnmapMemory(c.allocator, st.allocation);
        c.device.destroyCommandPool(cp);
        c.device.destroyDescriptorPool(pool);
        c.device.destroyPipeline(pipe);
        c.device.destroyPipelineLayout(pl);
        c.device.destroyDescriptorSetLayout(sl);
        c.device.destroyShaderModule(sm);
        drop(c.allocator, st);
        drop(c.allocator, o);
        drop(c.allocator, b);
        drop(c.allocator, a);
        vulkan_runtime::core::destroy_context(c);
        return 0;
    } catch (...) {
        return 2;
    }
}
