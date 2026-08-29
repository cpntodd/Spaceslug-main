#include "api/embedding_training_api.h"
#include "core/vk_setup.h"
#include "embedded_shaders.hpp"
#include "exec/engine.h"
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace emb {
struct B { vk::Buffer b{}; VmaAllocation a{}; };
B make(VmaAllocator al, vk::DeviceSize n, vk::BufferUsageFlags u, VmaMemoryUsage m, VmaAllocationCreateFlags f=0) {
    VkBufferCreateInfo ci{}; ci.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; ci.size=n; ci.usage=static_cast<VkBufferUsageFlags>(u); ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo ai{}; ai.usage=m; ai.flags=f; VkBuffer b{}; VmaAllocation a{};
    if (vmaCreateBuffer(al, &ci, &ai, &b, &a, nullptr) != VK_SUCCESS)
        throw std::runtime_error("embedding buffer");
    return {vk::Buffer(b), a};
}
void drop(VmaAllocator al,B& b){if(b.a){vmaDestroyBuffer(al,static_cast<VkBuffer>(b.b),b.a);b={};}}
struct GradPC { std::uint32_t vocab,hidden,rows,has_mask; };
struct SgdPC { std::uint32_t count; float lr; };
vk::Pipeline pipeline(vk::Device d,char const* name,vk::ShaderModule& sm,vk::DescriptorSetLayout sl,vk::PipelineLayout& pl,std::size_t pcs){
    auto z=vulkan_runtime::shaders::get(name); vk::ShaderModuleCreateInfo mi; mi.setCodeSize(z.size).setPCode(reinterpret_cast<std::uint32_t const*>(z.data)); sm=d.createShaderModule(mi);
    vk::PushConstantRange pc; pc.setStageFlags(vk::ShaderStageFlagBits::eCompute).setSize(static_cast<std::uint32_t>(pcs)); vk::PipelineLayoutCreateInfo li; li.setSetLayouts(sl).setPushConstantRanges(pc); pl=d.createPipelineLayout(li);
    vk::PipelineShaderStageCreateInfo ss; ss.setStage(vk::ShaderStageFlagBits::eCompute).setModule(sm).setPName("main"); vk::ComputePipelineCreateInfo ci; ci.setStage(ss).setLayout(pl); auto r=d.createComputePipeline({},ci); if(r.result!=vk::Result::eSuccess) throw std::runtime_error("embedding pipeline"); return r.value;
}
struct Graph {
    vulkan_runtime::core::VulkanContext c; std::unique_ptr<vulkan_runtime::exec::ExecEngine> e; std::uint32_t v,h; vk::DeviceSize wb,so;
    B ids,ds,mask,w,g,st; vk::ShaderModule gs{},ss{}; vk::DescriptorSetLayout gl{},sl{}; vk::PipelineLayout gpl{},spl{}; vk::Pipeline gp{},sp{}; vk::DescriptorPool dp{}; vk::DescriptorSet gd{},sd{};
    Graph(float const* in,std::uint32_t V,std::uint32_t H):c(vulkan_runtime::core::create_context("standalone-embedding-training")),e(std::make_unique<vulkan_runtime::exec::ExecEngine>(c,3,1)),v(V),h(H),wb(vk::DeviceSize(V)*H*4),so(128*4+vk::DeviceSize(128)*H*4+128*4),ids(make(c.allocator,128*4,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),ds(make(c.allocator,vk::DeviceSize(128)*H*4,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),mask(make(c.allocator,128*4,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),w(make(c.allocator,wb,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),g(make(c.allocator,wb,vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)),st(make(c.allocator,so+3*wb,vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst,VMA_MEMORY_USAGE_AUTO_PREFER_HOST,VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)){
        if (!in) throw std::runtime_error("null embedding weight");
        void* x{};
        if (vmaMapMemory(c.allocator, st.a, &x) != VK_SUCCESS) throw std::runtime_error("embedding map");
        std::memcpy(static_cast<char*>(x) + so, in, wb);
        vmaFlushAllocation(c.allocator, st.a, so, wb);
        vmaUnmapMemory(c.allocator, st.a);
        auto layout=[&](std::uint32_t n){std::vector<vk::DescriptorSetLayoutBinding>b(n);for(std::uint32_t i=0;i<n;++i)b[i].setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);vk::DescriptorSetLayoutCreateInfo q;q.setBindings(b);return c.device.createDescriptorSetLayout(q);}; gl=layout(4);sl=layout(2);gp=pipeline(c.device,"embedding_training_grad.spv",gs,gl,gpl,sizeof(GradPC));sp=pipeline(c.device,"embedding_training_sgd.spv",ss,sl,spl,sizeof(SgdPC));
        vk::DescriptorPoolSize ps;ps.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(6);vk::DescriptorPoolCreateInfo pi;pi.setMaxSets(2).setPoolSizes(ps);dp=c.device.createDescriptorPool(pi);std::array<vk::DescriptorSetLayout,2> ls{gl,sl};vk::DescriptorSetAllocateInfo ai;ai.setDescriptorPool(dp).setSetLayouts(ls);auto sets=c.device.allocateDescriptorSets(ai);gd=sets[0];sd=sets[1];std::array<vk::DescriptorBufferInfo,6> bi{{{ids.b,0,VK_WHOLE_SIZE},{ds.b,0,VK_WHOLE_SIZE},{mask.b,0,VK_WHOLE_SIZE},{g.b,0,VK_WHOLE_SIZE},{w.b,0,VK_WHOLE_SIZE},{g.b,0,VK_WHOLE_SIZE}}};std::array<vk::WriteDescriptorSet,6> wr{};for(std::uint32_t i=0;i<4;++i)wr[i].setDstSet(gd).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(bi[i]);for(std::uint32_t i=0;i<2;++i)wr[i+4].setDstSet(sd).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(bi[i+4]);c.device.updateDescriptorSets(wr,{});auto q=e->submit([&](vk::CommandBuffer cmd){cmd.copyBuffer(st.b,w.b,vk::BufferCopy{so,0,wb});});e->wait(q);
    }
    int step(std::uint32_t const* ti,float const* state,std::uint8_t const* ma,std::uint32_t rows,float lr){void*x{};if(vmaMapMemory(c.allocator,st.a,&x)!=VK_SUCCESS)return 2;auto*p=static_cast<char*>(x);std::memcpy(p,ti,rows*4);std::memcpy(p+128*4,state,vk::DeviceSize(rows)*h*4);std::array<std::uint32_t,128>mw{};for(std::uint32_t i=0;i<rows;++i)mw[i]=ma?ma[i]!=0:1;std::memcpy(p+128*4+vk::DeviceSize(128)*h*4,mw.data(),rows*4);vmaFlushAllocation(c.allocator,st.a,0,so);vmaUnmapMemory(c.allocator,st.a);auto q=e->submit([&](vk::CommandBuffer cmd){cmd.copyBuffer(st.b,ids.b,vk::BufferCopy{0,0,rows*4});cmd.copyBuffer(st.b,ds.b,vk::BufferCopy{128*4,0,vk::DeviceSize(rows)*h*4});cmd.copyBuffer(st.b,mask.b,vk::BufferCopy{128*4+vk::DeviceSize(128)*h*4,0,rows*4});vk::MemoryBarrier a;a.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead);cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,vk::PipelineStageFlagBits::eComputeShader,{},a,{},{});cmd.bindPipeline(vk::PipelineBindPoint::eCompute,gp);cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,gpl,0,gd,{});GradPC pc{v,h,rows,ma?1u:0u};cmd.pushConstants(gpl,vk::ShaderStageFlagBits::eCompute,0,sizeof(pc),&pc);cmd.dispatch((v*h+255)/256,1,1);vk::MemoryBarrier b;b.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead);cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader,{},b,{},{});cmd.bindPipeline(vk::PipelineBindPoint::eCompute,sp);cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,spl,0,sd,{});SgdPC s{v*h,lr};cmd.pushConstants(spl,vk::ShaderStageFlagBits::eCompute,0,sizeof(s),&s);cmd.dispatch((v*h+255)/256,1,1);});e->wait(q);return 0;}
    int read(float*outw,float*outg){auto q=e->submit([&](vk::CommandBuffer cmd){vk::MemoryBarrier b;b.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead);cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eTransfer,{},b,{},{});cmd.copyBuffer(w.b,st.b,vk::BufferCopy{0,so+wb,wb});cmd.copyBuffer(g.b,st.b,vk::BufferCopy{0,so+2*wb,wb});});e->wait(q);void*x{};if(vmaMapMemory(c.allocator,st.a,&x)!=VK_SUCCESS)return 2;vmaInvalidateAllocation(c.allocator,st.a,so+wb,2*wb);std::memcpy(outw,static_cast<char*>(x)+so+wb,wb);std::memcpy(outg,static_cast<char*>(x)+so+2*wb,wb);vmaUnmapMemory(c.allocator,st.a);return 0;}
    int update(float const*in){void*x{};if(vmaMapMemory(c.allocator,st.a,&x)!=VK_SUCCESS)return 2;std::memcpy(static_cast<char*>(x)+so,in,wb);vmaFlushAllocation(c.allocator,st.a,so,wb);vmaUnmapMemory(c.allocator,st.a);auto q=e->submit([&](vk::CommandBuffer cmd){cmd.copyBuffer(st.b,w.b,vk::BufferCopy{so,0,wb});});e->wait(q);return 0;}
    ~Graph(){e->drain();c.device.destroyDescriptorPool(dp);c.device.destroyPipeline(sp);c.device.destroyPipeline(gp);c.device.destroyPipelineLayout(spl);c.device.destroyPipelineLayout(gpl);c.device.destroyDescriptorSetLayout(sl);c.device.destroyDescriptorSetLayout(gl);c.device.destroyShaderModule(ss);c.device.destroyShaderModule(gs);e.reset();drop(c.allocator,st);drop(c.allocator,g);drop(c.allocator,w);drop(c.allocator,mask);drop(c.allocator,ds);drop(c.allocator,ids);vulkan_runtime::core::destroy_context(c);}
};}
struct spaceslug_embedding_training{emb::Graph*g;};
extern "C" const char* spaceslug_embedding_training_capability(){return "standalone_fp32_embedding_training_V259_H64_deterministic_sparse_sgd_no_tiny_graph_integration";}
extern "C" spaceslug_embedding_training* spaceslug_embedding_training_create(float const*w,std::uint32_t v,std::uint32_t h){if(!w||!v||!h||v>259||h>64)return nullptr;try{return new spaceslug_embedding_training{new emb::Graph(w,v,h)};}catch(...){return nullptr;}}
extern "C" void spaceslug_embedding_training_destroy(spaceslug_embedding_training*x){if(x){delete x->g;delete x;}}
extern "C" int spaceslug_embedding_training_step(spaceslug_embedding_training*x,std::uint32_t const*i,float const*s,std::uint8_t const*m,std::uint32_t r,float lr){return(!x||!x->g||!i||!s||r>128||!(lr>0))?1:x->g->step(i,s,m,r,lr);}
extern "C" int spaceslug_embedding_training_readback(spaceslug_embedding_training*x,float*w,float*g){return(!x||!x->g||!w||!g)?1:x->g->read(w,g);}
extern "C" int spaceslug_embedding_training_update(spaceslug_embedding_training*x,float const*w){return(!x||!x->g||!w)?1:x->g->update(w);}