#include "api/tiny_forward_persistent.h"
#include "api/causal_loss_api.h"
#include "api/reduced_precision_storage.h"
#include "embedded_shaders.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vulkan_runtime::tiny {
namespace {
struct TinyBuffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{nullptr};
};
TinyBuffer make_buffer(core::VulkanContext const& c,
                       vk::DeviceSize bytes,
                       vk::BufferUsageFlags usage,
                       VmaMemoryUsage memory,
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
        throw std::runtime_error("vmaCreateBuffer failed");
    return {vk::Buffer(raw), allocation};
}
void drop(core::VulkanContext const& c, TinyBuffer& b) {
    if (b.allocation) {
        vmaDestroyBuffer(c.allocator, b.buffer, b.allocation);
        b.allocation = nullptr;
    }
}
void upload(core::VulkanContext const& c, vk::Buffer dst, void const* data, vk::DeviceSize bytes) {
    TinyBuffer s = make_buffer(c,
                               bytes,
                               vk::BufferUsageFlagBits::eTransferSrc,
                               VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                               VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    void* mapped = nullptr;
    if (vmaMapMemory(c.allocator, s.allocation, &mapped) != VK_SUCCESS) {
        throw std::runtime_error("map failed");
    }
    std::memcpy(mapped, data, static_cast<size_t>(bytes));
    vmaFlushAllocation(c.allocator, s.allocation, 0, bytes);
    vmaUnmapMemory(c.allocator, s.allocation);
    vk::CommandPool pool =
        c.device.createCommandPool(vk::CommandPoolCreateInfo{}.setQueueFamilyIndex(c.computeQueueFamily));
    vk::CommandBuffer cmd = c.device
                                .allocateCommandBuffers(vk::CommandBufferAllocateInfo{}
                                                            .setCommandPool(pool)
                                                            .setLevel(vk::CommandBufferLevel::ePrimary)
                                                            .setCommandBufferCount(1))
                                .front();
    cmd.begin(vk::CommandBufferBeginInfo{});
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                        vk::PipelineStageFlagBits::eTransfer,
                        {},
                        vk::MemoryBarrier(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                              vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                          vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite),
                        {},
                        {});
    cmd.copyBuffer(s.buffer, dst, vk::BufferCopy{}.setSize(bytes));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
                        {},
                        {},
                        vk::BufferMemoryBarrier{}
                            .setBuffer(dst)
                            .setSize(bytes)
                            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                            .setDstAccessMask(vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderRead),
                        {});
    cmd.end();
    c.computeQueue.submit(vk::SubmitInfo{}.setCommandBuffers(cmd));
    c.computeQueue.waitIdle();
    c.device.destroyCommandPool(pool);
    drop(c, s);
}
struct ForwardPC {
    uint32_t seq_length;
    uint32_t final_only;
};
struct LossPC {
    uint32_t rows;
    uint32_t vocab;
    uint32_t logits_stride;
};
struct MetricsPC {
    uint32_t rows;
};
struct LmBackwardPC {
    uint32_t rows, vocab, hidden;
    uint32_t logits_stride, weight_stride, output_stride;
};
struct LmHeadGradientPC {
    uint32_t rows, tcap, vocab, padded_vocab, hidden, accumulate;
};
struct LmHeadSgdPC {
    uint32_t count;
    float learning_rate;
};
struct OutputProjectionGradientPC {
    uint32_t rows, tcap, hidden;
};
struct OutputProjectionSgdPC {
    uint32_t count;
    float learning_rate;
};
struct QkvGradientPC {
    uint32_t rows, tcap, hidden;
};
struct QkvSgdPC {
    uint32_t count;
    float learning_rate;
};
struct EmbeddingGradientPC {
    uint32_t vocab, hidden, rows, has_mask;
};
struct EmbeddingSgdPC {
    uint32_t count;
    float learning_rate;
};
struct PositionGradientPC {
    uint32_t hidden, rows;
};
struct PositionSgdPC {
    uint32_t hidden, rows;
    float learning_rate;
};
struct FFNForwardPC {
    uint32_t rows, hidden, intermediate;
};
struct RmsNormPC { uint32_t rows, hidden; };
struct TinyRmsForwardPC { uint32_t seq_length, final_only, state_only; };
struct FfnAdamwPC {
    uint32_t count, step;
    float learning_rate, beta1, beta2, epsilon, weight_decay;
};
struct FfnGroupPC {
    uint32_t count, step;
    float learning_rate, beta1, beta2, epsilon, weight_decay;
};
struct FfnGradPC {
    uint32_t rows, hidden, intermediate;
};
struct PositionAdamwPC {
    uint32_t count, hidden, step;
    float learning_rate, beta1, beta2, epsilon, weight_decay;
};
struct QkvAdamwPC {
    uint32_t count, hidden, step;
    float learning_rate, beta1, beta2, epsilon, weight_decay;
};
struct LmHeadForwardPC { uint32_t rows, hidden, vocab, stride; };
struct LmHeadAdamwPC {
    uint32_t count, step;
    float learning_rate, beta1, beta2, epsilon, weight_decay;
};
struct OutputAdamwPC {
    uint32_t count, step;
    float learning_rate, beta1, beta2, epsilon, weight_decay;
};
struct AttentionBackwardPC {
    uint32_t tokens, hidden, mode;
};
struct ProjectionBackwardPC {
    uint32_t rows, input_size, output_size, accumulate;
};
struct LoraPC {
    uint32_t rows, hidden, rank, target, accumulate, input_row;
};
struct LoraSgdPC {
    uint32_t rank;
    float learning_rate;
};
struct LoraClearPC {
    uint32_t elements;
};
struct LoraFinalizePC {
    uint32_t rank;
    float learning_rate;
    float normalizer;
};
struct LoraAdamwPC {
    uint32_t rank, step;
    float learning_rate, beta1, beta2, epsilon, weight_decay, normalizer;
};
constexpr vk::DeviceSize lora_a_bytes(std::uint32_t rank) {
    return 4u * H * rank * sizeof(float);
}
constexpr vk::DeviceSize lora_b_bytes(std::uint32_t rank) {
    return 4u * rank * H * sizeof(float);
}
constexpr vk::DeviceSize adamw_state_bytes(std::uint32_t rank) {
    return 3u * lora_a_bytes(rank) + 3u * lora_b_bytes(rank);
}
constexpr vk::DeviceSize adamw_staging_offset(std::uint32_t rank) {
    auto const trainingBytes = Tcap * sizeof(uint32_t) + 8u + H * sizeof(float) + Vp * sizeof(float) + sizeof(float) +
                               (4u * H + Tcap * H + 16u * H * rank) * sizeof(float);
    auto const retainedLossBytes = Tcap * sizeof(uint32_t) * 3 + Tcap * Vp * sizeof(float) + Tcap * sizeof(float);
    return std::max(trainingBytes, retainedLossBytes);
}

vk::Pipeline make_compute_pipeline(core::VulkanContext const& c, vk::ShaderModule shader, vk::PipelineLayout layout) {
    auto result = c.device.createComputePipeline({},
                                                 vk::ComputePipelineCreateInfo{}
                                                     .setStage(vk::PipelineShaderStageCreateInfo{}
                                                                   .setStage(vk::ShaderStageFlagBits::eCompute)
                                                                   .setModule(shader)
                                                                   .setPName("main"))
                                                     .setLayout(layout));
    if (result.result != vk::Result::eSuccess)
        throw std::runtime_error("compute pipeline failed");
    return result.value;
}
std::vector<float> identity() {
    std::vector<float> x(H * H, 0.0f);
    for (uint32_t i = 0; i < H; ++i)
        x[i * H + i] = 1.0f;
    return x;
}
} // namespace
struct ForwardResourceGraph::Buffer : TinyBuffer {};
ForwardResourceGraph::ForwardResourceGraph(core::VulkanContext const& c,
                                           float const* e,
                                           float const* p,
                                           float const* q,
                                           float const* k,
                                           float const* v,
                                           float const* o,
                                           float const* lm,
                                           std::uint32_t rank)
    : context_(c), engine_(c, 3, 1), loraRank_(rank) {
    if (!lora_rank_supported(rank))
        throw std::invalid_argument("unsupported Tiny LoRA rank");
    initialize(e, p, q, k, v, o, lm);
}
ForwardResourceGraph::ForwardResourceGraph(core::VulkanContext const& c,
                                           std::uint16_t const* e,
                                           std::uint16_t const* p,
                                           std::uint16_t const* q,
                                           std::uint16_t const* k,
                                           std::uint16_t const* v,
                                           std::uint16_t const* o,
                                           std::uint16_t const* lm,
                                           std::uint32_t rank)
    : context_(c), engine_(c, 3, 1), loraRank_(rank) {
    if (!e || !p || !q || !k || !v || !o || !lm)
        throw std::invalid_argument("Tiny FP16 storage inputs must not be null");
    std::vector<float> ef(V * H), pf(Tcap * H), qf(H * H), kf(H * H), vf(H * H), of(H * H), lmf(H * Vp);
    storage::fp16_to_fp32(e, ef.data(), ef.size());
    storage::fp16_to_fp32(p, pf.data(), pf.size());
    storage::fp16_to_fp32(q, qf.data(), qf.size());
    storage::fp16_to_fp32(k, kf.data(), kf.size());
    storage::fp16_to_fp32(v, vf.data(), vf.size());
    storage::fp16_to_fp32(o, of.data(), of.size());
    storage::fp16_to_fp32(lm, lmf.data(), lmf.size());
    if (!lora_rank_supported(rank))
        throw std::invalid_argument("unsupported Tiny LoRA rank");
    initialize(ef.data(), pf.data(), qf.data(), kf.data(), vf.data(), of.data(), lmf.data());
}
ForwardResourceGraph::ForwardResourceGraph(core::VulkanContext const& c,
                                           float const* e,
                                           float const* p,
                                           float const* lm,
                                           std::uint32_t rank)
    : context_(c), engine_(c, 3, 1), loraRank_(rank) {
    if (!lora_rank_supported(rank))
        throw std::invalid_argument("unsupported Tiny LoRA rank");
    if (!e || !p || !lm)
        throw std::invalid_argument("Tiny reduced forward inputs must not be null");
    auto id = identity();
    std::vector<float> full(H * Vp, 0.0f);
    for (uint32_t v = 0; v < V; ++v)
        for (uint32_t h = 0; h < H; ++h)
            full[h * Vp + v] = lm[v * H + h];
    initialize(e, p, id.data(), id.data(), id.data(), id.data(), full.data());
}
void ForwardResourceGraph::initialize(float const* e,
                                      float const* p,
                                      float const* q,
                                      float const* k,
                                      float const* v,
                                      float const* o,
                                      float const* lm) {
    if (!e || !p || !q || !k || !v || !o || !lm)
        throw std::invalid_argument("Tiny forward inputs must not be null");
    auto alloc = [&](size_t bytes, vk::BufferUsageFlags u) {
        auto b = std::make_unique<Buffer>();
        *b = {make_buffer(
            context_,
            bytes,
            u | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE)};
        return b;
    };
    embeddings_ = alloc(V * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    positions_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    // Graph-owned trainable normalization state; gamma is bound by the fixed forward ABI.
    gamma_ = alloc(H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    gammaGradient_ = alloc(H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    gammaM_ = alloc(H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    gammaV_ = alloc(H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    constexpr size_t FFN_I = 4 * H;
    ffnW1_ = alloc(H * FFN_I * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnB1_ = alloc(FFN_I * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnW2_ = alloc(FFN_I * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnB2_ = alloc(H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnW1Gradient_ = alloc(H * FFN_I * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnB1Gradient_ = alloc(FFN_I * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnW2Gradient_ = alloc(FFN_I * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnB2Gradient_ = alloc(H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnW1M_ = alloc(H * FFN_I * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnB1M_ = alloc(FFN_I * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnW2M_ = alloc(FFN_I * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnB2M_ = alloc(H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnW1V_ = alloc(H * FFN_I * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnB1V_ = alloc(FFN_I * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnW2V_ = alloc(FFN_I * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnB2V_ = alloc(H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    ffnOutput_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    ffnDx_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    query_ = alloc(H * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    key_ = alloc(H * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    value_ = alloc(H * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    output_ = alloc(H * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    lmHead_ = alloc(H * Vp * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    tokens_ = alloc(Tcap * 4,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                        vk::BufferUsageFlagBits::eTransferDst);
    targets_ = alloc(Tcap * 4,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eTransferSrc);
    mask_ = alloc(Tcap * 4,
                  vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                      vk::BufferUsageFlagBits::eTransferSrc);
    states_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    activations_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    gammaInput_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    rmsRaw_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    rmsInv_ = alloc(Tcap * 4, vk::BufferUsageFlagBits::eStorageBuffer);
     rmsDx_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    projected_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    qRows_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    kRows_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    vRows_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer);
    logits_ = alloc(Tcap * Vp * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    dlogits_ = alloc(Tcap * Vp * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    dprojected_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    embeddingGradient_ =
        alloc(V * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    positionGradient_ =
        alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    positionM_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    positionV_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    lmHeadGradient_ =
        alloc(H * Vp * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    outputGradient_ = alloc(H * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    outputM_ = alloc(H * H * 4,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eTransferDst);
    outputV_ = alloc(H * H * 4,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eTransferDst);
    qkvGradientQ_ = alloc(H * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    qkvGradientK_ = alloc(H * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    qkvGradientV_ = alloc(H * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    qkvM_ = alloc(3 * H * H * 4,
                  vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                      vk::BufferUsageFlagBits::eTransferDst);
    qkvV_ = alloc(3 * H * H * 4,
                  vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                      vk::BufferUsageFlagBits::eTransferDst);
    lmHeadM_ = alloc(H * Vp * 4,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eTransferDst);
    lmHeadV_ = alloc(H * Vp * 4,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eTransferDst);
    doutput_ = alloc(Tcap * H * 4,
                     vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eTransferSrc);
    dcontext_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    dquery_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    dkey_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    dvalue_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    dstates_ = alloc(Tcap * H * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    loraA_ = alloc(4 * H * loraRank_ * 4,
                   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                       vk::BufferUsageFlagBits::eTransferSrc);
    loraB_ = alloc(4 * loraRank_ * H * 4,
                   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                       vk::BufferUsageFlagBits::eTransferSrc);
    loraDA_ = alloc(4 * H * loraRank_ * 4,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                        vk::BufferUsageFlagBits::eTransferDst);
    loraDB_ = alloc(4 * loraRank_ * H * 4,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                        vk::BufferUsageFlagBits::eTransferDst);
    loraMA_ = alloc(4 * H * loraRank_ * 4,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                        vk::BufferUsageFlagBits::eTransferDst);
    loraVA_ = alloc(4 * H * loraRank_ * 4,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                        vk::BufferUsageFlagBits::eTransferDst);
    loraMB_ = alloc(4 * loraRank_ * H * 4,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                        vk::BufferUsageFlagBits::eTransferDst);
    loraVB_ = alloc(4 * loraRank_ * H * 4,
                    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                        vk::BufferUsageFlagBits::eTransferDst);
    rowLoss_ = alloc(Tcap * 4, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    metrics_ = alloc(8, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);
    trainingStaging_ = std::make_unique<Buffer>();
    // One host-visible allocation carries inputs and all backward readbacks.
    *trainingStaging_ = {make_buffer(
        context_,
        adamw_staging_offset(loraRank_) +
            std::max<vk::DeviceSize>(adamw_state_bytes(loraRank_), std::max<vk::DeviceSize>(2 * H * Vp * sizeof(float), 3 * (H * 4 * H + 4 * H + 4 * H * H + H) * sizeof(float) + 3 * H * sizeof(float) + H * sizeof(float) + (H * 4 * H + 4 * H + 4 * H * H + H) * sizeof(float) + 4 * Tcap * H * sizeof(float))) + sizeof(std::uint64_t),
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)};
    readback_ = std::make_unique<Buffer>();
    *readback_ = {make_buffer(context_,
                              Tcap * Vp * 4,
                              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst |
                                  vk::BufferUsageFlagBits::eTransferSrc,
                              VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                              VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT)};
    upload(context_, embeddings_->buffer, e, V * H * 4);
    upload(context_, positions_->buffer, p, Tcap * H * 4);
    std::vector<float> gamma_init(H, 1.0f), gamma_zero(H, 0.0f);
    upload(context_, gamma_->buffer, gamma_init.data(), H * sizeof(float));
    upload(context_, gammaGradient_->buffer, gamma_zero.data(), H * sizeof(float));
    upload(context_, gammaM_->buffer, gamma_zero.data(), H * sizeof(float));
    upload(context_, gammaV_->buffer, gamma_zero.data(), H * sizeof(float));
    std::vector<float> ffn_zero(H * FFN_I, 0.0f);
    upload(context_, ffnW1_->buffer, ffn_zero.data(), ffn_zero.size() * sizeof(float));
    upload(context_, ffnW2_->buffer, ffn_zero.data(), ffn_zero.size() * sizeof(float));
    upload(context_, ffnW1Gradient_->buffer, ffn_zero.data(), ffn_zero.size() * sizeof(float));
    upload(context_, ffnW2Gradient_->buffer, ffn_zero.data(), ffn_zero.size() * sizeof(float));
    std::vector<float> ffn_bias_zero(FFN_I + H, 0.0f);
    upload(context_, ffnB1_->buffer, ffn_bias_zero.data(), FFN_I * sizeof(float));
    upload(context_, ffnB2_->buffer, ffn_bias_zero.data(), H * sizeof(float));
    upload(context_, ffnB1Gradient_->buffer, ffn_bias_zero.data(), FFN_I * sizeof(float));
    upload(context_, ffnB2Gradient_->buffer, ffn_bias_zero.data(), H * sizeof(float));
    upload(context_, ffnW1M_->buffer, ffn_zero.data(), ffn_zero.size() * sizeof(float));
    upload(context_, ffnW2M_->buffer, ffn_zero.data(), ffn_zero.size() * sizeof(float));
    upload(context_, ffnW1V_->buffer, ffn_zero.data(), ffn_zero.size() * sizeof(float));
    upload(context_, ffnW2V_->buffer, ffn_zero.data(), ffn_zero.size() * sizeof(float));
    upload(context_, ffnB1M_->buffer, ffn_bias_zero.data(), FFN_I * sizeof(float));
    upload(context_, ffnB2M_->buffer, ffn_bias_zero.data(), H * sizeof(float));
    upload(context_, ffnB1V_->buffer, ffn_bias_zero.data(), FFN_I * sizeof(float));
    upload(context_, ffnB2V_->buffer, ffn_bias_zero.data(), H * sizeof(float));
    upload(context_, query_->buffer, q, H * H * 4);
    upload(context_, key_->buffer, k, H * H * 4);
    upload(context_, value_->buffer, v, H * H * 4);
    upload(context_, output_->buffer, o, H * H * 4);
    upload(context_, lmHead_->buffer, lm, H * Vp * 4);
    std::vector<float> lm_head_zero(H * Vp, 0.0f);
    upload(context_, lmHeadM_->buffer, lm_head_zero.data(), lm_head_zero.size() * sizeof(float));
    upload(context_, lmHeadV_->buffer, lm_head_zero.data(), lm_head_zero.size() * sizeof(float));
    upload(context_, outputM_->buffer, std::vector<float>(H * H, 0.0f).data(), H * H * sizeof(float));
    upload(context_, outputV_->buffer, std::vector<float>(H * H, 0.0f).data(), H * H * sizeof(float));
    upload(context_, positionM_->buffer, std::vector<float>(Tcap * H, 0.0f).data(), Tcap * H * sizeof(float));
    upload(context_, positionV_->buffer, std::vector<float>(Tcap * H, 0.0f).data(), Tcap * H * sizeof(float));
    upload(context_, qkvM_->buffer, std::vector<float>(3 * H * H, 0.0f).data(), 3 * H * H * sizeof(float));
    upload(context_, qkvV_->buffer, std::vector<float>(3 * H * H, 0.0f).data(), 3 * H * H * sizeof(float));
    std::vector<float> lora_zero(4 * H * loraRank_, 0.0f);
    for (size_t i = 0; i < lora_zero.size(); ++i)
        lora_zero[i] = 0.001f * float(int(i % 17) - 8);
    upload(context_, loraA_->buffer, lora_zero.data(), lora_zero.size() * sizeof(float));
    upload(context_, loraB_->buffer, lora_zero.data(), lora_zero.size() * sizeof(float));
    upload(
        context_, loraMA_->buffer, std::vector<float>(lora_zero.size(), 0.0f).data(), lora_zero.size() * sizeof(float));
    upload(
        context_, loraVA_->buffer, std::vector<float>(lora_zero.size(), 0.0f).data(), lora_zero.size() * sizeof(float));
    upload(
        context_, loraMB_->buffer, std::vector<float>(lora_zero.size(), 0.0f).data(), lora_zero.size() * sizeof(float));
    upload(
        context_, loraVB_->buffer, std::vector<float>(lora_zero.size(), 0.0f).data(), lora_zero.size() * sizeof(float));
    // True RMSNorm training forward is staged separately; the legacy pipeline remains active.
    auto rms_blob = shaders::get("tiny_train_forward_rmsnorm.spv");
    if (!rms_blob.data) throw std::runtime_error("tiny RMSNorm training shader missing");
    rmsForwardShader_ = context_.device.createShaderModule(vk::ShaderModuleCreateInfo{}.setCodeSize(rms_blob.size).setPCode(reinterpret_cast<uint32_t const*>(rms_blob.data)));
    auto blob = shaders::get("tiny_forward_logits.spv");
    if (!blob.data)
        throw std::runtime_error("tiny shader missing");
    shader_ = context_.device.createShaderModule(
        vk::ShaderModuleCreateInfo{}.setCodeSize(blob.size).setPCode(reinterpret_cast<uint32_t const*>(blob.data)));
    std::array<vk::DescriptorSetLayoutBinding, 17> bs{};
    for (uint32_t i = 0; i < 17; ++i)
        bs[i] = vk::DescriptorSetLayoutBinding{}
                    .setBinding(i)
                    .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                    .setDescriptorCount(1)
                    .setStageFlags(vk::ShaderStageFlagBits::eCompute);
    descriptorLayout_ = context_.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(bs));
    pipelineLayout_ = context_.device.createPipelineLayout(
        vk::PipelineLayoutCreateInfo{}
            .setSetLayouts(descriptorLayout_)
            .setPushConstantRanges(
                vk::PushConstantRange{}.setStageFlags(vk::ShaderStageFlagBits::eCompute).setSize(sizeof(ForwardPC))));
    auto pr = context_.device.createComputePipeline({},
                                                    vk::ComputePipelineCreateInfo{}
                                                        .setStage(vk::PipelineShaderStageCreateInfo{}
                                                                      .setStage(vk::ShaderStageFlagBits::eCompute)
                                                                      .setModule(shader_)
                                                                      .setPName("main"))
                                                        .setLayout(pipelineLayout_));
    if (pr.result != vk::Result::eSuccess)
        throw std::runtime_error("tiny pipeline failed");
    pipeline_ = pr.value;
     // Build the complete staged true-RMSNorm descriptor contract. Construction is
     // complete here; execution remains deferred until downstream consumers are
     // atomic and the placement contract is resolved.
     std::array<vk::DescriptorSetLayoutBinding, 19> rms_bs{};
     for (uint32_t i = 0; i < 19; ++i)
         rms_bs[i] = vk::DescriptorSetLayoutBinding{}.setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
     rmsForwardLayout_ = context_.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(rms_bs));
     rmsForwardPipelineLayout_ = context_.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{}.setSetLayouts(rmsForwardLayout_).setPushConstantRanges(vk::PushConstantRange{}.setStageFlags(vk::ShaderStageFlagBits::eCompute).setSize(sizeof(TinyRmsForwardPC))));
     auto rms_pr = context_.device.createComputePipeline({}, vk::ComputePipelineCreateInfo{}.setStage(vk::PipelineShaderStageCreateInfo{}.setStage(vk::ShaderStageFlagBits::eCompute).setModule(rmsForwardShader_).setPName("main")).setLayout(rmsForwardPipelineLayout_));
     if (rms_pr.result != vk::Result::eSuccess) throw std::runtime_error("tiny RMSNorm forward pipeline failed");
     rmsForwardPipeline_ = rms_pr.value;
     rmsForwardPool_ = context_.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(19)));
     rmsForwardSet_ = context_.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}.setDescriptorPool(rmsForwardPool_).setSetLayouts(rmsForwardLayout_)).front();
     std::array<vk::Buffer, 19> rms_bufs{embeddings_->buffer, positions_->buffer, query_->buffer, key_->buffer, value_->buffer, output_->buffer, lmHead_->buffer, tokens_->buffer, states_->buffer, activations_->buffer, projected_->buffer, logits_->buffer, readback_->buffer, qRows_->buffer, kRows_->buffer, vRows_->buffer, gamma_->buffer, rmsRaw_->buffer, rmsInv_->buffer};
     std::array<vk::DescriptorBufferInfo, 19> rms_infos{};
     std::array<vk::WriteDescriptorSet, 19> rms_writes{};
     for (uint32_t i = 0; i < 19; ++i) { rms_infos[i] = vk::DescriptorBufferInfo{}.setBuffer(rms_bufs[i]).setRange(VK_WHOLE_SIZE); rms_writes[i] = vk::WriteDescriptorSet{}.setDstSet(rmsForwardSet_).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(rms_infos[i]); }
     context_.device.updateDescriptorSets(rms_writes, {});
      // Dedicated post-attention fixed-profile RMSNorm consumes projected input
      // and produces normalized activations plus raw/inv statistics under mask.
      auto rms_fixed_blob = shaders::get("rmsnorm_forward_fixed_profile.spv");
      if (!rms_fixed_blob.data) throw std::runtime_error("fixed-profile RMSNorm shader missing");
      rmsFixedForwardShader_ = context_.device.createShaderModule(vk::ShaderModuleCreateInfo{}.setCodeSize(rms_fixed_blob.size).setPCode(reinterpret_cast<uint32_t const*>(rms_fixed_blob.data)));
      std::array<vk::DescriptorSetLayoutBinding, 6> rms_fixed_bs{};
      for (uint32_t i = 0; i < 6; ++i) rms_fixed_bs[i] = vk::DescriptorSetLayoutBinding{}.setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
      rmsFixedForwardLayout_ = context_.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(rms_fixed_bs));
      rmsFixedForwardPipelineLayout_ = context_.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{}.setSetLayouts(rmsFixedForwardLayout_).setPushConstantRanges(vk::PushConstantRange{}.setStageFlags(vk::ShaderStageFlagBits::eCompute).setSize(sizeof(RmsNormPC))));
      rmsFixedForwardPipeline_ = make_compute_pipeline(context_, rmsFixedForwardShader_, rmsFixedForwardPipelineLayout_);
      rmsFixedForwardPool_ = context_.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(6)));
      rmsFixedForwardSet_ = context_.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}.setDescriptorPool(rmsFixedForwardPool_).setSetLayouts(rmsFixedForwardLayout_)).front();
      std::array<vk::Buffer, 6> rms_fixed_bufs{projected_->buffer, gamma_->buffer, activations_->buffer, rmsRaw_->buffer, rmsInv_->buffer, mask_->buffer};
      std::array<vk::DescriptorBufferInfo, 6> rms_fixed_infos{}; std::array<vk::WriteDescriptorSet, 6> rms_fixed_writes{};
      for (uint32_t i = 0; i < 6; ++i) { rms_fixed_infos[i] = vk::DescriptorBufferInfo{}.setBuffer(rms_fixed_bufs[i]).setRange(VK_WHOLE_SIZE); rms_fixed_writes[i] = vk::WriteDescriptorSet{}.setDstSet(rmsFixedForwardSet_).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(rms_fixed_infos[i]); }
      context_.device.updateDescriptorSets(rms_fixed_writes, {});
      auto rms_back_blob = shaders::get("rmsnorm_backward_state.spv");
      if (!rms_back_blob.data) throw std::runtime_error("RMSNorm backward state shader missing");
      rmsBackwardStateShader_ = context_.device.createShaderModule(vk::ShaderModuleCreateInfo{}.setCodeSize(rms_back_blob.size).setPCode(reinterpret_cast<uint32_t const*>(rms_back_blob.data)));
      std::array<vk::DescriptorSetLayoutBinding, 6> rms_back_bs{};
      for (uint32_t i = 0; i < 6; ++i) rms_back_bs[i] = vk::DescriptorSetLayoutBinding{}.setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
      rmsBackwardStateLayout_ = context_.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(rms_back_bs));
      rmsBackwardStatePipelineLayout_ = context_.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{}.setSetLayouts(rmsBackwardStateLayout_).setPushConstantRanges(vk::PushConstantRange{}.setStageFlags(vk::ShaderStageFlagBits::eCompute).setSize(sizeof(RmsNormPC))));
      auto rms_back_pr = context_.device.createComputePipeline({}, vk::ComputePipelineCreateInfo{}.setStage(vk::PipelineShaderStageCreateInfo{}.setStage(vk::ShaderStageFlagBits::eCompute).setModule(rmsBackwardStateShader_).setPName("main")).setLayout(rmsBackwardStatePipelineLayout_));
      if (rms_back_pr.result != vk::Result::eSuccess) throw std::runtime_error("RMSNorm backward state pipeline failed");
      rmsBackwardStatePipeline_ = rms_back_pr.value;
      rmsBackwardStatePool_ = context_.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(6)));
      rmsBackwardStateSet_ = context_.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}.setDescriptorPool(rmsBackwardStatePool_).setSetLayouts(rmsBackwardStateLayout_)).front();
      std::array<vk::Buffer, 6> rms_back_bufs{rmsRaw_->buffer, gamma_->buffer, dprojected_->buffer, mask_->buffer, rmsInv_->buffer, rmsDx_->buffer};
      std::array<vk::DescriptorBufferInfo, 6> rms_back_infos{}; std::array<vk::WriteDescriptorSet, 6> rms_back_writes{};
      for (uint32_t i = 0; i < 6; ++i) { rms_back_infos[i] = vk::DescriptorBufferInfo{}.setBuffer(rms_back_bufs[i]).setRange(VK_WHOLE_SIZE); rms_back_writes[i] = vk::WriteDescriptorSet{}.setDstSet(rmsBackwardStateSet_).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(rms_back_infos[i]); }
      context_.device.updateDescriptorSets(rms_back_writes, {});
       auto rms_dg_blob = shaders::get("rmsnorm_dgamma_state.spv");
       if (!rms_dg_blob.data) throw std::runtime_error("RMSNorm dgamma state shader missing");
       rmsDgammaStateShader_ = context_.device.createShaderModule(vk::ShaderModuleCreateInfo{}.setCodeSize(rms_dg_blob.size).setPCode(reinterpret_cast<uint32_t const*>(rms_dg_blob.data)));
       std::array<vk::DescriptorSetLayoutBinding, 5> rms_dg_bs{};
       for (uint32_t i = 0; i < 5; ++i) rms_dg_bs[i] = vk::DescriptorSetLayoutBinding{}.setBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
       rmsDgammaStateLayout_ = context_.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(rms_dg_bs));
       rmsDgammaStatePipelineLayout_ = context_.device.createPipelineLayout(vk::PipelineLayoutCreateInfo{}.setSetLayouts(rmsDgammaStateLayout_).setPushConstantRanges(vk::PushConstantRange{}.setStageFlags(vk::ShaderStageFlagBits::eCompute).setSize(sizeof(RmsNormPC))));
       auto rms_dg_pr = context_.device.createComputePipeline({}, vk::ComputePipelineCreateInfo{}.setStage(vk::PipelineShaderStageCreateInfo{}.setStage(vk::ShaderStageFlagBits::eCompute).setModule(rmsDgammaStateShader_).setPName("main")).setLayout(rmsDgammaStatePipelineLayout_));
       if (rms_dg_pr.result != vk::Result::eSuccess) throw std::runtime_error("RMSNorm dgamma state pipeline failed");
       rmsDgammaStatePipeline_ = rms_dg_pr.value;
       rmsDgammaStatePool_ = context_.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(5)));
       rmsDgammaStateSet_ = context_.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}.setDescriptorPool(rmsDgammaStatePool_).setSetLayouts(rmsDgammaStateLayout_)).front();
       std::array<vk::Buffer, 5> rms_dg_bufs{rmsRaw_->buffer, dprojected_->buffer, mask_->buffer, rmsInv_->buffer, gammaGradient_->buffer};
       std::array<vk::DescriptorBufferInfo, 5> rms_dg_infos{}; std::array<vk::WriteDescriptorSet, 5> rms_dg_writes{};
       for (uint32_t i = 0; i < 5; ++i) { rms_dg_infos[i] = vk::DescriptorBufferInfo{}.setBuffer(rms_dg_bufs[i]).setRange(VK_WHOLE_SIZE); rms_dg_writes[i] = vk::WriteDescriptorSet{}.setDstSet(rmsDgammaStateSet_).setDstBinding(i).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setBufferInfo(rms_dg_infos[i]); }
       context_.device.updateDescriptorSets(rms_dg_writes, {});
     descriptorPool_ = context_.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}.setMaxSets(1).setPoolSizes(
        vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(17)));
    descriptorSet_ =
        context_.device
            .allocateDescriptorSets(
                vk::DescriptorSetAllocateInfo{}.setDescriptorPool(descriptorPool_).setSetLayouts(descriptorLayout_))
            .front();
    std::array<vk::Buffer, 17> bufs{embeddings_->buffer,
                                    positions_->buffer,
                                    query_->buffer,
                                    key_->buffer,
                                    value_->buffer,
                                    output_->buffer,
                                    lmHead_->buffer,
                                    tokens_->buffer,
                                    states_->buffer,
                                    activations_->buffer,
                                    projected_->buffer,
                                    logits_->buffer,
                                    readback_->buffer,
                                    qRows_->buffer,
                                    kRows_->buffer,
                                    vRows_->buffer};
    bufs[16] = gamma_->buffer;
    std::array<vk::DescriptorBufferInfo, 17> infos{};
    std::array<vk::WriteDescriptorSet, 17> writes{};
    for (uint32_t i = 0; i < 17; ++i) {
        infos[i] = vk::DescriptorBufferInfo{}.setBuffer(bufs[i]).setRange(VK_WHOLE_SIZE);
        writes[i] = vk::WriteDescriptorSet{}
                        .setDstSet(descriptorSet_)
                        .setDstBinding(i)
                        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setDescriptorCount(1)
                        .setBufferInfo(infos[i]);
    }
    context_.device.updateDescriptorSets(writes, {});

    auto setup_training = [&](char const* shader_name,
                              uint32_t count,
                              vk::ShaderModule& shader,
                              vk::DescriptorSetLayout& layout,
                              vk::PipelineLayout& pipeline_layout,
                              vk::Pipeline& pipeline,
                              vk::DescriptorPool& pool,
                              vk::DescriptorSet& set,
                              std::initializer_list<vk::Buffer> buffers,
                              uint32_t pc_size) {
        auto b = shaders::get(shader_name);
        shader = context_.device.createShaderModule(
            vk::ShaderModuleCreateInfo{}.setCodeSize(b.size).setPCode(reinterpret_cast<uint32_t const*>(b.data)));
        std::vector<vk::DescriptorSetLayoutBinding> lb(count);
        for (uint32_t i = 0; i < count; ++i)
            lb[i] = vk::DescriptorSetLayoutBinding{}
                        .setBinding(i)
                        .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                        .setDescriptorCount(1)
                        .setStageFlags(vk::ShaderStageFlagBits::eCompute);
        layout = context_.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{}.setBindings(lb));
        pipeline_layout = context_.device.createPipelineLayout(
            vk::PipelineLayoutCreateInfo{}.setSetLayouts(layout).setPushConstantRanges(
                vk::PushConstantRange{}.setStageFlags(vk::ShaderStageFlagBits::eCompute).setSize(pc_size)));
        pipeline = make_compute_pipeline(context_, shader, pipeline_layout);
        pool = context_.device.createDescriptorPool(vk::DescriptorPoolCreateInfo{}.setMaxSets(4).setPoolSizes(
            vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(count * 4)));
        set = context_.device
                  .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}.setDescriptorPool(pool).setSetLayouts(layout))
                  .front();
        std::vector<vk::DescriptorBufferInfo> infos;
        std::vector<vk::WriteDescriptorSet> writes2;
        infos.reserve(count);
        writes2.reserve(count);
        for (auto buffer : buffers) {
            infos.push_back(vk::DescriptorBufferInfo{}.setBuffer(buffer).setRange(VK_WHOLE_SIZE));
            writes2.push_back(vk::WriteDescriptorSet{}
                                  .setDstSet(set)
                                  .setDstBinding(static_cast<uint32_t>(writes2.size()))
                                  .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                                  .setDescriptorCount(1)
                                  .setBufferInfo(infos.back()));
        }
        context_.device.updateDescriptorSets(writes2, {});
    };
    setup_training("causal_loss.spv",
                   5,
                   lossShader_,
                   lossLayout_,
                   lossPipelineLayout_,
                   lossPipeline_,
                   lossPool_,
                   lossSet_,
                   {logits_->buffer, targets_->buffer, mask_->buffer, dlogits_->buffer, rowLoss_->buffer},
                   sizeof(LossPC));
    setup_training("metrics_reduce.spv",
                   3,
                   metricsShader_,
                   metricsLayout_,
                   metricsPipelineLayout_,
                   metricsPipeline_,
                   metricsPool_,
                   metricsSet_,
                   {rowLoss_->buffer, mask_->buffer, metrics_->buffer},
                   sizeof(MetricsPC));
    setup_training("lm_head_backward.spv",
                   3,
                   lmBackwardShader_,
                   lmBackwardLayout_,
                   lmBackwardPipelineLayout_,
                   lmBackwardPipeline_,
                   lmBackwardPool_,
                   lmBackwardSet_,
                   {dlogits_->buffer, lmHead_->buffer, dprojected_->buffer},
                   sizeof(LmBackwardPC));
    setup_training("lm_head_weight_grad.spv",
                   3,
                   lmHeadGradientShader_,
                   lmHeadGradientLayout_,
                   lmHeadGradientPipelineLayout_,
                   lmHeadGradientPipeline_,
                   lmHeadGradientPool_,
                   lmHeadGradientSet_,
                   {projected_->buffer, dlogits_->buffer, lmHeadGradient_->buffer},
                   sizeof(LmHeadGradientPC));
    setup_training("lm_head_sgd.spv",
                   2,
                   lmHeadSgdShader_,
                   lmHeadSgdLayout_,
                   lmHeadSgdPipelineLayout_,
                   lmHeadSgdPipeline_,
                   lmHeadSgdPool_,
                   lmHeadSgdSet_,
                   {lmHead_->buffer, lmHeadGradient_->buffer},
                   sizeof(LmHeadSgdPC));
    setup_training("lm_head_adamw.spv",
                   4,
                   lmHeadAdamwShader_,
                   lmHeadAdamwLayout_,
                   lmHeadAdamwPipelineLayout_,
                   lmHeadAdamwPipeline_,
                   lmHeadAdamwPool_,
                   lmHeadAdamwSet_,
                   {lmHead_->buffer, lmHeadGradient_->buffer, lmHeadM_->buffer, lmHeadV_->buffer},
                   sizeof(LmHeadAdamwPC));
    setup_training("output_projection_weight_grad.spv",
                   3,
                   outputGradientShader_,
                   outputGradientLayout_,
                   outputGradientPipelineLayout_,
                   outputGradientPipeline_,
                   outputGradientPool_,
                   outputGradientSet_,
                   {activations_->buffer, dprojected_->buffer, outputGradient_->buffer},
                   sizeof(OutputProjectionGradientPC));
    setup_training("output_projection_sgd.spv",
                   2,
                   outputSgdShader_,
                   outputSgdLayout_,
                   outputSgdPipelineLayout_,
                   outputSgdPipeline_,
                   outputSgdPool_,
                   outputSgdSet_,
                   {output_->buffer, outputGradient_->buffer},
                   sizeof(OutputProjectionSgdPC));
    setup_training("output_projection_adamw.spv",
                   4,
                   outputAdamwShader_,
                   outputAdamwLayout_,
                   outputAdamwPipelineLayout_,
                   outputAdamwPipeline_,
                   outputAdamwPool_,
                   outputAdamwSet_,
                   {output_->buffer, outputGradient_->buffer, outputM_->buffer, outputV_->buffer},
                   sizeof(OutputAdamwPC));
    setup_training("qkv_projection_weight_grad.spv",
                   7,
                   qkvGradientShader_,
                   qkvGradientLayout_,
                   qkvGradientPipelineLayout_,
                   qkvGradientPipeline_,
                   qkvGradientPool_,
                   qkvGradientSet_,
                   {states_->buffer,
                    dquery_->buffer,
                    dkey_->buffer,
                    dvalue_->buffer,
                    qkvGradientQ_->buffer,
                    qkvGradientK_->buffer,
                    qkvGradientV_->buffer},
                   sizeof(QkvGradientPC));
    setup_training("embedding_training_grad.spv",
                    4,
                    embeddingGradientShader_, embeddingGradientLayout_, embeddingGradientPipelineLayout_,
                    embeddingGradientPipeline_, embeddingGradientPool_, embeddingGradientSet_,
                    {tokens_->buffer, dstates_->buffer, mask_->buffer, embeddingGradient_->buffer},
                    sizeof(EmbeddingGradientPC));
     // Staged true-dx artifacts remain separately tested; active graph stays legacy.
     if (!shaders::get("embedding_training_grad_rmsnorm.spv").data ||
          !shaders::get("position_training_grad_rmsnorm.spv").data)
          throw std::runtime_error("RMSNorm true-dx gradient shader missing");
      // The staged true-dx consumers are intentionally validated for presence
      // only. They are not bound to the legacy active chain until placement,
      // liveness, synchronization, and end-to-end parity gates are complete.
     setup_training("embedding_training_sgd.spv",
                    2,
                    embeddingSgdShader_, embeddingSgdLayout_, embeddingSgdPipelineLayout_,
                    embeddingSgdPipeline_, embeddingSgdPool_, embeddingSgdSet_,
                    {embeddings_->buffer, embeddingGradient_->buffer},
                    sizeof(EmbeddingSgdPC));
     setup_training("position_training_grad.spv",
                     4,
                     positionGradientShader_, positionGradientLayout_, positionGradientPipelineLayout_,
                     positionGradientPipeline_, positionGradientPool_, positionGradientSet_,
                     {dstates_->buffer, mask_->buffer, gamma_->buffer, positionGradient_->buffer},
                     sizeof(PositionGradientPC));
     setup_training("position_training_sgd.spv",
                     2,
                     positionSgdShader_, positionSgdLayout_, positionSgdPipelineLayout_,
                     positionSgdPipeline_, positionSgdPool_, positionSgdSet_,
                      {positions_->buffer, positionGradient_->buffer},
                      sizeof(PositionSgdPC));
      setup_training("ffn_gelu_backward.spv", 7, ffnBackwardShader_, ffnBackwardLayout_, ffnBackwardPipelineLayout_, ffnBackwardPipeline_, ffnBackwardPool_, ffnBackwardSet_, {activations_->buffer, ffnW1_->buffer, ffnB1_->buffer, ffnW2_->buffer, dprojected_->buffer, ffnDx_->buffer, mask_->buffer}, sizeof(FFNForwardPC));
      setup_training("rmsnorm_dgamma_masked.spv", 4, rmsNormGradientShader_, rmsNormGradientLayout_, rmsNormGradientPipelineLayout_, rmsNormGradientPipeline_, rmsNormGradientPool_, rmsNormGradientSet_, {gammaInput_->buffer, dprojected_->buffer, mask_->buffer, gammaGradient_->buffer}, sizeof(RmsNormPC));
       setup_training("scale_dgamma_masked.spv", 5, scaleDgammaShader_, scaleDgammaLayout_, scaleDgammaPipelineLayout_, scaleDgammaPipeline_, scaleDgammaPool_, scaleDgammaSet_, {states_->buffer, dstates_->buffer, mask_->buffer, gamma_->buffer, gammaGradient_->buffer}, sizeof(RmsNormPC));
      setup_training("ffn_param_grads_masked.spv", 10, ffnGradientShader_, ffnGradientLayout_, ffnGradientPipelineLayout_, ffnGradientPipeline_, ffnGradientPool_, ffnGradientSet_, {activations_->buffer, ffnW1_->buffer, ffnB1_->buffer, ffnW2_->buffer, dprojected_->buffer, ffnW1Gradient_->buffer, ffnB1Gradient_->buffer, ffnW2Gradient_->buffer, ffnB2Gradient_->buffer, mask_->buffer}, sizeof(FfnGradPC));
      setup_training("ffn_adamw.spv", 4, ffnB1AdamwShader_, ffnB1AdamwLayout_, ffnB1AdamwPipelineLayout_, ffnB1AdamwPipeline_, ffnB1AdamwPool_, ffnB1AdamwSet_, {ffnB1_->buffer, ffnB1Gradient_->buffer, ffnB1M_->buffer, ffnB1V_->buffer}, sizeof(FfnGroupPC));
      setup_training("ffn_adamw.spv", 4, ffnW2AdamwShader_, ffnW2AdamwLayout_, ffnW2AdamwPipelineLayout_, ffnW2AdamwPipeline_, ffnW2AdamwPool_, ffnW2AdamwSet_, {ffnW2_->buffer, ffnW2Gradient_->buffer, ffnW2M_->buffer, ffnW2V_->buffer}, sizeof(FfnGroupPC));
      setup_training("ffn_adamw.spv", 4, ffnB2AdamwShader_, ffnB2AdamwLayout_, ffnB2AdamwPipelineLayout_, ffnB2AdamwPipeline_, ffnB2AdamwPool_, ffnB2AdamwSet_, {ffnB2_->buffer, ffnB2Gradient_->buffer, ffnB2M_->buffer, ffnB2V_->buffer}, sizeof(FfnGroupPC));
      setup_training("ffn_adamw.spv", 4, ffnAdamwShader_, ffnAdamwLayout_, ffnAdamwPipelineLayout_, ffnAdamwPipeline_, ffnAdamwPool_, ffnAdamwSet_, {ffnW1_->buffer, ffnW1Gradient_->buffer, ffnW1M_->buffer, ffnW1V_->buffer}, sizeof(FfnAdamwPC));
      setup_training("ffn_gelu_forward.spv", 7, ffnForwardShader_, ffnForwardLayout_, ffnForwardPipelineLayout_, ffnForwardPipeline_, ffnForwardPool_, ffnForwardSet_, {activations_->buffer, ffnW1_->buffer, ffnB1_->buffer, ffnW2_->buffer, ffnB2_->buffer, ffnOutput_->buffer, mask_->buffer}, sizeof(FFNForwardPC));
       setup_training("lm_head_forward.spv", 3, lmHeadForwardShader_, lmHeadForwardLayout_, lmHeadForwardPipelineLayout_, lmHeadForwardPipeline_, lmHeadForwardPool_, lmHeadForwardSet_, {projected_->buffer, lmHead_->buffer, logits_->buffer}, sizeof(LmHeadForwardPC));
       setup_training("rmsnorm_gamma_adamw.spv", 4, gammaAdamwShader_, gammaAdamwLayout_, gammaAdamwPipelineLayout_, gammaAdamwPipeline_, gammaAdamwPool_, gammaAdamwSet_, {gamma_->buffer, gammaGradient_->buffer, gammaM_->buffer, gammaV_->buffer}, sizeof(PositionAdamwPC));
     setup_training("position_training_adamw.spv", 5, positionAdamwShader_, positionAdamwLayout_, positionAdamwPipelineLayout_,
                     positionAdamwPipeline_, positionAdamwPool_, positionAdamwSet_,
                     {positions_->buffer, positionGradient_->buffer, positionM_->buffer, positionV_->buffer, mask_->buffer},
                     sizeof(PositionAdamwPC));
     setup_training("qkv_projection_sgd.spv",
                   6,
                   qkvSgdShader_,
                   qkvSgdLayout_,
                   qkvSgdPipelineLayout_,
                   qkvSgdPipeline_,
                   qkvSgdPool_,
                   qkvSgdSet_,
                   {query_->buffer,
                    key_->buffer,
                    value_->buffer,
                    qkvGradientQ_->buffer,
                    qkvGradientK_->buffer,
                    qkvGradientV_->buffer},
                   sizeof(QkvSgdPC));
    setup_training("qkv_projection_adamw.spv",
                   8,
                   qkvAdamwShader_,
                   qkvAdamwLayout_,
                   qkvAdamwPipelineLayout_,
                   qkvAdamwPipeline_,
                   qkvAdamwPool_,
                   qkvAdamwSet_,
                   {query_->buffer,
                    key_->buffer,
                    value_->buffer,
                    qkvGradientQ_->buffer,
                    qkvGradientK_->buffer,
                    qkvGradientV_->buffer,
                    qkvM_->buffer,
                    qkvV_->buffer},
                   sizeof(QkvAdamwPC));
    setup_training("attention_causal_backward.spv",
                   7,
                   attentionBackwardShader_,
                   attentionBackwardLayout_,
                   attentionBackwardPipelineLayout_,
                   attentionBackwardPipeline_,
                   attentionBackwardPool_,
                   attentionBackwardSet_,
                   {qRows_->buffer,
                    kRows_->buffer,
                    vRows_->buffer,
                    doutput_->buffer,
                    dquery_->buffer,
                    dkey_->buffer,
                    dvalue_->buffer},
                   sizeof(AttentionBackwardPC));
    auto make_attention_set = [&](vk::Buffer output) {
        auto set = context_.device
                       .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                   .setDescriptorPool(attentionBackwardPool_)
                                                   .setSetLayouts(attentionBackwardLayout_))
                       .front();
        std::array<vk::Buffer, 7> buffers{
            qRows_->buffer, kRows_->buffer, vRows_->buffer, doutput_->buffer, output, output, output};
        std::array<vk::DescriptorBufferInfo, 7> infos{};
        std::array<vk::WriteDescriptorSet, 7> writes{};
        for (uint32_t i = 0; i < 7; ++i) {
            infos[i] = vk::DescriptorBufferInfo{}.setBuffer(buffers[i]).setRange(VK_WHOLE_SIZE);
            writes[i] = vk::WriteDescriptorSet{}
                            .setDstSet(set)
                            .setDstBinding(i)
                            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                            .setDescriptorCount(1)
                            .setBufferInfo(infos[i]);
        }
        context_.device.updateDescriptorSets(writes, {});
        return set;
    };
    attentionBackwardQSet_ = make_attention_set(dquery_->buffer);
    attentionBackwardKSet_ = make_attention_set(dkey_->buffer);
    attentionBackwardVSet_ = make_attention_set(dvalue_->buffer);
    setup_training("projection_backward.spv",
                   3,
                   projectionBackwardShader_,
                   projectionBackwardLayout_,
                   projectionBackwardPipelineLayout_,
                   projectionBackwardPipeline_,
                   projectionBackwardPool_,
                   projectionBackwardSet_,
                   {dprojected_->buffer, output_->buffer, dcontext_->buffer},
                   sizeof(ProjectionBackwardPC));
    setup_training("lora_gradients_persistent.spv",
                   11,
                   loraShader_,
                   loraLayout_,
                   loraPipelineLayout_,
                   loraPipeline_,
                   loraPool_,
                   loraSet_,
                   {states_->buffer,
                    dquery_->buffer,
                    dkey_->buffer,
                    dvalue_->buffer,
                    doutput_->buffer,
                    activations_->buffer,
                    dprojected_->buffer,
                    loraA_->buffer,
                    loraB_->buffer,
                    loraDA_->buffer,
                    loraDB_->buffer},
                   sizeof(LoraPC));
    setup_training("lora_sgd_multi.spv",
                   4,
                   loraSgdShader_,
                   loraSgdLayout_,
                   loraSgdPipelineLayout_,
                   loraSgdPipeline_,
                   loraSgdPool_,
                   loraSgdSet_,
                   {loraA_->buffer, loraB_->buffer, loraDA_->buffer, loraDB_->buffer},
                   sizeof(LoraSgdPC));
    setup_training("lora_adamw_multi.spv",
                   8,
                   loraAdamwShader_,
                   loraAdamwLayout_,
                   loraAdamwPipelineLayout_,
                   loraAdamwPipeline_,
                   loraAdamwPool_,
                   loraAdamwSet_,
                   {loraA_->buffer,
                    loraB_->buffer,
                    loraDA_->buffer,
                    loraDB_->buffer,
                    loraMA_->buffer,
                    loraVA_->buffer,
                    loraMB_->buffer,
                    loraVB_->buffer},
                   sizeof(LoraAdamwPC));
    setup_training("lora_accumulate_clear.spv",
                   2,
                   loraClearShader_,
                   loraClearLayout_,
                   loraClearPipelineLayout_,
                   loraClearPipeline_,
                   loraClearPool_,
                   loraClearSet_,
                   {loraDA_->buffer, loraDB_->buffer},
                   sizeof(LoraClearPC));
    setup_training("lora_finalize_sgd_multi.spv",
                   4,
                   loraFinalizeShader_,
                   loraFinalizeLayout_,
                   loraFinalizePipelineLayout_,
                   loraFinalizePipeline_,
                   loraFinalizePool_,
                   loraFinalizeSet_,
                   {loraA_->buffer, loraB_->buffer, loraDA_->buffer, loraDB_->buffer},
                   sizeof(LoraFinalizePC));
    auto make_projection_set = [&](vk::Buffer dy, vk::Buffer weight) {
        auto set = context_.device
                       .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                   .setDescriptorPool(projectionBackwardPool_)
                                                   .setSetLayouts(projectionBackwardLayout_))
                       .front();
        std::array<vk::DescriptorBufferInfo, 3> infos{
            vk::DescriptorBufferInfo{}.setBuffer(dy).setRange(VK_WHOLE_SIZE),
            vk::DescriptorBufferInfo{}.setBuffer(weight).setRange(VK_WHOLE_SIZE),
            vk::DescriptorBufferInfo{}.setBuffer(dstates_->buffer).setRange(VK_WHOLE_SIZE)};
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (uint32_t i = 0; i < 3; ++i)
            writes[i] = vk::WriteDescriptorSet{}
                            .setDstSet(set)
                            .setDstBinding(i)
                            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                            .setDescriptorCount(1)
                            .setBufferInfo(infos[i]);
        context_.device.updateDescriptorSets(writes, {});
        return set;
    };
    projectionBackwardKVSet_ = make_projection_set(dkey_->buffer, key_->buffer);
    projectionBackwardVVSet_ = make_projection_set(dvalue_->buffer, value_->buffer);
    projectionBackwardQkvSet_ = context_.device
                                    .allocateDescriptorSets(vk::DescriptorSetAllocateInfo{}
                                                                .setDescriptorPool(projectionBackwardPool_)
                                                                .setSetLayouts(projectionBackwardLayout_))
                                    .front();
    std::array<vk::DescriptorBufferInfo, 3> qkv_infos{
        vk::DescriptorBufferInfo{}.setBuffer(dquery_->buffer).setRange(VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo{}.setBuffer(query_->buffer).setRange(VK_WHOLE_SIZE),
        vk::DescriptorBufferInfo{}.setBuffer(dstates_->buffer).setRange(VK_WHOLE_SIZE)};
    std::array<vk::WriteDescriptorSet, 3> qkv_writes{};
    for (uint32_t i = 0; i < 3; ++i)
        qkv_writes[i] = vk::WriteDescriptorSet{}
                            .setDstSet(projectionBackwardQkvSet_)
                            .setDstBinding(i)
                            .setDescriptorType(vk::DescriptorType::eStorageBuffer)
                            .setDescriptorCount(1)
                            .setBufferInfo(qkv_infos[i]);
    context_.device.updateDescriptorSets(qkv_writes, {});

    // Fixed-shape retained subset: copy all token slots, execute the complete
    // forward dispatch, and read back all rows from the same command buffer.
    // The loss variant below uses the same retained command with device-buffer
    // targets/masks and is recorded separately to keep each immutable graph
    // semantically unambiguous.
    engine_.recordImmutable([this](vk::CommandBuffer cmd) {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(trainingStaging_->buffer)
                                .setSize(VK_WHOLE_SIZE)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                            {});
        cmd.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(fixedForwardStagingBytes));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                            {},
                            {});
        ForwardPC const pc{Tcap, 0};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        cmd.dispatch(5, Tcap, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
                            {},
                            {});
        cmd.copyBuffer(logits_->buffer, readback_->buffer, vk::BufferCopy{}.setSize(Tcap * Vp * sizeof(float)));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {},
                            vk::BufferMemoryBarrier{}.setBuffer(trainingStaging_->buffer).setSize(VK_WHOLE_SIZE)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite), {});
    });
    fixedForwardRecorded_ = true;

    // A separate immutable command buffer retains the production-bounded
    // forward+loss shape. Targets and masks are mutable device buffers fed by
    // the same host staging allocation; all dispatch dimensions and push
    // constants remain immutable at Tcap/V/Vp.
    engine_.recordImmutable([this](vk::CommandBuffer cmd) {
        constexpr vk::DeviceSize tokenBytes = Tcap * sizeof(std::uint32_t);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(trainingStaging_->buffer)
                                .setSize(VK_WHOLE_SIZE)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                            {});
        constexpr vk::DeviceSize targetOffset = tokenBytes;
        constexpr vk::DeviceSize maskOffset = targetOffset + Tcap * sizeof(std::uint32_t);
        constexpr vk::DeviceSize logitsOffset = maskOffset + Tcap * sizeof(std::uint32_t);
        constexpr vk::DeviceSize lossOffset = logitsOffset + Tcap * Vp * sizeof(float);
        cmd.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(tokenBytes));
        cmd.copyBuffer(trainingStaging_->buffer,
                       targets_->buffer,
                       vk::BufferCopy{}.setSrcOffset(targetOffset).setSize(Tcap * sizeof(std::uint32_t)));
        cmd.copyBuffer(trainingStaging_->buffer,
                       mask_->buffer,
                       vk::BufferCopy{}.setSrcOffset(maskOffset).setSize(Tcap * sizeof(std::uint32_t)));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(trainingStaging_->buffer)
                                .setSize(VK_WHOLE_SIZE)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
                            {});
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                            {},
                            {});
        ForwardPC const forwardPc{Tcap, 0};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(forwardPc), &forwardPc);
        cmd.dispatch(5, Tcap, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                            {},
                            {});
        FFNForwardPC const ff{Tcap, H, 4 * H};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ffnForwardPipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnForwardPipelineLayout_, 0, ffnForwardSet_, {});
        cmd.pushConstants(ffnForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ff), &ff);
        cmd.dispatch((Tcap * H + 255) / 256, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                            {},
                            {});
        LossPC const lossPc{Tcap, V, Vp};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
        cmd.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lossPc), &lossPc);
        cmd.dispatch(Tcap, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
                            {},
                            {});
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(trainingStaging_->buffer)
                                .setOffset(logitsOffset)
                                .setSize(Tcap * Vp * sizeof(float) + Tcap * sizeof(float))
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
                            {});
        cmd.copyBuffer(logits_->buffer,
                       trainingStaging_->buffer,
                       vk::BufferCopy{}.setDstOffset(logitsOffset).setSize(Tcap * Vp * sizeof(float)));
        cmd.copyBuffer(rowLoss_->buffer,
                       trainingStaging_->buffer,
                       vk::BufferCopy{}.setDstOffset(lossOffset).setSize(Tcap * sizeof(float)));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(trainingStaging_->buffer)
                                .setSize(VK_WHOLE_SIZE)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite),
                            {});
    }, true);
    fixedForwardLossRecorded_ = true;
}
std::unique_ptr<dataset::BatchBuffer> ForwardResourceGraph::create_dataset_batch(std::uint32_t windows,
                                                                                 std::uint32_t window_length) const {
    if (windows == 0 || window_length == 0 || window_length > Tcap)
        throw std::invalid_argument("dataset batch exceeds bounded Tiny window shape");
    return std::make_unique<dataset::BatchBuffer>(context_, windows, window_length);
}

int ForwardResourceGraph::train_dataset_batch(dataset::BatchBuffer const& batch,
                                              float learning_rate,
                                              float normalizer) noexcept {
    auto const view = batch.device_view();
    if (!view.tokens || !view.targets || !view.masks || view.window_count == 0 || view.window_count > 32 ||
        view.window_tokens == 0 || view.window_tokens > Tcap || !std::isfinite(learning_rate) ||
        learning_rate <= 0.0f || !std::isfinite(normalizer) || normalizer <= 0.0f)
        return 1;
    float const effectiveLearningRate = learning_rate / normalizer;
    vk::DeviceSize const windowBytes = vk::DeviceSize(view.window_tokens) * sizeof(std::uint32_t);
    LossPC const lossPc{view.window_tokens, V, Vp};
    LmHeadGradientPC const baseGradientPc{view.window_tokens, Tcap, V, Vp, H, 0};
    LmHeadSgdPC const sgdPc{H * Vp, effectiveLearningRate};
    lastSubmission_ = engine_.submit(
        [this, view, windowBytes, lossPc, baseGradientPc, sgdPc](vk::CommandBuffer cmd) {
            auto barrier = [&]() {
                cmd.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {},
                    vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                    {},
                    {});
            };
            for (std::uint32_t window = 0; window < view.window_count; ++window) {
                auto const sourceOffset = vk::DeviceSize(window) * windowBytes;
                cmd.copyBuffer(
                    view.tokens, tokens_->buffer, vk::BufferCopy{}.setSrcOffset(sourceOffset).setSize(windowBytes));
                cmd.copyBuffer(
                    view.targets, targets_->buffer, vk::BufferCopy{}.setSrcOffset(sourceOffset).setSize(windowBytes));
                cmd.copyBuffer(
                    view.masks, mask_->buffer, vk::BufferCopy{}.setSrcOffset(sourceOffset).setSize(windowBytes));
                cmd.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eComputeShader,
                    {},
                    vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                    {},
                    {});
                ForwardPC const forwardPc{view.window_tokens, 0};
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
                cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(forwardPc), &forwardPc);
                cmd.dispatch(5, forwardPc.seq_length, 1);
                barrier();
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
                cmd.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lossPc), &lossPc);
                cmd.dispatch(lossPc.rows, 1, 1);
                barrier();
                auto gradientPc = baseGradientPc;
                gradientPc.accumulate = window == 0 ? 0u : 1u;
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmHeadGradientPipeline_);
                cmd.bindDescriptorSets(
                    vk::PipelineBindPoint::eCompute, lmHeadGradientPipelineLayout_, 0, lmHeadGradientSet_, {});
                cmd.pushConstants(lmHeadGradientPipelineLayout_,
                                  vk::ShaderStageFlagBits::eCompute,
                                  0,
                                  sizeof(gradientPc),
                                  &gradientPc);
                cmd.dispatch((H * Vp + 255) / 256, 1, 1);
                barrier();
            }
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmHeadSgdPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmHeadSgdPipelineLayout_, 0, lmHeadSgdSet_, {});
            cmd.pushConstants(lmHeadSgdPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(sgdPc), &sgdPc);
            cmd.dispatch((sgdPc.count + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    return 0;
}

int ForwardResourceGraph::train_dataset_batch_full(dataset::BatchBuffer const& batch,
                                                     float learning_rate,
                                                     float normalizer) noexcept {
    // The complete LoRA path still requires graph-owned per-token backward
    // state. Reuse the bounded device submission only for validation of the
    // batch contract; never silently downgrade this API to LM-head-only SGD.
    auto const view = batch.device_view();
    if (!view.tokens || !view.targets || !view.masks || view.window_count == 0 || view.window_count > 32 ||
        view.window_tokens == 0 || view.window_tokens > Tcap || !std::isfinite(learning_rate) || learning_rate <= 0.0f ||
        !std::isfinite(normalizer) || normalizer <= 0.0f)
        return 1;
    // Full graph integration remains intentionally disabled until BatchBuffer
    // descriptors are wired into every backward resource. Keep the explicit
    // status stable so callers cannot mistake LM-head-only training for LoRA.
    return dataset_training_full_unsupported;
}

std::vector<float> ForwardResourceGraph::evaluate_dataset_batch(dataset::BatchBuffer const& batch) {
    auto const view = batch.device_view();
    if (!view.tokens || !view.targets || !view.masks || !view.results || view.window_count == 0 ||
        view.window_count > 32 || view.window_tokens == 0 || view.window_tokens > Tcap)
        throw std::invalid_argument("invalid Tiny dataset evaluation batch");
    auto const windowBytes = vk::DeviceSize(view.window_tokens) * sizeof(std::uint32_t);
    lastSubmission_ = engine_.submit([this, view, windowBytes](vk::CommandBuffer cmd) {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}.setBuffer(trainingStaging_->buffer).setSize(VK_WHOLE_SIZE)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                            {});
        for (std::uint32_t window = 0; window < view.window_count; ++window) {
            auto const sourceOffset = vk::DeviceSize(window) * windowBytes;
            cmd.copyBuffer(view.tokens, tokens_->buffer,
                           vk::BufferCopy{}.setSrcOffset(sourceOffset).setSize(windowBytes));
            cmd.copyBuffer(view.targets, targets_->buffer,
                           vk::BufferCopy{}.setSrcOffset(sourceOffset).setSize(windowBytes));
            cmd.copyBuffer(view.masks, mask_->buffer,
                           vk::BufferCopy{}.setSrcOffset(sourceOffset).setSize(windowBytes));
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite,
                                                  vk::AccessFlagBits::eShaderRead), {}, {});
            ForwardPC const fp{view.window_tokens, 0};
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fp), &fp);
            cmd.dispatch(5, view.window_tokens, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader, {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eShaderRead), {}, {});
            LossPC const lp{view.window_tokens, V, Vp};
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
            cmd.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lp), &lp);
            cmd.dispatch(view.window_tokens, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader, {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eShaderRead), {}, {});
            MetricsPC const mp{view.window_tokens};
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, metricsPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, metricsPipelineLayout_, 0, metricsSet_, {});
            cmd.pushConstants(metricsPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(mp), &mp);
            cmd.dispatch(1, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite,
                                                  vk::AccessFlagBits::eTransferRead), {}, {});
            auto const metricsOffset = vk::DeviceSize(window) * 2 * sizeof(float);
             cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {},
                                 vk::BufferMemoryBarrier{}.setBuffer(trainingStaging_->buffer).setOffset(metricsOffset).setSize(8)
                                     .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferWrite), {});
             cmd.copyBuffer(metrics_->buffer, trainingStaging_->buffer,
                            vk::BufferCopy{}.setDstOffset(metricsOffset).setSize(8));
             cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {},
                                 vk::BufferMemoryBarrier{}.setBuffer(trainingStaging_->buffer).setOffset(metricsOffset).setSize(8)
                                     .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite), {});
        }
    });
    engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    auto const resultBytes = vk::DeviceSize(view.window_count) * 2 * sizeof(float);
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("Tiny dataset evaluation readback map failed");
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, 0, resultBytes);
    std::vector<float> output(view.window_count * 2);
    auto const* bytes = static_cast<std::byte const*>(mapped);
    for (std::uint32_t window = 0; window < view.window_count; ++window) {
        std::uint32_t count = 0;
        std::memcpy(&output[window * 2], bytes + window * 8, sizeof(float));
        std::memcpy(&count, bytes + window * 8 + sizeof(float), sizeof(count));
        output[window * 2 + 1] = static_cast<float>(count);
    }
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    return output;
}

ForwardResourceGraph::~ForwardResourceGraph() {
    if (!context_.device)
        return;
    engine_.drain();
    context_.device.destroyDescriptorPool(ffnGradientPool_);
    context_.device.destroyPipeline(ffnGradientPipeline_);
    context_.device.destroyPipelineLayout(ffnGradientPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(ffnGradientLayout_);
    context_.device.destroyShaderModule(ffnGradientShader_);
    context_.device.destroyDescriptorPool(ffnB2AdamwPool_);
    context_.device.destroyPipeline(ffnB2AdamwPipeline_);
    context_.device.destroyPipelineLayout(ffnB2AdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(ffnB2AdamwLayout_);
    context_.device.destroyShaderModule(ffnB2AdamwShader_);
    context_.device.destroyDescriptorPool(ffnW2AdamwPool_);
    context_.device.destroyPipeline(ffnW2AdamwPipeline_);
    context_.device.destroyPipelineLayout(ffnW2AdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(ffnW2AdamwLayout_);
    context_.device.destroyShaderModule(ffnW2AdamwShader_);
    context_.device.destroyDescriptorPool(ffnB1AdamwPool_);
    context_.device.destroyPipeline(ffnB1AdamwPipeline_);
    context_.device.destroyPipelineLayout(ffnB1AdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(ffnB1AdamwLayout_);
    context_.device.destroyShaderModule(ffnB1AdamwShader_);
    context_.device.destroyDescriptorPool(rmsNormGradientPool_);
    context_.device.destroyPipeline(rmsNormGradientPipeline_);
    context_.device.destroyPipelineLayout(rmsNormGradientPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(rmsNormGradientLayout_);
    context_.device.destroyShaderModule(rmsNormGradientShader_);
    context_.device.destroyDescriptorPool(scaleDgammaPool_);
    context_.device.destroyPipeline(scaleDgammaPipeline_);
    context_.device.destroyPipelineLayout(scaleDgammaPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(scaleDgammaLayout_);
    context_.device.destroyShaderModule(scaleDgammaShader_);
    context_.device.destroyDescriptorPool(ffnAdamwPool_);
    context_.device.destroyPipeline(ffnAdamwPipeline_);
    context_.device.destroyPipelineLayout(ffnAdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(ffnAdamwLayout_);
    context_.device.destroyShaderModule(ffnAdamwShader_);
    context_.device.destroyDescriptorPool(ffnBackwardPool_);
    context_.device.destroyPipeline(ffnBackwardPipeline_);
    context_.device.destroyPipelineLayout(ffnBackwardPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(ffnBackwardLayout_);
    context_.device.destroyShaderModule(ffnBackwardShader_);
    context_.device.destroyDescriptorPool(ffnForwardPool_);
    context_.device.destroyPipeline(ffnForwardPipeline_);
    context_.device.destroyPipelineLayout(ffnForwardPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(ffnForwardLayout_);
    context_.device.destroyShaderModule(ffnForwardShader_);
     context_.device.destroyDescriptorPool(lmHeadForwardPool_);
     context_.device.destroyPipeline(lmHeadForwardPipeline_);
     context_.device.destroyPipelineLayout(lmHeadForwardPipelineLayout_);
     context_.device.destroyDescriptorSetLayout(lmHeadForwardLayout_);
     context_.device.destroyShaderModule(lmHeadForwardShader_);
    context_.device.destroyDescriptorPool(gammaAdamwPool_);
    context_.device.destroyPipeline(gammaAdamwPipeline_);
    context_.device.destroyPipelineLayout(gammaAdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(gammaAdamwLayout_);
    context_.device.destroyShaderModule(gammaAdamwShader_);
    for (auto* b :
         {trainingStaging_.get(), readback_.get(), metrics_.get(),   rowLoss_.get(),        dstates_.get(),
          dvalue_.get(),          dkey_.get(),      dquery_.get(),    qkvGradientV_.get(),   qkvGradientK_.get(),
          qkvGradientQ_.get(),    qkvM_.get(),      qkvV_.get(),      positionM_.get(),      positionV_.get(),
           dcontext_.get(),       projected_.get(),
          embeddingGradient_.get(), positionGradient_.get(), lmHeadGradient_.get(), lmHeadM_.get(), lmHeadV_.get(), outputGradient_.get(), doutput_.get(),
          dprojected_.get(),      loraVB_.get(),    loraMB_.get(),    loraVA_.get(),         loraMA_.get(),
          loraDB_.get(),          loraDA_.get(),    loraB_.get(),     loraA_.get(),          qRows_.get(),
          kRows_.get(),           vRows_.get(),     dlogits_.get(),   logits_.get(),         activations_.get(),
          states_.get(),          gammaInput_.get(), rmsRaw_.get(), rmsInv_.get(), rmsDx_.get(), mask_.get(),      targets_.get(),   tokens_.get(),         lmHead_.get(),
          output_.get(),          outputM_.get(),   outputV_.get(),   value_.get(),          key_.get(),
          query_.get(),           positions_.get(), embeddings_.get(), ffnDx_.get(), gammaV_.get(), gammaM_.get(), gammaGradient_.get(), gamma_.get(), ffnOutput_.get(), ffnB2V_.get(), ffnB1V_.get(), ffnW2V_.get(), ffnW1V_.get(), ffnB2M_.get(), ffnB1M_.get(), ffnW2M_.get(), ffnW1M_.get(), ffnB2Gradient_.get(), ffnB1Gradient_.get(), ffnW2Gradient_.get(), ffnW1Gradient_.get(), ffnB2_.get(), ffnB1_.get(), ffnW2_.get(), ffnW1_.get()})
        drop(context_, *b);
    context_.device.destroyDescriptorPool(projectionBackwardPool_);
    context_.device.destroyPipeline(projectionBackwardPipeline_);
    context_.device.destroyPipelineLayout(projectionBackwardPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(projectionBackwardLayout_);
    context_.device.destroyShaderModule(projectionBackwardShader_);
    context_.device.destroyDescriptorPool(loraPool_);
    context_.device.destroyPipeline(loraPipeline_);
    context_.device.destroyPipelineLayout(loraPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(loraLayout_);
    context_.device.destroyShaderModule(loraShader_);
    context_.device.destroyDescriptorPool(loraFinalizePool_);
    context_.device.destroyPipeline(loraFinalizePipeline_);
    context_.device.destroyPipelineLayout(loraFinalizePipelineLayout_);
    context_.device.destroyDescriptorSetLayout(loraFinalizeLayout_);
    context_.device.destroyShaderModule(loraFinalizeShader_);
    context_.device.destroyDescriptorPool(loraAdamwPool_);
    context_.device.destroyPipeline(loraAdamwPipeline_);
    context_.device.destroyPipelineLayout(loraAdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(loraAdamwLayout_);
    context_.device.destroyShaderModule(loraAdamwShader_);
    context_.device.destroyDescriptorPool(loraClearPool_);
    context_.device.destroyPipeline(loraClearPipeline_);
    context_.device.destroyPipelineLayout(loraClearPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(loraClearLayout_);
    context_.device.destroyShaderModule(loraClearShader_);
    context_.device.destroyDescriptorPool(loraSgdPool_);
    context_.device.destroyPipeline(loraSgdPipeline_);
    context_.device.destroyPipelineLayout(loraSgdPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(loraSgdLayout_);
    context_.device.destroyShaderModule(loraSgdShader_);
    context_.device.destroyDescriptorPool(embeddingSgdPool_);
    context_.device.destroyPipeline(embeddingSgdPipeline_);
    context_.device.destroyPipelineLayout(embeddingSgdPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(embeddingSgdLayout_);
    context_.device.destroyShaderModule(embeddingSgdShader_);
    context_.device.destroyPipeline(positionSgdPipeline_);
    context_.device.destroyPipelineLayout(positionSgdPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(positionSgdLayout_);
    context_.device.destroyDescriptorPool(positionSgdPool_);
    context_.device.destroyShaderModule(positionSgdShader_);
    context_.device.destroyDescriptorPool(positionAdamwPool_);
    context_.device.destroyPipeline(positionAdamwPipeline_);
    context_.device.destroyPipelineLayout(positionAdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(positionAdamwLayout_);
    context_.device.destroyShaderModule(positionAdamwShader_);
    context_.device.destroyPipeline(positionGradientPipeline_);
    context_.device.destroyPipelineLayout(positionGradientPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(positionGradientLayout_);
    context_.device.destroyDescriptorPool(positionGradientPool_);
    context_.device.destroyShaderModule(positionGradientShader_);
    context_.device.destroyDescriptorPool(embeddingGradientPool_);
    context_.device.destroyPipeline(embeddingGradientPipeline_);
    context_.device.destroyPipelineLayout(embeddingGradientPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(embeddingGradientLayout_);
    context_.device.destroyShaderModule(embeddingGradientShader_);
    context_.device.destroyDescriptorPool(qkvAdamwPool_);
    context_.device.destroyPipeline(qkvAdamwPipeline_);
    context_.device.destroyPipelineLayout(qkvAdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(qkvAdamwLayout_);
    context_.device.destroyShaderModule(qkvAdamwShader_);
    context_.device.destroyDescriptorPool(qkvSgdPool_);
    context_.device.destroyPipeline(qkvSgdPipeline_);
    context_.device.destroyPipelineLayout(qkvSgdPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(qkvSgdLayout_);
    context_.device.destroyShaderModule(qkvSgdShader_);
    context_.device.destroyDescriptorPool(qkvGradientPool_);
    context_.device.destroyPipeline(qkvGradientPipeline_);
    context_.device.destroyPipelineLayout(qkvGradientPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(qkvGradientLayout_);
    context_.device.destroyShaderModule(qkvGradientShader_);
    context_.device.destroyDescriptorPool(attentionBackwardPool_);
    context_.device.destroyPipeline(attentionBackwardPipeline_);
    context_.device.destroyPipelineLayout(attentionBackwardPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(attentionBackwardLayout_);
    context_.device.destroyShaderModule(attentionBackwardShader_);
    context_.device.destroyDescriptorPool(outputAdamwPool_);
    context_.device.destroyPipeline(outputAdamwPipeline_);
    context_.device.destroyPipelineLayout(outputAdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(outputAdamwLayout_);
    context_.device.destroyShaderModule(outputAdamwShader_);
    context_.device.destroyDescriptorPool(outputSgdPool_);
    context_.device.destroyPipeline(outputSgdPipeline_);
    context_.device.destroyPipelineLayout(outputSgdPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(outputSgdLayout_);
    context_.device.destroyShaderModule(outputSgdShader_);
    context_.device.destroyDescriptorPool(outputGradientPool_);
    context_.device.destroyPipeline(outputGradientPipeline_);
    context_.device.destroyPipelineLayout(outputGradientPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(outputGradientLayout_);
    context_.device.destroyShaderModule(outputGradientShader_);
    context_.device.destroyDescriptorPool(lmHeadAdamwPool_);
    context_.device.destroyPipeline(lmHeadAdamwPipeline_);
    context_.device.destroyPipelineLayout(lmHeadAdamwPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(lmHeadAdamwLayout_);
    context_.device.destroyShaderModule(lmHeadAdamwShader_);
    context_.device.destroyDescriptorPool(lmHeadSgdPool_);
    context_.device.destroyPipeline(lmHeadSgdPipeline_);
    context_.device.destroyPipelineLayout(lmHeadSgdPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(lmHeadSgdLayout_);
    context_.device.destroyShaderModule(lmHeadSgdShader_);
    context_.device.destroyDescriptorPool(lmHeadGradientPool_);
    context_.device.destroyPipeline(lmHeadGradientPipeline_);
    context_.device.destroyPipelineLayout(lmHeadGradientPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(lmHeadGradientLayout_);
    context_.device.destroyShaderModule(lmHeadGradientShader_);
    context_.device.destroyDescriptorPool(lmBackwardPool_);
    context_.device.destroyPipeline(lmBackwardPipeline_);
    context_.device.destroyPipelineLayout(lmBackwardPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(lmBackwardLayout_);
    context_.device.destroyShaderModule(lmBackwardShader_);
    context_.device.destroyDescriptorPool(metricsPool_);
    context_.device.destroyPipeline(metricsPipeline_);
    context_.device.destroyPipelineLayout(metricsPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(metricsLayout_);
    context_.device.destroyShaderModule(metricsShader_);
    context_.device.destroyDescriptorPool(lossPool_);
    context_.device.destroyPipeline(lossPipeline_);
    context_.device.destroyPipelineLayout(lossPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(lossLayout_);
    context_.device.destroyShaderModule(lossShader_);
    context_.device.destroyDescriptorPool(descriptorPool_);
    context_.device.destroyPipeline(pipeline_);
    context_.device.destroyPipelineLayout(pipelineLayout_);
    context_.device.destroyDescriptorSetLayout(descriptorLayout_);
    context_.device.destroyShaderModule(shader_);
    context_.device.destroyDescriptorPool(rmsFixedForwardPool_);
     context_.device.destroyPipeline(rmsFixedForwardPipeline_);
     context_.device.destroyPipelineLayout(rmsFixedForwardPipelineLayout_);
     context_.device.destroyDescriptorSetLayout(rmsFixedForwardLayout_);
     context_.device.destroyShaderModule(rmsFixedForwardShader_);
     context_.device.destroyDescriptorPool(rmsDgammaStatePool_);
    context_.device.destroyPipeline(rmsDgammaStatePipeline_);
    context_.device.destroyPipelineLayout(rmsDgammaStatePipelineLayout_);
    context_.device.destroyDescriptorSetLayout(rmsDgammaStateLayout_);
    context_.device.destroyShaderModule(rmsDgammaStateShader_);
    context_.device.destroyDescriptorPool(rmsBackwardStatePool_);
    context_.device.destroyPipeline(rmsBackwardStatePipeline_);
    context_.device.destroyPipelineLayout(rmsBackwardStatePipelineLayout_);
    context_.device.destroyDescriptorSetLayout(rmsBackwardStateLayout_);
    context_.device.destroyShaderModule(rmsBackwardStateShader_);
    context_.device.destroyDescriptorPool(rmsForwardPool_);
    context_.device.destroyPipeline(rmsForwardPipeline_);
    context_.device.destroyPipelineLayout(rmsForwardPipelineLayout_);
    context_.device.destroyDescriptorSetLayout(rmsForwardLayout_);
    context_.device.destroyShaderModule(rmsForwardShader_);
}
int ForwardResourceGraph::readback_base_checkpoint(BaseCheckpoint& checkpoint) noexcept {
    try {
        checkpoint = BaseCheckpoint{};
        checkpoint.version = 4;
        checkpoint.profile = profile_descriptor();
        checkpoint.group_mask = BaseCheckpointLmHead | BaseCheckpointOutput | BaseCheckpointQKV |
                                BaseCheckpointEmbeddings | BaseCheckpointPositions | BaseCheckpointNormalization | BaseCheckpointFfn;
        checkpoint.embeddings.resize(V * H);
        checkpoint.positions.resize(Tcap * H);
         checkpoint.positions_m.resize(Tcap * H);
         checkpoint.positions_v.resize(Tcap * H);
         checkpoint.gamma.resize(H); checkpoint.gamma_m.resize(H); checkpoint.gamma_v.resize(H);
        checkpoint.lm_head.resize(H * Vp);
        checkpoint.output.resize(H * H);
        checkpoint.query.resize(H * H);
        checkpoint.key.resize(H * H);
        checkpoint.value.resize(H * H);
        checkpoint.lm_head_m.resize(H * Vp);
        checkpoint.lm_head_v.resize(H * Vp);
        checkpoint.output_m.resize(H * H);
        checkpoint.output_v.resize(H * H);
        checkpoint.query_m.resize(H * H);
        checkpoint.key_m.resize(H * H);
        checkpoint.value_m.resize(H * H);
        checkpoint.query_v.resize(H * H);
        checkpoint.key_v.resize(H * H);
        checkpoint.value_v.resize(H * H);
        checkpoint.ffn_w1.resize(H*4*H); checkpoint.ffn_b1.resize(4*H); checkpoint.ffn_w2.resize(4*H*H); checkpoint.ffn_b2.resize(H);
        checkpoint.ffn_w1_m.resize(H*4*H); checkpoint.ffn_b1_m.resize(4*H); checkpoint.ffn_w2_m.resize(4*H*H); checkpoint.ffn_b2_m.resize(H);
        checkpoint.ffn_w1_v.resize(H*4*H); checkpoint.ffn_b1_v.resize(4*H); checkpoint.ffn_w2_v.resize(4*H*H); checkpoint.ffn_b2_v.resize(H);
        if (readback_base_train_embeddings(checkpoint.embeddings.data()) != 0 ||
            readback_base_train_positions(checkpoint.positions.data()) != 0 ||
            readback_base_train_qkv_adamw_state(checkpoint.query.data(),
                                                checkpoint.key.data(),
                                                checkpoint.value.data(),
                                                checkpoint.query_m.data(),
                                                checkpoint.key_m.data(),
                                                checkpoint.value_m.data(),
                                                checkpoint.query_v.data(),
                                                checkpoint.key_v.data(),
                                                checkpoint.value_v.data(),
                                                &checkpoint.adamw_step) != 0 ||
            readback_base_train_lm_head_adamw_state(checkpoint.lm_head.data(),
                                                    checkpoint.lm_head_m.data(),
                                                    checkpoint.lm_head_v.data(),
                                                    &checkpoint.adamw_step) != 0 ||
            readback_base_train_output_adamw_state(checkpoint.output.data(),
                                                   checkpoint.output_m.data(),
                                                   checkpoint.output_v.data(),
                                                   &checkpoint.adamw_step) != 0 ||
             readback_gamma_state(checkpoint.gamma.data(), checkpoint.gamma_m.data(), checkpoint.gamma_v.data(), &checkpoint.adamw_step) != 0)
            return 2;
        std::vector<float> ffn(3*(H*4*H+4*H+4*H*H+H));
        if (readback_ffn_state(ffn.data(), ffn.size(), &checkpoint.adamw_step) != 0) return 2;
        std::size_t at=0; auto split=[&](std::vector<float>& out){std::copy(ffn.begin()+at,ffn.begin()+at+out.size(),out.begin());at+=out.size();};
        split(checkpoint.ffn_w1); split(checkpoint.ffn_b1); split(checkpoint.ffn_w2); split(checkpoint.ffn_b2); split(checkpoint.ffn_w1_m); split(checkpoint.ffn_b1_m); split(checkpoint.ffn_w2_m); split(checkpoint.ffn_b2_m); split(checkpoint.ffn_w1_v); split(checkpoint.ffn_b1_v); split(checkpoint.ffn_w2_v); split(checkpoint.ffn_b2_v);
        return 0;
    } catch (...) {
        return 4;
    }
}
int ForwardResourceGraph::update_base_checkpoint(BaseCheckpoint const& checkpoint) noexcept {
    try {
        constexpr auto allGroups = BaseCheckpointLmHead | BaseCheckpointOutput | BaseCheckpointQKV |
                                   BaseCheckpointEmbeddings | BaseCheckpointPositions | BaseCheckpointNormalization | BaseCheckpointFfn;
        auto all_finite = [](std::vector<float> const& values) {
            return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
        };

        if (checkpoint.version != 4 || checkpoint.group_mask != allGroups ||
            !profile_supported(checkpoint.profile.hidden,
                               checkpoint.profile.vocab,
                               checkpoint.profile.padded_vocab,
                               checkpoint.profile.token_capacity,
                               checkpoint.profile.lora_rank) ||
            checkpoint.profile.lora_rank != loraRank_ || checkpoint.embeddings.size() != V * H ||
             checkpoint.positions.size() != Tcap * H || checkpoint.positions_m.size() != Tcap * H || checkpoint.positions_v.size() != Tcap * H || checkpoint.lm_head.size() != H * Vp ||
            checkpoint.output.size() != H * H || checkpoint.query.size() != H * H || checkpoint.key.size() != H * H ||
            checkpoint.value.size() != H * H || checkpoint.lm_head_m.size() != H * Vp ||
            checkpoint.lm_head_v.size() != H * Vp || checkpoint.output_m.size() != H * H ||
            checkpoint.output_v.size() != H * H || checkpoint.query_m.size() != H * H ||
            checkpoint.key_m.size() != H * H || checkpoint.value_m.size() != H * H ||
            checkpoint.query_v.size() != H * H || checkpoint.key_v.size() != H * H ||
            checkpoint.value_v.size() != H * H || checkpoint.gamma.size() != H || checkpoint.gamma_m.size() != H || checkpoint.gamma_v.size() != H || checkpoint.ffn_w1.size() != H*4*H || checkpoint.ffn_b1.size() != 4*H || checkpoint.ffn_w2.size() != 4*H*H || checkpoint.ffn_b2.size() != H || checkpoint.ffn_w1_m.size() != H*4*H || checkpoint.ffn_b1_m.size() != 4*H || checkpoint.ffn_w2_m.size() != 4*H*H || checkpoint.ffn_b2_m.size() != H || checkpoint.ffn_w1_v.size() != H*4*H || checkpoint.ffn_b1_v.size() != 4*H || checkpoint.ffn_w2_v.size() != 4*H*H || checkpoint.ffn_b2_v.size() != H || checkpoint.adamw_step > UINT32_MAX)
            return 1;
        if (!all_finite(checkpoint.embeddings) || !all_finite(checkpoint.positions) || !all_finite(checkpoint.positions_m) || !all_finite(checkpoint.positions_v) ||
             !all_finite(checkpoint.gamma) || !all_finite(checkpoint.gamma_m) || !all_finite(checkpoint.gamma_v) || !all_finite(checkpoint.lm_head) || !all_finite(checkpoint.output) ||
             !all_finite(checkpoint.query) || !all_finite(checkpoint.key) || !all_finite(checkpoint.value) || !all_finite(checkpoint.lm_head_m) || !all_finite(checkpoint.lm_head_v) ||
             !all_finite(checkpoint.output_m) || !all_finite(checkpoint.output_v) || !all_finite(checkpoint.query_m) || !all_finite(checkpoint.key_m) || !all_finite(checkpoint.value_m) ||
             !all_finite(checkpoint.query_v) || !all_finite(checkpoint.key_v) || !all_finite(checkpoint.value_v) || !all_finite(checkpoint.ffn_w1) || !all_finite(checkpoint.ffn_b1) ||
             !all_finite(checkpoint.ffn_w2) || !all_finite(checkpoint.ffn_b2) || !all_finite(checkpoint.ffn_w1_m) || !all_finite(checkpoint.ffn_b1_m) || !all_finite(checkpoint.ffn_w2_m) ||
             !all_finite(checkpoint.ffn_b2_m) || !all_finite(checkpoint.ffn_w1_v) || !all_finite(checkpoint.ffn_b1_v) || !all_finite(checkpoint.ffn_w2_v) || !all_finite(checkpoint.ffn_b2_v))
             return 1;
         std::vector<float> ffn_state;
        ffn_state.reserve(3*(H*4*H+4*H+4*H*H+H));
        auto append_ffn=[&](std::vector<float> const& x){ffn_state.insert(ffn_state.end(),x.begin(),x.end());};
        append_ffn(checkpoint.ffn_w1); append_ffn(checkpoint.ffn_b1); append_ffn(checkpoint.ffn_w2); append_ffn(checkpoint.ffn_b2); append_ffn(checkpoint.ffn_w1_m); append_ffn(checkpoint.ffn_b1_m); append_ffn(checkpoint.ffn_w2_m); append_ffn(checkpoint.ffn_b2_m); append_ffn(checkpoint.ffn_w1_v); append_ffn(checkpoint.ffn_b1_v); append_ffn(checkpoint.ffn_w2_v); append_ffn(checkpoint.ffn_b2_v);
        if (import_base_train_embeddings(checkpoint.embeddings.data()) != 0 ||
            update_base_train_positions_adamw_state(checkpoint.positions.data(), checkpoint.positions_m.data(), checkpoint.positions_v.data(), checkpoint.adamw_step) != 0 ||
            import_base_train_lm_head(checkpoint.lm_head.data()) != 0 ||
            import_base_train_output(checkpoint.output.data()) != 0 ||
            import_base_train_qkv(checkpoint.query.data(), checkpoint.key.data(), checkpoint.value.data()) != 0 ||
            update_base_train_lm_head_adamw_state(checkpoint.lm_head.data(),
                                                  checkpoint.lm_head_m.data(),
                                                  checkpoint.lm_head_v.data(),
                                                  checkpoint.adamw_step) != 0 ||
            update_base_train_qkv_adamw_state(checkpoint.query.data(),
                                              checkpoint.key.data(),
                                              checkpoint.value.data(),
                                              checkpoint.query_m.data(),
                                              checkpoint.key_m.data(),
                                              checkpoint.value_m.data(),
                                              checkpoint.query_v.data(),
                                              checkpoint.key_v.data(),
                                              checkpoint.value_v.data(),
                                              checkpoint.adamw_step) != 0 ||
            update_base_train_output_adamw_state(checkpoint.output.data(),
                                                 checkpoint.output_m.data(),
                                                 checkpoint.output_v.data(),
                                                 checkpoint.adamw_step) != 0 ||
             update_gamma_state(checkpoint.gamma.data(), checkpoint.gamma_m.data(), checkpoint.gamma_v.data(), checkpoint.adamw_step) != 0 ||
             update_ffn_state(ffn_state.data(), ffn_state.size(), checkpoint.adamw_step) != 0)
            return 2;
        return 0;
    } catch (...) {
        return 3;
    }
}
int ForwardResourceGraph::import_base_train_lm_head(float const* weight) {
    if (!weight)
        return 1;
    upload(context_, lmHead_->buffer, weight, H * Vp * sizeof(float));
    return 0;
}
int ForwardResourceGraph::import_base_train_output(float const* weight) {
    if (!weight)
        return 1;
    upload(context_, output_->buffer, weight, H * H * sizeof(float));
    return 0;
}
int ForwardResourceGraph::readback_base_train_output_adamw_state(float* weight,
                                                                 float* m,
                                                                 float* v,
                                                                 std::uint64_t* step) {
    if (!weight || !m || !v || !step)
        return 1;
    if (readback_base_train_output(weight) != 0)
        return 2;
    constexpr vk::DeviceSize bytes = H * H * sizeof(float);
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit(
        [this, base](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                {},
                                {vk::BufferMemoryBarrier{}
                                     .setBuffer(outputM_->buffer)
                                     .setSize(H * H * sizeof(float))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(outputV_->buffer)
                                     .setSize(H * H * sizeof(float))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead)},
                                {});
            cmd.copyBuffer(
                outputM_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(bytes));
            cmd.copyBuffer(
                outputV_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base + bytes).setSize(bytes));
        },
        0);
    engine_.wait(lastSubmission_);
    void* p = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &p) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, 2 * bytes);
    auto* state = static_cast<std::uint8_t*>(p) + base;
    std::memcpy(m, state, bytes);
    std::memcpy(v, state + bytes, bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    *step = adamwStep_;
    return 0;
}
int ForwardResourceGraph::update_base_train_output_adamw_state(float const* weight,
                                                               float const* m,
                                                               float const* v,
                                                               std::uint64_t step) {
    if (!weight || !m || !v || step > UINT32_MAX)
        return 1;
    if (import_base_train_output(weight) != 0)
        return 2;
    upload(context_, outputM_->buffer, m, H * H * sizeof(float));
    upload(context_, outputV_->buffer, v, H * H * sizeof(float));
    adamwStep_ = step;
    return 0;
}
int ForwardResourceGraph::readback_base_train_output(float* weight) {
    if (!weight)
        return 1;
    void* mapped = nullptr;
    lastSubmission_ = engine_.submit([this](vk::CommandBuffer cmd) {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(output_->buffer)
                                .setSize(H * H * sizeof(float))
                                .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                            {});
        cmd.copyBuffer(output_->buffer, readback_->buffer, vk::BufferCopy{}.setSize(H * H * sizeof(float)));
    });
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, readback_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, H * H * sizeof(float));
    std::memcpy(weight, mapped, H * H * sizeof(float));
    vmaUnmapMemory(context_.allocator, readback_->allocation);
    return 0;
}
int ForwardResourceGraph::train_lm_head_sgd(std::uint32_t const* tokens,
                                            std::uint32_t const* targets,
                                            std::uint32_t const* masks,
                                            std::uint32_t rows,
                                            float learning_rate) noexcept {
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap || !std::isfinite(learning_rate) || learning_rate <= 0.0f)
        return 1;
    for (std::uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u)
            return 1;
    constexpr vk::DeviceSize tokenBytes = Tcap * sizeof(std::uint32_t);
    constexpr vk::DeviceSize targetOffset = tokenBytes;
    constexpr vk::DeviceSize maskOffset = targetOffset + Tcap * sizeof(std::uint32_t);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    std::memset(bytes, 0, static_cast<size_t>(maskOffset + Tcap * sizeof(std::uint32_t)));
    std::memcpy(bytes, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(bytes + targetOffset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(bytes + maskOffset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, maskOffset + tokenBytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);

    ForwardPC forward_pc{rows, 0};
    LossPC loss_pc{rows, V, Vp};
    LmHeadGradientPC gradient_pc{rows, Tcap, V, Vp, H, 0};
    LmHeadSgdPC sgd_pc{H * Vp, learning_rate};
    lastSubmission_ = engine_.submit(
        [this, forward_pc, loss_pc, gradient_pc, sgd_pc, rows](vk::CommandBuffer cmd) {
            cmd.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(tokenBytes));
            cmd.copyBuffer(trainingStaging_->buffer,
                           targets_->buffer,
                           vk::BufferCopy{}.setSrcOffset(targetOffset).setSize(tokenBytes));
            cmd.copyBuffer(
                trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(maskOffset).setSize(tokenBytes));
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(forward_pc), &forward_pc);
            cmd.dispatch(5, forward_pc.seq_length, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
            cmd.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(loss_pc), &loss_pc);
            cmd.dispatch(rows, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmHeadGradientPipeline_);
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, lmHeadGradientPipelineLayout_, 0, lmHeadGradientSet_, {});
            cmd.pushConstants(
                lmHeadGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(gradient_pc), &gradient_pc);
            cmd.dispatch((H * Vp + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmHeadSgdPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmHeadSgdPipelineLayout_, 0, lmHeadSgdSet_, {});
            cmd.pushConstants(lmHeadSgdPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(sgd_pc), &sgd_pc);
            cmd.dispatch((sgd_pc.count + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::import_base_train_qkv(float const* query, float const* key, float const* value) {
    if (!query || !key || !value)
        return 1;
    upload(context_, query_->buffer, query, H * H * sizeof(float));
    upload(context_, key_->buffer, key, H * H * sizeof(float));
    upload(context_, value_->buffer, value, H * H * sizeof(float));
    return 0;
}
int ForwardResourceGraph::readback_base_train_qkv(float* query, float* key, float* value) {
    if (!query || !key || !value)
        return 1;
    for (auto pair :
         {std::pair<vk::Buffer, float*>{query_->buffer, query}, {key_->buffer, key}, {value_->buffer, value}}) {
        lastSubmission_ = engine_.submit([this, pair](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                {},
                                vk::BufferMemoryBarrier{}
                                    .setBuffer(pair.first)
                                    .setSize(H * H * sizeof(float))
                                    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                    .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                {});
            cmd.copyBuffer(pair.first, readback_->buffer, vk::BufferCopy{}.setSize(H * H * sizeof(float)));
        });
        engine_.wait(lastSubmission_);
        void* mapped = nullptr;
        if (vmaMapMemory(context_.allocator, readback_->allocation, &mapped) != VK_SUCCESS)
            return 2;
        vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, H * H * sizeof(float));
        std::memcpy(pair.second, mapped, H * H * sizeof(float));
        vmaUnmapMemory(context_.allocator, readback_->allocation);
    }
    return 0;
}
int ForwardResourceGraph::readback_base_train_qkv_gradients(float* query, float* key, float* value) {
    if (!query || !key || !value)
        return 1;
    for (auto pair : {std::pair<vk::Buffer, float*>{qkvGradientQ_->buffer, query},
                      {qkvGradientK_->buffer, key},
                      {qkvGradientV_->buffer, value}}) {
        lastSubmission_ = engine_.submit([this, pair](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                {},
                                vk::BufferMemoryBarrier{}
                                    .setBuffer(pair.first)
                                    .setSize(H * H * sizeof(float))
                                    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                    .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                {});
            cmd.copyBuffer(pair.first, readback_->buffer, vk::BufferCopy{}.setSize(H * H * sizeof(float)));
        });
        engine_.wait(lastSubmission_);
        void* mapped = nullptr;
        if (vmaMapMemory(context_.allocator, readback_->allocation, &mapped) != VK_SUCCESS)
            return 2;
        vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, H * H * sizeof(float));
        std::memcpy(pair.second, mapped, H * H * sizeof(float));
        vmaUnmapMemory(context_.allocator, readback_->allocation);
    }
    return 0;
}
int ForwardResourceGraph::readback_base_train_qkv_adamw_state(float* query,
                                                              float* key,
                                                              float* value,
                                                              float* qm,
                                                              float* km,
                                                              float* vm,
                                                              float* qv,
                                                              float* kv,
                                                              float* vv,
                                                              std::uint64_t* step) {
    if (!query || !key || !value || !qm || !km || !vm || !qv || !kv || !vv || !step)
        return 1;
    if (readback_base_train_qkv(query, key, value) != 0)
        return 2;
    constexpr vk::DeviceSize b = H * H * sizeof(float), total = 3 * b;
    lastSubmission_ = engine_.submit([this](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eTransfer,
                          {},
                          {},
                          {vk::BufferMemoryBarrier{}
                               .setBuffer(qkvM_->buffer)
                               .setSize(3 * H * H * sizeof(float))
                               .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                               .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                           vk::BufferMemoryBarrier{}
                               .setBuffer(qkvV_->buffer)
                               .setSize(3 * H * H * sizeof(float))
                               .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                               .setDstAccessMask(vk::AccessFlagBits::eTransferRead)},
                          {});
        c.copyBuffer(qkvM_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(0).setSize(total));
        c.copyBuffer(qkvV_->buffer,
                     trainingStaging_->buffer,
                     vk::BufferCopy{}.setSrcOffset(0).setDstOffset(total).setSize(total));
    });
    engine_.wait(lastSubmission_);
    void* p = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &p) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, 0, 2 * total);
    auto* x = static_cast<std::uint8_t*>(p);
    std::memcpy(qm, x, b);
    std::memcpy(km, x + b, b);
    std::memcpy(vm, x + 2 * b, b);
    std::memcpy(qv, x + total, b);
    std::memcpy(kv, x + total + b, b);
    std::memcpy(vv, x + total + 2 * b, b);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    *step = adamwStep_;
    return 0;
}
int ForwardResourceGraph::update_base_train_qkv_adamw_state(float const* query,
                                                            float const* key,
                                                            float const* value,
                                                            float const* qm,
                                                            float const* km,
                                                            float const* vm,
                                                            float const* qv,
                                                            float const* kv,
                                                            float const* vv,
                                                            std::uint64_t step) {
    if (!query || !key || !value || !qm || !km || !vm || !qv || !kv || !vv || step > UINT32_MAX)
        return 1;
    if (import_base_train_qkv(query, key, value) != 0)
        return 2;
    constexpr vk::DeviceSize b = H * H * sizeof(float), total = 3 * b;
    void* p = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &p) != VK_SUCCESS)
        return 2;
    auto* x = static_cast<std::uint8_t*>(p);
    std::memcpy(x, qm, b);
    std::memcpy(x + b, km, b);
    std::memcpy(x + 2 * b, vm, b);
    std::memcpy(x + total, qv, b);
    std::memcpy(x + total + b, kv, b);
    std::memcpy(x + total + 2 * b, vv, b);
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 2 * total);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    lastSubmission_ = engine_.submit([this, total](vk::CommandBuffer c) {
        c.copyBuffer(
            trainingStaging_->buffer, qkvM_->buffer, vk::BufferCopy{}.setSrcOffset(0).setDstOffset(0).setSize(total));
        c.copyBuffer(trainingStaging_->buffer,
                     qkvV_->buffer,
                     vk::BufferCopy{}.setSrcOffset(total).setDstOffset(0).setSize(total));
    });
    engine_.wait(lastSubmission_);
    adamwStep_ = step;
    return 0;
}
int ForwardResourceGraph::train_qkv_adamw_from_gradients(float learning_rate,
                                                         float beta1,
                                                         float beta2,
                                                         float epsilon,
                                                         float weight_decay) noexcept {
    if (!std::isfinite(learning_rate) || !std::isfinite(beta1) || !std::isfinite(beta2) || !std::isfinite(epsilon) ||
        !std::isfinite(weight_decay) || learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f ||
        beta2 >= 1.0f || epsilon <= 0.0f || weight_decay < 0.0f || adamwStep_ == UINT32_MAX)
        return 1;
    ++adamwStep_;
    QkvAdamwPC a{3 * H * H, H, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay};
    lastSubmission_ = engine_.submit(
        [this, a](vk::CommandBuffer c) {
            c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                              vk::PipelineStageFlagBits::eComputeShader,
                              {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                              {},
                              {});
            c.bindPipeline(vk::PipelineBindPoint::eCompute, qkvAdamwPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, qkvAdamwPipelineLayout_, 0, qkvAdamwSet_, {});
            c.pushConstants(qkvAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(a), &a);
            c.dispatch((a.count + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::train_qkv_sgd(std::uint32_t const* tokens,
                                        std::uint32_t const* targets,
                                        std::uint32_t const* masks,
                                        std::uint32_t rows,
                                        float learning_rate) noexcept {
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap || !std::isfinite(learning_rate) ||
        learning_rate <= 0.0f)
        return 1;
    for (uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1)
            return 1;
    constexpr vk::DeviceSize bytes = Tcap * sizeof(uint32_t), yo = bytes, mo = 2 * bytes;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* b = static_cast<uint8_t*>(mapped);
    std::memset(b, 0, 3 * bytes);
    std::memcpy(b, tokens, rows * sizeof(uint32_t));
    std::memcpy(b + yo, targets, rows * sizeof(uint32_t));
    std::memcpy(b + mo, masks, rows * sizeof(uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    ForwardPC f{rows, 0};
    LossPC l{rows, V, Vp};
    LmBackwardPC bw{rows, V, H, Vp, Vp, H};
    QkvGradientPC g{rows, Tcap, H};
    QkvSgdPC s{3 * H * H, learning_rate};
    lastSubmission_ = engine_.submit(
        [this, f, l, bw, g, s, rows](vk::CommandBuffer c) {
            c.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(bytes));
            c.copyBuffer(trainingStaging_->buffer, targets_->buffer, vk::BufferCopy{}.setSrcOffset(yo).setSize(bytes));
            c.copyBuffer(trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(mo).setSize(bytes));
            c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                              vk::PipelineStageFlagBits::eComputeShader,
                              {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                              {},
                              {});
            c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            c.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(f), &f);
            c.dispatch(5, f.seq_length, 1);
            auto bar = [&]() {
                c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  {},
                                  vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                  {},
                                  {});
            };
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
            c.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(l), &l);
            c.dispatch(rows, 1, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, lmBackwardPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmBackwardPipelineLayout_, 0, lmBackwardSet_, {});
            c.pushConstants(lmBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(bw), &bw);
            c.dispatch((rows * H + 255) / 256, 1, 1);
            bar();
            ProjectionBackwardPC pp{rows, H, H, 0};
            c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_);
            c.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardSet_, {});
            c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pp), &pp);
            c.dispatch((rows * H + 255) / 256, 1, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, attentionBackwardPipeline_);
            vk::DescriptorSet as[3]{attentionBackwardQSet_, attentionBackwardKSet_, attentionBackwardVSet_};
            for (uint32_t m = 0; m < 3; ++m) {
                c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, attentionBackwardPipelineLayout_, 0, as[m], {});
                AttentionBackwardPC ap{rows, H, m};
                c.pushConstants(
                    attentionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ap), &ap);
                c.dispatch((rows * H + 255) / 256, 1, 1);
                if (m < 2)
                    bar();
            }
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, qkvGradientPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, qkvGradientPipelineLayout_, 0, qkvGradientSet_, {});
            c.pushConstants(qkvGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(g), &g);
            c.dispatch((3 * H * H + 255) / 256, 1, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, qkvSgdPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, qkvSgdPipelineLayout_, 0, qkvSgdSet_, {});
            c.pushConstants(qkvSgdPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(s), &s);
            c.dispatch((s.count + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::train_embeddings_sgd(std::uint32_t const* tokens,
                                                std::uint32_t const* targets,
                                                std::uint32_t const* masks,
                                                std::uint32_t rows,
                                                float learning_rate) noexcept {
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap || !std::isfinite(learning_rate) || learning_rate <= 0.0f)
        return 1;
    for (std::uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u)
            return 1;
    constexpr vk::DeviceSize bytes = Tcap * sizeof(std::uint32_t), target_offset = bytes, mask_offset = 2 * bytes;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* staged = static_cast<std::uint8_t*>(mapped);
    std::memset(staged, 0, 3 * bytes);
    std::memcpy(staged, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(staged + target_offset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(staged + mask_offset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    ForwardPC f{rows, 0}; LossPC l{rows, V, Vp}; LmBackwardPC bw{rows, V, H, Vp, Vp, H};
    EmbeddingGradientPC eg{V, H, rows, 1}; EmbeddingSgdPC es{V * H, learning_rate};
    lastSubmission_ = engine_.submit([this, f, l, bw, eg, es, rows](vk::CommandBuffer c) {
        c.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(bytes));
        c.copyBuffer(trainingStaging_->buffer, targets_->buffer, vk::BufferCopy{}.setSrcOffset(target_offset).setSize(bytes));
        c.copyBuffer(trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(mask_offset).setSize(bytes));
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
                          vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        auto bar = [&]() { c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
                                              vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {}); };
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        c.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(f), &f); c.dispatch(5, f.seq_length, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
        c.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(l), &l); c.dispatch(l.rows, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, lmBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmBackwardPipelineLayout_, 0, lmBackwardSet_, {});
        c.pushConstants(lmBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(bw), &bw); c.dispatch((rows * H + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardSet_, {});
        ProjectionBackwardPC pp{rows, H, H, 0}; c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pp), &pp); c.dispatch((rows * H + 255) / 256, 1, 1); bar();
        vk::DescriptorSet as[3]{attentionBackwardQSet_, attentionBackwardKSet_, attentionBackwardVSet_};
        for (std::uint32_t m = 0; m < 3; ++m) { c.bindPipeline(vk::PipelineBindPoint::eCompute, attentionBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, attentionBackwardPipelineLayout_, 0, as[m], {}); AttentionBackwardPC ap{rows, H, m}; c.pushConstants(attentionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ap), &ap); c.dispatch((rows * H + 255) / 256, 1, 1); if (m != 2) bar(); }
        bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_);
        ProjectionBackwardPC qp{rows, H, H, 0};
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardQkvSet_, {});
        c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qp), &qp);
        c.dispatch((rows * H + 255) / 256, 1, 1); bar();
        ProjectionBackwardPC qk{rows, H, H, 1};
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardKVSet_, {});
        c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qk), &qk);
        c.dispatch((rows * H + 255) / 256, 1, 1); bar();
        ProjectionBackwardPC qv{rows, H, H, 1};
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardVVSet_, {});
        c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qv), &qv);
        c.dispatch((rows * H + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, embeddingGradientPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, embeddingGradientPipelineLayout_, 0, embeddingGradientSet_, {}); c.pushConstants(embeddingGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(eg), &eg); c.dispatch((V * H + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, embeddingSgdPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, embeddingSgdPipelineLayout_, 0, embeddingSgdSet_, {}); c.pushConstants(embeddingSgdPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(es), &es); c.dispatch((es.count + 255) / 256, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::readback_position_gradient(float* output, std::size_t count) noexcept {
    if (!output || count != Tcap * H) return 1;
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(positionGradient_->buffer).setSize(Tcap * H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(positionGradient_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(Tcap * H * sizeof(float)));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, Tcap * H * sizeof(float));
    std::memcpy(output, static_cast<std::uint8_t*>(mapped) + base, Tcap * H * sizeof(float));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    return 0;
}
int ForwardResourceGraph::readback_positions(float* output) {
    if (!output)
        return 1;
    lastSubmission_ = engine_.submit([this](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(positions_->buffer).setSize(Tcap * H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(positions_->buffer, readback_->buffer, vk::BufferCopy{}.setSize(Tcap * H * sizeof(float)));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, readback_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, Tcap * H * sizeof(float));
    std::memcpy(output, mapped, Tcap * H * sizeof(float));
    vmaUnmapMemory(context_.allocator, readback_->allocation);
    return 0;
}
int ForwardResourceGraph::train_positions_sgd(std::uint32_t const* tokens,
                                               std::uint32_t const* targets,
                                               std::uint32_t const* masks,
                                               std::uint32_t rows,
                                               float learning_rate) {
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap || !std::isfinite(learning_rate) || learning_rate <= 0.0f)
        return 1;
    for (std::uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u)
            return 1;
    constexpr vk::DeviceSize bytes = Tcap * sizeof(std::uint32_t), target_offset = bytes, mask_offset = 2 * bytes;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* staged = static_cast<std::uint8_t*>(mapped);
    std::memset(staged, 0, 3 * bytes);
    std::memcpy(staged, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(staged + target_offset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(staged + mask_offset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    ForwardPC f{rows, 0}; LossPC l{rows, V, Vp}; LmBackwardPC bw{rows, V, H, Vp, Vp, H};
    PositionGradientPC pg{H, rows}; PositionSgdPC ps{H, rows, learning_rate};
    lastSubmission_ = engine_.submit([this, f, l, bw, pg, ps](vk::CommandBuffer c) {
        c.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(bytes));
        c.copyBuffer(trainingStaging_->buffer, targets_->buffer, vk::BufferCopy{}.setSrcOffset(target_offset).setSize(bytes));
        c.copyBuffer(trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(mask_offset).setSize(bytes));
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
                          vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        auto bar = [&]() { c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
                                               vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {}); };
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        c.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(f), &f); c.dispatch(5, f.seq_length, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
        c.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(l), &l); c.dispatch(l.rows, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, lmBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmBackwardPipelineLayout_, 0, lmBackwardSet_, {});
        c.pushConstants(lmBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(bw), &bw); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardSet_, {});
        ProjectionBackwardPC pp{bw.rows, H, H, 0}; c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pp), &pp); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, vk::BufferMemoryBarrier{}.setBuffer(dcontext_->buffer).setSize(bw.rows * H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(dcontext_->buffer, doutput_->buffer, vk::BufferCopy{}.setSize(bw.rows * H * sizeof(float)));
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, vk::BufferMemoryBarrier{}.setBuffer(doutput_->buffer).setSize(bw.rows * H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eTransferWrite).setDstAccessMask(vk::AccessFlagBits::eShaderRead), {});
        vk::DescriptorSet as[3]{attentionBackwardQSet_, attentionBackwardKSet_, attentionBackwardVSet_};
        for (std::uint32_t m = 0; m < 3; ++m) { c.bindPipeline(vk::PipelineBindPoint::eCompute, attentionBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, attentionBackwardPipelineLayout_, 0, as[m], {}); AttentionBackwardPC ap{bw.rows, H, m}; c.pushConstants(attentionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ap), &ap); c.dispatch((bw.rows * H + 255) / 256, 1, 1); if (m != 2) bar(); }
        bar(); c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardQkvSet_, {});
        ProjectionBackwardPC qp{bw.rows, H, H, 0}; c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qp), &qp); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        ProjectionBackwardPC qk{bw.rows, H, H, 1}; c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardKVSet_, {}); c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qk), &qk); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        ProjectionBackwardPC qv{bw.rows, H, H, 1}; c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardVVSet_, {}); c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qv), &qv); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, positionGradientPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, positionGradientPipelineLayout_, 0, positionGradientSet_, {}); c.pushConstants(positionGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pg), &pg); c.dispatch((pg.rows * pg.hidden + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, positionSgdPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, positionSgdPipelineLayout_, 0, positionSgdSet_, {}); c.pushConstants(positionSgdPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ps), &ps); c.dispatch((ps.rows * ps.hidden + 255) / 256, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::readback_trainable_gamma(float* data) const noexcept {
    if (!data) return 1;
    auto* self=const_cast<ForwardResourceGraph*>(this); auto const base=adamw_staging_offset(self->loraRank_)+3*(H*4*H+4*H+4*H*H+H)*sizeof(float);
    self->lastSubmission_=self->engine_.submit([self,base](vk::CommandBuffer c){
        c.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {},
                          vk::MemoryBarrier(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                            vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                            vk::AccessFlagBits::eTransferWrite), {}, {});
        c.copyBuffer(self->gamma_->buffer,self->trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(base).setSize(H*sizeof(float)));
    },0); self->engine_.wait(self->lastSubmission_); void* mapped=nullptr; if(vmaMapMemory(context_.allocator,trainingStaging_->allocation,&mapped)!=VK_SUCCESS)return 2;vmaInvalidateAllocation(context_.allocator,trainingStaging_->allocation,base,H*sizeof(float));std::memcpy(data,static_cast<std::uint8_t*>(mapped)+base,H*sizeof(float));vmaUnmapMemory(context_.allocator,trainingStaging_->allocation);return 0;
}
int ForwardResourceGraph::update_trainable_gamma(float const* data) noexcept {if(!data)return 1;upload(context_,gamma_->buffer,data,H*sizeof(float));return 0;}
int ForwardResourceGraph::readback_gamma_state(float* gamma,float* m,float* v,std::uint64_t* step) noexcept {
    if(!gamma||!m||!v||!step)return 1;
    auto* self=this; auto const base=adamw_staging_offset(loraRank_)+3*(H*4*H+4*H+4*H*H+H)*sizeof(float);
    lastSubmission_=engine_.submit([self,base](vk::CommandBuffer c){
        c.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {},
                          vk::MemoryBarrier(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                            vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                            vk::AccessFlagBits::eTransferWrite), {}, {});
        c.copyBuffer(self->gamma_->buffer,self->trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(base).setSize(H*4));
        c.copyBuffer(self->gammaM_->buffer,self->trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(base+H*4).setSize(H*4));
        c.copyBuffer(self->gammaV_->buffer,self->trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(base+2*H*4).setSize(H*4));
    },0);engine_.wait(lastSubmission_);void*mapped=nullptr;if(vmaMapMemory(context_.allocator,trainingStaging_->allocation,&mapped)!=VK_SUCCESS)return 2;vmaInvalidateAllocation(context_.allocator,trainingStaging_->allocation,base,3*H*4);auto*bytes=static_cast<std::uint8_t*>(mapped);std::memcpy(gamma,bytes+base,H*4);std::memcpy(m,bytes+base+H*4,H*4);std::memcpy(v,bytes+base+2*H*4,H*4);vmaUnmapMemory(context_.allocator,trainingStaging_->allocation);*step=adamwStep_;return 0;
}
int ForwardResourceGraph::update_gamma_state(float const* gamma,float const* m,float const* v,std::uint64_t step) noexcept {if(!gamma||!m||!v||step>UINT32_MAX)return 1;upload(context_,gamma_->buffer,gamma,H*4);upload(context_,gammaM_->buffer,m,H*4);upload(context_,gammaV_->buffer,v,H*4);adamwStep_=step;return 0;}
int ForwardResourceGraph::readback_combined_gradients(float* gamma_gradient,
                                                       float* ffn_gradient,
                                                       std::size_t ffn_count,
                                                       float* ffn_output_gradient,
                                                       float* activations,
                                                       float* scaled_states,
                                                       float* dstate,
                                                       std::uint32_t rows) noexcept {
    constexpr std::size_t group = H * 4 * H + 4 * H + 4 * H * H + H;
    if (!gamma_gradient || !ffn_gradient || !ffn_output_gradient || !activations || !scaled_states || !dstate || rows == 0 || rows > Tcap || ffn_count != group)
        return 1;
    auto const base = adamw_staging_offset(loraRank_) + 3 * group * sizeof(float) + 3 * H * sizeof(float);
    auto const gamma_bytes = H * sizeof(float);
    auto const w1_bytes = H * 4 * H * sizeof(float);
    auto const b1_bytes = 4 * H * sizeof(float);
    auto const w2_bytes = 4 * H * H * sizeof(float);
    auto const b2_bytes = H * sizeof(float);
    auto const ffn_bytes = w1_bytes + b1_bytes + w2_bytes + b2_bytes;
    auto const dx_bytes = static_cast<vk::DeviceSize>(rows) * H * sizeof(float);
    auto const row_bytes = dx_bytes;
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
     lastSubmission_ = engine_.submit([this, base, gamma_bytes, w1_bytes, b1_bytes, w2_bytes, b2_bytes, ffn_bytes, dx_bytes, row_bytes](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eTransfer,
                          {},
                          {},
                          vk::BufferMemoryBarrier{}
                              .setBuffer(trainingStaging_->buffer)
                              .setSize(VK_WHOLE_SIZE)
                              .setSrcAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                                vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                              .setDstAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite),
                          {});
        auto copy_to_staging = [&](vk::Buffer src, vk::DeviceSize dst_offset, vk::DeviceSize size) {
            c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                                vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite), {}, {});
            c.copyBuffer(src, trainingStaging_->buffer,
                         vk::BufferCopy{}.setDstOffset(dst_offset).setSize(size));
        };
        copy_to_staging(gammaGradient_->buffer, base, gamma_bytes);
        copy_to_staging(ffnW1Gradient_->buffer, base + gamma_bytes, w1_bytes);
        copy_to_staging(ffnB1Gradient_->buffer, base + gamma_bytes + w1_bytes, b1_bytes);
        copy_to_staging(ffnW2Gradient_->buffer, base + gamma_bytes + w1_bytes + b1_bytes, w2_bytes);
        copy_to_staging(ffnB2Gradient_->buffer, base + gamma_bytes + w1_bytes + b1_bytes + w2_bytes, b2_bytes);
        copy_to_staging(ffnDx_->buffer, base + gamma_bytes + ffn_bytes, dx_bytes);
        copy_to_staging(activations_->buffer, base + gamma_bytes + ffn_bytes + row_bytes, dx_bytes);
        copy_to_staging(states_->buffer, base + gamma_bytes + ffn_bytes + 2 * row_bytes, dx_bytes);
        copy_to_staging(dstates_->buffer, base + gamma_bytes + ffn_bytes + 3 * row_bytes, dx_bytes);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eTransfer,
                          {},
                          {},
                          vk::BufferMemoryBarrier{}
                              .setBuffer(trainingStaging_->buffer)
                              .setSize(VK_WHOLE_SIZE)
                              .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                              .setDstAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite),
                          {});
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, gamma_bytes + ffn_bytes + 4 * dx_bytes);
    auto* bytes = static_cast<std::uint8_t*>(mapped) + base;
    std::memcpy(gamma_gradient, bytes, gamma_bytes);
    std::memcpy(ffn_gradient, bytes + gamma_bytes, ffn_bytes);
    std::memcpy(ffn_output_gradient, bytes + gamma_bytes + ffn_bytes, dx_bytes);
    std::memcpy(activations, bytes + gamma_bytes + ffn_bytes + dx_bytes, dx_bytes);
    std::memcpy(scaled_states, bytes + gamma_bytes + ffn_bytes + 2 * dx_bytes, dx_bytes);
    std::memcpy(dstate, bytes + gamma_bytes + ffn_bytes + 3 * dx_bytes, dx_bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    return 0;
}
int ForwardResourceGraph::readback_rmsnorm_staged_diagnostics(float* dx, float* dgamma, std::uint32_t rows) noexcept {
    if (!dx || !dgamma || rows == 0 || rows > Tcap) return 1;
    if (readback_rmsnorm_dx_staged(dx, std::size_t(rows) * H) != 0) return 2;
    if (readback_rmsnorm_dgamma_staged(dgamma, H) != 0) return 3;
    return 0;
}
int ForwardResourceGraph::readback_ffn_state(float* data, std::size_t count, std::uint64_t* step) noexcept {
    constexpr std::size_t w1n=H*4*H,b1n=4*H,w2n=4*H*H,b2n=H,group=w1n+b1n+w2n+b2n,total=3*group;
    if (!data || !step || count != total) return 1;
    constexpr vk::DeviceSize base=adamw_staging_offset(LoraRank4), w1o=base, b1o=w1o+w1n*4, w2o=b1o+b1n*4, b2o=w2o+w2n*4, mo=base+group*4, vo=base+2*group*4;
    lastSubmission_=engine_.submit([this,w1o,b1o,w2o,b2o,mo,vo](vk::CommandBuffer c){
        c.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {},
                          vk::MemoryBarrier(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                            vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                            vk::AccessFlagBits::eTransferWrite), {}, {});
        c.copyBuffer(ffnW1_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(w1o).setSize(w1n*4)); c.copyBuffer(ffnB1_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(b1o).setSize(b1n*4)); c.copyBuffer(ffnW2_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(w2o).setSize(w2n*4)); c.copyBuffer(ffnB2_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(b2o).setSize(b2n*4)); c.copyBuffer(ffnW1M_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(mo).setSize(w1n*4)); c.copyBuffer(ffnB1M_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(mo+w1n*4).setSize(b1n*4)); c.copyBuffer(ffnW2M_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(mo+(w1n+b1n)*4).setSize(w2n*4)); c.copyBuffer(ffnB2M_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(mo+(w1n+b1n+w2n)*4).setSize(b2n*4)); c.copyBuffer(ffnW1V_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(vo).setSize(w1n*4)); c.copyBuffer(ffnB1V_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(vo+w1n*4).setSize(b1n*4)); c.copyBuffer(ffnW2V_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(vo+(w1n+b1n)*4).setSize(w2n*4)); c.copyBuffer(ffnB2V_->buffer,trainingStaging_->buffer,vk::BufferCopy{}.setDstOffset(vo+(w1n+b1n+w2n)*4).setSize(b2n*4)); },0); engine_.wait(lastSubmission_); void* mapped=nullptr; if(vmaMapMemory(context_.allocator,trainingStaging_->allocation,&mapped)!=VK_SUCCESS)return 2;vmaInvalidateAllocation(context_.allocator,trainingStaging_->allocation,base,total*4);std::memcpy(data,static_cast<std::uint8_t*>(mapped)+base,total*4);vmaUnmapMemory(context_.allocator,trainingStaging_->allocation);*step=adamwStep_;return 0;
}
int ForwardResourceGraph::update_ffn_state(float const* data, std::size_t count, std::uint64_t step) noexcept {
    constexpr std::size_t group=H*4*H+4*H+4*H*H+H; if(!data||count!=3*group||step>UINT32_MAX)return 1; auto const*m=data+group;auto const*v=data+2*group; upload(context_,ffnW1_->buffer,data,H*4*H*4);upload(context_,ffnB1_->buffer,data+H*4*H,4*H*4);upload(context_,ffnW2_->buffer,data+H*4*H+4*H,4*H*H*4);upload(context_,ffnB2_->buffer,data+group-H,H*4);upload(context_,ffnW1M_->buffer,m,H*4*H*4);upload(context_,ffnB1M_->buffer,m+H*4*H,4*H*4);upload(context_,ffnW2M_->buffer,m+H*4*H+4*H,4*H*H*4);upload(context_,ffnB2M_->buffer,m+group-H,H*4);upload(context_,ffnW1V_->buffer,v,H*4*H*4);upload(context_,ffnB1V_->buffer,v+H*4*H,4*H*4);upload(context_,ffnW2V_->buffer,v+H*4*H+4*H,4*H*H*4);upload(context_,ffnB2V_->buffer,v+group-H,H*4);adamwStep_=step;return 0;
}
int ForwardResourceGraph::run_rmsnorm_forward_staged(std::uint32_t rows, bool final_only) noexcept {
    if (rows == 0 || rows > Tcap || !rmsForwardPipeline_ || !rmsForwardSet_) return 1;
    TinyRmsForwardPC pc{rows, final_only ? 1u : 0u, 0u};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsForwardPipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsForwardPipelineLayout_, 0, rmsForwardSet_, {});
        c.pushConstants(rmsForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        // Five wave64 workgroups cover fixed Vp=320; y indexes the bounded rows.
        c.dispatch(5, pc.seq_length, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::run_rmsnorm_state_only_staged(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > Tcap || !rmsForwardPipeline_ || !rmsForwardSet_) return 1;
    TinyRmsForwardPC pc{rows, 0u, 1u};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsForwardPipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsForwardPipelineLayout_, 0, rmsForwardSet_, {});
        c.pushConstants(rmsForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch(5, pc.seq_length, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::staged_rmsnorm_forward_is_valid(std::uint32_t rows) const noexcept {
    return rows > 0 && rows <= Tcap && rmsForwardPipeline_ && rmsForwardSet_ ? 1 : 0;
}
int ForwardResourceGraph::validate_rmsnorm_staged_rows(std::uint32_t const* mask, std::uint32_t rows) const noexcept {
    if (!mask || rows == 0 || rows > Tcap) return 1;
    bool seen_zero = false;
    for (std::uint32_t r = 0; r < rows; ++r) {
        if (mask[r] > 1u) return 2;
        if (mask[r] == 0u) seen_zero = true;
        else if (seen_zero) return 3;
    }
    return 0;
}
int ForwardResourceGraph::seed_rmsnorm_staged_dy(float const* dy, std::size_t count) noexcept {
    if (!dy || count == 0 || count > Tcap * H) return 1;
    upload(context_, dprojected_->buffer, dy, count * sizeof(float));
    return 0;
}
int ForwardResourceGraph::seed_rmsnorm_staged_gamma(float const* gamma, std::size_t count) noexcept {
    if (!gamma || count != H) return 1;
    upload(context_, gamma_->buffer, gamma, H * sizeof(float));
    return 0;
}
int ForwardResourceGraph::seed_rmsnorm_staged_mask(std::uint32_t const* mask, std::size_t count) noexcept {
    if (!mask || count == 0 || count > Tcap) return 1;
    for (std::size_t i = 0; i < count; ++i) if (mask[i] > 1u) return 2;
    upload(context_, mask_->buffer, mask, count * sizeof(std::uint32_t));
    return 0;
}
int ForwardResourceGraph::seed_rmsnorm_staged_mask_for_rows(std::uint32_t const* mask, std::uint32_t rows) noexcept {
    if (validate_rmsnorm_staged_rows(mask, rows) != 0) return 1;
    return seed_rmsnorm_staged_mask(mask, rows);
}
int ForwardResourceGraph::run_rmsnorm_staged_state_chain(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > Tcap) return 1;
    // The staged backward consumes dprojected_ as dy; callers must populate it
    // before invoking this diagnostic chain. Each helper waits before the next
    // stage, making the resource dependency explicit while it remains gated.
    if (run_rmsnorm_forward_staged(rows, false) != 0) return 2;
    if (run_rmsnorm_backward_staged(rows) != 0) return 3;
    if (run_rmsnorm_dgamma_staged(rows) != 0) return 4;
    return 0;
}
int ForwardResourceGraph::readback_rmsnorm_state_staged(float* raw, std::size_t raw_count, float* inv_rms, std::size_t inv_count) noexcept {
    if (!raw || !inv_rms || raw_count != Tcap * H || inv_count != Tcap) return 1;
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          {vk::BufferMemoryBarrier{}.setBuffer(rmsRaw_->buffer).setSize(Tcap * H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                           vk::BufferMemoryBarrier{}.setBuffer(rmsInv_->buffer).setSize(Tcap * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead)}, {});
        c.copyBuffer(rmsRaw_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(Tcap * H * sizeof(float)));
        c.copyBuffer(rmsInv_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base + Tcap * H * sizeof(float)).setSize(Tcap * sizeof(float)));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr; if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, Tcap * (H + 1) * sizeof(float));
    auto* bytes = static_cast<std::uint8_t*>(mapped); std::memcpy(raw, bytes + base, Tcap * H * sizeof(float)); std::memcpy(inv_rms, bytes + base + Tcap * H * sizeof(float), Tcap * sizeof(float)); vmaUnmapMemory(context_.allocator, trainingStaging_->allocation); return 0;
}
int ForwardResourceGraph::readback_rmsnorm_states_staged(float* states, std::size_t count) noexcept {
    if (!states || count != Tcap * H) return 1;
    auto const bytes = count * sizeof(float);
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base, bytes](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(states_->buffer).setSize(bytes).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(states_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(bytes));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr; if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, bytes);
    std::memcpy(states, static_cast<std::uint8_t*>(mapped) + base, bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation); return 0;
}
int ForwardResourceGraph::run_rmsnorm_backward_staged(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > Tcap || !rmsBackwardStatePipeline_ || !rmsBackwardStateSet_) return 1;
    RmsNormPC pc{rows, H};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsBackwardStatePipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsBackwardStatePipelineLayout_, 0, rmsBackwardStateSet_, {});
        c.pushConstants(rmsBackwardStatePipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch(pc.rows, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::readback_rmsnorm_dx_staged(float* dx, std::size_t count) noexcept {
    if (!dx || count == 0 || count > Tcap * H || count % H != 0) return 1;
    auto const bytes = count * sizeof(float);
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base, bytes](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(rmsDx_->buffer).setSize(bytes).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(rmsDx_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(bytes));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr; if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, bytes);
    std::memcpy(dx, static_cast<std::uint8_t*>(mapped) + base, bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation); return 0;
}
int ForwardResourceGraph::run_rmsnorm_dgamma_staged(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > Tcap || !rmsDgammaStatePipeline_ || !rmsDgammaStateSet_) return 1;
    RmsNormPC pc{rows, H};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsDgammaStatePipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsDgammaStatePipelineLayout_, 0, rmsDgammaStateSet_, {});
        c.pushConstants(rmsDgammaStatePipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch((pc.hidden + 255) / 256, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::readback_rmsnorm_dgamma_staged(float* dgamma, std::size_t count) noexcept {
    if (!dgamma || count != H) return 1;
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(gammaGradient_->buffer).setSize(H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(gammaGradient_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(H * sizeof(float)));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr; if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, H * sizeof(float));
    std::memcpy(dgamma, static_cast<std::uint8_t*>(mapped) + base, H * sizeof(float));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation); return 0;
}
int ForwardResourceGraph::staged_rmsnorm_dgamma_is_valid(std::uint32_t rows) const noexcept {
    return rows > 0 && rows <= Tcap && rmsDgammaStatePipeline_ && rmsDgammaStateSet_ ? 1 : 0;
}
int ForwardResourceGraph::readback_rmsnorm_gamma_state_staged(float* gamma, float* m, float* v, std::uint64_t* step) noexcept {
    return readback_gamma_state(gamma, m, v, step);
}
int ForwardResourceGraph::run_rmsnorm_gamma_adamw_staged(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay) noexcept {
    if (!gammaAdamwPipeline_ || !std::isfinite(learning_rate) || !std::isfinite(beta1) || !std::isfinite(beta2) ||
        !std::isfinite(epsilon) || !std::isfinite(weight_decay) || learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f ||
        beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f || weight_decay < 0.0f || adamwStep_ == UINT32_MAX)
        return 1;
    ++adamwStep_;
    FfnGroupPC pc{H, static_cast<std::uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, gammaAdamwPipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, gammaAdamwPipelineLayout_, 0, gammaAdamwSet_, {});
        c.pushConstants(gammaAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch(1, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::seed_ffn_input_staged(float const* input, std::size_t count) noexcept {
    if (!input || count == 0 || count > Tcap * H || count % H != 0) return 1;
    upload(context_, activations_->buffer, input, count * sizeof(float));
    return 0;
}
int ForwardResourceGraph::seed_ffn_output_gradient_staged(float const* dy, std::size_t count) noexcept {
    if (!dy || count == 0 || count > Tcap * H || count % H != 0) return 1;
    upload(context_, dprojected_->buffer, dy, count * sizeof(float));
    return 0;
}
int ForwardResourceGraph::run_ffn_backward_staged(std::uint32_t rows) noexcept {
    return run_ffn_backward(rows);
}
int ForwardResourceGraph::readback_ffn_dx_staged(float* dx, std::size_t count) noexcept {
    if (!dx || count == 0 || count > Tcap * H || count % H != 0) return 1;
    auto const bytes = count * sizeof(float), base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base, bytes](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(ffnDx_->buffer).setSize(bytes).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(ffnDx_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(bytes));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr; if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, bytes);
    std::memcpy(dx, static_cast<std::uint8_t*>(mapped) + base, bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation); return 0;
}
int ForwardResourceGraph::run_ffn_forward(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > Tcap || !ffnForwardPipeline_ || !projected_ || !ffnOutput_) return 1;
    FFNForwardPC pc{rows, H, 4 * H};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnForwardPipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnForwardPipelineLayout_, 0, ffnForwardSet_, {});
        c.pushConstants(ffnForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch((pc.rows * pc.hidden + 255) / 256, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::readback_ffn_output_staged(float* output, std::size_t count) noexcept {
    if (!output || count == 0 || count > Tcap * H || count % H != 0) return 1;
    auto const bytes = count * sizeof(float), base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base, bytes](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(ffnOutput_->buffer).setSize(bytes).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(ffnOutput_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(bytes));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr; if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, bytes);
    std::memcpy(output, static_cast<std::uint8_t*>(mapped) + base, bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation); return 0;
}
int ForwardResourceGraph::run_ffn_adamw_staged(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay) noexcept {
    return run_ffn_adamw_all(learning_rate, beta1, beta2, epsilon, weight_decay);
}
int ForwardResourceGraph::run_ffn_adamw_all(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay) noexcept {
    if (!std::isfinite(learning_rate) || !std::isfinite(beta1) || !std::isfinite(beta2) || !std::isfinite(epsilon) || !std::isfinite(weight_decay) || learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f || weight_decay < 0.0f || adamwStep_ == UINT32_MAX) return 1;
    ++adamwStep_;
    FfnGroupPC pc{0, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay};
    struct G { vk::Pipeline p; vk::PipelineLayout l; vk::DescriptorSet s; uint32_t n; };
    std::array<G,4> gs{{{ffnAdamwPipeline_,ffnAdamwPipelineLayout_,ffnAdamwSet_,H*4*H},{ffnB1AdamwPipeline_,ffnB1AdamwPipelineLayout_,ffnB1AdamwSet_,4*H},{ffnW2AdamwPipeline_,ffnW2AdamwPipelineLayout_,ffnW2AdamwSet_,4*H*H},{ffnB2AdamwPipeline_,ffnB2AdamwPipelineLayout_,ffnB2AdamwSet_,H}}};
    lastSubmission_ = engine_.submit([gs, pc](vk::CommandBuffer c) mutable { for (auto const& g : gs) { auto x=pc; x.count=g.n; c.bindPipeline(vk::PipelineBindPoint::eCompute,g.p); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute,g.l,0,g.s,{}); c.pushConstants(g.l,vk::ShaderStageFlagBits::eCompute,0,sizeof(x),&x); c.dispatch((x.count+255)/256,1,1); } },0); engine_.wait(lastSubmission_); return 0;
}
int ForwardResourceGraph::run_ffn_w1_adamw(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay) noexcept {
    if (!ffnAdamwPipeline_ || !std::isfinite(learning_rate) || !std::isfinite(beta1) || !std::isfinite(beta2) || !std::isfinite(epsilon) || !std::isfinite(weight_decay) || learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f || weight_decay < 0.0f || adamwStep_ == UINT32_MAX) return 1;
    ++adamwStep_;
    FfnAdamwPC pc{H * 4 * H, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnAdamwPipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnAdamwPipelineLayout_, 0, ffnAdamwSet_, {});
        c.pushConstants(ffnAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch((pc.count + 255) / 256, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::readback_ffn_gradients_staged(float* w1, float* b1, float* w2, float* b2) noexcept {
    if (!w1 || !b1 || !w2 || !b2) return 1;
    constexpr std::size_t ffn_i = 4 * H;
    constexpr std::size_t w1n = H * ffn_i, b1n = ffn_i, w2n = ffn_i * H, b2n = H;
    constexpr vk::DeviceSize total = (w1n + b1n + w2n + b2n) * sizeof(float);
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(ffnW1Gradient_->buffer).setSize(VK_WHOLE_SIZE).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        c.copyBuffer(ffnW1Gradient_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(H * ffn_i * sizeof(float)));
        c.copyBuffer(ffnB1Gradient_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base + H * ffn_i * sizeof(float)).setSize(ffn_i * sizeof(float)));
        c.copyBuffer(ffnW2Gradient_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base + (H * ffn_i + ffn_i) * sizeof(float)).setSize(ffn_i * H * sizeof(float)));
        c.copyBuffer(ffnB2Gradient_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base + (H * ffn_i + ffn_i + ffn_i * H) * sizeof(float)).setSize(H * sizeof(float)));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr; if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, total);
    auto* bytes = static_cast<std::uint8_t*>(mapped) + base;
    std::memcpy(w1, bytes, w1n * sizeof(float)); std::memcpy(b1, bytes + w1n * sizeof(float), b1n * sizeof(float));
    std::memcpy(w2, bytes + (w1n + b1n) * sizeof(float), w2n * sizeof(float)); std::memcpy(b2, bytes + (w1n + b1n + w2n) * sizeof(float), b2n * sizeof(float));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation); return 0;
}
int ForwardResourceGraph::run_rmsnorm_gamma_gradient(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > Tcap || !rmsNormGradientPipeline_) return 1;
    RmsNormPC pc{rows, H};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsNormGradientPipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsNormGradientPipelineLayout_, 0, rmsNormGradientSet_, {});
        c.pushConstants(rmsNormGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch(1, 1, 1);
    }, 0); engine_.wait(lastSubmission_); return 0;
}
int ForwardResourceGraph::run_ffn_parameter_gradients(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > Tcap || !ffnGradientPipeline_) return 1;
    FfnGradPC pc{rows, H, 4 * H};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnGradientPipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnGradientPipelineLayout_, 0, ffnGradientSet_, {});
        c.pushConstants(ffnGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch((H * 4 * H + 4 * H + 4 * H * H + H + 255) / 256, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::run_ffn_backward(std::uint32_t rows) noexcept {
    if (rows == 0 || rows > Tcap || !ffnBackwardPipeline_ || !projected_ || !ffnOutput_ || !ffnDx_) return 1;
    FFNForwardPC pc{rows, H, 4 * H};
    lastSubmission_ = engine_.submit([this, pc](vk::CommandBuffer c) {
        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnBackwardPipeline_);
        c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnBackwardPipelineLayout_, 0, ffnBackwardSet_, {});
        c.pushConstants(ffnBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        c.dispatch((pc.rows * pc.hidden + 255) / 256, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::train_ffn_forward_loss(std::uint32_t const* tokens,
                                                 std::uint32_t const* targets,
                                                 std::uint32_t const* masks,
                                                 std::uint32_t rows) noexcept {
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap) return 1;
    for (std::uint32_t i = 0; i < rows; ++i) if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u) return 1;
    constexpr vk::DeviceSize bytes = Tcap * sizeof(std::uint32_t), targetOffset = bytes, maskOffset = 2 * bytes;
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    auto* staged = static_cast<std::uint8_t*>(mapped);
    std::memset(staged, 0, 3 * bytes);
    std::memcpy(staged, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(staged + targetOffset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(staged + maskOffset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    ForwardPC fp{rows, 0};
    LossPC lp{rows, V, Vp};
    FFNForwardPC ffp{rows, H, 4 * H};
    lastSubmission_ = engine_.submit([this, fp, lp, ffp](vk::CommandBuffer c) {
        c.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(bytes));
        c.copyBuffer(trainingStaging_->buffer, targets_->buffer, vk::BufferCopy{}.setSrcOffset(targetOffset).setSize(bytes));
        c.copyBuffer(trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(maskOffset).setSize(bytes));
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {}); c.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fp), &fp); c.dispatch(5, fp.seq_length, 1);
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnForwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnForwardPipelineLayout_, 0, ffnForwardSet_, {}); c.pushConstants(ffnForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ffp), &ffp); c.dispatch((ffp.rows * ffp.hidden + 255) / 256, 1, 1); c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {}); c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsNormGradientPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsNormGradientPipelineLayout_, 0, rmsNormGradientSet_, {}); RmsNormPC rp{fp.seq_length, H}; c.pushConstants(rmsNormGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(rp), &rp); c.dispatch(1, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::train_positions_adamw(std::uint32_t const* tokens,
                                                  std::uint32_t const* targets,
                                                  std::uint32_t const* masks,
                                                  std::uint32_t rows,
                                                  float learning_rate,
                                                  float beta1,
                                                  float beta2,
                                                  float epsilon,
                                                  float weight_decay) noexcept {
    if (!std::isfinite(learning_rate) || !std::isfinite(beta1) || !std::isfinite(beta2) ||
        !std::isfinite(epsilon) || !std::isfinite(weight_decay) || learning_rate <= 0.0f ||
        beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f ||
        weight_decay < 0.0f || adamwStep_ == UINT32_MAX)
        return 1;
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap || !std::isfinite(learning_rate) || learning_rate <= 0.0f)
        return 1;
    for (std::uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u)
            return 1;
    // tiny_forward_logits uses causal keys and has no key-mask descriptor;
    // reject holes so masked rows cannot perturb later valid rows.
    bool seen_masked = false;
    for (std::uint32_t i = 0; i < rows; ++i) {
        if (masks[i] == 0u) seen_masked = true;
        else if (seen_masked) return 1;
    }
    const auto next_adamw_step = adamwStep_ + 1;
    constexpr vk::DeviceSize bytes = Tcap * sizeof(std::uint32_t), target_offset = bytes, mask_offset = 2 * bytes;
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    ++adamwStep_;
    auto* staged = static_cast<std::uint8_t*>(mapped);
    std::memset(staged, 0, 3 * bytes);
    std::memcpy(staged, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(staged + target_offset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(staged + mask_offset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    ForwardPC f{rows, 0}; LossPC l{rows, V, Vp}; LmBackwardPC bw{rows, V, H, Vp, Vp, H};
    PositionGradientPC pg{H, rows}; PositionAdamwPC ps{rows * H, H, static_cast<uint32_t>(next_adamw_step), learning_rate, beta1, beta2, epsilon, weight_decay};
    lastSubmission_ = engine_.submit([this, f, l, bw, pg, ps, learning_rate, beta1, beta2, epsilon, weight_decay](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {},
                          vk::MemoryBarrier(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                            vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                            vk::AccessFlagBits::eTransferWrite), {}, {});
        std::array<vk::BufferMemoryBarrier, 3> inputBarriers{
            vk::BufferMemoryBarrier{}.setBuffer(tokens_->buffer).setSize(bytes)
                .setSrcAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
            vk::BufferMemoryBarrier{}.setBuffer(targets_->buffer).setSize(bytes)
                .setSrcAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
            vk::BufferMemoryBarrier{}.setBuffer(mask_->buffer).setSize(bytes)
                .setSrcAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite)};
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eTransfer, {}, {}, inputBarriers, {});
        c.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(bytes));
        c.copyBuffer(trainingStaging_->buffer, targets_->buffer, vk::BufferCopy{}.setSrcOffset(target_offset).setSize(bytes));
        c.copyBuffer(trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(mask_offset).setSize(bytes));
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          vk::BufferMemoryBarrier{}.setBuffer(trainingStaging_->buffer).setSize(VK_WHOLE_SIZE)
                              .setSrcAccessMask(vk::AccessFlagBits::eTransferRead)
                              .setDstAccessMask(vk::AccessFlagBits::eTransferWrite), {});
        c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
                          vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        auto bar = [&]() { c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {},
                                               vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {}); };
        auto copy_after_compute = [&](vk::Buffer src, vk::Buffer dst, vk::DeviceSize size) {
            std::array<vk::BufferMemoryBarrier, 2> barriers{
                vk::BufferMemoryBarrier{}.setBuffer(src).setSize(size)
                    .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                    .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                vk::BufferMemoryBarrier{}.setBuffer(dst).setSize(size)
                    .setSrcAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead |
                                      vk::AccessFlagBits::eTransferWrite)
                    .setDstAccessMask(vk::AccessFlagBits::eTransferWrite)};
            c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
                              vk::PipelineStageFlagBits::eTransfer, {}, {}, barriers, {});
            c.copyBuffer(src, dst, vk::BufferCopy{}.setSize(size));
            c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        };
        c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        c.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(f), &f); c.dispatch(5, f.seq_length, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsFixedForwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsFixedForwardPipelineLayout_, 0, rmsFixedForwardSet_, {}); RmsNormPC rfp{bw.rows, H}; c.pushConstants(rmsFixedForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(rfp), &rfp); c.dispatch(bw.rows, 1, 1); bar(); // normalized activations
         // normalized activations already written by RMSNorm
         copy_after_compute(states_->buffer, gammaInput_->buffer, bw.rows * H * sizeof(float)); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnForwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnForwardPipelineLayout_, 0, ffnForwardSet_, {}); FFNForwardPC ff0{bw.rows, H, 4 * H}; c.pushConstants(ffnForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ff0), &ff0); c.dispatch((ff0.rows * ff0.hidden + 255) / 256, 1, 1); bar();
        copy_after_compute(ffnOutput_->buffer, projected_->buffer, bw.rows * H * sizeof(float)); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, lmHeadForwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmHeadForwardPipelineLayout_, 0, lmHeadForwardSet_, {}); LmHeadForwardPC lmf{bw.rows, H, V, Vp}; c.pushConstants(lmHeadForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lmf), &lmf); c.dispatch((bw.rows * Vp + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
        c.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(l), &l); c.dispatch(l.rows, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, lmBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmBackwardPipelineLayout_, 0, lmBackwardSet_, {});
        c.pushConstants(lmBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(bw), &bw); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnBackwardPipelineLayout_, 0, ffnBackwardSet_, {}); c.pushConstants(ffnBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ff0), &ff0); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        copy_after_compute(ffnDx_->buffer, dprojected_->buffer, bw.rows * H * sizeof(float)); bar();
         c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnGradientPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnGradientPipelineLayout_, 0, ffnGradientSet_, {}); FfnGradPC ffg{bw.rows, H, 4*H}; c.pushConstants(ffnGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ffg), &ffg); c.dispatch((H*4*H+4*H+4*H*H+H+255)/256, 1, 1); bar();
         c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsDgammaStatePipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsDgammaStatePipelineLayout_, 0, rmsDgammaStateSet_, {}); RmsNormPC rdp{bw.rows, H}; c.pushConstants(rmsDgammaStatePipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(rdp), &rdp); c.dispatch(1, 1, 1); bar();
         c.bindPipeline(vk::PipelineBindPoint::eCompute, rmsBackwardStatePipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rmsBackwardStatePipelineLayout_, 0, rmsBackwardStateSet_, {}); RmsNormPC rbp{bw.rows, H}; c.pushConstants(rmsBackwardStatePipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(rbp), &rbp); c.dispatch(bw.rows, 1, 1); bar();
         copy_after_compute(rmsDx_->buffer, dprojected_->buffer, bw.rows * H * sizeof(float)); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardSet_, {});
        ProjectionBackwardPC pp{bw.rows, H, H, 0}; c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pp), &pp); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        copy_after_compute(dcontext_->buffer, doutput_->buffer, bw.rows * H * sizeof(float));
        vk::DescriptorSet as[3]{attentionBackwardQSet_, attentionBackwardKSet_, attentionBackwardVSet_};
        for (std::uint32_t m = 0; m < 3; ++m) { c.bindPipeline(vk::PipelineBindPoint::eCompute, attentionBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, attentionBackwardPipelineLayout_, 0, as[m], {}); AttentionBackwardPC ap{bw.rows, H, m}; c.pushConstants(attentionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ap), &ap); c.dispatch((bw.rows * H + 255) / 256, 1, 1); if (m != 2) bar(); }
        bar(); c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardQkvSet_, {});
        ProjectionBackwardPC qp{bw.rows, H, H, 0}; c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qp), &qp); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        ProjectionBackwardPC qk{bw.rows, H, H, 1}; c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardKVSet_, {}); c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qk), &qk); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        ProjectionBackwardPC qv{bw.rows, H, H, 1}; c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardVVSet_, {}); c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(qv), &qv); c.dispatch((bw.rows * H + 255) / 256, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, positionGradientPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, positionGradientPipelineLayout_, 0, positionGradientSet_, {}); c.pushConstants(positionGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pg), &pg); c.dispatch((pg.rows * pg.hidden + 255) / 256, 1, 1); bar();
        copy_after_compute(dstates_->buffer, gammaInput_->buffer, bw.rows * H * sizeof(float)); bar();

        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnGradientPipeline_); c.bindPipeline(vk::PipelineBindPoint::eCompute, gammaAdamwPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, gammaAdamwPipelineLayout_, 0, gammaAdamwSet_, {}); FfnGroupPC ga{H, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay}; c.pushConstants(gammaAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ga), &ga); c.dispatch(1, 1, 1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnAdamwPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnAdamwPipelineLayout_, 0, ffnAdamwSet_, {}); FfnGroupPC fa{H*4*H, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay}; c.pushConstants(ffnAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fa), &fa); c.dispatch((fa.count+255)/256,1,1); bar(); c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnB1AdamwPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnB1AdamwPipelineLayout_, 0, ffnB1AdamwSet_, {}); FfnGroupPC fb{4*H, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay}; c.pushConstants(ffnB1AdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fb), &fb); c.dispatch((fb.count+255)/256,1,1); bar(); c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnW2AdamwPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnW2AdamwPipelineLayout_, 0, ffnW2AdamwSet_, {}); FfnGroupPC fw2{4*H*H, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay}; c.pushConstants(ffnW2AdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fw2), &fw2); c.dispatch((fw2.count+255)/256,1,1); bar(); c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnB2AdamwPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnB2AdamwPipelineLayout_, 0, ffnB2AdamwSet_, {}); FfnGroupPC fb2{H, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay}; c.pushConstants(ffnB2AdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fb2), &fb2); c.dispatch((fb2.count+255)/256,1,1); bar();
        c.bindPipeline(vk::PipelineBindPoint::eCompute, positionAdamwPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, positionAdamwPipelineLayout_, 0, positionAdamwSet_, {}); c.pushConstants(positionAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ps), &ps); c.dispatch((ps.count + 255) / 256, 1, 1);
    }, 0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::readback_base_train_positions_adamw_state(float* positions, float* m, float* v, std::uint64_t* step) {
    if (!positions || !m || !v || !step) return 1;
    if (readback_positions(positions) != 0) return 2;
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit([this, base](vk::CommandBuffer c) {
        c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                          {vk::BufferMemoryBarrier{}.setBuffer(positionM_->buffer).setSize(Tcap * H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                           vk::BufferMemoryBarrier{}.setBuffer(positionV_->buffer).setSize(Tcap * H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead)}, {});
        c.copyBuffer(positionM_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base).setSize(Tcap * H * sizeof(float)));
        c.copyBuffer(positionV_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(base + Tcap * H * sizeof(float)).setSize(Tcap * H * sizeof(float)));
    }, 0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr; if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS) return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, 2 * Tcap * H * sizeof(float));
    auto* bytes = static_cast<std::uint8_t*>(mapped); std::memcpy(m, bytes + base, Tcap * H * sizeof(float)); std::memcpy(v, bytes + base + Tcap * H * sizeof(float), Tcap * H * sizeof(float));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation); *step = adamwStep_; return 0;
}
int ForwardResourceGraph::update_base_train_positions_adamw_state(float const* positions, float const* m, float const* v, std::uint64_t step) {
    if (!positions || !m || !v || step > UINT32_MAX) return 1;
    if (import_base_train_positions(positions) != 0) return 2;
    upload(context_, positionM_->buffer, m, Tcap * H * sizeof(float)); upload(context_, positionV_->buffer, v, Tcap * H * sizeof(float)); adamwStep_ = step; return 0;
}
int ForwardResourceGraph::readback_graph_dstate(std::uint32_t const* tokens,
                                                   std::uint32_t const* targets,
                                                   std::uint32_t const* masks,
                                                   std::uint32_t rows,
                                                   float* dstates) {
    if (!tokens || !targets || !masks || !dstates || rows == 0 || rows > Tcap)
        return 1;
    for (std::uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u)
            return 1;
    constexpr vk::DeviceSize bytes = Tcap * sizeof(std::uint32_t), target_offset = bytes, mask_offset = 2 * bytes;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* staged = static_cast<std::uint8_t*>(mapped);
    std::memset(staged, 0, 3 * bytes);
    std::memcpy(staged, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(staged + target_offset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(staged + mask_offset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * bytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);

    ForwardPC f{rows, 0};
    LossPC l{rows, V, Vp};
    LmBackwardPC bw{rows, V, H, Vp, Vp, H};
    lastSubmission_ = engine_.submit(
        [this, f, l, bw, rows](vk::CommandBuffer c) {
            c.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(bytes));
            c.copyBuffer(trainingStaging_->buffer,
                         targets_->buffer,
                         vk::BufferCopy{}.setSrcOffset(target_offset).setSize(bytes));
            c.copyBuffer(trainingStaging_->buffer,
                         mask_->buffer,
                         vk::BufferCopy{}.setSrcOffset(mask_offset).setSize(bytes));
            c.fillBuffer(dstates_->buffer, 0, Tcap * H * sizeof(float), 0);
            auto bar = [&]() {
                c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  {},
                                  vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                  {},
                                  {});
            };
            c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                              vk::PipelineStageFlagBits::eComputeShader,
                              {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                              {},
                              {});
            c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            c.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(f), &f);
            c.dispatch(5, f.seq_length, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, ffnForwardPipeline_); c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnForwardPipelineLayout_, 0, ffnForwardSet_, {}); FFNForwardPC ff0{bw.rows, H, 4 * H}; c.pushConstants(ffnForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ff0), &ff0); c.dispatch((ff0.rows * ff0.hidden + 255) / 256, 1, 1); bar();
         c.copyBuffer(ffnOutput_->buffer, projected_->buffer, vk::BufferCopy{}.setSize(bw.rows * H * sizeof(float))); bar();
         c.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
            c.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(l), &l);
            c.dispatch(rows, 1, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, lmBackwardPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmBackwardPipelineLayout_, 0, lmBackwardSet_, {});
            c.pushConstants(lmBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(bw), &bw);
            c.dispatch((rows * H + 255) / 256, 1, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardSet_, {});
            ProjectionBackwardPC context_proj{rows, H, H, 0};
            c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(context_proj), &context_proj);
            c.dispatch((rows * H + 255) / 256, 1, 1);
            bar();
            c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead), {}, {});
            c.copyBuffer(dcontext_->buffer, doutput_->buffer, vk::BufferCopy{}.setSize(rows * H * sizeof(float)));
            c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
            c.bindPipeline(vk::PipelineBindPoint::eCompute, attentionBackwardPipeline_);
            vk::DescriptorSet attention_sets[3]{attentionBackwardQSet_, attentionBackwardKSet_, attentionBackwardVSet_};
            for (std::uint32_t mode = 0; mode < 3; ++mode) {
                c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, attentionBackwardPipelineLayout_, 0, attention_sets[mode], {});
                AttentionBackwardPC attention_pc{rows, H, mode};
                c.pushConstants(attentionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(attention_pc), &attention_pc);
                c.dispatch((rows * H + 255) / 256, 1, 1);
                if (mode != 2)
                    bar();
            }
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_);
            ProjectionBackwardPC q_proj{rows, H, H, 0};
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardQkvSet_, {});
            c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(q_proj), &q_proj);
            c.dispatch((rows * H + 255) / 256, 1, 1);
            bar();
            ProjectionBackwardPC k_proj{rows, H, H, 1};
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardKVSet_, {});
            c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(k_proj), &k_proj);
            c.dispatch((rows * H + 255) / 256, 1, 1);
            bar();
            ProjectionBackwardPC v_proj{rows, H, H, 1};
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardVVSet_, {});
            c.pushConstants(projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(v_proj), &v_proj);
            c.dispatch((rows * H + 255) / 256, 1, 1);
            c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                              vk::PipelineStageFlagBits::eTransfer,
                              {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
                              {},
                              {});
                         c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eTransfer,
                               {},
                               {},
                               vk::BufferMemoryBarrier{}
                                   .setBuffer(trainingStaging_->buffer)
                                   .setOffset(0)
                                   .setSize(Tcap * H * sizeof(float))
                                   .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                   .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
                               {});
c.copyBuffer(dstates_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(0).setSize(Tcap * H * sizeof(float)));
        },
        0);
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, 0, Tcap * H * sizeof(float));
    std::memcpy(dstates, mapped, Tcap * H * sizeof(float));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    return 0;
}
int ForwardResourceGraph::train_output_sgd(std::uint32_t const* tokens,
                                           std::uint32_t const* targets,
                                           std::uint32_t const* masks,
                                           std::uint32_t rows,
                                           float learning_rate) noexcept {
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap || !std::isfinite(learning_rate) ||
        learning_rate <= 0.0f)
        return 1;
    for (std::uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u)
            return 1;
    constexpr vk::DeviceSize tokenBytes = Tcap * sizeof(std::uint32_t), targetOffset = tokenBytes,
                             maskOffset = 2 * tokenBytes;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    std::memset(bytes, 0, 3 * tokenBytes);
    std::memcpy(bytes, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(bytes + targetOffset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(bytes + maskOffset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * tokenBytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    ForwardPC fpc{rows, 0};
    LossPC lpc{rows, V, Vp};
    LmBackwardPC bpc{rows, V, H, Vp, Vp, H};
    OutputProjectionGradientPC gpc{rows, Tcap, H};
    OutputProjectionSgdPC spc{H * H, learning_rate};
    lastSubmission_ = engine_.submit(
        [this, fpc, lpc, bpc, gpc, spc, rows](vk::CommandBuffer cmd) {
            cmd.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(tokenBytes));
            cmd.copyBuffer(trainingStaging_->buffer,
                           targets_->buffer,
                           vk::BufferCopy{}.setSrcOffset(targetOffset).setSize(tokenBytes));
            cmd.copyBuffer(
                trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(maskOffset).setSize(tokenBytes));
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fpc), &fpc);
            cmd.dispatch(5, fpc.seq_length, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
            cmd.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lpc), &lpc);
            cmd.dispatch(rows, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmBackwardPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmBackwardPipelineLayout_, 0, lmBackwardSet_, {});
            cmd.pushConstants(lmBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(bpc), &bpc);
            cmd.dispatch((rows * H + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, outputGradientPipeline_);
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, outputGradientPipelineLayout_, 0, outputGradientSet_, {});
            cmd.pushConstants(outputGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(gpc), &gpc);
            cmd.dispatch((H * H + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, outputSgdPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, outputSgdPipelineLayout_, 0, outputSgdSet_, {});
            cmd.pushConstants(outputSgdPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(spc), &spc);
            cmd.dispatch((H * H + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::train_output_adamw(std::uint32_t const* tokens,
                                             std::uint32_t const* targets,
                                             std::uint32_t const* masks,
                                             std::uint32_t rows,
                                             float learning_rate,
                                             float beta1,
                                             float beta2,
                                             float epsilon,
                                             float weight_decay) noexcept {
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap || !std::isfinite(learning_rate) ||
        !std::isfinite(beta1) || !std::isfinite(beta2) || !std::isfinite(epsilon) || !std::isfinite(weight_decay) ||
        learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f ||
        adamwStep_ == UINT32_MAX)
        return 1;
    for (std::uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u)
            return 1;
    constexpr vk::DeviceSize tokenBytes = Tcap * sizeof(std::uint32_t), targetOffset = tokenBytes,
                             maskOffset = 2 * tokenBytes;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    std::memset(bytes, 0, 3 * tokenBytes);
    std::memcpy(bytes, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(bytes + targetOffset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(bytes + maskOffset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * tokenBytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    ++adamwStep_;
    ForwardPC f{rows, 0};
    LossPC l{rows, V, Vp};
    LmBackwardPC b{rows, V, H, Vp, Vp, H};
    OutputProjectionGradientPC g{rows, Tcap, H};
    OutputAdamwPC a{H * H, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay};
    lastSubmission_ = engine_.submit(
        [this, f, l, b, g, a](vk::CommandBuffer c) {
            c.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(tokenBytes));
            c.copyBuffer(trainingStaging_->buffer,
                         targets_->buffer,
                         vk::BufferCopy{}.setSrcOffset(targetOffset).setSize(tokenBytes));
            c.copyBuffer(
                trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(maskOffset).setSize(tokenBytes));
            auto bar = [&]() {
                c.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  {},
                                  vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                  {},
                                  {});
            };
            c.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                              vk::PipelineStageFlagBits::eComputeShader,
                              {},
                              vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                              {},
                              {});
            c.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            c.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(f), &f);
            c.dispatch(5, f.seq_length, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
            c.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(l), &l);
            c.dispatch(l.rows, 1, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, lmBackwardPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmBackwardPipelineLayout_, 0, lmBackwardSet_, {});
            c.pushConstants(lmBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(b), &b);
            c.dispatch((b.rows * H + 255) / 256, 1, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, outputGradientPipeline_);
            c.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, outputGradientPipelineLayout_, 0, outputGradientSet_, {});
            c.pushConstants(outputGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(g), &g);
            c.dispatch((H * H + 255) / 256, 1, 1);
            bar();
            c.bindPipeline(vk::PipelineBindPoint::eCompute, outputAdamwPipeline_);
            c.bindDescriptorSets(vk::PipelineBindPoint::eCompute, outputAdamwPipelineLayout_, 0, outputAdamwSet_, {});
            c.pushConstants(outputAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(a), &a);
            c.dispatch((a.count + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    return 0;
}
int ForwardResourceGraph::train_lm_head_adamw(std::uint32_t const* tokens,
                                              std::uint32_t const* targets,
                                              std::uint32_t const* masks,
                                              std::uint32_t rows,
                                              float learning_rate,
                                              float beta1,
                                              float beta2,
                                              float epsilon,
                                              float weight_decay) noexcept {
    if (!tokens || !targets || !masks || rows == 0 || rows > Tcap || !std::isfinite(learning_rate) ||
        !std::isfinite(beta1) || !std::isfinite(beta2) || !std::isfinite(epsilon) || !std::isfinite(weight_decay) ||
        learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f)
        return 1;
    for (std::uint32_t i = 0; i < rows; ++i)
        if (tokens[i] >= V || targets[i] >= V || masks[i] > 1u)
            return 1;
    if (adamwStep_ == UINT32_MAX)
        return 1;
    constexpr vk::DeviceSize tokenBytes = Tcap * sizeof(std::uint32_t), targetOffset = tokenBytes,
                             maskOffset = 2 * tokenBytes;
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    std::memset(bytes, 0, 3 * tokenBytes);
    std::memcpy(bytes, tokens, rows * sizeof(std::uint32_t));
    std::memcpy(bytes + targetOffset, targets, rows * sizeof(std::uint32_t));
    std::memcpy(bytes + maskOffset, masks, rows * sizeof(std::uint32_t));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, 3 * tokenBytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    ++adamwStep_;
    ForwardPC fpc{rows, 0};
    LossPC lpc{rows, V, Vp};
    LmHeadGradientPC gpc{rows, Tcap, V, Vp, H, 0};
    LmHeadAdamwPC apc{H * Vp, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay};
    lastSubmission_ = engine_.submit(
        [this, fpc, lpc, gpc, apc](vk::CommandBuffer cmd) {
            cmd.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(tokenBytes));
            cmd.copyBuffer(trainingStaging_->buffer,
                           targets_->buffer,
                           vk::BufferCopy{}.setSrcOffset(targetOffset).setSize(tokenBytes));
            cmd.copyBuffer(
                trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(maskOffset).setSize(tokenBytes));
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fpc), &fpc);
            cmd.dispatch(5, fpc.seq_length, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
            cmd.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lpc), &lpc);
            cmd.dispatch(lpc.rows, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmHeadGradientPipeline_);
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, lmHeadGradientPipelineLayout_, 0, lmHeadGradientSet_, {});
            cmd.pushConstants(lmHeadGradientPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(gpc), &gpc);
            cmd.dispatch((H * Vp + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmHeadAdamwPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmHeadAdamwPipelineLayout_, 0, lmHeadAdamwSet_, {});
            cmd.pushConstants(lmHeadAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(apc), &apc);
            cmd.dispatch((apc.count + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    return 0;
}

int ForwardResourceGraph::readback_base_train_lm_head_adamw_state(float* weight,
                                                                  float* m,
                                                                  float* v,
                                                                  std::uint64_t* step) {
    if (!weight || !m || !v || !step)
        return 1;
    if (readback_base_train_lm_head(weight) != 0)
        return 2;
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit(
        [this, base](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                {},
                                {vk::BufferMemoryBarrier{}
                                     .setBuffer(lmHeadM_->buffer)
                                     .setSize(H * Vp * sizeof(float))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(lmHeadV_->buffer)
                                     .setSize(H * Vp * sizeof(float))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead)},
                                {});
            cmd.copyBuffer(lmHeadM_->buffer,
                           trainingStaging_->buffer,
                           vk::BufferCopy{}.setDstOffset(base).setSize(H * Vp * sizeof(float)));
            cmd.copyBuffer(
                lmHeadV_->buffer,
                trainingStaging_->buffer,
                vk::BufferCopy{}.setDstOffset(base + H * Vp * sizeof(float)).setSize(H * Vp * sizeof(float)));
        },
        0);
    engine_.wait(lastSubmission_);
    void* p = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &p) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, 2 * H * Vp * sizeof(float));
    auto* state = static_cast<std::uint8_t*>(p) + base;
    std::memcpy(m, state, H * Vp * sizeof(float));
    std::memcpy(v, state + H * Vp * sizeof(float), H * Vp * sizeof(float));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    *step = adamwStep_;
    return 0;
}
int ForwardResourceGraph::update_base_train_lm_head_adamw_state(float const* weight,
                                                                float const* m,
                                                                float const* v,
                                                                std::uint64_t step) {
    if (!weight || !m || !v || step > UINT32_MAX)
        return 1;
    if (import_base_train_lm_head(weight) != 0)
        return 2;
    upload(context_, lmHeadM_->buffer, m, H * Vp * sizeof(float));
    upload(context_, lmHeadV_->buffer, v, H * Vp * sizeof(float));
    adamwStep_ = step;
    return 0;
}
int ForwardResourceGraph::import_base_train_embeddings(float const* weight) {
    if (!weight)
        return 1;
    upload(context_, embeddings_->buffer, weight, V * H * sizeof(float));
    return 0;
}
int ForwardResourceGraph::import_base_train_positions(float const* positions) {
    if (!positions)
        return 1;
    upload(context_, positions_->buffer, positions, Tcap * H * sizeof(float));
    return 0;
}
int ForwardResourceGraph::readback_base_train_embeddings(float* weight) {
    if (!weight)
        return 1;
    void* mapped = nullptr;
    lastSubmission_ = engine_.submit([this](vk::CommandBuffer cmd) {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {},
                            vk::BufferMemoryBarrier{}.setBuffer(embeddings_->buffer).setSize(V * H * sizeof(float)).setSrcAccessMask(vk::AccessFlagBits::eShaderWrite).setDstAccessMask(vk::AccessFlagBits::eTransferRead), {});
        cmd.copyBuffer(embeddings_->buffer, readback_->buffer, vk::BufferCopy{}.setSize(V * H * sizeof(float)));
    });
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, readback_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, V * H * sizeof(float));
    std::memcpy(weight, mapped, V * H * sizeof(float));
    vmaUnmapMemory(context_.allocator, readback_->allocation);
    return 0;
}
int ForwardResourceGraph::readback_base_train_positions(float* positions) {
    return readback_positions(positions);
}
int ForwardResourceGraph::readback_base_train_lm_head(float* weight) {
    if (!weight)
        return 1;
    void* mapped = nullptr;
    lastSubmission_ = engine_.submit([this](vk::CommandBuffer cmd) {
        vk::BufferMemoryBarrier barrier;
        barrier.setBuffer(lmHead_->buffer)
            .setOffset(0)
            .setSize(H * Vp * sizeof(float))
            .setSrcAccessMask(vk::AccessFlagBits::eShaderRead)
            .setDstAccessMask(vk::AccessFlagBits::eTransferRead);
        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, barrier, {});
        cmd.copyBuffer(lmHead_->buffer, readback_->buffer, vk::BufferCopy{}.setSize(H * Vp * sizeof(float)));
    });
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, readback_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, H * Vp * sizeof(float));
    std::memcpy(weight, mapped, H * Vp * sizeof(float));
    vmaUnmapMemory(context_.allocator, readback_->allocation);
    return 0;
}

void ForwardResourceGraph::forward_fixed_retained(std::uint32_t const* toks, float* out) {
    if (!fixedForwardRecorded_ || !toks || !out)
        throw std::invalid_argument("invalid retained Tiny forward arguments");
    for (std::uint32_t i = 0; i < Tcap; ++i)
        if (toks[i] >= V)
            throw std::invalid_argument("invalid Tiny token");
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("retained forward staging map failed");
    std::memcpy(mapped, toks, fixedForwardStagingBytes);
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, fixedForwardStagingBytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    lastSubmission_ = engine_.submitImmutable();
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, readback_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("retained forward readback map failed");
    vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, Tcap * Vp * sizeof(float));
    std::memcpy(out, mapped, Tcap * Vp * sizeof(float));
    vmaUnmapMemory(context_.allocator, readback_->allocation);
}

void ForwardResourceGraph::forward_loss_fixed_retained(std::uint32_t const* toks,
                                                       std::uint32_t const* targets,
                                                       std::uint32_t const* masks,
                                                       float* out,
                                                       float* row_losses) {
    if (!fixedForwardLossRecorded_ || !toks || !targets || !masks || !out || !row_losses)
        throw std::invalid_argument("invalid retained Tiny forward-loss arguments");
    for (std::uint32_t i = 0; i < Tcap; ++i) {
        if (toks[i] >= V || targets[i] >= V || masks[i] > 1u)
            throw std::invalid_argument("invalid retained Tiny target or mask");
    }
    constexpr vk::DeviceSize tokenBytes = Tcap * sizeof(std::uint32_t);
    constexpr vk::DeviceSize targetOffset = tokenBytes;
    constexpr vk::DeviceSize maskOffset = targetOffset + tokenBytes;
    constexpr vk::DeviceSize logitsOffset = maskOffset + tokenBytes;
    constexpr vk::DeviceSize lossOffset = logitsOffset + Tcap * Vp * sizeof(float);
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("retained forward-loss staging map failed");
    std::memcpy(static_cast<char*>(mapped), toks, tokenBytes);
    std::memcpy(static_cast<char*>(mapped) + targetOffset, targets, tokenBytes);
    std::memcpy(static_cast<char*>(mapped) + maskOffset, masks, tokenBytes);
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, maskOffset + tokenBytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    lastSubmission_ = engine_.submitImmutable(0, true);
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("retained forward-loss readback map failed");
    vmaInvalidateAllocation(context_.allocator,
                            trainingStaging_->allocation,
                            logitsOffset,
                            Tcap * Vp * sizeof(float) + Tcap * sizeof(float));
    std::memcpy(out, static_cast<char*>(mapped) + logitsOffset, Tcap * Vp * sizeof(float));
    std::memcpy(row_losses, static_cast<char*>(mapped) + lossOffset, Tcap * sizeof(float));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
}

void ForwardResourceGraph::forward_loss_fixed_metrics(std::uint32_t const* toks,
                                                      std::uint32_t const* targets,
                                                      std::uint32_t const* masks,
                                                      float* loss,
                                                      std::uint32_t* count) {
    if (!fixedForwardLossRecorded_ || !toks || !targets || !masks || !loss || !count)
        throw std::invalid_argument("invalid Tiny metrics arguments");
    for (std::uint32_t i = 0; i < Tcap; ++i)
        if (toks[i] >= V || targets[i] >= V || masks[i] > 1u)
            throw std::invalid_argument("invalid Tiny metrics token, target, or mask");
    constexpr vk::DeviceSize tokenBytes = Tcap * sizeof(std::uint32_t);
    constexpr vk::DeviceSize targetOffset = tokenBytes;
    constexpr vk::DeviceSize maskOffset = targetOffset + tokenBytes;
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("Tiny metrics staging map failed");
    std::memcpy(mapped, toks, tokenBytes);
    std::memcpy(static_cast<char*>(mapped) + targetOffset, targets, tokenBytes);
    std::memcpy(static_cast<char*>(mapped) + maskOffset, masks, tokenBytes);
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, maskOffset + tokenBytes);
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    lastSubmission_ = engine_.submit([this](vk::CommandBuffer cmd) {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}.setBuffer(trainingStaging_->buffer).setSize(VK_WHOLE_SIZE)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                            {});
        constexpr vk::DeviceSize tokenBytes = Tcap * sizeof(std::uint32_t);
        cmd.copyBuffer(trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(tokenBytes));
        cmd.copyBuffer(
            trainingStaging_->buffer, targets_->buffer, vk::BufferCopy{}.setSrcOffset(tokenBytes).setSize(tokenBytes));
        cmd.copyBuffer(
            trainingStaging_->buffer, mask_->buffer, vk::BufferCopy{}.setSrcOffset(2 * tokenBytes).setSize(tokenBytes));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                            {},
                            {});
        ForwardPC const fp{Tcap, 0};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
        cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(fp), &fp);
        cmd.dispatch(5, Tcap, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        cmd.copyBuffer(projected_->buffer, activations_->buffer, vk::BufferCopy{}.setSize(Tcap * H * sizeof(float)));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        FFNForwardPC const ff{Tcap, H, 4 * H};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, ffnForwardPipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, ffnForwardPipelineLayout_, 0, ffnForwardSet_, {});
        cmd.pushConstants(ffnForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(ff), &ff);
        cmd.dispatch((Tcap * H + 255) / 256, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {});
         cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                             vk::PipelineStageFlagBits::eTransfer,
                             {},
                             {},
                             vk::BufferMemoryBarrier{}.setBuffer(projected_->buffer)
                                 .setSize(Tcap * H * sizeof(float))
                                 .setSrcAccessMask(vk::AccessFlagBits::eTransferRead)
                                 .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
                             {});
        cmd.copyBuffer(ffnOutput_->buffer, projected_->buffer, vk::BufferCopy{}.setSize(Tcap * H * sizeof(float)));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        LmHeadForwardPC const lmf{Tcap, H, V, Vp};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmHeadForwardPipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmHeadForwardPipelineLayout_, 0, lmHeadForwardSet_, {});
        cmd.pushConstants(lmHeadForwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lmf), &lmf);
        cmd.dispatch((Tcap * Vp + 255) / 256, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead), {}, {});
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                            {},
                            {});
        LossPC const lp{Tcap, V, Vp};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
        cmd.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lp), &lp);
        cmd.dispatch(Tcap, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                            {},
                            {});
        MetricsPC const mp{Tcap};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                            {},
                            {});
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, metricsPipeline_);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, metricsPipelineLayout_, 0, metricsSet_, {});
        cmd.pushConstants(metricsPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(mp), &mp);
        cmd.dispatch(1, 1, 1);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
                            {},
                            {});
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(trainingStaging_->buffer)
                                .setOffset(3 * tokenBytes)
                                .setSize(8)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferRead)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
                            {});
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(trainingStaging_->buffer)
                                .setOffset(3 * tokenBytes)
                                .setSize(8)
                                .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
                            {});
        cmd.pipelineBarrier2(vk::DependencyInfo{}.setBufferMemoryBarriers(
            vk::BufferMemoryBarrier2{}
                .setSrcStageMask(vk::PipelineStageFlagBits2::eCopy)
                .setSrcAccessMask(vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite)
                .setDstStageMask(vk::PipelineStageFlagBits2::eCopy)
                .setDstAccessMask(vk::AccessFlagBits2::eTransferWrite)
                .setBuffer(trainingStaging_->buffer)
                .setOffset(3 * tokenBytes)
                .setSize(8)));
        cmd.copyBuffer(
            metrics_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(3 * tokenBytes).setSize(8));
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {},
                            vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite,
                                              vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite), {}, {});
    });
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("Tiny metrics readback map failed");
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, 3 * tokenBytes, 8);
    std::memcpy(loss, static_cast<char*>(mapped) + 3 * tokenBytes, sizeof(float));
    std::memcpy(count, static_cast<char*>(mapped) + 3 * tokenBytes + sizeof(float), sizeof(std::uint32_t));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
}

void ForwardResourceGraph::readback_projected(float* out, std::uint32_t rows) {
    if (!out || rows == 0 || rows > Tcap)
        throw std::invalid_argument("invalid projected activation readback arguments");
    auto const bytes = vk::DeviceSize(rows) * H * sizeof(float);
    lastSubmission_ = engine_.submit([this, bytes](vk::CommandBuffer cmd) {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            {},
                            {},
                            vk::BufferMemoryBarrier{}
                                .setBuffer(projected_->buffer)
                                .setSize(bytes)
                                .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                            {});
        cmd.copyBuffer(projected_->buffer, readback_->buffer, vk::BufferCopy{}.setSize(bytes));
    });
    engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, readback_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("projected activation readback map failed");
    vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, bytes);
    std::memcpy(out, mapped, static_cast<size_t>(bytes));
    vmaUnmapMemory(context_.allocator, readback_->allocation);
}
void ForwardResourceGraph::forward(uint32_t const* toks, uint32_t length, float* out, bool final_only) {
    if (!toks || !out || length == 0 || length > Tcap)
        throw std::invalid_argument("invalid Tiny sequence");
    std::vector<uint32_t> t(toks, toks + length);
    for (uint32_t token : t)
        if (token >= V)
            throw std::invalid_argument("invalid Tiny token");
    upload(context_, tokens_->buffer, t.data(), length * 4);
    ForwardPC pc{length, final_only ? 1u : 0u};
    lastSubmission_ = engine_.submit(
        [this, pc](vk::CommandBuffer cmd) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd.dispatch(5, pc.seq_length, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
                                {},
                                {});
            cmd.copyBuffer(logits_->buffer,
                           readback_->buffer,
                           vk::BufferCopy{}.setSize((pc.final_only ? 1u : pc.seq_length) * Vp * 4));
        },
        0);
    engine_.wait(lastSubmission_);
    void* m = nullptr;
    if (vmaMapMemory(context_.allocator, readback_->allocation, &m) != VK_SUCCESS)
        throw std::runtime_error("readback map failed");
    vmaInvalidateAllocation(context_.allocator, readback_->allocation, 0, (final_only ? 1u : length) * Vp * 4);
    std::memcpy(out, m, (final_only ? 1u : length) * Vp * 4);
    vmaUnmapMemory(context_.allocator, readback_->allocation);
}
void ForwardResourceGraph::token_step(uint32_t token, uint32_t position, float* out) {
    if (token >= V || position >= Tcap)
        throw std::invalid_argument("invalid Tiny token");
    std::vector<uint32_t> toks(position + 1);
    for (uint32_t i = 0; i <= position; ++i)
        toks[i] = i == position ? token : 0;
    forward(toks.data(), position + 1, out, true);
}
int ForwardResourceGraph::token_step_training(uint32_t token,
                                              uint32_t position,
                                              uint32_t target,
                                              uint32_t mask,
                                              float* loss,
                                              float* dlogits,
                                              float* dprojected) {
    if (!loss || !dlogits || !dprojected || target >= V || mask > 1u)
        throw std::invalid_argument("invalid Tiny training arguments");
    constexpr vk::DeviceSize token_bytes = Tcap * sizeof(uint32_t);
    constexpr vk::DeviceSize target_offset = token_bytes;
    constexpr vk::DeviceSize mask_offset = target_offset + sizeof(uint32_t);
    constexpr vk::DeviceSize result_offset = mask_offset + sizeof(uint32_t);
    constexpr vk::DeviceSize result_bytes = Vp * sizeof(float) + sizeof(float) + H * sizeof(float);
    std::vector<uint8_t> input(static_cast<size_t>(result_offset));
    auto* staged_tokens = reinterpret_cast<uint32_t*>(input.data());
    for (uint32_t i = 0; i <= position; ++i)
        staged_tokens[i] = i == position ? token : 0;
    std::memcpy(input.data() + target_offset, &target, sizeof(target));
    std::memcpy(input.data() + mask_offset, &mask, sizeof(mask));
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("training staging map failed");
    std::memcpy(mapped, input.data(), input.size());
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, 0, input.size());
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);

    LossPC loss_pc{1, V, Vp};
    LmBackwardPC lm_pc{1, V, H, Vp, Vp, H};
    ForwardPC forward_pc{position + 1, 1};
    lastSubmission_ = engine_.submit(
        [this, forward_pc, loss_pc, lm_pc](vk::CommandBuffer cmd) {
            cmd.copyBuffer(
                trainingStaging_->buffer, tokens_->buffer, vk::BufferCopy{}.setSize(Tcap * sizeof(uint32_t)));
            cmd.copyBuffer(trainingStaging_->buffer,
                           targets_->buffer,
                           vk::BufferCopy{}.setSrcOffset(Tcap * sizeof(uint32_t)).setSize(sizeof(uint32_t)));
            cmd.copyBuffer(
                trainingStaging_->buffer,
                mask_->buffer,
                vk::BufferCopy{}.setSrcOffset(Tcap * sizeof(uint32_t) + sizeof(uint32_t)).setSize(sizeof(uint32_t)));
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout_, 0, descriptorSet_, {});
            cmd.pushConstants(pipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(forward_pc), &forward_pc);
            cmd.dispatch(5, forward_pc.seq_length, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.fillBuffer(dlogits_->buffer, 0, Vp * sizeof(float), 0);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderWrite),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lossPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lossPipelineLayout_, 0, lossSet_, {});
            cmd.pushConstants(lossPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(loss_pc), &loss_pc);
            cmd.dispatch(1, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, lmBackwardPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, lmBackwardPipelineLayout_, 0, lmBackwardSet_, {});
            cmd.pushConstants(lmBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lm_pc), &lm_pc);
            cmd.dispatch(1, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
                                {},
                                {});
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                {},
                                vk::BufferMemoryBarrier{}
                                    .setBuffer(trainingStaging_->buffer)
                                    .setOffset(result_offset)
                                    .setSize(result_bytes)
                                    .setSrcAccessMask(vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite)
                                    .setDstAccessMask(vk::AccessFlagBits::eTransferWrite),
                             {});
            cmd.copyBuffer(dlogits_->buffer,
                           trainingStaging_->buffer,
                           vk::BufferCopy{}.setDstOffset(result_offset).setSize(Vp * sizeof(float)));
            cmd.copyBuffer(rowLoss_->buffer,
                           trainingStaging_->buffer,
                           vk::BufferCopy{}.setDstOffset(result_offset + Vp * sizeof(float)).setSize(sizeof(float)));
            cmd.copyBuffer(dprojected_->buffer,
                           trainingStaging_->buffer,
                           vk::BufferCopy{}
                               .setDstOffset(result_offset + Vp * sizeof(float) + sizeof(float))
                               .setSize(H * sizeof(float)));
        },
        0);
    engine_.wait(lastSubmission_);
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("training readback map failed");
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, result_offset, result_bytes);
    auto* bytes = static_cast<char*>(mapped) + result_offset;
    std::memcpy(dlogits, bytes, Vp * sizeof(float));
    std::memcpy(loss, bytes + Vp * sizeof(float), sizeof(float));
    std::memcpy(dprojected, bytes + Vp * sizeof(float) + sizeof(float), H * sizeof(float));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    return 0;
}
int ForwardResourceGraph::begin_lora_accumulation() {
    lastSubmission_ = engine_.submit(
        [this](vk::CommandBuffer cmd) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, loraClearPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, loraClearPipelineLayout_, 0, loraClearSet_, {});
            LoraClearPC pc{4 * H * loraRank_};
            cmd.pushConstants(loraClearPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd.dispatch((pc.elements + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    accumulating_ = true;
    return 0;
}

int ForwardResourceGraph::token_step_training_backward_accumulate(std::uint32_t token,
                                                                  std::uint32_t position,
                                                                  std::uint32_t target,
                                                                  std::uint32_t mask,
                                                                  float const* doutput,
                                                                  float* loss,
                                                                  float* dlogits,
                                                                  float* dprojected,
                                                                  float* dquery,
                                                                  float* dkey,
                                                                  float* dvalue,
                                                                  float* dcontext,
                                                                  float* dstates) {
    if (!accumulating_)
        return 1;
    return token_step_training_backward(
        token, position, target, mask, doutput, loss, dlogits, dprojected, dquery, dkey, dvalue, dcontext, dstates);
}

int ForwardResourceGraph::token_windows_training_backward_accumulate(std::uint32_t const* tokens,
                                                                     std::uint32_t const* targets,
                                                                     std::uint32_t const* mask,
                                                                     std::uint32_t windows,
                                                                     std::uint32_t window_length,
                                                                     float* losses) {
    if (!accumulating_ || !tokens || !targets || !mask || !losses || windows == 0 || window_length == 0 ||
        window_length > Tcap)
        return 1;
    std::vector<float> doutput(H, 0.0f), loss(1), dlogits(Vp), dprojected(H), dquery(Tcap * H), dkey(Tcap * H),
        dvalue(Tcap * H), dcontext(Tcap * H), dstates(Tcap * H);
    try {
        for (std::uint32_t window = 0; window < windows; ++window) {
            auto const base = static_cast<size_t>(window) * window_length;
            for (std::uint32_t position = 0; position < window_length; ++position) {
                auto const index = base + position;
                if (token_step_training_backward_accumulate(tokens[index],
                                                            position,
                                                            targets[index],
                                                            mask[index],
                                                            doutput.data(),
                                                            loss.data(),
                                                            dlogits.data(),
                                                            dprojected.data(),
                                                            dquery.data(),
                                                            dkey.data(),
                                                            dvalue.data(),
                                                            dcontext.data(),
                                                            dstates.data()) != 0)
                    return 2;
                losses[index] = loss[0];
            }
        }
    } catch (...) {
        return 2;
    }
    return 0;
}

int ForwardResourceGraph::begin_lora_adamw() {
    return begin_lora_accumulation();
}
int ForwardResourceGraph::accumulate_lora_adamw(std::uint32_t token,
                                                std::uint32_t position,
                                                std::uint32_t target,
                                                std::uint32_t mask,
                                                float const* doutput,
                                                float* loss,
                                                float* dlogits,
                                                float* dprojected,
                                                float* dquery,
                                                float* dkey,
                                                float* dvalue,
                                                float* dcontext,
                                                float* dstates) {
    return token_step_training_backward_accumulate(
        token, position, target, mask, doutput, loss, dlogits, dprojected, dquery, dkey, dvalue, dcontext, dstates);
}
int ForwardResourceGraph::finalize_lora_adamw(float learning_rate,
                                              float beta1,
                                              float beta2,
                                              float epsilon,
                                              float weight_decay,
                                              float normalizer) {
    if (!accumulating_ || !std::isfinite(learning_rate) || !std::isfinite(beta1) || !std::isfinite(beta2) ||
        !std::isfinite(epsilon) || !std::isfinite(weight_decay) || !std::isfinite(normalizer) ||
        learning_rate <= 0.0f || beta1 < 0.0f || beta1 >= 1.0f || beta2 < 0.0f || beta2 >= 1.0f || epsilon <= 0.0f ||
        normalizer <= 0.0f)
        return 1;
    ++adamwStep_;
    LoraAdamwPC pc{
        loraRank_, static_cast<uint32_t>(adamwStep_), learning_rate, beta1, beta2, epsilon, weight_decay, normalizer};
    lastSubmission_ = engine_.submit(
        [this, pc](vk::CommandBuffer cmd) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, loraAdamwPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, loraAdamwPipelineLayout_, 0, loraAdamwSet_, {});
            cmd.pushConstants(loraAdamwPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd.dispatch((4 * H * loraRank_ + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    accumulating_ = false;
    return 0;
}
int ForwardResourceGraph::readback_lora_adamw_state(float* adapters, float* m, float* v, std::uint64_t* step) {
    if (!adapters || !m || !v || !step)
        return 1;
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    auto const base = adamw_staging_offset(loraRank_);
    lastSubmission_ = engine_.submit(
        [this, base](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                {},
                                {vk::BufferMemoryBarrier{}
                                     .setBuffer(loraA_->buffer)
                                     .setSize(lora_a_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraB_->buffer)
                                     .setSize(lora_b_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraMA_->buffer)
                                     .setSize(lora_a_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraVA_->buffer)
                                     .setSize(lora_a_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraMB_->buffer)
                                     .setSize(lora_b_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraVB_->buffer)
                                     .setSize(lora_b_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eShaderWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eTransferRead)},
                                {});
            auto copy = [&](vk::Buffer src, vk::DeviceSize src_offset, vk::DeviceSize dst_offset, vk::DeviceSize size) {
                cmd.copyBuffer(src,
                               trainingStaging_->buffer,
                               vk::BufferCopy{}.setSrcOffset(src_offset).setDstOffset(base + dst_offset).setSize(size));
            };
            copy(loraA_->buffer, 0, 0, lora_a_bytes(loraRank_));
            copy(loraB_->buffer, 0, lora_a_bytes(loraRank_), lora_b_bytes(loraRank_));
            copy(loraMA_->buffer, 0, lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_), lora_a_bytes(loraRank_));
            copy(loraVA_->buffer, 0, 2 * lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_), lora_a_bytes(loraRank_));
            copy(
                loraMB_->buffer, 0, 2 * lora_a_bytes(loraRank_) + 2 * lora_b_bytes(loraRank_), lora_b_bytes(loraRank_));
            copy(
                loraVB_->buffer, 0, 3 * lora_a_bytes(loraRank_) + 2 * lora_b_bytes(loraRank_), lora_b_bytes(loraRank_));
        },
        0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    vmaInvalidateAllocation(context_.allocator, trainingStaging_->allocation, base, adamw_state_bytes(loraRank_));
    auto* bytes = static_cast<char*>(mapped) + base;
    std::memcpy(adapters, bytes, lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_));
    std::memcpy(m, bytes + lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_), lora_a_bytes(loraRank_));
    std::memcpy(m + lora_a_bytes(loraRank_) / sizeof(float),
                bytes + 2 * lora_a_bytes(loraRank_) + 2 * lora_b_bytes(loraRank_),
                lora_b_bytes(loraRank_));
    std::memcpy(v, bytes + 2 * lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_), lora_a_bytes(loraRank_));
    std::memcpy(v + lora_a_bytes(loraRank_) / sizeof(float),
                bytes + 3 * lora_a_bytes(loraRank_) + 2 * lora_b_bytes(loraRank_),
                lora_b_bytes(loraRank_));
    *step = adamwStep_;
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    return 0;
}
int ForwardResourceGraph::update_lora_adamw_state(float const* adapters,
                                                  float const* m,
                                                  float const* v,
                                                  std::uint64_t step) {
    if (!adapters || !m || !v || step > UINT32_MAX)
        return 1;
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    auto const base = adamw_staging_offset(loraRank_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        return 2;
    auto* bytes = static_cast<char*>(mapped) + base;
    std::memcpy(bytes, adapters, lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_));
    std::memcpy(bytes + lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_), m, lora_a_bytes(loraRank_));
    std::memcpy(bytes + 2 * lora_a_bytes(loraRank_) + 2 * lora_b_bytes(loraRank_),
                m + lora_a_bytes(loraRank_) / sizeof(float),
                lora_b_bytes(loraRank_));
    std::memcpy(bytes + 2 * lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_), v, lora_a_bytes(loraRank_));
    std::memcpy(bytes + 3 * lora_a_bytes(loraRank_) + 2 * lora_b_bytes(loraRank_),
                v + lora_a_bytes(loraRank_) / sizeof(float),
                lora_b_bytes(loraRank_));
    vmaFlushAllocation(context_.allocator, trainingStaging_->allocation, base, adamw_state_bytes(loraRank_));
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    lastSubmission_ = engine_.submit(
        [this, base](vk::CommandBuffer cmd) {
            auto copy = [&](vk::Buffer dst, vk::DeviceSize src_offset, vk::DeviceSize size) {
                cmd.copyBuffer(
                    trainingStaging_->buffer, dst, vk::BufferCopy{}.setSrcOffset(base + src_offset).setSize(size));
            };
            copy(loraA_->buffer, 0, lora_a_bytes(loraRank_));
            copy(loraB_->buffer, lora_a_bytes(loraRank_), lora_b_bytes(loraRank_));
            copy(loraMA_->buffer, lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_), lora_a_bytes(loraRank_));
            copy(loraVA_->buffer, 2 * lora_a_bytes(loraRank_) + lora_b_bytes(loraRank_), lora_a_bytes(loraRank_));
            copy(loraMB_->buffer, 2 * lora_a_bytes(loraRank_) + 2 * lora_b_bytes(loraRank_), lora_b_bytes(loraRank_));
            copy(loraVB_->buffer, 3 * lora_a_bytes(loraRank_) + 2 * lora_b_bytes(loraRank_), lora_b_bytes(loraRank_));
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                {},
                                {vk::BufferMemoryBarrier{}
                                     .setBuffer(loraA_->buffer)
                                     .setSize(lora_a_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eShaderRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraB_->buffer)
                                     .setSize(lora_b_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eShaderRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraMA_->buffer)
                                     .setSize(lora_a_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eShaderRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraVA_->buffer)
                                     .setSize(lora_a_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eShaderRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraMB_->buffer)
                                     .setSize(lora_b_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eShaderRead),
                                 vk::BufferMemoryBarrier{}
                                     .setBuffer(loraVB_->buffer)
                                     .setSize(lora_b_bytes(loraRank_))
                                     .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                                     .setDstAccessMask(vk::AccessFlagBits::eShaderRead)},
                                {});
        },
        0);
    engine_.wait(lastSubmission_);
    adamwStep_ = step;
    return 0;
}
int ForwardResourceGraph::finalize_lora_sgd(float learning_rate, float normalizer) {
    if (!accumulating_ || !std::isfinite(learning_rate) || !std::isfinite(normalizer) || learning_rate <= 0.0f ||
        normalizer <= 0.0f)
        return 1;
    lastSubmission_ = engine_.submit(
        [this, learning_rate, normalizer](vk::CommandBuffer cmd) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, loraFinalizePipeline_);
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, loraFinalizePipelineLayout_, 0, loraFinalizeSet_, {});
            LoraFinalizePC pc{loraRank_, learning_rate, normalizer};
            cmd.pushConstants(loraFinalizePipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmd.dispatch((4 * H * loraRank_ + 255) / 256, 1, 1);
        },
        0);
    engine_.wait(lastSubmission_);
    accumulating_ = false;
    return 0;
}

int ForwardResourceGraph::token_step_training_backward(std::uint32_t token,
                                                       std::uint32_t position,
                                                       std::uint32_t target,
                                                       std::uint32_t mask,
                                                       float const* doutput,
                                                       float* loss,
                                                       float* dlogits,
                                                       float* dprojected,
                                                       float* dquery,
                                                       float* dkey,
                                                       float* dvalue,
                                                       float* dcontext,
                                                       float* dstates) {
    if (!doutput || !loss || !dlogits || !dprojected || !dquery || !dkey || !dvalue || !dcontext || !dstates ||
        token >= V || position >= Tcap || target >= V || mask > 1u)
        throw std::invalid_argument("invalid Tiny backward arguments");
    std::vector<float> logits(Vp), upstream(Tcap * H, 0.0f), q(Tcap * H), k(Tcap * H), v(Tcap * H), refq(H), refk(H),
        refv(H);
    if (token_step_training(token, position, target, mask, loss, dlogits, dprojected) != 0)
        return 2;
    token_step(token, position, logits.data());
    std::memcpy(upstream.data() + position * H, doutput, H * sizeof(float));
    // The persistent forward graph leaves projected Q/K/V in device buffers. Run
    // the reusable attention pipeline after the existing loss/projection graph.
    upload(context_, doutput_->buffer, upstream.data(), upstream.size() * sizeof(float));
    AttentionBackwardPC pc{position + 1, H, 0};
    ProjectionBackwardPC proj{position + 1, H, H, 1};
    lastSubmission_ = engine_.submit(
        [this, pc, proj, position](vk::CommandBuffer cmd) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, attentionBackwardPipeline_);
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, attentionBackwardPipelineLayout_, 0, attentionBackwardSet_, {});
            for (uint32_t mode = 0; mode < 3; ++mode) {
                AttentionBackwardPC mode_pc{pc.tokens, pc.hidden, mode};
                cmd.pushConstants(
                    attentionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(mode_pc), &mode_pc);
                cmd.dispatch((pc.tokens * pc.hidden + 255) / 256, 1, 1);
                if (mode != 2)
                    cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        {},
                        vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderWrite),
                        {},
                        {});
            }
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, projectionBackwardPipeline_);
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardSet_, {});
            ProjectionBackwardPC context_proj{1, H, H, 0};
            cmd.pushConstants(projectionBackwardPipelineLayout_,
                              vk::ShaderStageFlagBits::eCompute,
                              0,
                              sizeof(context_proj),
                              &context_proj);
            cmd.dispatch(1, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderWrite),
                                {},
                                {});
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardQkvSet_, {});
            cmd.pushConstants(
                projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(proj), &proj);
            cmd.dispatch((proj.rows * H + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderWrite),
                                {},
                                {});
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardKVSet_, {});
            cmd.pushConstants(
                projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(proj), &proj);
            cmd.dispatch((proj.rows * H + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderWrite),
                                {},
                                {});
            cmd.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, projectionBackwardPipelineLayout_, 0, projectionBackwardVVSet_, {});
            cmd.pushConstants(
                projectionBackwardPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(proj), &proj);
            cmd.dispatch((proj.rows * H + 255) / 256, 1, 1);
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
                                {},
                                {});
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, loraPipeline_);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, loraPipelineLayout_, 0, loraSet_, {});
            for (uint32_t target = 0; target < 4; ++target) {
                LoraPC lpc{target == 3u ? 1u : position + 1u, H, loraRank_, target, accumulating_ ? 1u : 0u, position};
                cmd.pushConstants(loraPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(lpc), &lpc);
                cmd.dispatch((H * loraRank_ + 255) / 256, 1, 1);
                if (target != 3)
                    cmd.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        {},
                        vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderWrite),
                        {},
                        {});
            }
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eComputeShader,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead),
                                {},
                                {});
            if (!accumulating_) {
                cmd.bindPipeline(vk::PipelineBindPoint::eCompute, loraSgdPipeline_);
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, loraSgdPipelineLayout_, 0, loraSgdSet_, {});
                LoraSgdPC sgd_pc{loraRank_, 0.01f};
                cmd.pushConstants(
                    loraSgdPipelineLayout_, vk::ShaderStageFlagBits::eCompute, 0, sizeof(sgd_pc), &sgd_pc);
                cmd.dispatch((4 * H * loraRank_ + 255) / 256, 1, 1);
            }
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eTransfer,
                                {},
                                vk::MemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead),
                                {},
                                {});
            cmd.copyBuffer(
                dquery_->buffer, trainingStaging_->buffer, vk::BufferCopy{}.setDstOffset(0).setSize(H * sizeof(float)));
            cmd.copyBuffer(dkey_->buffer,
                           trainingStaging_->buffer,
                           vk::BufferCopy{}
                               .setSrcOffset(position * H * sizeof(float))
                               .setDstOffset(H * sizeof(float))
                               .setSize(H * sizeof(float)));
            cmd.copyBuffer(dvalue_->buffer,
                           trainingStaging_->buffer,
                           vk::BufferCopy{}
                               .setSrcOffset(position * H * sizeof(float))
                               .setDstOffset(2 * H * sizeof(float))
                               .setSize(H * sizeof(float)));
            cmd.copyBuffer(dcontext_->buffer,
                           trainingStaging_->buffer,
                           vk::BufferCopy{}.setDstOffset(3 * H * sizeof(float)).setSize(H * sizeof(float)));
            cmd.copyBuffer(dstates_->buffer,
                           trainingStaging_->buffer,
                           vk::BufferCopy{}.setDstOffset(4 * H * sizeof(float)).setSize(Tcap * H * sizeof(float)));
            for (uint32_t target = 0; target < 4; ++target) {
                vk::BufferCopy ca{};
                ca.setSrcOffset(target * H * loraRank_ * sizeof(float))
                    .setDstOffset((4 * H + Tcap * H + target * H * loraRank_) * sizeof(float))
                    .setSize(H * loraRank_ * sizeof(float));
                cmd.copyBuffer(loraDA_->buffer, trainingStaging_->buffer, ca);
                vk::BufferCopy cb{};
                cb.setSrcOffset(target * loraRank_ * H * sizeof(float))
                    .setDstOffset((4 * H + Tcap * H + 4 * H * loraRank_ + target * loraRank_ * H) * sizeof(float))
                    .setSize(loraRank_ * H * sizeof(float));
                cmd.copyBuffer(loraDB_->buffer, trainingStaging_->buffer, cb);
                vk::BufferCopy aa{};
                aa.setSrcOffset(target * H * loraRank_ * sizeof(float))
                    .setDstOffset((4 * H + Tcap * H + 8 * H * loraRank_ + target * H * loraRank_) * sizeof(float))
                    .setSize(H * loraRank_ * sizeof(float));
                cmd.copyBuffer(loraA_->buffer, trainingStaging_->buffer, aa);
                vk::BufferCopy bb{};
                bb.setSrcOffset(target * loraRank_ * H * sizeof(float))
                    .setDstOffset((4 * H + Tcap * H + 12 * H * loraRank_ + target * loraRank_ * H) * sizeof(float))
                    .setSize(loraRank_ * H * sizeof(float));
                cmd.copyBuffer(loraB_->buffer, trainingStaging_->buffer, bb);
            }
        },
        0);
    engine_.wait(lastSubmission_);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, trainingStaging_->allocation, &mapped) != VK_SUCCESS)
        throw std::runtime_error("training readback map failed");
    vmaInvalidateAllocation(
        context_.allocator, trainingStaging_->allocation, 0, (4 * H + Tcap * H + 16 * H * loraRank_) * sizeof(float));
    std::memcpy(dquery, mapped, H * sizeof(float));
    std::memcpy(dkey, static_cast<char*>(mapped) + H * sizeof(float), H * sizeof(float));
    std::memcpy(dvalue, static_cast<char*>(mapped) + 2 * H * sizeof(float), H * sizeof(float));
    std::memcpy(dcontext, static_cast<char*>(mapped) + 3 * H * sizeof(float), H * sizeof(float));
    std::memcpy(dstates, static_cast<char*>(mapped) + 4 * H * sizeof(float), Tcap * H * sizeof(float));
    auto* rb = static_cast<char*>(mapped) + (4 * H + Tcap * H) * sizeof(float);
    for (uint32_t i = 0; i < 4; ++i) {
        loraReadback_[2 * i].assign(reinterpret_cast<float*>(rb + i * H * loraRank_ * sizeof(float)),
                                    reinterpret_cast<float*>(rb + (i + 1) * H * loraRank_ * sizeof(float)));
    }
    rb += 4 * H * loraRank_ * sizeof(float);
    for (uint32_t i = 0; i < 4; ++i) {
        loraReadback_[2 * i + 1].assign(reinterpret_cast<float*>(rb + i * loraRank_ * H * sizeof(float)),
                                        reinterpret_cast<float*>(rb + (i + 1) * loraRank_ * H * sizeof(float)));
    }
    rb += 4 * loraRank_ * H * sizeof(float);
    for (uint32_t i = 0; i < 4; ++i) {
        loraAdapterReadback_[2 * i].assign(reinterpret_cast<float*>(rb + i * H * loraRank_ * sizeof(float)),
                                           reinterpret_cast<float*>(rb + (i + 1) * H * loraRank_ * sizeof(float)));
    }
    rb += 4 * H * loraRank_ * sizeof(float);
    for (uint32_t i = 0; i < 4; ++i) {
        loraAdapterReadback_[2 * i + 1].assign(reinterpret_cast<float*>(rb + i * loraRank_ * H * sizeof(float)),
                                               reinterpret_cast<float*>(rb + (i + 1) * loraRank_ * H * sizeof(float)));
    }
    vmaUnmapMemory(context_.allocator, trainingStaging_->allocation);
    if (token_step_training(token, position, target, mask, loss, dlogits, dprojected) != 0)
        return 2;
    return 0;
}
int ForwardResourceGraph::readback_lora_gradients(float* dquery_a,
                                                  float* dquery_b,
                                                  float* dkey_a,
                                                  float* dkey_b,
                                                  float* dvalue_a,
                                                  float* dvalue_b,
                                                  float* doutput_a,
                                                  float* doutput_b) {
    if (!dquery_a || !dquery_b || !dkey_a || !dkey_b || !dvalue_a || !dvalue_b || !doutput_a || !doutput_b)
        return 1;
    std::memcpy(dquery_a, loraReadback_[0].data(), loraReadback_[0].size() * sizeof(float));
    std::memcpy(dquery_b, loraReadback_[1].data(), loraReadback_[1].size() * sizeof(float));
    std::memcpy(dkey_a, loraReadback_[2].data(), loraReadback_[2].size() * sizeof(float));
    std::memcpy(dkey_b, loraReadback_[3].data(), loraReadback_[3].size() * sizeof(float));
    std::memcpy(dvalue_a, loraReadback_[4].data(), loraReadback_[4].size() * sizeof(float));
    std::memcpy(dvalue_b, loraReadback_[5].data(), loraReadback_[5].size() * sizeof(float));
    std::memcpy(doutput_a, loraReadback_[6].data(), loraReadback_[6].size() * sizeof(float));
    std::memcpy(doutput_b, loraReadback_[7].data(), loraReadback_[7].size() * sizeof(float));
    return 0;
}
int ForwardResourceGraph::readback_lora_adapters(float* query_a,
                                                 float* query_b,
                                                 float* key_a,
                                                 float* key_b,
                                                 float* value_a,
                                                 float* value_b,
                                                 float* output_a,
                                                 float* output_b) {
    if (!query_a || !query_b || !key_a || !key_b || !value_a || !value_b || !output_a || !output_b)
        return 1;
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    auto const a_bytes = H * loraRank_ * sizeof(float);
    auto const b_bytes = loraRank_ * H * sizeof(float);
    TinyBuffer staging = make_buffer(context_, 4 * (a_bytes + b_bytes), vk::BufferUsageFlagBits::eTransferDst,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    vk::CommandPool pool = context_.device.createCommandPool(
        vk::CommandPoolCreateInfo{}.setQueueFamilyIndex(context_.computeQueueFamily));
    auto cmd = context_.device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{}
                                                           .setCommandPool(pool)
                                                           .setLevel(vk::CommandBufferLevel::ePrimary)
                                                           .setCommandBufferCount(1))
                   .front();
    cmd.begin(vk::CommandBufferBeginInfo{});
    for (uint32_t target = 0; target < 4; ++target) {
        auto const base = target * (a_bytes + b_bytes);
        cmd.copyBuffer(loraA_->buffer, staging.buffer,
                       vk::BufferCopy{}.setSrcOffset(target * a_bytes).setDstOffset(base).setSize(a_bytes));
        cmd.copyBuffer(loraB_->buffer, staging.buffer,
                       vk::BufferCopy{}.setSrcOffset(target * b_bytes).setDstOffset(base + a_bytes).setSize(b_bytes));
    }
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                        {}, {},
                        vk::BufferMemoryBarrier{}
                            .setBuffer(staging.buffer)
                            .setSize(4 * (a_bytes + b_bytes))
                            .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                            .setDstAccessMask(vk::AccessFlagBits::eHostRead), {});
    cmd.end();
    context_.computeQueue.submit(vk::SubmitInfo{}.setCommandBuffers(cmd));
    context_.computeQueue.waitIdle();
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, staging.allocation, &mapped) != VK_SUCCESS) {
        context_.device.freeCommandBuffers(pool, cmd);
        context_.device.destroyCommandPool(pool);
        drop(context_, staging);
        return 2;
    }
    vmaInvalidateAllocation(context_.allocator, staging.allocation, 0, 4 * (a_bytes + b_bytes));
    auto* bytes = static_cast<char*>(mapped);
    float* outputs[] = {query_a, query_b, key_a, key_b, value_a, value_b, output_a, output_b};
    for (uint32_t target = 0; target < 4; ++target) {
        auto const base = target * (a_bytes + b_bytes);
        std::memcpy(outputs[2 * target], bytes + base, a_bytes);
        std::memcpy(outputs[2 * target + 1], bytes + base + a_bytes, b_bytes);
    }
    vmaUnmapMemory(context_.allocator, staging.allocation);
    context_.device.freeCommandBuffers(pool, cmd);
    context_.device.destroyCommandPool(pool);
    drop(context_, staging);
    return 0;
}
int ForwardResourceGraph::update_lora_adapters(float const* query_a,
                                               float const* query_b,
                                               float const* key_a,
                                               float const* key_b,
                                               float const* value_a,
                                               float const* value_b,
                                               float const* output_a,
                                               float const* output_b) {
    if (!query_a || !query_b || !key_a || !key_b || !value_a || !value_b || !output_a || !output_b)
        return 1;
    if (lastSubmission_ != 0)
        engine_.wait(lastSubmission_);
    std::array<float const*, 8> factors{query_a, query_b, key_a, key_b, value_a, value_b, output_a, output_b};
    vk::DeviceSize const a_bytes = H * loraRank_ * sizeof(float);
    vk::DeviceSize const b_bytes = loraRank_ * H * sizeof(float);
    TinyBuffer staging = make_buffer(context_,
                                     4 * (a_bytes + b_bytes),
                                     vk::BufferUsageFlagBits::eTransferSrc,
                                     VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                     VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    void* mapped = nullptr;
    if (vmaMapMemory(context_.allocator, staging.allocation, &mapped) != VK_SUCCESS)
        return 2;
    for (uint32_t target = 0; target < 4; ++target) {
        auto* dst = static_cast<char*>(mapped) + target * (a_bytes + b_bytes);
        std::memcpy(dst, factors[2 * target], a_bytes);
        std::memcpy(dst + a_bytes, factors[2 * target + 1], b_bytes);
    }
    vmaFlushAllocation(context_.allocator, staging.allocation, 0, 4 * (a_bytes + b_bytes));
    vmaUnmapMemory(context_.allocator, staging.allocation);
    vk::CommandPool pool =
        context_.device.createCommandPool(vk::CommandPoolCreateInfo{}.setQueueFamilyIndex(context_.computeQueueFamily));
    auto cmd = context_.device
                   .allocateCommandBuffers(vk::CommandBufferAllocateInfo{}
                                               .setCommandPool(pool)
                                               .setLevel(vk::CommandBufferLevel::ePrimary)
                                               .setCommandBufferCount(1))
                   .front();
    cmd.begin(vk::CommandBufferBeginInfo{});
    for (uint32_t target = 0; target < 4; ++target) {
        vk::DeviceSize const src = target * (a_bytes + b_bytes);
        cmd.copyBuffer(staging.buffer,
                       loraA_->buffer,
                       vk::BufferCopy{}.setSrcOffset(src).setDstOffset(target * a_bytes).setSize(a_bytes));
        cmd.copyBuffer(staging.buffer,
                       loraB_->buffer,
                       vk::BufferCopy{}.setSrcOffset(src + a_bytes).setDstOffset(target * b_bytes).setSize(b_bytes));
    }
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader,
                        {},
                        {},
                        {vk::BufferMemoryBarrier{}
                             .setBuffer(loraA_->buffer)
                             .setSize(4 * a_bytes)
                             .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                             .setDstAccessMask(vk::AccessFlagBits::eShaderRead),
                         vk::BufferMemoryBarrier{}
                             .setBuffer(loraB_->buffer)
                             .setSize(4 * b_bytes)
                             .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                             .setDstAccessMask(vk::AccessFlagBits::eShaderRead)},
                        {});
    cmd.end();
    context_.computeQueue.submit(vk::SubmitInfo{}.setCommandBuffers(cmd));
    context_.computeQueue.waitIdle();
    context_.device.destroyCommandPool(pool);
    drop(context_, staging);
    loraAdapterReadback_[0].assign(query_a, query_a + H * loraRank_);
    loraAdapterReadback_[1].assign(query_b, query_b + loraRank_ * H);
    loraAdapterReadback_[2].assign(key_a, key_a + H * loraRank_);
    loraAdapterReadback_[3].assign(key_b, key_b + loraRank_ * H);
    loraAdapterReadback_[4].assign(value_a, value_a + H * loraRank_);
    loraAdapterReadback_[5].assign(value_b, value_b + loraRank_ * H);
    loraAdapterReadback_[6].assign(output_a, output_a + H * loraRank_);
    loraAdapterReadback_[7].assign(output_b, output_b + loraRank_ * H);
    return 0;
}
} // namespace vulkan_runtime::tiny
struct spaceslug_tiny_forward_graph {
    vulkan_runtime::core::VulkanContext context;
    std::unique_ptr<vulkan_runtime::tiny::ForwardResourceGraph> graph;
};
struct spaceslug_tiny_base_checkpoint {
    vulkan_runtime::tiny::BaseCheckpoint snapshot;
};
extern "C" spaceslug_tiny_base_checkpoint* spaceslug_tiny_base_checkpoint_create() {
    try {
        return new spaceslug_tiny_base_checkpoint;
    } catch (...) {
        return nullptr;
    }
}
extern "C" void spaceslug_tiny_base_checkpoint_destroy(spaceslug_tiny_base_checkpoint* checkpoint) {
    delete checkpoint;
}
extern "C" uint32_t spaceslug_tiny_base_checkpoint_group_mask(spaceslug_tiny_base_checkpoint const* checkpoint) {
    return checkpoint ? checkpoint->snapshot.group_mask : 0;
}
extern "C" uint64_t spaceslug_tiny_base_checkpoint_adamw_step(spaceslug_tiny_base_checkpoint const* checkpoint) {
    return checkpoint ? checkpoint->snapshot.adamw_step : 0;
}
extern "C" uint32_t spaceslug_tiny_base_checkpoint_profile_rank(spaceslug_tiny_base_checkpoint const* checkpoint) {
    return checkpoint ? checkpoint->snapshot.profile.lora_rank : 0;
}
extern "C" uint32_t spaceslug_tiny_base_checkpoint_float_count(spaceslug_tiny_base_checkpoint const* checkpoint,
                                                               uint32_t group) {
    if (!checkpoint)
        return 0;
    switch (group) {
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_LM_HEAD:
        return static_cast<uint32_t>(checkpoint->snapshot.lm_head.size());
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_OUTPUT:
        return static_cast<uint32_t>(checkpoint->snapshot.output.size());
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_QKV:
        return static_cast<uint32_t>(checkpoint->snapshot.query.size() * 3);
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_NORMALIZATION:
        return static_cast<uint32_t>(checkpoint->snapshot.gamma.size());
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_FFN:
        return static_cast<uint32_t>(checkpoint->snapshot.ffn_w1.size() + checkpoint->snapshot.ffn_b1.size() + checkpoint->snapshot.ffn_w2.size() + checkpoint->snapshot.ffn_b2.size());
    default:
        return 0;
    }
}
extern "C" float const* spaceslug_tiny_base_checkpoint_weights(spaceslug_tiny_base_checkpoint const* checkpoint,
                                                               uint32_t group) {
    if (!checkpoint)
        return nullptr;
    switch (group) {
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_LM_HEAD:
        return checkpoint->snapshot.lm_head.data();
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_OUTPUT:
        return checkpoint->snapshot.output.data();
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_QKV:
        return checkpoint->snapshot.query.data();
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_NORMALIZATION:
        return checkpoint->snapshot.gamma.data();
    case SPACESLUG_TINY_BASE_TRAIN_GROUP_FFN:
        return checkpoint->snapshot.ffn_w1.data();
    default:
        return nullptr;
    }
}
extern "C" uint32_t spaceslug_tiny_base_checkpoint_state_float_count(spaceslug_tiny_base_checkpoint const* checkpoint, uint32_t group) {
    return spaceslug_tiny_base_checkpoint_float_count(checkpoint, group);
}
extern "C" float const* spaceslug_tiny_base_checkpoint_normalization_state(spaceslug_tiny_base_checkpoint const* checkpoint, uint32_t state) {
    if (!checkpoint || state > 2) return nullptr;
    auto const& s = checkpoint->snapshot;
    return state == 0 ? s.gamma.data() : (state == 1 ? s.gamma_m.data() : s.gamma_v.data());
}
extern "C" float const* spaceslug_tiny_base_checkpoint_ffn_state(spaceslug_tiny_base_checkpoint const* checkpoint, uint32_t component, uint32_t state) {
    if (!checkpoint || component > 3 || state > 2) return nullptr;
    auto const& s = checkpoint->snapshot;
    if (component == 0) return state == 0 ? s.ffn_w1.data() : (state == 1 ? s.ffn_w1_m.data() : s.ffn_w1_v.data());
    if (component == 1) return state == 0 ? s.ffn_b1.data() : (state == 1 ? s.ffn_b1_m.data() : s.ffn_b1_v.data());
    if (component == 2) return state == 0 ? s.ffn_w2.data() : (state == 1 ? s.ffn_w2_m.data() : s.ffn_w2_v.data());
    return state == 0 ? s.ffn_b2.data() : (state == 1 ? s.ffn_b2_m.data() : s.ffn_b2_v.data());
}
extern "C" uint32_t spaceslug_tiny_base_checkpoint_positions_float_count(spaceslug_tiny_base_checkpoint const* checkpoint) {
    return checkpoint ? static_cast<uint32_t>(checkpoint->snapshot.positions.size()) : 0;
}
extern "C" float const* spaceslug_tiny_base_checkpoint_positions(spaceslug_tiny_base_checkpoint const* checkpoint) {
    return checkpoint ? checkpoint->snapshot.positions.data() : nullptr;
}
extern "C" uint32_t spaceslug_tiny_base_checkpoint_embeddings_float_count(spaceslug_tiny_base_checkpoint const* checkpoint) {
    return checkpoint ? static_cast<uint32_t>(checkpoint->snapshot.embeddings.size()) : 0;
}
extern "C" float const* spaceslug_tiny_base_checkpoint_embeddings(spaceslug_tiny_base_checkpoint const* checkpoint) {
    return checkpoint ? checkpoint->snapshot.embeddings.data() : nullptr;
}
extern "C" float const* spaceslug_tiny_base_checkpoint_qkv_weights(spaceslug_tiny_base_checkpoint const* checkpoint,
                                                                   uint32_t projection) {
    if (!checkpoint)
        return nullptr;
    switch (projection) {
    case 0:
        return checkpoint->snapshot.query.data();
    case 1:
        return checkpoint->snapshot.key.data();
    case 2:
        return checkpoint->snapshot.value.data();
    default:
        return nullptr;
    }
}
extern "C" float const* spaceslug_tiny_base_checkpoint_adamw_m(spaceslug_tiny_base_checkpoint const* checkpoint,
                                                               uint32_t group) {
    if (!checkpoint)
        return nullptr;
    return group == SPACESLUG_TINY_BASE_TRAIN_GROUP_LM_HEAD  ? checkpoint->snapshot.lm_head_m.data()
           : group == SPACESLUG_TINY_BASE_TRAIN_GROUP_OUTPUT ? checkpoint->snapshot.output_m.data()
                                                             : nullptr;
}
extern "C" float const* spaceslug_tiny_base_checkpoint_adamw_v(spaceslug_tiny_base_checkpoint const* checkpoint,
                                                               uint32_t group) {
    if (!checkpoint)
        return nullptr;
    return group == SPACESLUG_TINY_BASE_TRAIN_GROUP_LM_HEAD  ? checkpoint->snapshot.lm_head_v.data()
           : group == SPACESLUG_TINY_BASE_TRAIN_GROUP_OUTPUT ? checkpoint->snapshot.output_v.data()
                                                             : nullptr;
}
extern "C" const char* spaceslug_tiny_forward_capability() {
    return vulkan_runtime::tiny::capability;
}
extern "C" const char* spaceslug_tiny_forward_command_buffer_capability() {
    return vulkan_runtime::tiny::command_buffer_capability;
}
extern "C" uint32_t spaceslug_tiny_profile_count() {
    return static_cast<uint32_t>(vulkan_runtime::tiny::ProfileCount);
}
extern "C" int spaceslug_tiny_profile_query(uint32_t index, spaceslug_tiny_profile_descriptor* out) {
    if (!out)
        return SPACESLUG_TINY_PROFILE_INVALID_ARGUMENT;
    if (index >= vulkan_runtime::tiny::ProfileCount)
        return SPACESLUG_TINY_PROFILE_UNSUPPORTED;
    auto const& profile = vulkan_runtime::tiny::Profiles[index];
    *out = {
        profile.name, profile.hidden, profile.vocab, profile.padded_vocab, profile.token_capacity, profile.lora_rank};
    return SPACESLUG_TINY_PROFILE_SUPPORTED;
}
extern "C" int spaceslug_tiny_profile_validate(uint32_t hidden,
                                               uint32_t vocab,
                                               uint32_t padded_vocab,
                                               uint32_t token_capacity,
                                               uint32_t rank) {
    if (hidden == 0 || vocab == 0 || padded_vocab == 0 || token_capacity == 0 || rank == 0)
        return SPACESLUG_TINY_PROFILE_INVALID_ARGUMENT;
    return vulkan_runtime::tiny::profile_supported(hidden, vocab, padded_vocab, token_capacity, rank)
               ? SPACESLUG_TINY_PROFILE_SUPPORTED
               : SPACESLUG_TINY_PROFILE_UNSUPPORTED;
}
extern "C" int spaceslug_tiny_forward_lora_rank_supported(uint32_t rank) {
    return vulkan_runtime::tiny::lora_rank_supported(rank) ? 1 : 0;
}
extern "C" const char* spaceslug_tiny_forward_base_train_capability(void) {
    return vulkan_runtime::tiny::base_train_capability;
}
extern "C" const char* spaceslug_tiny_forward_full_base_training_capability(void) {
    return vulkan_runtime::tiny::full_base_training_capability;
}
extern "C" const char* spaceslug_tiny_forward_bounded_full_graph_training_capability(void) {
    return vulkan_runtime::tiny::bounded_full_graph_training_capability;
}
extern "C" const char* spaceslug_tiny_forward_retained_backward_optimizer_capability(void) {
    return vulkan_runtime::tiny::retained_backward_optimizer_capability;
}
extern "C" int spaceslug_tiny_forward_retained_backward_optimizer_supported(void) {
    return vulkan_runtime::tiny::retained_backward_optimizer_supported ? 1 : 0;
}
extern "C" const char* spaceslug_tiny_forward_arbitrary_shape_full_base_capability(void) {
    return vulkan_runtime::tiny::arbitrary_shape_full_base_capability;
}
extern "C" int spaceslug_tiny_forward_arbitrary_shape_full_base_supported(void) {
    return vulkan_runtime::tiny::arbitrary_shape_full_base_supported ? 1 : 0;
}
extern "C" int spaceslug_tiny_forward_readback_gamma_state(spaceslug_tiny_forward_graph* g, float* gamma, float* m, float* v, uint64_t* step) {
    if (!g) return -1;
    return g->graph->readback_gamma_state(gamma, m, v, step);
}
extern "C" int spaceslug_tiny_forward_update_gamma_state(spaceslug_tiny_forward_graph* g, float const* gamma, float const* m, float const* v, uint64_t step) {
    if (!g) return -1;
    return g->graph->update_gamma_state(gamma, m, v, step);
}
extern "C" const char* spaceslug_tiny_forward_ffn_capability(void) { return vulkan_runtime::tiny::ForwardResourceGraph::trainable_ffn_capability(); }
extern "C" int spaceslug_tiny_forward_readback_combined_gradients(spaceslug_tiny_forward_graph* g,
                                                                   float* gamma_gradient,
                                                                   float* ffn_gradient,
                                                                   size_t ffn_count,
                                                                   float* ffn_output_gradient,
                                                                   float* activations,
                                                                   float* scaled_states,
                                                                   float* dstate,
                                                                   uint32_t rows) {
    if (!g) return -1;
    return g->graph->readback_combined_gradients(gamma_gradient, ffn_gradient, ffn_count, ffn_output_gradient, activations, scaled_states, dstate, rows);
}
extern "C" int spaceslug_tiny_forward_readback_ffn_state(spaceslug_tiny_forward_graph* g, float* data, size_t count, uint64_t* step) {
    if (!g) return -1;
    return g->graph->readback_ffn_state(data, count, step);
}
extern "C" int spaceslug_tiny_forward_update_ffn_state(spaceslug_tiny_forward_graph* g, float const* data, size_t count, uint64_t step) {
    if (!g) return -1;
    return g->graph->update_ffn_state(data, count, step);
}
extern "C" int spaceslug_tiny_forward_base_train_group_supported(uint32_t group) {
    return vulkan_runtime::tiny::base_train_group_supported(static_cast<vulkan_runtime::tiny::BaseTrainGroup>(group))
               ? 1
               : 0;
}
extern "C" const char* spaceslug_tiny_forward_graph_embedding_training_capability(void) {
    return vulkan_runtime::tiny::graph_embedding_training_capability;
}
extern "C" int spaceslug_tiny_forward_graph_embedding_training_status(void) {
    return vulkan_runtime::tiny::graph_embedding_training_status;
}
extern "C" int spaceslug_tiny_forward_readback_positions(spaceslug_tiny_forward_graph* g, float* positions) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_positions(positions);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_readback_position_gradient(spaceslug_tiny_forward_graph* g, float* gradient, size_t count) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_position_gradient(gradient, count);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_train_positions_sgd(spaceslug_tiny_forward_graph* g,
                                                              const uint32_t* tokens,
                                                              const uint32_t* targets,
                                                              const uint32_t* masks,
                                                              uint32_t rows,
                                                              float learning_rate) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->train_positions_sgd(tokens, targets, masks, rows, learning_rate);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_train_embeddings_sgd(spaceslug_tiny_forward_graph* g,
                                                             const uint32_t* tokens,
                                                             const uint32_t* targets,
                                                             const uint32_t* masks,
                                                             uint32_t rows,
                                                             float learning_rate) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->train_embeddings_sgd(tokens, targets, masks, rows, learning_rate);
    } catch (...) {
        return 2;
    }
}
extern "C" spaceslug_tiny_forward_graph*
spaceslug_tiny_forward_create_rank(float const* e, float const* p, float const* w, uint32_t rank) {
    try {
        auto* g = new spaceslug_tiny_forward_graph;
        g->context = vulkan_runtime::core::create_context("tiny-forward-persistent");
        g->graph = std::make_unique<vulkan_runtime::tiny::ForwardResourceGraph>(g->context, e, p, w, rank);
        return g;
    } catch (...) {
        return nullptr;
    }
}
extern "C" spaceslug_tiny_forward_graph* spaceslug_tiny_forward_create(float const* e, float const* p, float const* w) {
    try {
        auto* g = new spaceslug_tiny_forward_graph;
        g->context = vulkan_runtime::core::create_context("tiny-forward-persistent");
        g->graph = std::make_unique<vulkan_runtime::tiny::ForwardResourceGraph>(g->context, e, p, w);
        return g;
    } catch (...) {
        return nullptr;
    }
}
extern "C" spaceslug_tiny_forward_graph* spaceslug_tiny_forward_create_full(float const* e,
                                                                            float const* p,
                                                                            float const* q,
                                                                            float const* k,
                                                                            float const* v,
                                                                            float const* o,
                                                                            float const* lm) {
    try {
        auto* g = new spaceslug_tiny_forward_graph;
        g->context = vulkan_runtime::core::create_context("tiny-forward-full");
        g->graph = std::make_unique<vulkan_runtime::tiny::ForwardResourceGraph>(g->context, e, p, q, k, v, o, lm);
        return g;
    } catch (...) {
        return nullptr;
    }
}
extern "C" int spaceslug_tiny_forward(const spaceslug_tiny_forward_graph* g,
                                      const uint32_t* t,
                                      uint32_t n,
                                      float* out,
                                      uint32_t final_only) {
    if (!g || !g->graph || !t || !out)
        return 1;
    try {
        g->graph->forward(t, n, out, final_only != 0);
        return 0;
    } catch (...) {
        return 2;
    }
}
extern "C" int
spaceslug_tiny_forward_fixed_retained(spaceslug_tiny_forward_graph* g, const uint32_t* tokens, float* out) {
    if (!g || !g->graph || !tokens || !out)
        return 1;
    try {
        g->graph->forward_fixed_retained(tokens, out);
        return 0;
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_loss_fixed_retained(spaceslug_tiny_forward_graph* g,
                                                          const uint32_t* tokens,
                                                          const uint32_t* targets,
                                                          const uint32_t* masks,
                                                          float* logits,
                                                          float* row_losses) {
    if (!g || !g->graph || !tokens || !targets || !masks || !logits || !row_losses)
        return 1;
    try {
        g->graph->forward_loss_fixed_retained(tokens, targets, masks, logits, row_losses);
        return 0;
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_loss_fixed_metrics(spaceslug_tiny_forward_graph* g,
                                                         const uint32_t* tokens,
                                                         const uint32_t* targets,
                                                         const uint32_t* masks,
                                                         float* loss,
                                                         uint32_t* count) {
    if (!g || !g->graph || !tokens || !targets || !masks || !loss || !count)
        return 1;
    try {
        g->graph->forward_loss_fixed_metrics(tokens, targets, masks, loss, count);
        return 0;
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_import_base_train_embeddings(spaceslug_tiny_forward_graph* g, float const* values) {
    if (!g || !g->graph)
        return 1;
    try { return g->graph->import_base_train_embeddings(values); } catch (...) { return 2; }
}
extern "C" int spaceslug_tiny_forward_import_base_train_positions(spaceslug_tiny_forward_graph* g, float const* values) {
    if (!g || !g->graph)
        return 1;
    try { return g->graph->import_base_train_positions(values); } catch (...) { return 2; }
}
extern "C" int spaceslug_tiny_forward_readback_base_train_embeddings(spaceslug_tiny_forward_graph* g, float* values) {
    if (!g || !g->graph)
        return 1;
    try { return g->graph->readback_base_train_embeddings(values); } catch (...) { return 2; }
}
extern "C" int spaceslug_tiny_forward_readback_base_train_positions(spaceslug_tiny_forward_graph* g, float* values) {
    if (!g || !g->graph)
        return 1;
    try { return g->graph->readback_base_train_positions(values); } catch (...) { return 2; }
}
extern "C" int spaceslug_tiny_forward_readback_base_checkpoint(spaceslug_tiny_forward_graph* g,
                                                               spaceslug_tiny_base_checkpoint* checkpoint) {
    if (!g || !g->graph || !checkpoint)
        return 1;
    try {
        return g->graph->readback_base_checkpoint(checkpoint->snapshot);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_update_base_checkpoint(spaceslug_tiny_forward_graph* g,
                                                             spaceslug_tiny_base_checkpoint const* checkpoint) {
    if (!g || !g->graph || !checkpoint)
        return 1;
    try {
        return g->graph->update_base_checkpoint(checkpoint->snapshot);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_import_base_train_lm_head(spaceslug_tiny_forward_graph* g, float const* weight) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->import_base_train_lm_head(weight);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_import_base_train_output(spaceslug_tiny_forward_graph* g, float const* weight) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->import_base_train_output(weight);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_readback_base_train_output(spaceslug_tiny_forward_graph* g, float* weight) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_base_train_output(weight);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_readback_base_train_lm_head(spaceslug_tiny_forward_graph* g, float* weight) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_base_train_lm_head(weight);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_readback_base_train_positions_adamw_state(spaceslug_tiny_forward_graph* g, float* p, float* m, float* v, uint64_t* step) {
    if (!g || !g->graph) return 1;
    try { return g->graph->readback_base_train_positions_adamw_state(p, m, v, step); } catch (...) { return 2; }
}
extern "C" int spaceslug_tiny_forward_update_base_train_positions_adamw_state(spaceslug_tiny_forward_graph* g, float const* p, float const* m, float const* v, uint64_t step) {
    if (!g || !g->graph) return 1;
    try { return g->graph->update_base_train_positions_adamw_state(p, m, v, step); } catch (...) { return 2; }
}
extern "C" int spaceslug_tiny_forward_readback_base_train_lm_head_adamw_state(spaceslug_tiny_forward_graph* g,
                                                                              float* w,
                                                                              float* m,
                                                                              float* v,
                                                                              uint64_t* step) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_base_train_lm_head_adamw_state(w, m, v, step);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_update_base_train_lm_head_adamw_state(spaceslug_tiny_forward_graph* g,
                                                                            float const* w,
                                                                            float const* m,
                                                                            float const* v,
                                                                            uint64_t step) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->update_base_train_lm_head_adamw_state(w, m, v, step);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_train_lm_head_sgd(spaceslug_tiny_forward_graph* g,
                                                        const uint32_t* t,
                                                        const uint32_t* y,
                                                        const uint32_t* m,
                                                        uint32_t rows,
                                                        float lr) {
    if (!g || !g->graph)
        return 1;
    return g->graph->train_lm_head_sgd(t, y, m, rows, lr);
}
extern "C" int spaceslug_tiny_forward_train_output_sgd(spaceslug_tiny_forward_graph* g,
                                                       const uint32_t* t,
                                                       const uint32_t* y,
                                                       const uint32_t* m,
                                                       uint32_t rows,
                                                       float lr) {
    if (!g || !g->graph)
        return 1;
    return g->graph->train_output_sgd(t, y, m, rows, lr);
}
extern "C" int spaceslug_tiny_forward_readback_base_train_output_adamw_state(spaceslug_tiny_forward_graph* g,
                                                                             float* w,
                                                                             float* m,
                                                                             float* v,
                                                                             uint64_t* step) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_base_train_output_adamw_state(w, m, v, step);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_update_base_train_output_adamw_state(spaceslug_tiny_forward_graph* g,
                                                                           float const* w,
                                                                           float const* m,
                                                                           float const* v,
                                                                           uint64_t step) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->update_base_train_output_adamw_state(w, m, v, step);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_train_output_adamw(spaceslug_tiny_forward_graph* g,
                                                         const uint32_t* t,
                                                         const uint32_t* y,
                                                         const uint32_t* m,
                                                         uint32_t rows,
                                                         float lr,
                                                         float b1,
                                                         float b2,
                                                         float eps,
                                                         float wd) {
    if (!g || !g->graph)
        return 1;
    return g->graph->train_output_adamw(t, y, m, rows, lr, b1, b2, eps, wd);
}
extern "C" int spaceslug_tiny_forward_import_base_train_qkv(spaceslug_tiny_forward_graph* g,
                                                            float const* q,
                                                            float const* k,
                                                            float const* v) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->import_base_train_qkv(q, k, v);
    } catch (...) {
        return 2;
    }
}
extern "C" int
spaceslug_tiny_forward_readback_base_train_qkv(spaceslug_tiny_forward_graph* g, float* q, float* k, float* v) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_base_train_qkv(q, k, v);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_readback_base_train_qkv_gradients(spaceslug_tiny_forward_graph* g,
                                                                        float* q,
                                                                        float* k,
                                                                        float* v) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_base_train_qkv_gradients(q, k, v);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_readback_base_train_qkv_adamw_state(spaceslug_tiny_forward_graph* g,
                                                                          float* q,
                                                                          float* k,
                                                                          float* v,
                                                                          float* qm,
                                                                          float* km,
                                                                          float* vm,
                                                                          float* qv,
                                                                          float* kv,
                                                                          float* vv,
                                                                          uint64_t* s) {
    if (!g || !g->graph)
        return 1;
    return g->graph->readback_base_train_qkv_adamw_state(q, k, v, qm, km, vm, qv, kv, vv, s);
}
extern "C" int spaceslug_tiny_forward_update_base_train_qkv_adamw_state(spaceslug_tiny_forward_graph* g,
                                                                        float const* q,
                                                                        float const* k,
                                                                        float const* v,
                                                                        float const* qm,
                                                                        float const* km,
                                                                        float const* vm,
                                                                        float const* qv,
                                                                        float const* kv,
                                                                        float const* vv,
                                                                        uint64_t s) {
    if (!g || !g->graph)
        return 1;
    return g->graph->update_base_train_qkv_adamw_state(q, k, v, qm, km, vm, qv, kv, vv, s);
}
extern "C" int spaceslug_tiny_forward_train_qkv_adamw_from_gradients(spaceslug_tiny_forward_graph* g,
                                                                     float lr,
                                                                     float b1,
                                                                     float b2,
                                                                     float eps,
                                                                     float wd) {
    if (!g || !g->graph)
        return 1;
    return g->graph->train_qkv_adamw_from_gradients(lr, b1, b2, eps, wd);
}
extern "C" int spaceslug_tiny_forward_train_positions_adamw(spaceslug_tiny_forward_graph* g, const uint32_t* t, const uint32_t* y, const uint32_t* m, uint32_t rows, float lr, float b1, float b2, float eps, float decay) { if (!g || !g->graph) return 1; try { return g->graph->train_positions_adamw(t, y, m, rows, lr, b1, b2, eps, decay); } catch (...) { return 2; } }
extern "C" int spaceslug_tiny_forward_train_bounded_full_graph_adamw(spaceslug_tiny_forward_graph* g, const uint32_t* t, const uint32_t* y, const uint32_t* m, uint32_t rows, float lr, float b1, float b2, float eps, float decay) { if (!g || !g->graph || !t || !y || !m || rows == 0 || rows > vulkan_runtime::tiny::Tcap || !std::isfinite(lr) || !std::isfinite(b1) || !std::isfinite(b2) || !std::isfinite(eps) || !std::isfinite(decay) || lr <= 0.0f || b1 < 0.0f || b1 >= 1.0f || b2 < 0.0f || b2 >= 1.0f || eps <= 0.0f || decay < 0.0f) return 1; try { return g->graph->train_positions_adamw(t, y, m, rows, lr, b1, b2, eps, decay); } catch (...) { return 2; } }
extern "C" int spaceslug_tiny_forward_train_qkv_sgd(spaceslug_tiny_forward_graph* g,
                                                    const uint32_t* t,
                                                    const uint32_t* y,
                                                    const uint32_t* m,
                                                    uint32_t rows,
                                                    float lr) {
    if (!g || !g->graph)
        return 1;
    return g->graph->train_qkv_sgd(t, y, m, rows, lr);
}
extern "C" int spaceslug_tiny_forward_train_lm_head_adamw(spaceslug_tiny_forward_graph* g,
                                                          const uint32_t* t,
                                                          const uint32_t* y,
                                                          const uint32_t* m,
                                                          uint32_t rows,
                                                          float lr,
                                                          float b1,
                                                          float b2,
                                                          float eps,
                                                          float wd) {
    if (!g || !g->graph)
        return 1;
    return g->graph->train_lm_head_adamw(t, y, m, rows, lr, b1, b2, eps, wd);
}
extern "C" void*
spaceslug_tiny_forward_create_dataset_batch(spaceslug_tiny_forward_graph* g, uint32_t windows, uint32_t window_length) {
    if (!g || !g->graph || windows == 0 || windows > 32 || window_length == 0 ||
        window_length > vulkan_runtime::tiny::Tcap)
        return nullptr;
    try {
        return g->graph->create_dataset_batch(windows, window_length).release();
    } catch (...) {
        return nullptr;
    }
}
extern "C" void spaceslug_tiny_forward_destroy_dataset_batch(void* batch) {
    delete static_cast<vulkan_runtime::dataset::BatchBuffer*>(batch);
}
extern "C" int spaceslug_tiny_forward_upload_dataset_batch(void* batch,
                                                           const uint32_t* tokens,
                                                           const uint32_t* targets,
                                                           const uint32_t* masks,
                                                           const uint32_t* controls) {
    if (!batch || !tokens || !targets || !masks || !controls)
        return 1;
    try {
        auto* b = static_cast<vulkan_runtime::dataset::BatchBuffer*>(batch);
        std::size_t n = std::size_t(b->window_count()) * b->window_tokens();
        b->upload(
            {tokens, tokens + n}, {targets, targets + n}, {masks, masks + n}, {controls, controls + b->window_count()});
        return 0;
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_train_dataset_batch(spaceslug_tiny_forward_graph* g,
                                                          void* batch,
                                                          float learning_rate,
                                                          float normalizer) {
    if (!g || !g->graph || !batch)
        return 1;
    try {
        return g->graph->train_dataset_batch(
            *static_cast<vulkan_runtime::dataset::BatchBuffer*>(batch), learning_rate, normalizer);
    } catch (...) {
        return 2;
    }
}
extern "C" const char* spaceslug_tiny_forward_dataset_training_capability(void) {
    return vulkan_runtime::tiny::ForwardResourceGraph::dataset_capability();
}
extern "C" int spaceslug_tiny_forward_train_dataset_batch_full(spaceslug_tiny_forward_graph* g,
                                                               void* batch,
                                                               float learning_rate,
                                                               float normalizer) {
    if (!g || !g->graph || !batch)
        return 1;
    try {
        return g->graph->train_dataset_batch_full(
            *static_cast<vulkan_runtime::dataset::BatchBuffer*>(batch), learning_rate, normalizer);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_token_step(spaceslug_tiny_forward_graph* g, uint32_t t, uint32_t p, float* out) {
    if (!g || !g->graph)
        return 1;
    try {
        g->graph->token_step(t, p, out);
        return 0;
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_token_step_training(spaceslug_tiny_forward_graph* g,
                                                          uint32_t token,
                                                          uint32_t position,
                                                          uint32_t target,
                                                          uint32_t mask,
                                                          float* loss,
                                                          float* dlogits,
                                                          float* dprojected) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->token_step_training(token, position, target, mask, loss, dlogits, dprojected);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_begin_lora_accumulation(spaceslug_tiny_forward_graph* g) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->begin_lora_accumulation();
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_token_step_training_backward_accumulate(spaceslug_tiny_forward_graph* g,
                                                                              uint32_t token,
                                                                              uint32_t position,
                                                                              uint32_t target,
                                                                              uint32_t mask,
                                                                              float const* doutput,
                                                                              float* loss,
                                                                              float* dlogits,
                                                                              float* dprojected,
                                                                              float* dquery,
                                                                              float* dkey,
                                                                              float* dvalue,
                                                                              float* dcontext,
                                                                              float* dstates) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->token_step_training_backward_accumulate(
            token, position, target, mask, doutput, loss, dlogits, dprojected, dquery, dkey, dvalue, dcontext, dstates);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_token_windows_training_backward_accumulate(spaceslug_tiny_forward_graph* g,
                                                                                 const uint32_t* tokens,
                                                                                 const uint32_t* targets,
                                                                                 const uint32_t* mask,
                                                                                 uint32_t windows,
                                                                                 uint32_t window_length,
                                                                                 float* losses) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->token_windows_training_backward_accumulate(
            tokens, targets, mask, windows, window_length, losses);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_begin_lora_adamw(spaceslug_tiny_forward_graph* g) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->begin_lora_adamw();
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_finalize_lora_adamw(spaceslug_tiny_forward_graph* g,
                                                          float lr,
                                                          float b1,
                                                          float b2,
                                                          float eps,
                                                          float wd,
                                                          float norm) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->finalize_lora_adamw(lr, b1, b2, eps, wd, norm);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_readback_lora_adamw_state(spaceslug_tiny_forward_graph* g,
                                                                float* a,
                                                                float* m,
                                                                float* v,
                                                                uint64_t* step) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_lora_adamw_state(a, m, v, step);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_update_lora_adamw_state(spaceslug_tiny_forward_graph* g,
                                                              float const* a,
                                                              float const* m,
                                                              float const* v,
                                                              uint64_t step) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->update_lora_adamw_state(a, m, v, step);
    } catch (...) {
        return 2;
    }
}
extern "C" int
spaceslug_tiny_forward_finalize_lora_sgd(spaceslug_tiny_forward_graph* g, float learning_rate, float normalizer) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->finalize_lora_sgd(learning_rate, normalizer);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_token_step_training_backward(spaceslug_tiny_forward_graph* g,
                                                                   uint32_t token,
                                                                   uint32_t position,
                                                                   uint32_t target,
                                                                   uint32_t mask,
                                                                   float const* doutput,
                                                                   float* loss,
                                                                   float* dlogits,
                                                                   float* dprojected,
                                                                   float* dquery,
                                                                   float* dkey,
                                                                   float* dvalue,
                                                                   float* dcontext,
                                                                   float* dstates) {
    if (!g || !g->graph)
        return 1;
    return g->graph->token_step_training_backward(
        token, position, target, mask, doutput, loss, dlogits, dprojected, dquery, dkey, dvalue, dcontext, dstates);
}
extern "C" int spaceslug_tiny_forward_readback_graph_dstate(spaceslug_tiny_forward_graph* g,
                                                              const uint32_t* tokens,
                                                              const uint32_t* targets,
                                                              const uint32_t* masks,
                                                              uint32_t rows,
                                                              float* dstates) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_graph_dstate(tokens, targets, masks, rows, dstates);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_readback_lora_adapters(spaceslug_tiny_forward_graph* g,
                                                             float* query_a,
                                                             float* query_b,
                                                             float* key_a,
                                                             float* key_b,
                                                             float* value_a,
                                                             float* value_b,
                                                             float* output_a,
                                                             float* output_b) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->readback_lora_adapters(query_a, query_b, key_a, key_b, value_a, value_b, output_a, output_b);
    } catch (...) {
        return 2;
    }
}
extern "C" int spaceslug_tiny_forward_update_lora_adapters(spaceslug_tiny_forward_graph* g,
                                                           float const* query_a,
                                                           float const* query_b,
                                                           float const* key_a,
                                                           float const* key_b,
                                                           float const* value_a,
                                                           float const* value_b,
                                                           float const* output_a,
                                                           float const* output_b) {
    if (!g || !g->graph)
        return 1;
    try {
        return g->graph->update_lora_adapters(query_a, query_b, key_a, key_b, value_a, value_b, output_a, output_b);
    } catch (...) {
        return 2;
    }
}
extern "C" void spaceslug_tiny_forward_destroy(spaceslug_tiny_forward_graph* g) {
    if (!g)
        return;
    g->graph.reset();
    vulkan_runtime::core::destroy_context(g->context);
    delete g;
}
