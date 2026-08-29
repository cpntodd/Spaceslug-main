// M6c-1: attention_i8kv — hybrid int8-cache + fp16-new attention on the GPU vs
// cactus_attention_hybrid_int8_fp16 (via the bridge). Mirrors the reference
// dequant (int8 * per-group-32 scale for cached, fp16 for new) + causal
// attention with position_offset == cache_len.

#include "m6c1_common.h"

#include <cmath>
#include <vector>

namespace {

struct PushConstants {
    std::uint32_t batch;
    std::uint32_t seq_len;
    std::uint32_t cache_len;
    std::uint32_t new_len;
    std::uint32_t num_q_heads;
    std::uint32_t num_kv_heads;
    std::uint32_t head_dim;
    std::uint32_t v_head_dim;
    std::uint32_t position_offset;
    std::uint32_t is_causal;
    std::uint32_t window_size;
    std::uint32_t group_size;
    std::uint32_t num_groups_k;
    std::uint32_t num_groups_v;
    float scale;
};

constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kSeqLen = 2;
constexpr std::uint32_t kCacheLen = 4;
constexpr std::uint32_t kNewLen = 2;
constexpr std::uint32_t kNumQHeads = 2;
constexpr std::uint32_t kNumKvHeads = 1;
constexpr std::uint32_t kHeadDim = 32;
constexpr std::uint32_t kVHeadDim = 32;
constexpr std::uint32_t kGroupSize = 32;

std::vector<std::uint16_t> to_fp16(std::vector<float> const& f) {
    std::vector<std::uint16_t> h(f.size());
    cactus_bridge_fp32_to_fp16(f.data(), h.data(), f.size());
    return h;
}

std::vector<std::int32_t> widen_i8(std::vector<std::int8_t> const& q) {
    std::vector<std::int32_t> w(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) w[i] = q[i];
    return w;
}

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-attention-i8kv");

        std::uint32_t const numGroupsK = (kHeadDim + kGroupSize - 1) / kGroupSize;
        std::uint32_t const numGroupsV = (kVHeadDim + kGroupSize - 1) / kGroupSize;

        std::size_t const qN = static_cast<std::size_t>(kBatch) * kSeqLen * kNumQHeads * kHeadDim;
        std::size_t const cacheN = static_cast<std::size_t>(kBatch) * kCacheLen * kNumKvHeads;
        std::size_t const newN = static_cast<std::size_t>(kBatch) * kNewLen * kNumKvHeads;

        std::vector<std::uint16_t> Q = to_fp16(m6c1::gen_f32(qN, 0xB77F0001u, -1.0f, 1.0f));
        std::vector<std::uint16_t> Kcache = to_fp16(m6c1::gen_f32(cacheN * kHeadDim, 0xB77F0002u, -1.0f, 1.0f));
        std::vector<std::uint16_t> Vcache = to_fp16(m6c1::gen_f32(cacheN * kVHeadDim, 0xB77F0003u, -1.0f, 1.0f));
        std::vector<std::uint16_t> Knew = to_fp16(m6c1::gen_f32(newN * kHeadDim, 0xB77F0004u, -1.0f, 1.0f));
        std::vector<std::uint16_t> Vnew = to_fp16(m6c1::gen_f32(newN * kVHeadDim, 0xB77F0005u, -1.0f, 1.0f));

        // quantize cached K/V
        std::vector<std::int8_t> Kq(cacheN * kHeadDim), Vq(cacheN * kVHeadDim);
        std::vector<float> Ksc(cacheN * numGroupsK), Vsc(cacheN * numGroupsV);
        cactus_bridge_quantize_kv_fp16_to_int8(Kcache.data(), Kq.data(), Ksc.data(),
                                               kCacheLen * kNumKvHeads, 1, kHeadDim, kGroupSize);
        cactus_bridge_quantize_kv_fp16_to_int8(Vcache.data(), Vq.data(), Vsc.data(),
                                               kCacheLen * kNumKvHeads, 1, kVHeadDim, kGroupSize);

        std::size_t const oN = static_cast<std::size_t>(kBatch) * kSeqLen * kNumQHeads * kVHeadDim;
        std::vector<std::uint16_t> ref_h(oN);
        cactus_bridge_attention_hybrid_int8_fp16(
            Q.data(), Kq.data(), Vq.data(), Ksc.data(), Vsc.data(), Knew.data(), Vnew.data(),
            ref_h.data(), kBatch, kSeqLen, kCacheLen, kNewLen, kNumQHeads, kNumKvHeads,
            kHeadDim, 1.0f / std::sqrt(static_cast<float>(kHeadDim)),
            kCacheLen /* position_offset */, true /* is_causal */, 0, kGroupSize, kVHeadDim);

        std::vector<std::uint32_t> q_w = m6c1::widen16(Q);
        std::vector<std::int32_t> kq_w = widen_i8(Kq);
        std::vector<std::int32_t> vq_w = widen_i8(Vq);
        std::vector<std::uint32_t> knew_w = m6c1::widen16(Knew);
        std::vector<std::uint32_t> vnew_w = m6c1::widen16(Vnew);
        vk::DeviceSize qBytes = q_w.size() * 4;
        vk::DeviceSize kqBytes = kq_w.size() * 4;
        vk::DeviceSize vqBytes = vq_w.size() * 4;
        vk::DeviceSize kscBytes = Ksc.size() * sizeof(float);
        vk::DeviceSize vscBytes = Vsc.size() * sizeof(float);
        vk::DeviceSize knewBytes = knew_w.size() * 4;
        vk::DeviceSize vnewBytes = vnew_w.size() * 4;
        vk::DeviceSize oBytes = oN * 4;

        auto mk = [&](vk::DeviceSize s, vk::BufferUsageFlags u, VmaMemoryUsage mu) {
            return m6c1::create_buffer(ctx.allocator, s, u, mu);
        };
        m6c1::Buffer dev_q = mk(qBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_kq = mk(kqBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_vq = mk(vqBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_ksc = mk(kscBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_vsc = mk(vscBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_knew = mk(knewBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_vnew = mk(vnewBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_o = mk(oBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize const inSize = qBytes + kqBytes + vqBytes + kscBytes + vscBytes + knewBytes + vnewBytes;
        m6c1::Buffer staging = m6c1::create_buffer(
            ctx.allocator, inSize + oBytes,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            char* p = static_cast<char*>(m);
            std::memcpy(p, q_w.data(), qBytes);
            std::memcpy(p + qBytes, kq_w.data(), kqBytes);
            std::memcpy(p + qBytes + kqBytes, vq_w.data(), vqBytes);
            std::memcpy(p + qBytes + kqBytes + vqBytes, Ksc.data(), kscBytes);
            std::memcpy(p + qBytes + kqBytes + vqBytes + kscBytes, Vsc.data(), vscBytes);
            std::memcpy(p + qBytes + kqBytes + vqBytes + kscBytes + vscBytes, knew_w.data(), knewBytes);
            std::memcpy(p + qBytes + kqBytes + vqBytes + kscBytes + vscBytes + knewBytes, vnew_w.data(), vnewBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inSize);
        }

        PushConstants pc{kBatch, kSeqLen, kCacheLen, kNewLen, kNumQHeads, kNumKvHeads,
                         kHeadDim, kVHeadDim, kCacheLen, 1, 0, kGroupSize, numGroupsK, numGroupsV,
                         1.0f / std::sqrt(static_cast<float>(kHeadDim))};
        std::uint32_t const total = kBatch * kNumQHeads * kSeqLen;
        m6c1::run_kernel(ctx, "attention_i8kv.spv",
                         {dev_q, dev_kq, dev_vq, dev_ksc, dev_vsc, dev_knew, dev_vnew, dev_o},
                         staging,
                         {{0, qBytes}, {qBytes, kqBytes}, {qBytes + kqBytes, vqBytes},
                          {qBytes + kqBytes + vqBytes, kscBytes},
                          {qBytes + kqBytes + vqBytes + kscBytes, vscBytes},
                          {qBytes + kqBytes + vqBytes + kscBytes + vscBytes, knewBytes},
                          {qBytes + kqBytes + vqBytes + kscBytes + vscBytes + knewBytes, vnewBytes}},
                         {{inSize, oBytes}},
                         &pc, sizeof(pc), (total + 63u) / 64u, 1, 1);

        std::vector<std::uint32_t> out_w(oN);
        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, inSize, oBytes);
            std::memcpy(out_w.data(), static_cast<char*>(m) + inSize, oBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
        }
        std::vector<std::uint16_t> out_h = m6c1::narrow16(out_w);

        bool ok = m6c1::compare_fp16("attention_i8kv", out_h, ref_h, 1e-2, 2e-2);

        m6c1::destroy_buffer(ctx.allocator, staging);
        m6c1::destroy_buffer(ctx.allocator, dev_o);
        m6c1::destroy_buffer(ctx.allocator, dev_vnew);
        m6c1::destroy_buffer(ctx.allocator, dev_knew);
        m6c1::destroy_buffer(ctx.allocator, dev_vsc);
        m6c1::destroy_buffer(ctx.allocator, dev_ksc);
        m6c1::destroy_buffer(ctx.allocator, dev_vq);
        m6c1::destroy_buffer(ctx.allocator, dev_kq);
        m6c1::destroy_buffer(ctx.allocator, dev_q);
        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
