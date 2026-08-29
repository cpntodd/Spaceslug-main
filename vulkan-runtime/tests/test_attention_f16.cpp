// M6c-1: attention_f16 — masked/causal GQA attention (fp16 q/k/v/o) on the GPU
// vs cactus_attention_f16 (via the bridge). Mirrors attention_naive semantics:
// multiplicative mask (0 = masked), causal, GQA.

#include "m6c1_common.h"

#include <cmath>
#include <vector>

namespace {

struct PushConstants {
    std::uint32_t batch;
    std::uint32_t seq_len;
    std::uint32_t kv_seq_len;
    std::uint32_t num_q_heads;
    std::uint32_t num_kv_heads;
    std::uint32_t head_dim;
    std::uint32_t v_head_dim;
    std::uint32_t position_offset;
    std::uint32_t window_size;
    std::uint32_t is_causal;
    std::uint32_t has_mask;
    std::uint32_t mask_is_additive;
    std::uint32_t mask_per_head;
    float scale;
    float logit_cap;
};

constexpr std::uint32_t kBatch = 1;
constexpr std::uint32_t kSeqLen = 16;
constexpr std::uint32_t kKvSeqLen = 16;
constexpr std::uint32_t kNumQHeads = 2;
constexpr std::uint32_t kNumKvHeads = 1;
constexpr std::uint32_t kHeadDim = 32;
constexpr std::uint32_t kVHeadDim = 32;

std::vector<std::uint16_t> to_fp16(std::vector<float> const& f) {
    std::vector<std::uint16_t> h(f.size());
    cactus_bridge_fp32_to_fp16(f.data(), h.data(), f.size());
    return h;
}

bool run_case(vulkan_runtime::core::VulkanContext& ctx, char const* name,
              std::vector<std::uint16_t> const& Q, std::vector<std::uint16_t> const& K,
              std::vector<std::uint16_t> const& V, std::vector<std::uint16_t> const& mask,
              std::uint32_t is_causal, std::uint32_t has_mask) {
    std::size_t const oN = static_cast<std::size_t>(kBatch) * kSeqLen * kNumQHeads * kVHeadDim;

    std::vector<std::uint16_t> ref_h(oN);
    cactus_bridge_attention_f16(
        Q.data(), K.data(), V.data(), ref_h.data(), kBatch, kSeqLen, kKvSeqLen,
        kNumQHeads, kNumKvHeads, kHeadDim,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)),
        has_mask ? mask.data() : nullptr, 0, 0, is_causal != 0, false, false, kVHeadDim, 0.0f);

    std::vector<std::uint32_t> q_w = m6c1::widen16(Q);
    std::vector<std::uint32_t> k_w = m6c1::widen16(K);
    std::vector<std::uint32_t> v_w = m6c1::widen16(V);
    std::vector<std::uint32_t> mask_w = m6c1::widen16(mask);
    vk::DeviceSize qBytes = q_w.size() * sizeof(std::uint32_t);
    vk::DeviceSize kBytes = k_w.size() * sizeof(std::uint32_t);
    vk::DeviceSize vBytes = v_w.size() * sizeof(std::uint32_t);
    vk::DeviceSize maskBytes = mask_w.size() * sizeof(std::uint32_t);
    vk::DeviceSize oBytes = oN * sizeof(std::uint32_t);

    m6c1::Buffer dev_q = m6c1::create_buffer(
        ctx.allocator, qBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m6c1::Buffer dev_k = m6c1::create_buffer(
        ctx.allocator, kBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m6c1::Buffer dev_v = m6c1::create_buffer(
        ctx.allocator, vBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m6c1::Buffer dev_o = m6c1::create_buffer(
        ctx.allocator, oBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m6c1::Buffer dev_mask = m6c1::create_buffer(
        ctx.allocator, maskBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    vk::DeviceSize const inSize = qBytes + kBytes + vBytes + maskBytes;
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
        std::memcpy(p + qBytes, k_w.data(), kBytes);
        std::memcpy(p + qBytes + kBytes, v_w.data(), vBytes);
        std::memcpy(p + qBytes + kBytes + vBytes, mask_w.data(), maskBytes);
        vmaUnmapMemory(ctx.allocator, staging.allocation);
        vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inSize);
    }

    PushConstants pc{kBatch, kSeqLen, kKvSeqLen, kNumQHeads, kNumKvHeads, kHeadDim,
                     kVHeadDim, 0, 0, is_causal, has_mask, 0, 0,
                     1.0f / std::sqrt(static_cast<float>(kHeadDim)), 0.0f};
    std::uint32_t const total = kBatch * kNumQHeads * kSeqLen;
    m6c1::run_kernel(ctx, "attention_f16.spv",
                     {dev_q, dev_k, dev_v, dev_mask, dev_o}, staging,
                     {{0, qBytes}, {qBytes, kBytes}, {qBytes + kBytes, vBytes},
                      {qBytes + kBytes + vBytes, maskBytes}},
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

    bool ok = m6c1::compare_fp16(name, out_h, ref_h, 1e-2, 2e-2);

    m6c1::destroy_buffer(ctx.allocator, staging);
    m6c1::destroy_buffer(ctx.allocator, dev_mask);
    m6c1::destroy_buffer(ctx.allocator, dev_o);
    m6c1::destroy_buffer(ctx.allocator, dev_v);
    m6c1::destroy_buffer(ctx.allocator, dev_k);
    m6c1::destroy_buffer(ctx.allocator, dev_q);
    return ok;
}

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-attention-f16");
        bool ok = true;

        std::size_t const qN = static_cast<std::size_t>(kBatch) * kSeqLen * kNumQHeads * kHeadDim;
        std::size_t const kvN = static_cast<std::size_t>(kBatch) * kKvSeqLen * kNumKvHeads * kHeadDim;
        std::size_t const vvN = static_cast<std::size_t>(kBatch) * kKvSeqLen * kNumKvHeads * kVHeadDim;

        std::vector<std::uint16_t> Q = to_fp16(m6c1::gen_f32(qN, 0xA77F0001u, -1.0f, 1.0f));
        std::vector<std::uint16_t> K = to_fp16(m6c1::gen_f32(kvN, 0xA77F0002u, -1.0f, 1.0f));
        std::vector<std::uint16_t> V = to_fp16(m6c1::gen_f32(vvN, 0xA77F0003u, -1.0f, 1.0f));

        // causal
        std::vector<std::uint16_t> dummy_mask(1, 0);
        ok &= run_case(ctx, "attention_f16_causal", Q, K, V, dummy_mask, 1, 0);

        // explicit multiplicative mask (0/1)
        std::vector<float> mask_f = m6c1::gen_f32(static_cast<std::size_t>(kSeqLen) * kKvSeqLen,
                                                  0xA77F0004u, 0.0f, 1.0f);
        std::vector<std::uint16_t> mask(mask_f.size());
        for (std::size_t i = 0; i < mask_f.size(); ++i) {
            mask[i] = (mask_f[i] < 0.5f) ? 0 : 1;
        }
        ok &= run_case(ctx, "attention_f16_masked", Q, K, V, mask, 0, 1);

        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
