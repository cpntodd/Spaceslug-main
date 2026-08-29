// M6c-1: kvquantize — fp16 -> int8 + per-group-32 scales on the GPU vs
// cactus_quantize_kv_fp16_to_int8 (via the bridge), matching the scale formula
// (max_abs/127, floored 1e-10, round-half-away) EXACTLY. Also checks the
// int8*scale roundtrip reconstruction.

#include "m6c1_common.h"

#include <cmath>
#include <vector>

namespace {

struct PushConstants {
    std::uint32_t seq_len;
    std::uint32_t kv_heads;
    std::uint32_t head_dim;
    std::uint32_t group_size;
    std::uint32_t num_groups;
};

constexpr std::uint32_t kSeqLen = 2;
constexpr std::uint32_t kKvHeads = 1;
constexpr std::uint32_t kHeadDim = 64;
constexpr std::uint32_t kGroupSize = 32;

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-kvquantize");

        std::uint32_t const rows = kSeqLen * kKvHeads;
        std::uint32_t const numGroups = (kHeadDim + kGroupSize - 1) / kGroupSize;
        std::size_t const n = static_cast<std::size_t>(rows) * kHeadDim;

        std::vector<float> src_f = m6c1::gen_f32(n, 0x51510001u, -1.0f, 1.0f);
        std::vector<std::uint16_t> src_h(n);
        cactus_bridge_fp32_to_fp16(src_f.data(), src_h.data(), n);

        // CPU reference
        std::vector<std::int8_t> ref_q(n);
        std::vector<float> ref_scales(static_cast<std::size_t>(rows) * numGroups);
        cactus_bridge_quantize_kv_fp16_to_int8(src_h.data(), ref_q.data(), ref_scales.data(),
                                               kSeqLen, kKvHeads, kHeadDim, kGroupSize);

        std::vector<std::uint32_t> src_w = m6c1::widen16(src_h);
        vk::DeviceSize srcBytes = src_w.size() * sizeof(std::uint32_t);
        vk::DeviceSize qBytes = n * sizeof(std::int32_t);
        vk::DeviceSize scBytes = static_cast<std::size_t>(rows) * numGroups * sizeof(float);

        m6c1::Buffer dev_src = m6c1::create_buffer(
            ctx.allocator, srcBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_q = m6c1::create_buffer(
            ctx.allocator, qBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_sc = m6c1::create_buffer(
            ctx.allocator, scBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer staging = m6c1::create_buffer(
            ctx.allocator, srcBytes + qBytes + scBytes,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            std::memcpy(m, src_w.data(), srcBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, srcBytes);
        }

        PushConstants pc{kSeqLen, kKvHeads, kHeadDim, kGroupSize, numGroups};
        m6c1::run_kernel(ctx, "kvquantize.spv", {dev_src, dev_q, dev_sc}, staging,
                         {{0, srcBytes}}, {{srcBytes, qBytes}, {srcBytes + qBytes, scBytes}},
                         &pc, sizeof(pc), numGroups, rows, 1);

        std::vector<std::int32_t> out_q(n);
        std::vector<float> out_sc(static_cast<std::size_t>(rows) * numGroups);
        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, srcBytes, qBytes + scBytes);
            std::memcpy(out_q.data(), static_cast<char*>(m) + srcBytes, qBytes);
            std::memcpy(out_sc.data(), static_cast<char*>(m) + srcBytes + qBytes, scBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
        }

        bool ok = m6c1::compare_int8_exact("kvquantize_int8", out_q, ref_q);
        ok &= m6c1::compare_f32_exact("kvquantize_scales", out_sc, ref_scales);

        // Roundtrip reconstruction: dequantize int8 * scale vs original fp16.
        std::vector<std::uint16_t> deq_h(n);
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t g = 0; g < numGroups; ++g) {
                std::uint32_t gstart = g * kGroupSize;
                std::uint32_t gcount = std::min(kGroupSize, kHeadDim - gstart);
                for (std::uint32_t d = 0; d < gcount; ++d) {
                    float v = static_cast<float>(ref_q[static_cast<std::size_t>(row) * kHeadDim + gstart + d])
                              * ref_scales[static_cast<std::size_t>(row) * numGroups + g];
                    std::uint16_t hb;
                    cactus_bridge_fp32_to_fp16(&v, &hb, 1);
                    deq_h[static_cast<std::size_t>(row) * kHeadDim + gstart + d] = hb;
                }
            }
        }
        ok &= m6c1::compare_fp16("kvquantize_roundtrip", deq_h, src_h, 2e-2, 2e-2);

        m6c1::destroy_buffer(ctx.allocator, staging);
        m6c1::destroy_buffer(ctx.allocator, dev_sc);
        m6c1::destroy_buffer(ctx.allocator, dev_q);
        m6c1::destroy_buffer(ctx.allocator, dev_src);
        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
