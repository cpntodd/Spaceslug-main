// M6c-1: embed_cq4 — CQ4 orthogonal embedding row dequant + gather on the GPU
// vs cactus_quant_dequantize_orthogonal_embedding_row (via the bridge).
//   out[m][j] = (sum_i codebook[idx(row,i)] * rotation[j][i]) * norm[row] * recip[j]

#include "m6c1_common.h"

#include <vector>

namespace {

struct PushConstants {
    std::uint32_t bits;
    std::uint32_t K;
    std::uint32_t num_rows;
    std::uint32_t M;
    std::uint32_t has_recip;
};

constexpr std::uint32_t kBits = 4;
constexpr std::uint32_t kK = 512;
constexpr std::uint32_t kNumRows = 8;
constexpr std::uint32_t kM = 4;

std::vector<std::uint16_t> to_fp16(std::vector<float> const& f) {
    std::vector<std::uint16_t> h(f.size());
    cactus_bridge_fp32_to_fp16(f.data(), h.data(), f.size());
    return h;
}

std::vector<std::uint8_t> pack_4bit_lsb(std::vector<std::uint32_t> const& idx) {
    std::vector<std::uint8_t> bytes((idx.size() + 1) / 2, 0);
    for (std::size_t k = 0; k < idx.size(); ++k) {
        if (k & 1u) bytes[k / 2] |= static_cast<std::uint8_t>((idx[k] & 0xFu) << 4);
        else bytes[k / 2] |= static_cast<std::uint8_t>(idx[k] & 0xFu);
    }
    return bytes;
}

std::vector<std::uint32_t> widen_u8(std::vector<std::uint8_t> const& b) {
    std::vector<std::uint32_t> w(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) w[i] = b[i];
    return w;
}

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-embed-cq4");

        std::uint32_t const pgb = (kK * kBits + 7) / 8;

        std::vector<std::uint16_t> codebook = to_fp16(m6c1::gen_f32(16, 0xE00C0001u, -1.0f, 1.0f));
        std::vector<std::uint16_t> norms = to_fp16(m6c1::gen_f32(kNumRows, 0xE00C0002u, 0.5f, 1.5f));
        std::vector<std::uint16_t> recip = to_fp16(m6c1::gen_f32(kK, 0xE00C0003u, 0.5f, 1.5f));
        std::vector<std::uint16_t> rotation = to_fp16(m6c1::gen_f32(
            static_cast<std::size_t>(kK) * kK, 0xE00C0004u, -0.5f, 0.5f));

        // random packed indices per row, 4-bit LSB
        std::vector<std::uint8_t> packed(static_cast<std::size_t>(kNumRows) * pgb, 0);
        {
            std::uint32_t s = 0xE00C0005u;
            for (std::uint32_t r = 0; r < kNumRows; ++r) {
                std::vector<std::uint32_t> idx(kK);
                for (std::uint32_t i = 0; i < kK; ++i) {
                    s = s * 1664525u + 1013904223u;
                    idx[i] = (s >> 8) % 16u;
                }
                std::vector<std::uint8_t> pb = pack_4bit_lsb(idx);
                std::memcpy(packed.data() + static_cast<std::size_t>(r) * pgb, pb.data(), pgb);
            }
        }

        std::vector<std::uint32_t> ids = {3, 0, 5, 1};

        // CPU reference: dequantize each gathered row.
        std::vector<std::uint16_t> ref_h(static_cast<std::size_t>(kM) * kK);
        for (std::uint32_t m = 0; m < kM; ++m) {
            cactus_bridge_dequantize_orthogonal_embedding_row(
                kBits, kK, ids[m], packed.data(), codebook.data(), norms.data(), recip.data(),
                rotation.data(), 0, ref_h.data() + static_cast<std::size_t>(m) * kK);
        }

        // GPU buffers
        std::vector<std::uint32_t> packed_w = widen_u8(packed);
        std::vector<std::uint32_t> codebook_w = m6c1::widen16(codebook);
        std::vector<std::uint32_t> norms_w = m6c1::widen16(norms);
        std::vector<std::uint32_t> recip_w = m6c1::widen16(recip);
        std::vector<std::uint32_t> rot_w = m6c1::widen16(rotation);
        vk::DeviceSize packedBytes = packed_w.size() * 4;
        vk::DeviceSize codebookBytes = codebook_w.size() * 4;
        vk::DeviceSize normsBytes = norms_w.size() * 4;
        vk::DeviceSize recipBytes = recip_w.size() * 4;
        vk::DeviceSize rotBytes = rot_w.size() * 4;
        vk::DeviceSize idsBytes = ids.size() * 4;
        vk::DeviceSize oBytes = static_cast<std::size_t>(kM) * kK * 4;

        auto mk = [&](vk::DeviceSize s, vk::BufferUsageFlags u, VmaMemoryUsage mu) {
            return m6c1::create_buffer(ctx.allocator, s, u, mu);
        };
        std::vector<m6c1::Buffer> dev(7);
        dev[0] = mk(packedBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        dev[1] = mk(codebookBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        dev[2] = mk(normsBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        dev[3] = mk(recipBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        dev[4] = mk(rotBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        dev[5] = mk(idsBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        dev[6] = mk(oBytes, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

        vk::DeviceSize const inSize = packedBytes + codebookBytes + normsBytes + recipBytes + rotBytes + idsBytes;
        m6c1::Buffer staging = m6c1::create_buffer(
            ctx.allocator, inSize + oBytes,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            char* p = static_cast<char*>(m);
            std::memcpy(p, packed_w.data(), packedBytes);
            std::memcpy(p + packedBytes, codebook_w.data(), codebookBytes);
            std::memcpy(p + packedBytes + codebookBytes, norms_w.data(), normsBytes);
            std::memcpy(p + packedBytes + codebookBytes + normsBytes, recip_w.data(), recipBytes);
            std::memcpy(p + packedBytes + codebookBytes + normsBytes + recipBytes, rot_w.data(), rotBytes);
            std::memcpy(p + packedBytes + codebookBytes + normsBytes + recipBytes + rotBytes, ids.data(), idsBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inSize);
        }

        PushConstants pc{kBits, kK, kNumRows, kM, 1};
        std::uint32_t const total = kM * kK;
        m6c1::run_kernel(ctx, "embed_cq4.spv", dev, staging,
                         {{0, packedBytes}, {packedBytes, codebookBytes},
                          {packedBytes + codebookBytes, normsBytes},
                          {packedBytes + codebookBytes + normsBytes, recipBytes},
                          {packedBytes + codebookBytes + normsBytes + recipBytes, rotBytes},
                          {packedBytes + codebookBytes + normsBytes + recipBytes + rotBytes, idsBytes}},
                         {{inSize, oBytes}},
                         &pc, sizeof(pc), (total + 63u) / 64u, 1, 1);

        std::vector<std::uint32_t> out_w(static_cast<std::size_t>(kM) * kK);
        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, inSize, oBytes);
            std::memcpy(out_w.data(), static_cast<char*>(m) + inSize, oBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
        }
        std::vector<std::uint16_t> out_h = m6c1::narrow16(out_w);

        bool ok = m6c1::compare_fp16("embed_cq4", out_h, ref_h, 2e-2, 1e-2);

        m6c1::destroy_buffer(ctx.allocator, staging);
        for (auto& b : dev) m6c1::destroy_buffer(ctx.allocator, b);
        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
