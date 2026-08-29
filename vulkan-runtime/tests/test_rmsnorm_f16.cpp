// M6c-1: rmsnorm_f16 — fp16 RMSNorm (no bias) on the GPU vs
// cactus_rms_norm_f16 (via the bridge). fp16 storage, fp32 accumulation.

#include "m6c1_common.h"

#include <vector>

namespace {

struct PushConstants {
    std::uint32_t batch;
    std::uint32_t dims;
    float eps;
};

constexpr std::uint32_t kBatch = 4;
constexpr std::uint32_t kDims = 128;
constexpr float kEps = 1e-5f;

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-rmsnorm-f16");

        std::size_t const n = static_cast<std::size_t>(kBatch) * kDims;
        std::vector<float> x_f = m6c1::gen_f32(n, 0x4E4D0001u, -1.0f, 1.0f);
        std::vector<float> w_f = m6c1::gen_f32(kDims, 0x4E4D0002u, 0.5f, 1.5f);

        std::vector<std::uint16_t> x_h(n), w_h(kDims);
        cactus_bridge_fp32_to_fp16(x_f.data(), x_h.data(), n);
        cactus_bridge_fp32_to_fp16(w_f.data(), w_h.data(), kDims);

        std::vector<std::uint16_t> ref_h(n);
        cactus_bridge_rms_norm_f16(x_h.data(), w_h.data(), ref_h.data(), kBatch, kDims, kEps);

        std::vector<std::uint32_t> x_w = m6c1::widen16(x_h);
        std::vector<std::uint32_t> w_w = m6c1::widen16(w_h);
        vk::DeviceSize xBytes = x_w.size() * sizeof(std::uint32_t);
        vk::DeviceSize wBytes = w_w.size() * sizeof(std::uint32_t);
        vk::DeviceSize yBytes = n * sizeof(std::uint32_t);

        m6c1::Buffer dev_x = m6c1::create_buffer(
            ctx.allocator, xBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_w = m6c1::create_buffer(
            ctx.allocator, wBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_y = m6c1::create_buffer(
            ctx.allocator, yBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer staging = m6c1::create_buffer(
            ctx.allocator, xBytes + wBytes + yBytes,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            std::memcpy(m, x_w.data(), xBytes);
            std::memcpy(static_cast<char*>(m) + xBytes, w_w.data(), wBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, xBytes + wBytes);
        }

        PushConstants pc{kBatch, kDims, kEps};
        m6c1::run_kernel(ctx, "rmsnorm_f16.spv", {dev_x, dev_w, dev_y}, staging,
                         {{0, xBytes}, {xBytes, wBytes}}, {{xBytes + wBytes, yBytes}},
                         &pc, sizeof(pc), kBatch, 1, 1);

        std::vector<std::uint32_t> out_w(n);
        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, xBytes + wBytes, yBytes);
            std::memcpy(out_w.data(), static_cast<char*>(m) + xBytes + wBytes, yBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
        }
        std::vector<std::uint16_t> out_h = m6c1::narrow16(out_w);

        bool ok = m6c1::compare_fp16("rmsnorm_f16", out_h, ref_h, 1e-3, 1e-2);

        m6c1::destroy_buffer(ctx.allocator, staging);
        m6c1::destroy_buffer(ctx.allocator, dev_y);
        m6c1::destroy_buffer(ctx.allocator, dev_w);
        m6c1::destroy_buffer(ctx.allocator, dev_x);
        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
