// M6c-1: sgemm_f16 — fp16 GEMM (C = A x B^T with B PRE-TRANSPOSED) on the GPU
// vs cactus_matmul_f16 (via the bridge). 64x64 tile, fp32 accumulation, fp16
// storage. Layout mirrors cactus_matmul_f16: A row-major [M][K], B_transposed
// [N][K], C row-major [M][N].

#include "m6c1_common.h"

#include <vector>

namespace {

struct PushConstants {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

constexpr std::uint32_t kM = 64;
constexpr std::uint32_t kN = 64;
constexpr std::uint32_t kK = 128;

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-sgemm-f16");

        std::size_t const aN = static_cast<std::size_t>(kM) * kK;
        std::size_t const bN = static_cast<std::size_t>(kN) * kK; // pre-transposed [N][K]
        std::size_t const cN = static_cast<std::size_t>(kM) * kN;

        std::vector<float> a_f = m6c1::gen_f32(aN, 0x66650001u, -1.0f, 1.0f);
        std::vector<float> b_f = m6c1::gen_f32(bN, 0x66650002u, -1.0f, 1.0f);

        std::vector<std::uint16_t> a_h(aN), b_h(bN);
        cactus_bridge_fp32_to_fp16(a_f.data(), a_h.data(), aN);
        cactus_bridge_fp32_to_fp16(b_f.data(), b_h.data(), bN);

        std::vector<std::uint16_t> ref_h(cN);
        cactus_bridge_matmul_f16(a_h.data(), b_h.data(), ref_h.data(), kM, kK, kN);

        std::vector<std::uint32_t> a_w = m6c1::widen16(a_h);
        std::vector<std::uint32_t> b_w = m6c1::widen16(b_h);
        vk::DeviceSize aBytes = a_w.size() * sizeof(std::uint32_t);
        vk::DeviceSize bBytes = b_w.size() * sizeof(std::uint32_t);
        vk::DeviceSize cBytes = cN * sizeof(std::uint32_t);

        m6c1::Buffer dev_a = m6c1::create_buffer(
            ctx.allocator, aBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_b = m6c1::create_buffer(
            ctx.allocator, bBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer dev_c = m6c1::create_buffer(
            ctx.allocator, cBytes,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        m6c1::Buffer staging = m6c1::create_buffer(
            ctx.allocator, aBytes + bBytes + cBytes,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            std::memcpy(m, a_w.data(), aBytes);
            std::memcpy(static_cast<char*>(m) + aBytes, b_w.data(), bBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
            vmaFlushAllocation(ctx.allocator, staging.allocation, 0, aBytes + bBytes);
        }

        PushConstants pc{kM, kN, kK};
        m6c1::run_kernel(ctx, "sgemm_f16.spv", {dev_a, dev_b, dev_c}, staging,
                         {{0, aBytes}, {aBytes, bBytes}}, {{aBytes + bBytes, cBytes}},
                         &pc, sizeof(pc), kN / 64, kM / 64, 1);

        std::vector<std::uint32_t> out_w(cN);
        {
            void* m = nullptr;
            vmaMapMemory(ctx.allocator, staging.allocation, &m);
            vmaInvalidateAllocation(ctx.allocator, staging.allocation, aBytes + bBytes, cBytes);
            std::memcpy(out_w.data(), static_cast<char*>(m) + aBytes + bBytes, cBytes);
            vmaUnmapMemory(ctx.allocator, staging.allocation);
        }
        std::vector<std::uint16_t> out_h = m6c1::narrow16(out_w);

        bool ok = m6c1::compare_fp16("sgemm_f16", out_h, ref_h, 1e-3, 1e-2);

        m6c1::destroy_buffer(ctx.allocator, staging);
        m6c1::destroy_buffer(ctx.allocator, dev_c);
        m6c1::destroy_buffer(ctx.allocator, dev_b);
        m6c1::destroy_buffer(ctx.allocator, dev_a);
        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
