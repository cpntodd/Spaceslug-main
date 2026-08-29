// M6c-2: sgemm_f16_am — fp16 GEMM with ARBITRARY M and N (K still %32==0) on
// the GPU vs cactus_matmul_f16 (via the bridge). Bounds-checked 64x64 tile.
// Covers the cactus decode path (M==1) plus partial edge tiles (M=3, 130) and
// the full-tile fast path (M=64). Layout mirrors cactus_matmul_f16: A row-major
// [M][K], B_transposed [N][K], C row-major [M][N].

#include "m6c1_common.h"

#include <vector>

namespace {

struct PushConstants {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

std::uint32_t ceil_div(std::uint32_t a, std::uint32_t b) { return (a + b - 1u) / b; }

bool run_case(vulkan_runtime::core::VulkanContext& ctx, std::uint32_t M, std::uint32_t N,
              std::uint32_t K, std::uint32_t seed) {
    std::size_t const aN = static_cast<std::size_t>(M) * K;
    std::size_t const bN = static_cast<std::size_t>(N) * K; // pre-transposed [N][K]
    std::size_t const cN = static_cast<std::size_t>(M) * N;

    std::vector<float> a_f = m6c1::gen_f32(aN, seed, -1.0f, 1.0f);
    std::vector<float> b_f = m6c1::gen_f32(bN, seed + 1u, -1.0f, 1.0f);

    std::vector<std::uint16_t> a_h(aN), b_h(bN);
    cactus_bridge_fp32_to_fp16(a_f.data(), a_h.data(), aN);
    cactus_bridge_fp32_to_fp16(b_f.data(), b_h.data(), bN);

    std::vector<std::uint16_t> ref_h(cN);
    cactus_bridge_matmul_f16(a_h.data(), b_h.data(), ref_h.data(), M, K, N);

    // M7a: A, B, and C are packed fp16 (2 per uint32). C has ceil(M*N/2) words.
    std::vector<std::uint32_t> a_w = m6c1::pack16(a_h);
    std::vector<std::uint32_t> b_w = m6c1::pack16(b_h);
    vk::DeviceSize aBytes = a_w.size() * sizeof(std::uint32_t);
    vk::DeviceSize bBytes = b_w.size() * sizeof(std::uint32_t);
    vk::DeviceSize cBytes = static_cast<vk::DeviceSize>((cN + 1u) / 2u) * sizeof(std::uint32_t);

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

    PushConstants pc{M, N, K};
    m6c1::run_kernel(ctx, "sgemm_f16_am.spv", {dev_a, dev_b, dev_c}, staging,
                     {{0, aBytes}, {aBytes, bBytes}}, {{aBytes + bBytes, cBytes}},
                     &pc, sizeof(pc), ceil_div(N, 64u), ceil_div(M, 64u), 1);

    std::vector<std::uint32_t> out_w((cN + 1u) / 2u);
    {
        void* m = nullptr;
        vmaMapMemory(ctx.allocator, staging.allocation, &m);
        vmaInvalidateAllocation(ctx.allocator, staging.allocation, aBytes + bBytes, cBytes);
        std::memcpy(out_w.data(), static_cast<char*>(m) + aBytes + bBytes, cBytes);
        vmaUnmapMemory(ctx.allocator, staging.allocation);
    }
    std::vector<std::uint16_t> out_h = m6c1::unpack16(out_w, cN);

    char name[64];
    std::snprintf(name, sizeof(name), "sgemm_f16_am M=%u N=%u K=%u", M, N, K);
    bool ok = m6c1::compare_fp16(name, out_h, ref_h, 1e-3, 1e-2);

    m6c1::destroy_buffer(ctx.allocator, staging);
    m6c1::destroy_buffer(ctx.allocator, dev_c);
    m6c1::destroy_buffer(ctx.allocator, dev_b);
    m6c1::destroy_buffer(ctx.allocator, dev_a);
    return ok;
}

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-sgemm-f16-am");

        bool ok = true;
        // decode (M=1), partial edge tiles (M=3, 130), full tile (M=64), and
        // non-tile N (130).
        ok &= run_case(ctx, 1, 64, 128, 0xA6A10001u);
        ok &= run_case(ctx, 3, 64, 128, 0xA6A10002u);
        ok &= run_case(ctx, 64, 64, 128, 0xA6A10003u);
        ok &= run_case(ctx, 130, 64, 128, 0xA6A10004u);
        ok &= run_case(ctx, 3, 130, 128, 0xA6A10005u);
        ok &= run_case(ctx, 1, 130, 128, 0xA6A10006u);
        ok &= run_case(ctx, 5, 256, 512, 0xA6A10007u);
        // Multi-tile N / partial-tile regression locks (the C-store tile-N word
        // offset): N=128 (2 full tiles), N=192 (3 full tiles), N=260 (partial
        // last tile), M=66 (partial M alongside multi-tile N).
        ok &= run_case(ctx, 1, 128, 128, 0xA6A10008u);
        ok &= run_case(ctx, 64, 192, 128, 0xA6A10009u);
        ok &= run_case(ctx, 3, 260, 128, 0xA6A1000Au);
        ok &= run_case(ctx, 66, 130, 128, 0xA6A1000Bu);

        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
