// M6c-2: fp16pack — packed <-> word layout bridging for fp16 (u16) and int8.
// Roundtrip-exact: u16 -> u32 -> u16 and i8 -> i32 -> i8 are bit-exact
// (clamp only matters for out-of-range int32 which the test also exercises).

#include "m6c1_common.h"

#include <vector>

namespace {

struct PushConstants {
    std::uint32_t mode;  // 0 u16->u32, 1 u32->u16, 2 i8->i32, 3 i32->i8
    std::uint32_t count; // elements
};

std::uint32_t ceil_div(std::uint32_t a, std::uint32_t b) { return (a + b - 1u) / b; }

// Run one pack/unpack pass: inputs (uint32 words) -> outputs (uint32 words).
std::vector<std::uint32_t> run(vulkan_runtime::core::VulkanContext& ctx, std::uint32_t mode,
                               std::uint32_t count, std::vector<std::uint32_t> const& input,
                               std::uint32_t outElems) {
    vk::DeviceSize inBytes = input.size() * sizeof(std::uint32_t);
    vk::DeviceSize outBytes = outElems * sizeof(std::uint32_t);

    m6c1::Buffer dev_in = m6c1::create_buffer(
        ctx.allocator, inBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m6c1::Buffer dev_out = m6c1::create_buffer(
        ctx.allocator, outBytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m6c1::Buffer staging = m6c1::create_buffer(
        ctx.allocator, inBytes + outBytes,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    {
        void* m = nullptr;
        vmaMapMemory(ctx.allocator, staging.allocation, &m);
        std::memcpy(m, input.data(), inBytes);
        vmaUnmapMemory(ctx.allocator, staging.allocation);
        vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inBytes);
    }

    PushConstants pc{mode, count};
    std::uint32_t gx = 0;
    if (mode == 0u || mode == 2u) {
        gx = ceil_div(count, 256u);       // one thread per output element
    } else if (mode == 1u) {
        gx = ceil_div((count + 1u) / 2u, 256u); // one thread per output u16-pair word
    } else {
        gx = ceil_div((count + 3u) / 4u, 256u); // one thread per output i8-quad word
    }
    if (gx == 0) gx = 1;
    m6c1::run_kernel(ctx, "fp16pack.spv", {dev_in, dev_out}, staging,
                     {{0, inBytes}}, {{inBytes, outBytes}}, &pc, sizeof(pc), gx, 1, 1);

    std::vector<std::uint32_t> out(outElems);
    {
        void* m = nullptr;
        vmaMapMemory(ctx.allocator, staging.allocation, &m);
        vmaInvalidateAllocation(ctx.allocator, staging.allocation, inBytes, outBytes);
        std::memcpy(out.data(), static_cast<char*>(m) + inBytes, outBytes);
        vmaUnmapMemory(ctx.allocator, staging.allocation);
    }

    m6c1::destroy_buffer(ctx.allocator, staging);
    m6c1::destroy_buffer(ctx.allocator, dev_out);
    m6c1::destroy_buffer(ctx.allocator, dev_in);
    return out;
}

bool test_fp16_roundtrip(vulkan_runtime::core::VulkanContext& ctx, std::uint32_t n) {
    // packed u16 words: ceil(n/2) uint32, each holding 2 fp16 (low/high halves)
    std::vector<std::uint32_t> packed((n + 1u) / 2u);
    std::uint32_t s = 0xF160001Au;
    std::vector<std::uint16_t> orig(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        orig[i] = static_cast<std::uint16_t>(s >> 8);
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t lane = i & 1u;
        packed[i >> 1u] |= static_cast<std::uint32_t>(orig[i]) << (lane * 16u);
    }

    std::vector<std::uint32_t> widened = run(ctx, 0u, n, packed, n); // u16 -> u32
    std::vector<std::uint32_t> repacked = run(ctx, 1u, n, widened, (n + 1u) / 2u); // u32 -> u16

    bool ok = repacked.size() == packed.size();
    for (std::size_t i = 0; ok && i < packed.size(); ++i) ok = (repacked[i] == packed[i]);
    std::cout << "test: fp16pack u16 roundtrip n=" << n << (ok ? " PASS" : " FAIL")
              << " (max_rel=0, max_abs=" << (ok ? 0 : 1) << ")\n";
    return ok;
}

bool test_i8_roundtrip(vulkan_runtime::core::VulkanContext& ctx, std::uint32_t n) {
    std::vector<std::int8_t> orig(n);
    std::uint32_t s = 0x1800001Bu;
    for (std::uint32_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        orig[i] = static_cast<std::int8_t>(static_cast<int>(s >> 8) - 128);
    }
    std::vector<std::uint32_t> packed((n + 3u) / 4u, 0u);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t lane = i & 3u;
        std::uint32_t byte = static_cast<std::uint32_t>(orig[i]) & 0xFFu;
        packed[i >> 2u] |= byte << (lane * 8u);
    }

    std::vector<std::uint32_t> widened = run(ctx, 2u, n, packed, n);        // i8 -> i32
    std::vector<std::uint32_t> repacked = run(ctx, 3u, n, widened, (n + 3u) / 4u); // i32 -> i8

    bool ok = repacked.size() == packed.size();
    for (std::size_t i = 0; ok && i < packed.size(); ++i) ok = (repacked[i] == packed[i]);
    std::cout << "test: fp16pack i8 roundtrip n=" << n << (ok ? " PASS" : " FAIL")
              << " (max_rel=0, max_abs=" << (ok ? 0 : 1) << ")\n";
    return ok;
}

bool test_u8_roundtrip(vulkan_runtime::core::VulkanContext& ctx, std::uint32_t n) {
    std::vector<std::uint8_t> orig(n);
    std::uint32_t s = 0x0E800001u;
    for (std::uint32_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        orig[i] = static_cast<std::uint8_t>(s >> 8); // full 0..255 range, incl >= 0x80
    }
    std::vector<std::uint32_t> packed((n + 3u) / 4u, 0u);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t lane = i & 3u;
        packed[i >> 2u] |= static_cast<std::uint32_t>(orig[i]) << (lane * 8u);
    }

    std::vector<std::uint32_t> widened = run(ctx, 4u, n, packed, n);         // u8 -> u32
    std::vector<std::uint32_t> repacked = run(ctx, 5u, n, widened, (n + 3u) / 4u); // u32 -> u8

    bool ok = repacked.size() == packed.size();
    for (std::size_t i = 0; ok && i < packed.size(); ++i) ok = (repacked[i] == packed[i]);
    std::cout << "test: fp16pack u8 roundtrip n=" << n << (ok ? " PASS" : " FAIL")
              << " (max_rel=0, max_abs=" << (ok ? 0 : 1) << ")\n";
    return ok;
}

bool test_i8_clamp(vulkan_runtime::core::VulkanContext& ctx) {
    // int32 values outside [-128, 127] clamp on narrow (mode 3).
    std::uint32_t n = 5;
    std::vector<std::uint32_t> in = {0u, 127u, static_cast<std::uint32_t>(-1), 200u,
                                     static_cast<std::uint32_t>(-200)};
    std::vector<std::uint32_t> repacked = run(ctx, 3u, n, in, (n + 3u) / 4u);

    std::int8_t got[5];
    for (std::uint32_t i = 0; i < n; ++i) {
        got[i] = static_cast<std::int8_t>((repacked[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu);
    }
    bool ok = got[0] == 0 && got[1] == 127 && got[2] == -1 && got[3] == 127 && got[4] == -128;
    std::cout << "test: fp16pack i8 clamp (mode 3)" << (ok ? " PASS" : " FAIL")
              << " (max_rel=0, max_abs=" << (ok ? 0 : 1) << ")\n";
    return ok;
}

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-fp16pack");

        bool ok = true;
        ok &= test_fp16_roundtrip(ctx, 1);
        ok &= test_fp16_roundtrip(ctx, 3);
        ok &= test_fp16_roundtrip(ctx, 64);
        ok &= test_fp16_roundtrip(ctx, 129);
        ok &= test_i8_roundtrip(ctx, 1);
        ok &= test_i8_roundtrip(ctx, 5);
        ok &= test_i8_roundtrip(ctx, 64);
        ok &= test_i8_roundtrip(ctx, 130);
        ok &= test_i8_clamp(ctx);
        ok &= test_u8_roundtrip(ctx, 1);
        ok &= test_u8_roundtrip(ctx, 5);
        ok &= test_u8_roundtrip(ctx, 64);
        ok &= test_u8_roundtrip(ctx, 130);

        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
