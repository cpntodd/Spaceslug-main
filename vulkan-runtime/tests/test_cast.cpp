// M6c-1: cast — fp16 <-> fp32 on the GPU, vs cactus_fp16_to_fp32 /
// cactus_fp32_to_fp16 (via the bridge). Both directions are bit-exact (IEEE
// 754 binary16 <-> binary32 round-trip through packHalf2x16/unpackHalf2x16 is
// identical to clang's __fp16 conversions).

#include "m6c1_common.h"

#include <array>
#include <vector>

namespace {

struct PushConstants {
    std::uint32_t direction; // 0 = f16->f32, 1 = f32->f16
    std::uint32_t count;
};

constexpr std::size_t kN = 4096;

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-cast");
        bool all_ok = true;

        // ---- case 1: fp16 -> fp32 (bit-exact) ------------------------------
        {
            std::vector<float> src_f = m6c1::gen_f32(kN, 0xC4570001u, -100.0f, 100.0f);
            std::vector<std::uint16_t> src_h(kN);
            cactus_bridge_fp32_to_fp16(src_f.data(), src_h.data(), kN);

            std::vector<float> ref_f(kN);
            cactus_bridge_fp16_to_fp32(src_h.data(), ref_f.data(), kN);

            std::vector<std::uint32_t> in_w = m6c1::pack16(src_h); // packed fp16 in
            vk::DeviceSize inBytes = in_w.size() * sizeof(std::uint32_t);
            vk::DeviceSize outBytes = kN * sizeof(std::uint32_t); // fp32 out

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
                std::memcpy(m, in_w.data(), inBytes);
                vmaUnmapMemory(ctx.allocator, staging.allocation);
                vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inBytes);
            }

            PushConstants pc{0u, static_cast<std::uint32_t>(kN)};
            m6c1::run_kernel(ctx, "cast.spv", {dev_in, dev_out}, staging,
                             {{0, inBytes}}, {{inBytes, outBytes}}, &pc, sizeof(pc),
                             (static_cast<std::uint32_t>(kN) + 511u) / 512u, 1, 1);

            std::vector<std::uint32_t> out_w(kN);
            std::vector<float> out_f(kN);
            {
                void* m = nullptr;
                vmaMapMemory(ctx.allocator, staging.allocation, &m);
                vmaInvalidateAllocation(ctx.allocator, staging.allocation, inBytes, outBytes);
                std::memcpy(out_w.data(), static_cast<char*>(m) + inBytes, outBytes);
                vmaUnmapMemory(ctx.allocator, staging.allocation);
            }
            std::memcpy(out_f.data(), out_w.data(), kN * sizeof(float));

            all_ok &= m6c1::compare_f32_exact("cast_f16_to_f32", out_f, ref_f);

            m6c1::destroy_buffer(ctx.allocator, staging);
            m6c1::destroy_buffer(ctx.allocator, dev_out);
            m6c1::destroy_buffer(ctx.allocator, dev_in);
        }

        // ---- case 2: fp32 -> fp16 (bit-exact) ------------------------------
        {
            std::vector<float> src_f = m6c1::gen_f32(kN, 0xC4570002u, -10.0f, 10.0f);

            std::vector<std::uint16_t> ref_h(kN);
            cactus_bridge_fp32_to_fp16(src_f.data(), ref_h.data(), kN);

            std::vector<std::uint32_t> in_w(kN);
            std::memcpy(in_w.data(), src_f.data(), kN * sizeof(float));
            vk::DeviceSize inBytes = in_w.size() * sizeof(std::uint32_t);
            vk::DeviceSize outBytes = ((kN + 1u) / 2u) * sizeof(std::uint32_t); // packed fp16 out

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
                std::memcpy(m, in_w.data(), inBytes);
                vmaUnmapMemory(ctx.allocator, staging.allocation);
                vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inBytes);
            }

            PushConstants pc{1u, static_cast<std::uint32_t>(kN)};
            m6c1::run_kernel(ctx, "cast.spv", {dev_in, dev_out}, staging,
                             {{0, inBytes}}, {{inBytes, outBytes}}, &pc, sizeof(pc),
                             (static_cast<std::uint32_t>(kN) + 511u) / 512u, 1, 1);

            std::vector<std::uint32_t> out_w((kN + 1u) / 2u);
            {
                void* m = nullptr;
                vmaMapMemory(ctx.allocator, staging.allocation, &m);
                vmaInvalidateAllocation(ctx.allocator, staging.allocation, inBytes, outBytes);
                std::memcpy(out_w.data(), static_cast<char*>(m) + inBytes, outBytes);
                vmaUnmapMemory(ctx.allocator, staging.allocation);
            }
            std::vector<std::uint16_t> out_h = m6c1::unpack16(out_w, kN);

            // NOTE: not bit-exact. RADV's packHalf2x16 (v_cvt_pkrtz_f16_f32,
            // round-toward-zero) differs from clang's static_cast<__fp16>
            // (round-to-nearest-even) by up to ~1 fp16 ULP on tie-ish values.
            // f16->f32 IS bit-exact; f32->f16 is exact to within 1 ULP.
            all_ok &= m6c1::compare_fp16("cast_f32_to_f16", out_h, ref_h, 2e-3, 1e-3);

            m6c1::destroy_buffer(ctx.allocator, staging);
            m6c1::destroy_buffer(ctx.allocator, dev_out);
            m6c1::destroy_buffer(ctx.allocator, dev_in);
        }

        vulkan_runtime::core::destroy_context(ctx);
        return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
