// M6c-1: unary — elementwise unary/scalar activations on the GPU (op via push
// constant) vs the cactus_x86 functions (via the bridge): sigmoid, clamp, exp,
// cos, sin, sqrt, log, abs, pow(x,s), relu, tanh, silu, softcap.

#include "m6c1_common.h"

#include <vector>

namespace {

struct PushConstants {
    std::uint32_t op;
    std::uint32_t count;
    float p0;
    float p1;
};

constexpr std::size_t kN = 2048;

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-unary");
        bool all_ok = true;

        // Device buffers are reused across every op. Packed fp16 (M7a): 2 values
        // per uint32, so ceil(kN/2) words each side.
        vk::DeviceSize inBytes = ((kN + 1u) / 2u) * sizeof(std::uint32_t);
        vk::DeviceSize outBytes = ((kN + 1u) / 2u) * sizeof(std::uint32_t);
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

        // {name, op, lo, hi, p0, p1, refKind}
        struct Case {
            char const* name;
            std::uint32_t op;
            float lo, hi;
            float p0, p1;
            int refKind; // 0 sigmoid, 1 clamp, 2 scalar_op, 3 relu, 4 tanh, 5 silu, 6 softcap
            int scalarOp;
        };
        Case const cases[] = {
            {"unary_sigmoid", 0u,  -5.0f, 5.0f, 0.0f,  0.0f, 0, 0},
            {"unary_clamp",   1u,  -3.0f, 3.0f, 0.5f,  2.0f, 1, 0},
            {"unary_exp",     2u,  -2.0f, 2.0f, 0.0f,  0.0f, 2, CACTUS_BRIDGE_OP_EXP},
            {"unary_cos",     3u,  -3.0f, 3.0f, 0.0f,  0.0f, 2, CACTUS_BRIDGE_OP_COS},
            {"unary_sin",     4u,  -3.0f, 3.0f, 0.0f,  0.0f, 2, CACTUS_BRIDGE_OP_SIN},
            {"unary_sqrt",    5u,   0.1f, 2.0f, 0.0f, 0.0f, 2, CACTUS_BRIDGE_OP_SQRT},
            {"unary_log",     6u,   0.1f, 2.0f, 0.0f,  0.0f, 2, CACTUS_BRIDGE_OP_LOG},
            {"unary_abs",     7u,  -2.0f, 2.0f, 0.0f,  0.0f, 2, CACTUS_BRIDGE_OP_ABS},
            {"unary_pow",     8u,   0.2f, 2.0f, 1.7f,  0.0f, 2, CACTUS_BRIDGE_OP_POW},
            {"unary_relu",    9u,  -2.0f, 2.0f, 0.0f,  0.0f, 3, 0},
            {"unary_tanh",   10u,  -3.0f, 3.0f, 0.0f,  0.0f, 4, 0},
            {"unary_silu",   11u,  -5.0f, 5.0f, 0.0f,  0.0f, 5, 0},
            {"unary_softcap", 12u, -5.0f, 5.0f, 10.0f, 1.0f, 6, 0},
        };

        std::uint32_t seed = 0x8E4A0001u;
        for (Case const& c : cases) {
            std::vector<float> src_f = m6c1::gen_f32(kN, seed++, c.lo, c.hi);
            std::vector<std::uint16_t> src_h(kN);
            cactus_bridge_fp32_to_fp16(src_f.data(), src_h.data(), kN);

            std::vector<std::uint16_t> ref_h(kN);
            switch (c.refKind) {
                case 0: cactus_bridge_sigmoid_f16(src_h.data(), ref_h.data(), kN); break;
                case 1: cactus_bridge_clamp_f16(src_h.data(), ref_h.data(), kN, c.p0, c.p1); break;
                case 2: cactus_bridge_scalar_op_f16(src_h.data(), ref_h.data(), kN, c.p0, c.scalarOp); break;
                case 3: cactus_bridge_relu_f16(src_h.data(), ref_h.data(), kN); break;
                case 4: cactus_bridge_tanh_f16(src_h.data(), ref_h.data(), kN); break;
                case 5: cactus_bridge_silu_f16(src_h.data(), ref_h.data(), kN); break;
                case 6: cactus_bridge_softcap_f16(src_h.data(), ref_h.data(), kN, c.p0, c.p1); break;
                default: break;
            }

            std::vector<std::uint32_t> in_w = m6c1::pack16(src_h);
            {
                void* m = nullptr;
                vmaMapMemory(ctx.allocator, staging.allocation, &m);
                std::memcpy(m, in_w.data(), inBytes);
                vmaUnmapMemory(ctx.allocator, staging.allocation);
                vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inBytes);
            }

            PushConstants pc{c.op, static_cast<std::uint32_t>(kN), c.p0, c.p1};
            m6c1::run_kernel(ctx, "unary.spv", {dev_in, dev_out}, staging,
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

            all_ok &= m6c1::compare_fp16(c.name, out_h, ref_h, 1e-3, 1e-2);
        }

        m6c1::destroy_buffer(ctx.allocator, staging);
        m6c1::destroy_buffer(ctx.allocator, dev_out);
        m6c1::destroy_buffer(ctx.allocator, dev_in);
        vulkan_runtime::core::destroy_context(ctx);
        return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
