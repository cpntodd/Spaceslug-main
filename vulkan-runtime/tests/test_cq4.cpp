// M6c-1: CQ4 GEMM — decomposed correctness-first pipeline on the GPU vs
// cactus_quant_matmul (via the bridge):
//   transform_act (fp16 A -> fp32 A') + dequant_w (CQ4 -> fp16 W', pre-transposed
//   N x K) + sgemm_f16 (A' x W' -> fp16 C).
//
// Three fixtures mirror tests/test_cactus_x86.cpp:
//   1. INTERLEAVED_4ROW, gs=128, no signs/perm, input_scale_recip
//   2. non-interleaved hadamard, signs + permutation + input_scale
//   3. ORTHOGONAL (single group spanning K, rotation)
// transform_act is also checked standalone against a duplicated CPU transform
// (double precision, mirroring cactus_x86.cpp transform_hadamard_group and the
// orthogonal A_rot math).

#include "m6c1_common.h"

#include <cmath>
#include <vector>

namespace {

struct TAPush {
    std::uint32_t mode;
    std::uint32_t M;
    std::uint32_t K;
    std::uint32_t group_size;
    std::uint32_t num_groups;
    std::uint32_t has_left;
    std::uint32_t has_right;
    std::uint32_t has_perm;
};

struct DWPush {
    std::uint32_t mode; // 0 hadamard, 1 interleaved, 2 orthogonal
    std::uint32_t bits;
    std::uint32_t K;
    std::uint32_t N;
    std::uint32_t group_size;
    std::uint32_t num_groups;
};

struct GEMMPush {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

// --- fixture helpers (mirror test_cactus_x86.cpp) -------------------------

std::vector<std::uint16_t> to_fp16(std::vector<float> const& f) {
    std::vector<std::uint16_t> h(f.size());
    cactus_bridge_fp32_to_fp16(f.data(), h.data(), f.size());
    return h;
}

double f16d(std::uint16_t h) {
    float f;
    cactus_bridge_fp16_to_fp32(&h, &f, 1);
    return static_cast<double>(f);
}

std::vector<std::uint32_t> widen_u8(std::vector<std::uint8_t> const& b) {
    std::vector<std::uint32_t> w(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) w[i] = b[i];
    return w;
}

std::vector<std::int32_t> widen_i8(std::vector<std::int8_t> const& q) {
    std::vector<std::int32_t> w(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) w[i] = q[i];
    return w;
}

std::vector<std::uint32_t> gen_permutation(std::uint32_t n, std::uint32_t seed) {
    std::vector<std::uint32_t> p(n);
    for (std::uint32_t i = 0; i < n; ++i) p[i] = i;
    std::uint32_t s = seed;
    for (std::uint32_t i = n; i > 1; --i) {
        s = s * 1664525u + 1013904223u;
        std::swap(p[i - 1], p[(s >> 8) % i]);
    }
    return p;
}

std::vector<std::int8_t> gen_signs(std::uint32_t n, std::uint32_t seed) {
    std::vector<std::int8_t> s(n);
    std::uint32_t st = seed;
    for (std::uint32_t i = 0; i < n; ++i) {
        st = st * 1664525u + 1013904223u;
        s[i] = ((st >> 8) & 1u) ? 1 : -1;
    }
    return s;
}

std::vector<std::uint8_t> pack_4bit_lsb(std::vector<std::uint32_t> const& idx) {
    std::vector<std::uint8_t> bytes((idx.size() + 1) / 2, 0);
    for (std::size_t k = 0; k < idx.size(); ++k) {
        if (k & 1u) bytes[k / 2] |= static_cast<std::uint8_t>((idx[k] & 0xFu) << 4);
        else bytes[k / 2] |= static_cast<std::uint8_t>(idx[k] & 0xFu);
    }
    return bytes;
}

// Interleaved 4-row panel packing over `ng` groups. idx[row] spans K = ng*gs
// columns (idx[row][g*gs+k]); each panel contributes ng group panels of
// panel_bytes each, laid out as (panel*ng+g)*panel_bytes. ng=1 reduces to the
// single-group layout.
std::vector<std::uint8_t> pack_interleaved_4row(
    std::vector<std::vector<std::uint32_t>> const& idx, std::uint32_t N, std::uint32_t gs,
    std::uint32_t ng) {
    std::uint32_t const pgb = (gs * 4u + 7u) / 8u;
    std::size_t const panel_bytes = 4u * pgb;
    std::uint32_t const num_panels = N / 4u;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(num_panels) * ng * panel_bytes, 0);
    for (std::uint32_t panel = 0; panel < num_panels; ++panel) {
        for (std::uint32_t g = 0; g < ng; ++g) {
            std::uint8_t* p = packed.data() +
                (static_cast<std::size_t>(panel) * ng + g) * panel_bytes;
            for (std::uint32_t r = 0; r < 4; ++r) {
                std::uint32_t row = panel * 4u + r;
                for (std::uint32_t k = 0; k < gs; ++k) {
                    std::uint32_t chunk = k / 8u, sub = k & 7u;
                    std::uint8_t& byte = p[chunk * 16u + r * 4u + (sub & 3u)];
                    std::uint32_t col = g * gs + k;
                    if (sub & 4u) byte |= static_cast<std::uint8_t>((idx[row][col] & 0xFu) << 4);
                    else byte |= static_cast<std::uint8_t>(idx[row][col] & 0xFu);
                }
            }
        }
    }
    return packed;
}

// Reorder logical norms [n*ng+g] -> interleaved panel-major [(n/4*ng+g)*4 + n%4]
// (the layout dequant_w mode 1 consumes, matching the on-disk format written by
// cq.py). ng=1 is the identity: the panel-major index (n/4)*4 + n%4 == n.
// The CPU reference (cactus_bridge_quant_matmul) reads the STORED/panel layout,
// so the same reordered buffer is passed to both GPU and CPU reference.
std::vector<std::uint16_t> layout_norms(std::vector<std::uint16_t> const& logical,
                                        std::uint32_t N, std::uint32_t ng, bool interleaved) {
    if (!interleaved) return logical;
    std::vector<std::uint16_t> out(logical.size());
    for (std::uint32_t n = 0; n < N; ++n)
        for (std::uint32_t g = 0; g < ng; ++g)
            out[((n / 4u) * ng + g) * 4u + (n & 3u)] = logical[n * ng + g];
    return out;
}

void fwht_d(std::vector<double>& x) {
    std::uint32_t const n = static_cast<std::uint32_t>(x.size());
    for (std::uint32_t h = 1; h < n; h <<= 1) {
        for (std::uint32_t i = 0; i < n; i += (h << 1)) {
            for (std::uint32_t j = i; j < i + h; ++j) {
                double a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    double inv = 1.0 / std::sqrt(static_cast<double>(n));
    for (double& v : x) v *= inv;
}

// --- generic upload/run/read (everything as uint32 words) ------------------

struct RunResult {
    std::vector<std::vector<std::uint32_t>> outputs;
};

RunResult run_cq4(vulkan_runtime::core::VulkanContext& ctx, char const* shaderName,
                  std::vector<std::vector<std::uint32_t>> const& inputs,
                  std::vector<vk::DeviceSize> const& outSizes, void const* push,
                  std::uint32_t pushSize, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
    std::uint32_t const nIn = static_cast<std::uint32_t>(inputs.size());
    std::uint32_t const nOut = static_cast<std::uint32_t>(outSizes.size());

    std::vector<vk::DeviceSize> inSizes(nIn);
    vk::DeviceSize inTotal = 0;
    for (std::uint32_t i = 0; i < nIn; ++i) {
        inSizes[i] = inputs[i].size() * sizeof(std::uint32_t);
        inTotal += inSizes[i];
    }
    vk::DeviceSize outTotal = 0;
    for (auto s : outSizes) outTotal += s;

    std::vector<m6c1::Buffer> dev(nIn + nOut);
    for (std::uint32_t i = 0; i < nIn; ++i)
        dev[i] = m6c1::create_buffer(ctx.allocator, inSizes[i],
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    for (std::uint32_t i = 0; i < nOut; ++i)
        dev[nIn + i] = m6c1::create_buffer(ctx.allocator, outSizes[i],
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    m6c1::Buffer staging = m6c1::create_buffer(ctx.allocator, inTotal + outTotal,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    {
        void* m = nullptr;
        vmaMapMemory(ctx.allocator, staging.allocation, &m);
        char* p = static_cast<char*>(m);
        vk::DeviceSize off = 0;
        for (std::uint32_t i = 0; i < nIn; ++i) {
            std::memcpy(p + off, inputs[i].data(), inSizes[i]);
            off += inSizes[i];
        }
        vmaUnmapMemory(ctx.allocator, staging.allocation);
        vmaFlushAllocation(ctx.allocator, staging.allocation, 0, inTotal);
    }

    std::vector<m6c1::Copy> inC, outC;
    vk::DeviceSize off = 0;
    for (auto s : inSizes) { inC.push_back({off, s}); off += s; }
    off = inTotal;
    for (auto s : outSizes) { outC.push_back({off, s}); off += s; }

    m6c1::run_kernel(ctx, shaderName, dev, staging, inC, outC, push, pushSize, gx, gy, gz);

    RunResult r;
    r.outputs.resize(nOut);
    {
        void* m = nullptr;
        vmaMapMemory(ctx.allocator, staging.allocation, &m);
        vmaInvalidateAllocation(ctx.allocator, staging.allocation, inTotal, outTotal);
        char* p = static_cast<char*>(m) + inTotal;
        vk::DeviceSize ooff = 0;
        for (std::uint32_t i = 0; i < nOut; ++i) {
            r.outputs[i].resize(outSizes[i] / sizeof(std::uint32_t));
            std::memcpy(r.outputs[i].data(), p + ooff, outSizes[i]);
            ooff += outSizes[i];
        }
        vmaUnmapMemory(ctx.allocator, staging.allocation);
    }

    m6c1::destroy_buffer(ctx.allocator, staging);
    for (auto& b : dev) m6c1::destroy_buffer(ctx.allocator, b);
    return r;
}

// --- CPU transform references (double, mirroring cactus_x86.cpp) ------------

std::vector<double> cpu_transform_hadamard(
    std::vector<std::uint16_t> const& A, std::vector<float> const& recip,
    std::vector<std::int8_t> const& left, std::vector<std::int8_t> const& right,
    std::vector<std::uint32_t> const& perm, std::uint32_t M, std::uint32_t K,
    std::uint32_t gs, std::uint32_t numGroups, bool hasLeft, bool hasRight, bool hasPerm) {
    std::vector<double> out(static_cast<std::size_t>(M) * K);
    for (std::uint32_t m = 0; m < M; ++m) {
        for (std::uint32_t g = 0; g < numGroups; ++g) {
            std::uint32_t base = g * gs;
            std::vector<double> work(gs);
            for (std::uint32_t k = 0; k < gs; ++k) {
                double sign = hasLeft ? static_cast<double>(left[k]) : 1.0;
                work[k] = f16d(A[static_cast<std::size_t>(m) * K + base + k])
                          * static_cast<double>(recip[base + k]) * sign;
            }
            fwht_d(work);
            for (std::uint32_t k = 0; k < gs; ++k) {
                if (hasRight) work[k] *= static_cast<double>(right[k]);
            }
            if (hasPerm) {
                for (std::uint32_t j = 0; j < gs; ++j)
                    out[static_cast<std::size_t>(m) * K + base + j] = work[perm[j]];
            } else {
                for (std::uint32_t j = 0; j < gs; ++j)
                    out[static_cast<std::size_t>(m) * K + base + j] = work[j];
            }
        }
    }
    return out;
}

std::vector<double> cpu_transform_orthogonal(
    std::vector<std::uint16_t> const& A, std::vector<float> const& recip,
    std::vector<std::uint16_t> const& rotation, std::uint32_t M, std::uint32_t K) {
    std::vector<double> out(static_cast<std::size_t>(M) * K, 0.0);
    for (std::uint32_t m = 0; m < M; ++m)
        for (std::uint32_t k = 0; k < K; ++k) {
            double av = f16d(A[static_cast<std::size_t>(m) * K + k]) * static_cast<double>(recip[k]);
            for (std::uint32_t i = 0; i < K; ++i)
                out[static_cast<std::size_t>(m) * K + i] += av * f16d(rotation[static_cast<std::size_t>(k) * K + i]);
        }
    return out;
}

// ---------------------------------------------------------------------------
// Full GPU pipeline: transform_act -> dequant_w -> sgemm_f16 -> C (fp16).
// Also returns A' (fp32) for the standalone transform check.
// ---------------------------------------------------------------------------
std::vector<std::uint16_t> gpu_pipeline(
    vulkan_runtime::core::VulkanContext& ctx,
    std::vector<std::uint16_t> const& A, std::vector<float> const& recip,
    std::vector<std::int8_t> const& left, std::vector<std::int8_t> const& right,
    std::vector<std::uint32_t> const& perm, std::vector<std::uint16_t> const& rotation,
    std::vector<std::uint8_t> const& packed, std::vector<std::uint16_t> const& codebook,
    std::vector<std::uint16_t> const& norms,
    std::uint32_t M, std::uint32_t K, std::uint32_t N, std::uint32_t gs,
    std::uint32_t numGroups, std::uint32_t bits, std::uint32_t flags,
    std::vector<std::uint16_t>* A_out /* optional, unpacked fp16 A' */) {
    bool const orthogonal = (flags & CACTUS_BRIDGE_QUANT_FLAG_ORTHOGONAL) != 0;
    bool const interleaved = (flags & CACTUS_BRIDGE_QUANT_FLAG_INTERLEAVED_4ROW) != 0;

    std::vector<std::uint32_t> A_w = m6c1::widen16(A);
    std::vector<std::uint32_t> rot_w = m6c1::widen16(rotation);
    std::vector<std::int32_t> left_w = widen_i8(left);
    std::vector<std::int32_t> right_w = widen_i8(right);
    // recip (float) -> uint32 bit-cast
    std::vector<std::uint32_t> recip_w(recip.size());
    std::memcpy(recip_w.data(), recip.data(), recip.size() * sizeof(float));

    // dummy 1-element buffers for absent signs/perm/rotation
    std::vector<std::uint32_t> one(1, 0);

    // --- transform_act (M7a: output is PACKED fp16 A', M*K/2 words) ---
    std::uint32_t taMode = orthogonal ? 1u : 0u;
    TAPush tap{taMode, M, K, gs, numGroups,
               left.empty() ? 0u : 1u, right.empty() ? 0u : 1u, perm.empty() ? 0u : 1u};
    // mode 0: one thread per (m, group); mode 1: one thread per output-column pair.
    std::uint32_t taTotal = orthogonal ? (M * (K / 2u)) : (M * numGroups);
    std::vector<std::uint32_t> left_uw(left_w.begin(), left_w.end());
    std::vector<std::uint32_t> right_uw(right_w.begin(), right_w.end());
    RunResult ta = run_cq4(
        ctx, "transform_act.spv",
        {A_w, recip_w, left_uw.empty() ? one : left_uw,
         right_uw.empty() ? one : right_uw,
         perm.empty() ? one : perm, rot_w.empty() ? one : rot_w},
        {static_cast<vk::DeviceSize>(M) * K / 2u * sizeof(std::uint32_t)},
        &tap, sizeof(tap), (taTotal + 63u) / 64u, 1, 1);
    std::vector<std::uint32_t> const& aPacked = ta.outputs[0];
    std::vector<std::uint16_t> aOutH = m6c1::unpack16(aPacked, static_cast<std::size_t>(M) * K);
    if (A_out) *A_out = aOutH;

    // --- dequant_w (M7a: output is PACKED fp16 W', N*K/2 words) ---
    std::vector<std::uint32_t> packed_w = widen_u8(packed);
    std::vector<std::uint32_t> codebook_w = m6c1::widen16(codebook);
    std::vector<std::uint32_t> norms_w = m6c1::widen16(norms);
    std::uint32_t dwMode = orthogonal ? 2u : (interleaved ? 1u : 0u);
    DWPush dwp{dwMode, bits, K, N, gs, numGroups};
    RunResult dw = run_cq4(
        ctx, "dequant_w.spv",
        {packed_w, codebook_w, norms_w},
        {static_cast<vk::DeviceSize>(N) * K / 2u * sizeof(std::uint32_t)},
        &dwp, sizeof(dwp), (N * K / 2u + 63u) / 64u, 1, 1);
    std::vector<std::uint32_t> const& wPacked = dw.outputs[0];

    // --- sgemm_f16_am: packed A' x packed W' (pre-transposed) -> packed C ---
    // (No cast step: A' and W' are already packed fp16 from the two kernels above.)
    GEMMPush gp{M, N, K};
    RunResult gm = run_cq4(
        ctx, "sgemm_f16_am.spv",
        {aPacked, wPacked},
        {static_cast<vk::DeviceSize>((static_cast<std::size_t>(M) * N + 1u) / 2u) *
         sizeof(std::uint32_t)},
        &gp, sizeof(gp), (N + 63u) / 64u, (M + 63u) / 64u, 1);
    return m6c1::unpack16(gm.outputs[0], static_cast<std::size_t>(M) * N);
}

// ---------------------------------------------------------------------------
// One full fixture test.
// ---------------------------------------------------------------------------
bool run_fixture(vulkan_runtime::core::VulkanContext& ctx, char const* name,
                 std::uint32_t M, std::uint32_t N, std::uint32_t gs, std::uint32_t numGroups,
                 bool interleaved, bool useSignsPerm, bool useInputScale, bool orthogonal) {
    std::uint32_t const bits = 4;
    std::uint32_t const K = gs * numGroups;

    std::uint32_t seedBase = orthogonal ? 0x0C170000u : 0x0C170100u;

    std::vector<std::uint16_t> codebook = to_fp16(m6c1::gen_f32(16, seedBase + 1, -1.0f, 1.0f));
    // Logical norm layout [n*numGroups+g] (what the CPU reference consumes).
    std::vector<std::uint16_t> norms = to_fp16(m6c1::gen_f32(
        orthogonal ? N : N * numGroups, seedBase + 2, 0.5f, 1.5f));
    // GPU-side norm layout: interleaved 4-row panels for mode 1, logical otherwise.
    std::vector<std::uint16_t> normsGpu = layout_norms(norms, N, numGroups, interleaved && !orthogonal);

    std::vector<std::uint16_t> recipH, inputScaleH;
    std::vector<float> recipF(K);
    if (orthogonal || !useInputScale) {
        recipH = to_fp16(m6c1::gen_f32(K, seedBase + 3, 0.5f, 1.5f));
        for (std::uint32_t i = 0; i < K; ++i) recipF[i] = static_cast<float>(f16d(recipH[i]));
    } else {
        inputScaleH = to_fp16(m6c1::gen_f32(K, seedBase + 3, 0.5f, 1.5f));
        for (std::uint32_t i = 0; i < K; ++i) recipF[i] = 1.0f / static_cast<float>(f16d(inputScaleH[i]));
    }

    std::vector<std::int8_t> leftSigns, rightSigns;
    std::vector<std::uint32_t> permutation;
    if (useSignsPerm) {
        leftSigns = gen_signs(gs, seedBase + 4);
        rightSigns = gen_signs(gs, seedBase + 5);
        permutation = gen_permutation(gs, seedBase + 6);
    }

    // random indices (N rows x K cols, single group)
    std::vector<std::vector<std::uint32_t>> idx(N, std::vector<std::uint32_t>(K));
    {
        std::uint32_t s = seedBase + 7;
        for (std::uint32_t n = 0; n < N; ++n)
            for (std::uint32_t k = 0; k < K; ++k) {
                s = s * 1664525u + 1013904223u;
                idx[n][k] = (s >> 8) % 16u;
            }
    }
    std::vector<std::uint8_t> packed;
    if (interleaved) {
        packed = pack_interleaved_4row(idx, N, gs, numGroups);
    } else {
        std::vector<std::uint32_t> flat(N * K);
        for (std::uint32_t n = 0; n < N; ++n)
            for (std::uint32_t k = 0; k < K; ++k) flat[n * K + k] = idx[n][k];
        packed = pack_4bit_lsb(flat);
    }

    std::vector<std::uint16_t> rotation = to_fp16(m6c1::gen_f32(
        static_cast<std::size_t>(K) * K, seedBase + 8, -0.5f, 0.5f));

    std::vector<std::uint16_t> A = to_fp16(m6c1::gen_f32(static_cast<std::size_t>(M) * K, seedBase + 9, -1.0f, 1.0f));

    // Build bridge matrix + CPU reference.
    CactusQuantMatrixBridge W{};
    W.bits = bits;
    W.K = K;
    W.N = N;
    W.group_size = gs;
    W.num_groups = numGroups;
    W.flags = 0;
    if (interleaved) W.flags |= CACTUS_BRIDGE_QUANT_FLAG_INTERLEAVED_4ROW;
    if (orthogonal) W.flags |= CACTUS_BRIDGE_QUANT_FLAG_ORTHOGONAL;
    W.codebook = codebook.data();
    W.input_scale = inputScaleH.empty() ? nullptr : inputScaleH.data();
    W.input_scale_recip = recipH.empty() ? nullptr : recipH.data();
    W.norms = normsGpu.data();
    W.packed_indices = packed.data();
    W.left_signs = leftSigns.empty() ? nullptr : leftSigns.data();
    W.right_signs = rightSigns.empty() ? nullptr : rightSigns.data();
    W.permutation = permutation.empty() ? nullptr : permutation.data();
    W.rotation = orthogonal ? rotation.data() : nullptr;
    W.expanded = nullptr;
    W.norm_f32 = nullptr;

    std::vector<std::uint16_t> C_ref(static_cast<std::size_t>(M) * N);
    cactus_bridge_quant_matmul(&W, A.data(), M, C_ref.data());

    // GPU pipeline.
    std::vector<std::uint16_t> aOutH;
    std::vector<std::uint16_t> C = gpu_pipeline(
        ctx, A, recipF, leftSigns, rightSigns, permutation, rotation,
        packed, codebook, normsGpu, M, K, N, gs, numGroups, bits, W.flags, &aOutH);

    bool ok = m6c1::compare_fp16(name, C, C_ref, 2e-2, 1e-2);

    // Standalone transform check: the GPU A' is now fp16 (rounded by
    // transform_act's packHalf2x16), so round the CPU reference to fp16 too and
    // compare fp16-vs-fp16 (the fp32->fp16 rounding is what now dominates; the
    // FWHT fp32-vs-double drift is ~1e-6, far below fp16's 1e-3).
    std::vector<double> taRef;
    if (orthogonal) {
        taRef = cpu_transform_orthogonal(A, recipF, rotation, M, K);
    } else {
        taRef = cpu_transform_hadamard(A, recipF, leftSigns, rightSigns, permutation,
                                       M, K, gs, numGroups, !leftSigns.empty(), !rightSigns.empty(),
                                       !permutation.empty());
    }
    std::vector<float> taRefF(taRef.size());
    for (std::size_t i = 0; i < taRef.size(); ++i) taRefF[i] = static_cast<float>(taRef[i]);
    std::vector<std::uint16_t> taRefH(taRef.size());
    cactus_bridge_fp32_to_fp16(taRefF.data(), taRefH.data(), taRefF.size());
    std::string tname = std::string(name) + "_transform";
    ok &= m6c1::compare_fp16(tname.c_str(), aOutH, taRefH, 1e-3, 1e-2);

    return ok;
}

} // namespace

int main() {
    try {
        vulkan_runtime::core::VulkanContext ctx =
            vulkan_runtime::core::create_context("vulkan-runtime-cq4");
        bool ok = true;

        ok &= run_fixture(ctx, "cq4_interleaved", 64, 64, 128, 1, true, false, false, false);
        ok &= run_fixture(ctx, "cq4_hadamard_signs", 64, 64, 128, 1, false, true, true, false);
        ok &= run_fixture(ctx, "cq4_orthogonal", 64, 64, 64, 1, false, false, false, true);
        ok &= run_fixture(ctx, "cq4_interleaved_multigroup_m1", 1, 64, 128, 4, true, false, false, false);
        ok &= run_fixture(ctx, "cq4_interleaved_m1_n512", 1, 512, 128, 4, true, false, false, false);

        vulkan_runtime::core::destroy_context(ctx);
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
