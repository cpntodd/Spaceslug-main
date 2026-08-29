// M6b-1: correctness tests for the x86 CPU Cactus kernel library
// (src/cactus/cactus_x86.{h,cpp}). Every public function in the subset is
// exercised against a double-precision CPU reference. Inputs are generated
// with a deterministic LCG and cast to __fp16, so the double reference sees
// exactly the half-precision values the kernel operates on.
//
// This test is CPU-only (no Vulkan, no GPU): it passes identically on RADV and
// lavapipe. Compiled with clang++ because it uses __fp16 (see CMakeLists.txt).

#include "cactus/cactus_x86.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Deterministic inputs + comparison helpers
// ---------------------------------------------------------------------------

bool g_failed = false;

void report(const char* name, bool ok, double max_rel, double max_abs) {
    if (ok) {
        std::cout << "cactus_x86: " << name << " PASS"
                  << " (max_rel=" << max_rel << ", max_abs=" << max_abs << ")\n";
    } else {
        std::cout << "cactus_x86: " << name << " FAIL"
                  << " (max_rel=" << max_rel << ", max_abs=" << max_abs << ")\n";
        g_failed = true;
    }
}

// Numerical Recipes LCG -> doubles in [lo, hi).
std::vector<double> gen_double(size_t n, uint32_t seed, double lo, double hi) {
    std::vector<double> v(n);
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        double u = static_cast<double>(s >> 8) * (1.0 / 16777216.0);
        v[i] = lo + u * (hi - lo);
    }
    return v;
}

std::vector<__fp16> gen_fp16(size_t n, uint32_t seed, double lo, double hi) {
    std::vector<double> d = gen_double(n, seed, lo, hi);
    std::vector<__fp16> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<__fp16>(d[i]);
    return v;
}

std::vector<double> to_double(const std::vector<__fp16>& v) {
    std::vector<double> d(v.size());
    for (size_t i = 0; i < v.size(); ++i) d[i] = static_cast<double>(v[i]);
    return d;
}

std::vector<__fp16> from_double(const std::vector<double>& d) {
    std::vector<__fp16> v(d.size());
    for (size_t i = 0; i < d.size(); ++i) v[i] = static_cast<__fp16>(d[i]);
    return v;
}

// Compare a __fp16 result buffer against a double reference. `abs_tol` and
// `rel_tol` are the combined pass criteria (err <= abs_tol || err <= rel_tol*|ref|).
bool check_fp16(const char* name, const std::vector<__fp16>& got, const std::vector<double>& ref,
                double abs_tol, double rel_tol) {
    double max_rel = 0.0, max_abs = 0.0;
    bool ok = true;
    const size_t n = std::min(got.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        const double g = static_cast<double>(got[i]);
        const double r = ref[i];
        const double err = std::fabs(g - r);
        max_abs = std::max(max_abs, err);
        max_rel = std::max(max_rel, err / std::max(1.0, std::fabs(r)));
        if (err > abs_tol && err > rel_tol * std::fabs(r)) ok = false;
    }
    report(name, ok, max_rel, max_abs);
    return ok;
}

// Bitwise-exact comparison of two __fp16 buffers (used for pure data movement).
bool check_fp16_exact(const char* name, const std::vector<__fp16>& got, const std::vector<__fp16>& ref) {
    bool ok = got.size() == ref.size() && std::memcmp(got.data(), ref.data(), got.size() * sizeof(__fp16)) == 0;
    report(name, ok, ok ? 0.0 : 1.0, ok ? 0.0 : 1.0);
    return ok;
}

// --- quant helpers (mirror the implementation's decode math, in double) -----

uint32_t ref_extract_idx_lsb(const uint8_t* packed, uint32_t k, uint32_t bits) {
    switch (bits) {
        case 4: return (packed[k / 2] >> ((k & 1u) * 4u)) & 0xFu;
        case 2: return (packed[k / 4] >> ((k & 3u) * 2u)) & 0x3u;
        case 1: return (packed[k / 8] >> (k & 7u)) & 0x1u;
        case 3: {
            const uint32_t bit_pos = k * 3u;
            const uint32_t byte_idx = bit_pos / 8u;
            const uint32_t bit_idx = bit_pos & 7u;
            uint32_t val = static_cast<uint32_t>(packed[byte_idx]) >> bit_idx;
            if (bit_idx > 5u) val |= static_cast<uint32_t>(packed[byte_idx + 1u]) << (8u - bit_idx);
            return val & 0x7u;
        }
        default: return 0;
    }
}

void ref_fwht(std::vector<double>& x) {
    const uint32_t n = static_cast<uint32_t>(x.size());
    for (uint32_t h = 1; h < n; h <<= 1) {
        for (uint32_t i = 0; i < n; i += (h << 1)) {
            for (uint32_t j = i; j < i + h; ++j) {
                const double a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const double inv = 1.0 / std::sqrt(static_cast<double>(n));
    for (double& v : x) v *= inv;
}

// Dequantize one Hadamard-embedded weight row into the raw (input) space.
std::vector<double> ref_dequantize_hadamard_row(
    uint32_t bits, uint32_t hidden_dim, uint32_t gs, uint32_t num_groups, size_t row,
    const uint8_t* packed, const std::vector<double>& codebook, const std::vector<double>& norms,
    const std::vector<double>& recip, const std::vector<double>& input_scale,
    const int8_t* left_signs, const int8_t* right_signs, const uint32_t* perm, bool interleaved) {
    const uint32_t pgb = (gs * bits + 7u) / 8u;
    const size_t panel_bytes = 4u * pgb;
    std::vector<double> out(hidden_dim, 0.0);
    for (uint32_t g = 0; g < num_groups; ++g) {
        std::vector<double> rotated(gs, 0.0);
        for (uint32_t k = 0; k < gs; ++k) {
            uint32_t idx = 0;
            if (interleaved) {
                const uint8_t* panel = packed + ((row / 4u) * num_groups + g) * panel_bytes;
                const uint32_t chunk = k / 8u, sub = k & 7u;
                const uint8_t byte = panel[chunk * 16u + (row & 3u) * 4u + (sub & 3u)];
                idx = (sub & 4u) ? static_cast<uint32_t>(byte >> 4)
                                 : static_cast<uint32_t>(byte & 0x0Fu);
            } else {
                const uint8_t* p = packed + (row * num_groups + g) * pgb;
                idx = ref_extract_idx_lsb(p, k, bits);
            }
            const uint32_t dst = perm ? perm[k] : k;
            const double rs = right_signs ? static_cast<double>(right_signs[dst]) : 1.0;
            rotated[dst] = codebook[idx] * rs;
        }
        ref_fwht(rotated);
        // INTERLEAVED_4ROW norms are panel-major (cq.py: index (nb*ng+g)*4+ni);
        // non-interleaved are row-major [row*ng+g]. The fixture stores panel-major
        // for interleaved, so the reference must read the same layout.
        const double norm = interleaved
            ? norms[((row / 4u) * num_groups + g) * 4u + (row & 3u)]
            : norms[row * num_groups + g];
        for (uint32_t k = 0; k < gs; ++k) {
            const uint32_t col = g * gs + k;
            const double ls = left_signs ? static_cast<double>(left_signs[k]) : 1.0;
            double sc = 1.0;
            if (!recip.empty()) sc = recip[col];
            else if (!input_scale.empty()) sc = 1.0 / input_scale[col];
            out[col] = rotated[k] * ls * norm * sc;
        }
    }
    return out;
}

// Reference panel decode for orthogonal INTERLEAVED_4ROW (single group, gs==K).
// Mirrors cactus_x86 extract_idx_interleaved_4row + the panel addressing in
// upstream tq_extract_interleaved_4row_4bit.
uint32_t ref_extract_idx_interleaved_4row(const uint8_t* packed_base, uint32_t K,
                                          size_t row, uint32_t k) {
    const uint32_t pgb = (K * 4u + 7u) / 8u;
    const size_t panel_bytes = 4u * pgb;
    const uint8_t* panel = packed_base + (row / 4u) * panel_bytes;
    const uint32_t chunk = k / 8u, sub = k & 7u;
    const uint8_t byte = panel[chunk * 16u + (row & 3u) * 4u + (sub & 3u)];
    return (sub & 4u) ? static_cast<uint32_t>(byte >> 4)
                      : static_cast<uint32_t>(byte & 0x0Fu);
}

std::vector<uint32_t> gen_permutation(uint32_t n, uint32_t seed) {
    std::vector<uint32_t> p(n);
    for (uint32_t i = 0; i < n; ++i) p[i] = i;
    uint32_t s = seed;
    for (uint32_t i = n; i > 1; --i) {
        s = s * 1664525u + 1013904223u;
        std::swap(p[i - 1], p[(s >> 8) % i]);
    }
    return p;
}

std::vector<int8_t> gen_signs(uint32_t n, uint32_t seed) {
    std::vector<int8_t> s(n);
    uint32_t st = seed;
    for (uint32_t i = 0; i < n; ++i) {
        st = st * 1664525u + 1013904223u;
        s[i] = ((st >> 8) & 1u) ? 1 : -1;
    }
    return s;
}

// Pack 4-bit indices LSB-first.
std::vector<uint8_t> pack_4bit_lsb(const std::vector<uint32_t>& idx) {
    std::vector<uint8_t> bytes((idx.size() + 1) / 2, 0);
    for (size_t k = 0; k < idx.size(); ++k) {
        if (k & 1u) bytes[k / 2] |= static_cast<uint8_t>((idx[k] & 0xFu) << 4);
        else bytes[k / 2] |= static_cast<uint8_t>(idx[k] & 0xFu);
    }
    return bytes;
}

// Pack 4-bit indices in the INTERLEAVED_4ROW layout: N rows (N % 4 == 0) x gs
// cols per group over ng groups, organized as (N/4) panels of 4 rows each.
// idx[row] spans K = ng*gs columns (idx[row][g*gs+k]); each panel contributes
// ng group panels of panel_bytes each, laid out as (panel*ng+g)*panel_bytes.
std::vector<uint8_t> pack_interleaved_4row(const std::vector<std::vector<uint32_t>>& idx,
                                           uint32_t N, uint32_t gs, uint32_t ng) {
    const uint32_t pgb = (gs * 4u + 7u) / 8u;
    const size_t panel_bytes = 4u * pgb;
    const uint32_t num_panels = N / 4u;
    std::vector<uint8_t> packed(static_cast<size_t>(num_panels) * ng * panel_bytes, 0);
    for (uint32_t panel = 0; panel < num_panels; ++panel) {
        for (uint32_t g = 0; g < ng; ++g) {
            uint8_t* p = packed.data() + (static_cast<size_t>(panel) * ng + g) * panel_bytes;
            for (uint32_t r = 0; r < 4; ++r) {
                const uint32_t row = panel * 4u + r;
                for (uint32_t k = 0; k < gs; ++k) {
                    const uint32_t chunk = k / 8u, sub = k & 7u;
                    uint8_t& byte = p[chunk * 16u + r * 4u + (sub & 3u)];
                    const uint32_t col = g * gs + k;
                    if (sub & 4u) byte |= static_cast<uint8_t>((idx[row][col] & 0xFu) << 4);
                    else byte |= static_cast<uint8_t>(idx[row][col] & 0xFu);
                }
            }
        }
    }
    return packed;
}

// ---------------------------------------------------------------------------
// matmul_f16
// ---------------------------------------------------------------------------
void test_matmul() {
    const size_t M = 8, K = 16, N = 12;
    std::vector<__fp16> a = gen_fp16(M * K, 0x11111111u, -1.0, 1.0);
    std::vector<__fp16> bt = gen_fp16(N * K, 0x22222222u, -1.0, 1.0); // pre-transposed: [n][k]
    std::vector<double> ad = to_double(a), btd = to_double(bt);

    std::vector<__fp16> c(M * N);
    cactus_matmul_f16(a.data(), bt.data(), c.data(), M, K, N);

    std::vector<double> ref(M * N, 0.0);
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n)
            for (size_t k = 0; k < K; ++k)
                ref[m * N + n] += ad[m * K + k] * btd[n * K + k];

    check_fp16("matmul_f16", c, ref, 1e-3, 1e-2);
}

// ---------------------------------------------------------------------------
// rms_norm_f16
// ---------------------------------------------------------------------------
void test_rms_norm() {
    const size_t rows = 4, cols = 64;
    const float eps = 1e-6f;
    std::vector<__fp16> x = gen_fp16(rows * cols, 0x33333333u, -1.0, 1.0);
    std::vector<__fp16> w = gen_fp16(cols, 0x44444444u, 0.5, 1.5);
    std::vector<double> xd = to_double(x), wd = to_double(w);

    std::vector<__fp16> out(rows * cols);
    cactus_rms_norm_f16(x.data(), w.data(), out.data(), rows, cols, eps);

    std::vector<double> ref(rows * cols, 0.0);
    for (size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (size_t i = 0; i < cols; ++i) sum += xd[r * cols + i] * xd[r * cols + i];
        const double inv = 1.0 / std::sqrt(sum / cols + eps);
        for (size_t i = 0; i < cols; ++i) ref[r * cols + i] = xd[r * cols + i] * wd[i] * inv;
    }

    check_fp16("rms_norm_f16", out, ref, 1e-3, 1e-2);
}

// ---------------------------------------------------------------------------
// quant_matmul (three configurations)
// ---------------------------------------------------------------------------

// Reorder logical norms [n*ng+g] -> interleaved panel-major [(n/4*ng+g)*4 + n%4]
// (the on-disk layout cq.py writes: norms.reshape(n//4,4,groups).transpose(0,2,1)).
std::vector<double> layout_norms_panel_major(const std::vector<double>& logical,
                                             uint32_t N, uint32_t ng) {
    std::vector<double> out(logical.size());
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t g = 0; g < ng; ++g)
            out[((n / 4u) * ng + g) * 4u + (n & 3u)] = logical[n * ng + g];
    return out;
}

// Build a synthetic codebook + norms + activation and run one quant_matmul
// configuration against the double dequantize reference.
void run_quant_matmul_case(const char* name, bool interleaved, bool use_signs_perm,
                           bool use_input_scale /* else input_scale_recip */,
                           uint32_t num_groups) {
    const uint32_t bits = 4, gs = 128, K = gs * num_groups, N = 64, M = 2;

    std::vector<double> codebook_d = gen_double(16, 0x55550001u, -1.0, 1.0);
    std::vector<__fp16> codebook = from_double(codebook_d);

    // Logical norms [n*ng+g]; stored panel-major for interleaved (matches cq.py).
    std::vector<double> norms_d = gen_double(N * num_groups, 0x55550002u, 0.5, 1.5);
    if (interleaved) norms_d = layout_norms_panel_major(norms_d, N, num_groups);
    std::vector<__fp16> norms = from_double(norms_d);

    std::vector<double> recip_d, input_scale_d;
    std::vector<__fp16> recip, input_scale;
    if (use_input_scale) {
        input_scale_d = gen_double(K, 0x55550003u, 0.5, 1.5);
        input_scale = from_double(input_scale_d);
    } else {
        recip_d = gen_double(K, 0x55550003u, 0.5, 1.5);
        recip = from_double(recip_d);
    }

    std::vector<int8_t> left_signs, right_signs;
    std::vector<uint32_t> permutation;
    const int8_t* ls_ptr = nullptr;
    const int8_t* rs_ptr = nullptr;
    const uint32_t* perm_ptr = nullptr;
    if (use_signs_perm) {
        left_signs = gen_signs(gs, 0x55550004u);
        right_signs = gen_signs(gs, 0x55550005u);
        permutation = gen_permutation(gs, 0x55550006u);
        ls_ptr = left_signs.data();
        rs_ptr = right_signs.data();
        perm_ptr = permutation.data();
    }

    // random indices (N rows x K cols, ng groups)
    std::vector<std::vector<uint32_t>> idx(N, std::vector<uint32_t>(K));
    {
        uint32_t s = 0x55550007u;
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t k = 0; k < K; ++k) {
                s = s * 1664525u + 1013904223u;
                idx[n][k] = (s >> 8) % 16u;
            }
    }

    std::vector<uint8_t> packed;
    if (interleaved) {
        packed = pack_interleaved_4row(idx, N, gs, num_groups);
    } else {
        std::vector<uint32_t> flat(N * K);
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t k = 0; k < K; ++k) flat[n * K + k] = idx[n][k];
        packed = pack_4bit_lsb(flat);
    }

    CactusQuantMatrix W{};
    W.bits = bits;
    W.K = K;
    W.N = N;
    W.group_size = gs;
    W.num_groups = num_groups;
    W.flags = interleaved ? CACTUS_QUANT_FLAG_INTERLEAVED_4ROW : 0;
    W.codebook = codebook.data();
    W.input_scale = use_input_scale ? input_scale.data() : nullptr;
    W.input_scale_recip = use_input_scale ? nullptr : recip.data();
    W.norms = norms.data();
    W.packed_indices = packed.data();
    W.left_signs = ls_ptr;
    W.right_signs = rs_ptr;
    W.permutation = perm_ptr;
    W.rotation = nullptr;
    W.expanded = nullptr;
    W.norm_f32 = nullptr;

    std::vector<__fp16> A = gen_fp16(M * K, 0x55550008u, -1.0, 1.0);
    std::vector<double> Ad = to_double(A);

    std::vector<__fp16> C(M * N);
    cactus_quant_matmul(&W, A.data(), M, C.data());

    // double reference: dequantize each weight row, dot with raw activation.
    std::vector<double> ref(M * N, 0.0);
    for (uint32_t n = 0; n < N; ++n) {
        std::vector<double> wd = ref_dequantize_hadamard_row(
            bits, K, gs, num_groups, n, packed.data(), codebook_d, norms_d, recip_d, input_scale_d,
            ls_ptr, rs_ptr, perm_ptr, interleaved);
        for (uint32_t m = 0; m < M; ++m) {
            double acc = 0.0;
            for (uint32_t k = 0; k < K; ++k) acc += wd[k] * Ad[m * K + k];
            ref[m * N + n] = acc;
        }
    }

    // abs floor 2e-2: the fp16 codebook/norms rounding + fp32 FWHT accumulation
    // produce ~0.3% absolute error at the ~O(7) output scale; near-zero outputs
    // need this floor (rel 1e-2 is the meaningful bound for typical values).
    check_fp16(name, C, ref, 2e-2, 1e-2);
}

void test_quant_matmul_interleaved() {
    // ng=4 so panel-major != row-major (ng=1 is degenerate); K = gs*ng = 512.
    run_quant_matmul_case("quant_matmul_interleaved", true, false, false, 4);
}

void test_quant_matmul_hadamard() {
    run_quant_matmul_case("quant_matmul_hadamard", false, true, true, 1);
}

void test_quant_matmul_orthogonal() {
    const uint32_t bits = 4, K = 64, N = 32, M = 2;

    std::vector<double> codebook_d = gen_double(16, 0x55560001u, -1.0, 1.0);
    std::vector<__fp16> codebook = from_double(codebook_d);
    std::vector<double> norms_d = gen_double(N, 0x55560002u, 0.5, 1.5);
    std::vector<__fp16> norms = from_double(norms_d);
    std::vector<double> recip_d = gen_double(K, 0x55560003u, 0.5, 1.5);
    std::vector<__fp16> recip = from_double(recip_d);
    // rotation: K x K, row-major [k][i]
    std::vector<__fp16> rotation = gen_fp16(K * K, 0x55560004u, -0.5, 0.5);

    std::vector<std::vector<uint32_t>> idx(N, std::vector<uint32_t>(K));
    {
        uint32_t s = 0x55560005u;
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t k = 0; k < K; ++k) {
                s = s * 1664525u + 1013904223u;
                idx[n][k] = (s >> 8) % 16u;
            }
    }
    std::vector<uint32_t> flat(N * K);
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t k = 0; k < K; ++k) flat[n * K + k] = idx[n][k];
    std::vector<uint8_t> packed = pack_4bit_lsb(flat);

    CactusQuantMatrix W{};
    W.bits = bits;
    W.K = K;
    W.N = N;
    W.group_size = K; // single group spans K (orthogonal tied embedding)
    W.num_groups = 1;
    W.flags = CACTUS_QUANT_FLAG_ORTHOGONAL;
    W.codebook = codebook.data();
    W.input_scale = nullptr;
    W.input_scale_recip = recip.data();
    W.norms = norms.data();
    W.packed_indices = packed.data();
    W.left_signs = nullptr;
    W.right_signs = nullptr;
    W.permutation = nullptr;
    W.rotation = rotation.data();
    W.expanded = nullptr;
    W.norm_f32 = nullptr;

    std::vector<__fp16> A = gen_fp16(M * K, 0x55560006u, -1.0, 1.0);
    std::vector<double> Ad = to_double(A);

    std::vector<__fp16> C(M * N);
    cactus_quant_matmul(&W, A.data(), M, C.data());

    // A_rot[m][i] = sum_k A[m][k] * recip[k] * rotation[k][i]
    std::vector<double> A_rot(M * K, 0.0);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k) {
            const double av = Ad[m * K + k] * recip_d[k];
            for (uint32_t i = 0; i < K; ++i)
                A_rot[m * K + i] += av * static_cast<double>(rotation[k * K + i]);
        }
    // C[m][n] = norm[n] * sum_i cb[idx(n,i)] * A_rot[m][i]
    std::vector<double> ref(M * N, 0.0);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (uint32_t i = 0; i < K; ++i)
                acc += codebook_d[idx[n][i]] * A_rot[m * K + i];
            ref[m * N + n] = acc * norms_d[n];
        }

    check_fp16("quant_matmul_orthogonal", C, ref, 2e-2, 1e-2);
}

void test_quant_matmul_orthogonal_interleaved() {
    // Orthogonal + INTERLEAVED_4ROW lm_head (single group, gs == K). Weights are
    // 4-row interleaved panels; norms are panel-major (n/4)*4 + (n&3) which for
    // ng=1 equals n, so the logical norm order is used directly.
    const uint32_t bits = 4, K = 64, N = 32, M = 2;

    std::vector<double> codebook_d = gen_double(16, 0x55570001u, -1.0, 1.0);
    std::vector<__fp16> codebook = from_double(codebook_d);
    std::vector<double> norms_d = gen_double(N, 0x55570002u, 0.5, 1.5);
    std::vector<__fp16> norms = from_double(norms_d);
    std::vector<double> recip_d = gen_double(K, 0x55570003u, 0.5, 1.5);
    std::vector<__fp16> recip = from_double(recip_d);
    // rotation: K x K, row-major [k][i]
    std::vector<__fp16> rotation = gen_fp16(K * K, 0x55570004u, -0.5, 0.5);

    std::vector<std::vector<uint32_t>> idx(N, std::vector<uint32_t>(K));
    {
        uint32_t s = 0x55570005u;
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t k = 0; k < K; ++k) {
                s = s * 1664525u + 1013904223u;
                idx[n][k] = (s >> 8) % 16u;
            }
    }
    // interleaved 4-row panel packing (single group, gs == K).
    std::vector<uint8_t> packed = pack_interleaved_4row(idx, N, K, 1);

    CactusQuantMatrix W{};
    W.bits = bits;
    W.K = K;
    W.N = N;
    W.group_size = K; // single group spans K (orthogonal tied embedding)
    W.num_groups = 1;
    W.flags = CACTUS_QUANT_FLAG_ORTHOGONAL | CACTUS_QUANT_FLAG_INTERLEAVED_4ROW;
    W.codebook = codebook.data();
    W.input_scale = nullptr;
    W.input_scale_recip = recip.data();
    W.norms = norms.data();
    W.packed_indices = packed.data();
    W.left_signs = nullptr;
    W.right_signs = nullptr;
    W.permutation = nullptr;
    W.rotation = rotation.data();
    W.expanded = nullptr;
    W.norm_f32 = nullptr;

    std::vector<__fp16> A = gen_fp16(M * K, 0x55570006u, -1.0, 1.0);
    std::vector<double> Ad = to_double(A);

    std::vector<__fp16> C(M * N);
    cactus_quant_matmul(&W, A.data(), M, C.data());

    // A_rot[m][i] = sum_k A[m][k] * recip[k] * rotation[k][i]
    std::vector<double> A_rot(M * K, 0.0);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k) {
            const double av = Ad[m * K + k] * recip_d[k];
            for (uint32_t i = 0; i < K; ++i)
                A_rot[m * K + i] += av * static_cast<double>(rotation[k * K + i]);
        }
    // C[m][n] = norm[n] * sum_i cb[idx(n,i)] * A_rot[m][i], idx via panel decode.
    std::vector<double> ref(M * N, 0.0);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (uint32_t i = 0; i < K; ++i)
                acc += codebook_d[ref_extract_idx_interleaved_4row(packed.data(), K, n, i)] *
                       A_rot[m * K + i];
            ref[m * N + n] = acc * norms_d[n];
        }

    check_fp16("quant_matmul_orthogonal_interleaved", C, ref, 2e-2, 1e-2);
}

// ---------------------------------------------------------------------------
// embedding dequantization rows
// ---------------------------------------------------------------------------
void test_dequantize_hadamard_embedding() {
    const uint32_t bits = 4, hidden_dim = 512, gs = 128, num_groups = 4, num_rows = 8, row = 2;

    std::vector<double> codebook_d = gen_double(16, 0x66660001u, -1.0, 1.0);
    std::vector<__fp16> codebook = from_double(codebook_d);
    std::vector<double> norms_d = gen_double(num_rows * num_groups, 0x66660002u, 0.5, 1.5);
    std::vector<__fp16> norms = from_double(norms_d);
    std::vector<double> recip_d = gen_double(hidden_dim, 0x66660003u, 0.5, 1.5);
    std::vector<__fp16> recip = from_double(recip_d);
    std::vector<int8_t> left_signs = gen_signs(gs, 0x66660004u);
    std::vector<int8_t> right_signs = gen_signs(gs, 0x66660005u);
    std::vector<uint32_t> permutation = gen_permutation(gs, 0x66660006u);

    const uint32_t pgb = (gs * bits + 7u) / 8u;
    std::vector<uint8_t> packed(num_rows * num_groups * pgb);
    {
        uint32_t s = 0x66660007u;
        for (uint32_t r = 0; r < num_rows; ++r)
            for (uint32_t g = 0; g < num_groups; ++g) {
                std::vector<uint32_t> idx(gs);
                for (uint32_t k = 0; k < gs; ++k) {
                    s = s * 1664525u + 1013904223u;
                    idx[k] = (s >> 8) % 16u;
                }
                std::vector<uint8_t> pb = pack_4bit_lsb(idx);
                std::memcpy(packed.data() + (r * num_groups + g) * pgb, pb.data(), pgb);
            }
    }

    std::vector<__fp16> out(hidden_dim);
    cactus_quant_dequantize_hadamard_embedding_row(
        bits, hidden_dim, gs, num_groups, row, packed.data(), codebook.data(), norms.data(),
        recip.data(), left_signs.data(), right_signs.data(), permutation.data(), out.data());

    std::vector<double> ref = ref_dequantize_hadamard_row(
        bits, hidden_dim, gs, num_groups, row, packed.data(), codebook_d, norms_d, recip_d, {},
        left_signs.data(), right_signs.data(), permutation.data(), false);

    check_fp16("dequantize_hadamard_embedding_row", out, ref, 5e-3, 1e-2);
}

void test_dequantize_orthogonal_embedding() {
    const uint32_t bits = 4, K = 512, num_rows = 8, row = 3;

    std::vector<double> codebook_d = gen_double(16, 0x66670001u, -1.0, 1.0);
    std::vector<__fp16> codebook = from_double(codebook_d);
    std::vector<double> norms_d = gen_double(num_rows, 0x66670002u, 0.5, 1.5);
    std::vector<__fp16> norms = from_double(norms_d);
    std::vector<double> recip_d = gen_double(K, 0x66670003u, 0.5, 1.5);
    std::vector<__fp16> recip = from_double(recip_d);
    std::vector<__fp16> rotation = gen_fp16(K * K, 0x66670004u, -0.5, 0.5);

    const uint32_t pgb = (K * bits + 7u) / 8u;
    std::vector<uint8_t> packed(num_rows * pgb);
    std::vector<uint32_t> idx(K);
    {
        uint32_t s = 0x66670005u;
        for (uint32_t k = 0; k < K; ++k) {
            s = s * 1664525u + 1013904223u;
            idx[k] = (s >> 8) % 16u;
        }
    }
    std::vector<uint8_t> pb = pack_4bit_lsb(idx);
    std::memcpy(packed.data() + row * pgb, pb.data(), pgb);

    std::vector<__fp16> out(K);
    cactus_quant_dequantize_orthogonal_embedding_row(
        bits, K, row, packed.data(), codebook.data(), norms.data(), recip.data(), rotation.data(),
        0, out.data());

    // out[j] = (sum_i cb[idx(row,i)] * rotation[j*K+i]) * norm[row] * recip[j]
    std::vector<double> ref(K, 0.0);
    for (uint32_t j = 0; j < K; ++j) {
        double acc = 0.0;
        for (uint32_t i = 0; i < K; ++i)
            acc += codebook_d[idx[i]] * static_cast<double>(rotation[j * K + i]);
        ref[j] = acc * norms_d[row] * recip_d[j];
    }

    check_fp16("dequantize_orthogonal_embedding_row", out, ref, 2e-2, 1e-2);
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

std::vector<double> ref_attention(const std::vector<double>& Q, const std::vector<double>& K,
                                  const std::vector<double>& V, size_t batch, size_t seq_len,
                                  size_t kv_seq_len, size_t n_qh, size_t n_kvh, size_t head_dim,
                                  size_t v_head_dim, double scale, const std::vector<double>& mask,
                                  bool has_mask, bool mask_is_additive, size_t position_offset,
                                  size_t window_size, bool is_causal) {
    const size_t gqa = n_qh / n_kvh;
    std::vector<double> O(batch * seq_len * n_qh * v_head_dim, 0.0);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t qh = 0; qh < n_qh; ++qh) {
            const size_t kh = qh / gqa;
            for (size_t q = 0; q < seq_len; ++q) {
                const size_t abs_q = position_offset + q;
                size_t kv_start = 0, kv_end = kv_seq_len;
                if (window_size > 0 && window_size < kv_seq_len) {
                    if (abs_q > window_size) kv_start = abs_q - window_size;
                    if (is_causal) kv_end = std::min(kv_end, abs_q + 1);
                } else if (is_causal) {
                    kv_end = std::min(kv_end, abs_q + 1);
                }

                double max_score = -std::numeric_limits<double>::infinity();
                std::vector<double> scores(kv_seq_len, -std::numeric_limits<double>::infinity());
                for (size_t kv = 0; kv < kv_seq_len; ++kv) {
                    if (kv < kv_start || kv >= kv_end) continue;
                    double s = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) {
                        const double qv = Q[b * seq_len * n_qh * head_dim + q * n_qh * head_dim + qh * head_dim + d];
                        const double kk = K[b * kv_seq_len * n_kvh * head_dim + kv * n_kvh * head_dim + kh * head_dim + d];
                        s += qv * kk;
                    }
                    s *= scale;
                    if (has_mask) {
                        const double m = mask[q * kv_seq_len + kv];
                        if (mask_is_additive) {
                            if (std::isfinite(m)) s += m; else s = -std::numeric_limits<double>::infinity();
                        } else if (m == 0.0) {
                            s = -std::numeric_limits<double>::infinity();
                        }
                    }
                    scores[kv] = s;
                    max_score = std::max(max_score, s);
                }

                std::vector<double> acc(v_head_dim, 0.0);
                double sum = 0.0;
                if (std::isfinite(max_score)) {
                    for (size_t kv = 0; kv < kv_seq_len; ++kv) {
                        if (scores[kv] == -std::numeric_limits<double>::infinity()) continue;
                        const double w = std::exp(scores[kv] - max_score);
                        sum += w;
                        for (size_t d = 0; d < v_head_dim; ++d) {
                            const double vv = V[b * kv_seq_len * n_kvh * v_head_dim + kv * n_kvh * v_head_dim + kh * v_head_dim + d];
                            acc[d] += w * vv;
                        }
                    }
                }
                double* o = O.data() + b * seq_len * n_qh * v_head_dim + q * n_qh * v_head_dim + qh * v_head_dim;
                if (sum > 0.0) {
                    for (size_t d = 0; d < v_head_dim; ++d) o[d] = acc[d] / sum;
                }
            }
        }
    }
    return O;
}

void test_attention() {
    const size_t batch = 1, seq_len = 16, kv_seq_len = 16, n_qh = 2, n_kvh = 1, head_dim = 32;
    const size_t v_head_dim = 32;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    std::vector<__fp16> Q = gen_fp16(batch * seq_len * n_qh * head_dim, 0x77770001u, -1.0, 1.0);
    std::vector<__fp16> K = gen_fp16(batch * kv_seq_len * n_kvh * head_dim, 0x77770002u, -1.0, 1.0);
    std::vector<__fp16> V = gen_fp16(batch * kv_seq_len * n_kvh * v_head_dim, 0x77770003u, -1.0, 1.0);
    std::vector<double> Qd = to_double(Q), Kd = to_double(K), Vd = to_double(V);

    { // causal
        std::vector<__fp16> out(batch * seq_len * n_qh * v_head_dim);
        cactus_attention_f16(Q.data(), K.data(), V.data(), out.data(), batch, seq_len, kv_seq_len,
                             n_qh, n_kvh, head_dim, static_cast<float>(scale), nullptr, 0, 0, true,
                             false, false, v_head_dim, 0.0f);
        std::vector<double> ref = ref_attention(Qd, Kd, Vd, batch, seq_len, kv_seq_len, n_qh, n_kvh,
                                                head_dim, v_head_dim, scale, {}, false, false, 0, 0, true);
        check_fp16("attention_f16_causal", out, ref, 1e-2, 2e-2);
    }

    { // explicit multiplicative mask (0 = masked)
        std::vector<__fp16> mask = gen_fp16(seq_len * kv_seq_len, 0x77770004u, 0.0, 1.0);
        for (size_t i = 0; i < mask.size(); ++i)
            mask[i] = (static_cast<double>(mask[i]) < 0.5) ? static_cast<__fp16>(0.0f)
                                                           : static_cast<__fp16>(1.0f);
        std::vector<double> maskd = to_double(mask);
        std::vector<__fp16> out(batch * seq_len * n_qh * v_head_dim);
        cactus_attention_f16(Q.data(), K.data(), V.data(), out.data(), batch, seq_len, kv_seq_len,
                             n_qh, n_kvh, head_dim, static_cast<float>(scale), mask.data(), 0, 0, false,
                             false, false, v_head_dim, 0.0f);
        std::vector<double> ref = ref_attention(Qd, Kd, Vd, batch, seq_len, kv_seq_len, n_qh, n_kvh,
                                                head_dim, v_head_dim, scale, maskd, true, false, 0, 0, false);
        check_fp16("attention_f16_masked", out, ref, 1e-2, 2e-2);
    }
}

void test_attention_hybrid() {
    const size_t batch = 1, seq_len = 2, cache_len = 4, new_len = 2, n_qh = 2, n_kvh = 1;
    const size_t head_dim = 32, v_head_dim = 32, group_size = 32;
    const size_t kv_seq_len = cache_len + new_len;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    const size_t num_groups = (head_dim + group_size - 1) / group_size;

    std::vector<__fp16> Q = gen_fp16(batch * seq_len * n_qh * head_dim, 0x77780001u, -1.0, 1.0);
    // cached K/V (fp16, to be quantized)
    std::vector<__fp16> Kcache = gen_fp16(batch * cache_len * n_kvh * head_dim, 0x77780002u, -1.0, 1.0);
    std::vector<__fp16> Vcache = gen_fp16(batch * cache_len * n_kvh * v_head_dim, 0x77780003u, -1.0, 1.0);
    // new K/V (fp16)
    std::vector<__fp16> Knew = gen_fp16(batch * new_len * n_kvh * head_dim, 0x77780004u, -1.0, 1.0);
    std::vector<__fp16> Vnew = gen_fp16(batch * new_len * n_kvh * v_head_dim, 0x77780005u, -1.0, 1.0);

    // quantize cache -> int8 + scales
    std::vector<int8_t> Kq(batch * cache_len * n_kvh * head_dim);
    std::vector<int8_t> Vq(batch * cache_len * n_kvh * v_head_dim);
    std::vector<float> Ksc(batch * cache_len * n_kvh * num_groups);
    std::vector<float> Vsc(batch * cache_len * n_kvh * num_groups);
    cactus_quantize_kv_fp16_to_int8(Kcache.data(), Kq.data(), Ksc.data(), cache_len * n_kvh, 1, head_dim, group_size);
    // (v: same head_dim for this test)
    {
        const size_t num_vgroups = (v_head_dim + group_size - 1) / group_size;
        std::vector<float> Vsc_tmp(batch * cache_len * n_kvh * num_vgroups);
        cactus_quantize_kv_fp16_to_int8(Vcache.data(), Vq.data(), Vsc_tmp.data(), cache_len * n_kvh, 1, v_head_dim, group_size);
        Vsc = std::move(Vsc_tmp);
    }

    std::vector<__fp16> out(batch * seq_len * n_qh * v_head_dim);
    cactus_attention_hybrid_int8_fp16(Q.data(), Kq.data(), Vq.data(), Ksc.data(), Vsc.data(),
                                      Knew.data(), Vnew.data(), out.data(), batch, seq_len, cache_len,
                                      new_len, n_qh, n_kvh, head_dim, static_cast<float>(scale),
                                      cache_len /* position_offset */, true /* is_causal */, 0, group_size,
                                      v_head_dim);

    // Build full fp32 K/V: cached (dequantized) then new.
    std::vector<double> Kfull(batch * kv_seq_len * n_kvh * head_dim, 0.0);
    std::vector<double> Vfull(batch * kv_seq_len * n_kvh * v_head_dim, 0.0);
    for (size_t kv = 0; kv < kv_seq_len; ++kv) {
        const bool cached = kv < cache_len;
        const size_t src = cached ? kv : kv - cache_len;
        for (size_t h = 0; h < n_kvh; ++h) {
            for (size_t d = 0; d < head_dim; ++d) {
                const size_t dst = kv * n_kvh * head_dim + h * head_dim + d;
                if (cached) {
                    const size_t si = src * n_kvh * head_dim + h * head_dim + d;
                    const double scl = static_cast<double>(Ksc[src * num_groups + d / group_size]);
                    Kfull[dst] = static_cast<double>(Kq[si]) * scl;
                } else {
                    const size_t si = src * n_kvh * head_dim + h * head_dim + d;
                    Kfull[dst] = static_cast<double>(Knew[si]);
                }
            }
            for (size_t d = 0; d < v_head_dim; ++d) {
                const size_t dst = kv * n_kvh * v_head_dim + h * v_head_dim + d;
                if (cached) {
                    const size_t si = src * n_kvh * v_head_dim + h * v_head_dim + d;
                    const size_t vnum_groups = (v_head_dim + group_size - 1) / group_size;
                    const double scl = static_cast<double>(Vsc[src * vnum_groups + d / group_size]);
                    Vfull[dst] = static_cast<double>(Vq[si]) * scl;
                } else {
                    const size_t si = src * n_kvh * v_head_dim + h * v_head_dim + d;
                    Vfull[dst] = static_cast<double>(Vnew[si]);
                }
            }
        }
    }

    std::vector<double> Qd = to_double(Q);
    std::vector<double> ref = ref_attention(Qd, Kfull, Vfull, batch, seq_len, kv_seq_len, n_qh, n_kvh,
                                            head_dim, v_head_dim, scale, {}, false, false, cache_len, 0, true);
    check_fp16("attention_hybrid_int8_fp16", out, ref, 1e-2, 2e-2);
}

// ---------------------------------------------------------------------------
// kv quantize roundtrip
// ---------------------------------------------------------------------------
void test_kv_quantize_roundtrip() {
    const size_t seq_len = 2, kv_heads = 1, head_dim = 64, group_size = 32;
    std::vector<__fp16> src = gen_fp16(seq_len * kv_heads * head_dim, 0x88880001u, -1.0, 1.0);
    const size_t num_groups = (head_dim + group_size - 1) / group_size;

    std::vector<int8_t> q(seq_len * kv_heads * head_dim);
    std::vector<float> scales(seq_len * kv_heads * num_groups);
    cactus_quantize_kv_fp16_to_int8(src.data(), q.data(), scales.data(), seq_len, kv_heads, head_dim, group_size);

    std::vector<__fp16> deq(seq_len * kv_heads * head_dim);
    for (size_t idx = 0; idx < seq_len * kv_heads; ++idx) {
        const float* s = scales.data() + idx * num_groups;
        for (size_t g = 0; g < num_groups; ++g) {
            const size_t gstart = g * group_size;
            const size_t gcount = std::min(group_size, head_dim - gstart);
            cactus_int8_to_fp16(q.data() + idx * head_dim + gstart, deq.data() + idx * head_dim + gstart, gcount, s[g]);
        }
    }

    std::vector<double> ref = to_double(src);
    // int8 group quantization inherently loses ~max_abs/127/2 ~ 0.4% absolute;
    // bound the reconstruction by abs 2e-2 (rel 2e-2 for typical magnitudes).
    check_fp16("kv_quantize_roundtrip", deq, ref, 2e-2, 2e-2);
}

// ---------------------------------------------------------------------------
// casts + max_abs
// ---------------------------------------------------------------------------
void test_casts() {
    { // fp16 -> fp32 -> fp16 is bitwise exact
        std::vector<__fp16> src = gen_fp16(1024, 0x99990001u, -100.0, 100.0);
        std::vector<float> f(src.size());
        cactus_fp16_to_fp32(src.data(), f.data(), src.size());
        std::vector<__fp16> back(src.size());
        cactus_fp32_to_fp16(f.data(), back.data(), src.size());
        check_fp16_exact("cast_fp16_fp32_roundtrip", back, src);
    }

    { // fp32 -> fp16 relative error
        std::vector<double> d = gen_double(1024, 0x99990002u, -10.0, 10.0);
        std::vector<float> f(d.size());
        for (size_t i = 0; i < d.size(); ++i) f[i] = static_cast<float>(d[i]);
        std::vector<__fp16> h(f.size());
        cactus_fp32_to_fp16(f.data(), h.data(), f.size());
        std::vector<double> ref(f.size());
        for (size_t i = 0; i < f.size(); ++i) ref[i] = static_cast<double>(f[i]);
        check_fp16("cast_fp32_to_fp16", h, ref, 1e-3, 1e-3);
    }

    { // fp16_max_abs
        std::vector<__fp16> src = gen_fp16(512, 0x99990003u, -5.0, 5.0);
        float got = cactus_fp16_max_abs(src.data(), src.size());
        double ref = 0.0;
        for (auto& v : src) ref = std::max(ref, std::fabs(static_cast<double>(v)));
        const double err = std::fabs(static_cast<double>(got) - ref);
        const bool ok = err <= 1e-4;
        report("fp16_max_abs", ok, ok ? 0.0 : err, err);
    }
}

// ---------------------------------------------------------------------------
// elementwise ops
// ---------------------------------------------------------------------------
void test_elementwise() {
    const size_t n = 4096;
    std::vector<__fp16> a = gen_fp16(n, 0xaaaa0001u, -1.0, 1.0);
    std::vector<__fp16> b = gen_fp16(n, 0xaaaa0002u, 0.5, 1.5); // non-zero for divide
    std::vector<double> ad = to_double(a), bd = to_double(b);
    std::vector<__fp16> out(n);

    cactus_add_f16(a.data(), b.data(), out.data(), n);
    {
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = ad[i] + bd[i];
        check_fp16("add_f16", out, ref, 1e-3, 1e-2);
    }
    cactus_subtract_f16(a.data(), b.data(), out.data(), n);
    {
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = ad[i] - bd[i];
        check_fp16("subtract_f16", out, ref, 1e-3, 1e-2);
    }
    cactus_multiply_f16(a.data(), b.data(), out.data(), n);
    {
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = ad[i] * bd[i];
        check_fp16("multiply_f16", out, ref, 1e-3, 1e-2);
    }
    cactus_divide_f16(a.data(), b.data(), out.data(), n);
    {
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = ad[i] / bd[i];
        check_fp16("divide_f16", out, ref, 1e-3, 1e-2);
    }

    { // add_scaled
        const float scale = 0.7f;
        cactus_add_scaled_f16(a.data(), b.data(), out.data(), n, scale);
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = ad[i] + bd[i] * scale;
        check_fp16("add_scaled_f16", out, ref, 1e-3, 1e-2);
    }
}

void test_scalar_op() {
    // exercise every ScalarOpType enum value (ADD..LOG)
    struct Case { ScalarOpType op; double lo, hi; double scalar; const char* name; };
    const Case cases[] = {
        {ScalarOpType::ADD, -2.0, 2.0, 0.5, "scalar_op_ADD"},
        {ScalarOpType::SUBTRACT, -2.0, 2.0, 0.5, "scalar_op_SUBTRACT"},
        {ScalarOpType::MULTIPLY, -2.0, 2.0, 0.5, "scalar_op_MULTIPLY"},
        {ScalarOpType::DIVIDE, -2.0, 2.0, 0.5, "scalar_op_DIVIDE"},
        {ScalarOpType::ABS, -2.0, 2.0, 0.0, "scalar_op_ABS"},
        {ScalarOpType::EXP, -2.0, 2.0, 0.0, "scalar_op_EXP"},
        {ScalarOpType::POW, 0.2, 2.0, 1.7, "scalar_op_POW"},
        {ScalarOpType::SQRT, 0.1, 2.0, 0.0, "scalar_op_SQRT"},
        {ScalarOpType::COS, -3.0, 3.0, 0.0, "scalar_op_COS"},
        {ScalarOpType::SIN, -3.0, 3.0, 0.0, "scalar_op_SIN"},
        {ScalarOpType::LOG, 0.1, 2.0, 0.0, "scalar_op_LOG"},
    };

    const size_t n = 2048;
    for (const Case& c : cases) {
        std::vector<__fp16> in = gen_fp16(n, 0xaaaa1000u, c.lo, c.hi);
        std::vector<double> ind = to_double(in);
        std::vector<__fp16> out(n);
        cactus_scalar_op_f16(in.data(), out.data(), n, static_cast<float>(c.scalar), c.op);

        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) {
            const double x = ind[i];
            switch (c.op) {
                case ScalarOpType::ADD: ref[i] = x + c.scalar; break;
                case ScalarOpType::SUBTRACT: ref[i] = x - c.scalar; break;
                case ScalarOpType::MULTIPLY: ref[i] = x * c.scalar; break;
                case ScalarOpType::DIVIDE: ref[i] = x / c.scalar; break;
                case ScalarOpType::ABS: ref[i] = std::fabs(x); break;
                case ScalarOpType::EXP: ref[i] = std::exp(x); break;
                case ScalarOpType::POW: ref[i] = std::pow(x, c.scalar); break;
                case ScalarOpType::SQRT: ref[i] = std::sqrt(x); break;
                case ScalarOpType::COS: ref[i] = std::cos(x); break;
                case ScalarOpType::SIN: ref[i] = std::sin(x); break;
                case ScalarOpType::LOG: ref[i] = std::log(x); break;
            }
        }
        check_fp16(c.name, out, ref, 1e-3, 2e-2);
    }
}

void test_broadcast() {
    // output_shape {2,3}; a = [2] broadcast over axis1, b = [3] broadcast over axis0
    const size_t out_shape[2] = {2, 3};
    const size_t a_strides[2] = {1, 0};
    const size_t b_strides[2] = {0, 1};
    const size_t ndim = 2;

    std::vector<__fp16> a = gen_fp16(2, 0xaaaa2001u, -1.0, 1.0);
    std::vector<__fp16> b = gen_fp16(3, 0xaaaa2002u, 0.5, 1.5);
    std::vector<double> ad = to_double(a), bd = to_double(b);
    std::vector<__fp16> out(6);

    auto ref2 = [&](char op) {
        std::vector<double> r(6);
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j) {
                const double x = ad[i], y = bd[j];
                switch (op) {
                    case '+': r[i * 3 + j] = x + y; break;
                    case '-': r[i * 3 + j] = x - y; break;
                    case '*': r[i * 3 + j] = x * y; break;
                    case '/': r[i * 3 + j] = x / y; break;
                }
            }
        return r;
    };

    cactus_add_broadcast_f16(a.data(), b.data(), out.data(), a_strides, b_strides, out_shape, ndim);
    check_fp16("add_broadcast_f16", out, ref2('+'), 1e-3, 1e-2);
    cactus_subtract_broadcast_f16(a.data(), b.data(), out.data(), a_strides, b_strides, out_shape, ndim);
    check_fp16("subtract_broadcast_f16", out, ref2('-'), 1e-3, 1e-2);
    cactus_multiply_broadcast_f16(a.data(), b.data(), out.data(), a_strides, b_strides, out_shape, ndim);
    check_fp16("multiply_broadcast_f16", out, ref2('*'), 1e-3, 1e-2);
    cactus_divide_broadcast_f16(a.data(), b.data(), out.data(), a_strides, b_strides, out_shape, ndim);
    check_fp16("divide_broadcast_f16", out, ref2('/'), 1e-3, 1e-2);
}

void test_activations() {
    const size_t n = 2048;

    { // clamp
        std::vector<__fp16> in = gen_fp16(n, 0xaaaa3001u, -3.0, 3.0);
        std::vector<double> ind = to_double(in);
        std::vector<__fp16> out(n);
        const float lo = 0.5f, hi = 2.0f;
        cactus_clamp_f16(in.data(), out.data(), n, lo, hi);
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = std::max(static_cast<double>(lo), std::min(static_cast<double>(hi), ind[i]));
        check_fp16("clamp_f16", out, ref, 1e-3, 1e-2);
    }
    { // sigmoid
        std::vector<__fp16> in = gen_fp16(n, 0xaaaa3002u, -5.0, 5.0);
        std::vector<double> ind = to_double(in);
        std::vector<__fp16> out(n);
        cactus_sigmoid_f16(in.data(), out.data(), n);
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = 1.0 / (1.0 + std::exp(-ind[i]));
        check_fp16("sigmoid_f16", out, ref, 1e-3, 2e-2);
    }
    { // softcap
        std::vector<__fp16> in = gen_fp16(n, 0xaaaa3003u, -5.0, 5.0);
        std::vector<double> ind = to_double(in);
        std::vector<__fp16> out(n);
        const float cap = 10.0f, input_scale = 1.0f;
        cactus_softcap_f16(in.data(), out.data(), n, cap, input_scale);
        std::vector<double> ref(n);
        for (size_t i = 0; i < n; ++i) ref[i] = static_cast<double>(cap) * std::tanh(ind[i] * input_scale / cap);
        check_fp16("softcap_f16", out, ref, 1e-3, 2e-2);
    }
}

// ---------------------------------------------------------------------------
// layout: transpose / concat
// ---------------------------------------------------------------------------
void test_transpose() {
    const size_t R = 5, C = 7;
    std::vector<__fp16> src = gen_fp16(R * C, 0xbbbb0001u, -1.0, 1.0);

    { // transpose_2d full
        std::vector<__fp16> dst(C * R);
        cactus_transpose_2d_f16(src.data(), dst.data(), R, C, 0, R);
        std::vector<__fp16> ref(C * R);
        for (size_t r = 0; r < R; ++r)
            for (size_t c = 0; c < C; ++c) ref[c * R + r] = src[r * C + c];
        check_fp16_exact("transpose_2d_f16", dst, ref);
    }
    { // transpose_f16 2D perm {1,0}
        std::vector<__fp16> dst(C * R);
        const size_t shape[2] = {R, C};
        const size_t perm[2] = {1, 0};
        cactus_transpose_f16(src.data(), dst.data(), shape, perm, 2, 0, R * C);
        std::vector<__fp16> ref(C * R);
        for (size_t r = 0; r < R; ++r)
            for (size_t c = 0; c < C; ++c) ref[c * R + r] = src[r * C + c];
        check_fp16_exact("transpose_f16_2d", dst, ref);
    }
    { // transpose_f16 3D identity permutation -> bitwise copy
        const size_t shape[3] = {2, 3, 4};
        const size_t perm[3] = {0, 1, 2};
        std::vector<__fp16> s3 = gen_fp16(2 * 3 * 4, 0xbbbb0002u, -1.0, 1.0);
        std::vector<__fp16> dst(s3.size());
        cactus_transpose_f16(s3.data(), dst.data(), shape, perm, 3, 0, s3.size());
        check_fp16_exact("transpose_f16_identity", dst, s3);
    }
}

void test_concat() {
    // 2D concat along axis 0: shapes {2,3} + {4,3} -> {6,3}
    {
        std::vector<__fp16> a = gen_fp16(2 * 3, 0xbbbb1001u, -1.0, 1.0);
        std::vector<__fp16> b = gen_fp16(4 * 3, 0xbbbb1002u, -1.0, 1.0);
        std::vector<__fp16> out(6 * 3);
        const size_t s1[2] = {2, 3}, s2[2] = {4, 3}, os[2] = {6, 3};
        cactus_concat_f16(a.data(), b.data(), out.data(), s1, s2, os, 2, 0);
        std::vector<__fp16> ref(6 * 3);
        std::memcpy(ref.data(), a.data(), 2 * 3 * sizeof(__fp16));
        std::memcpy(ref.data() + 2 * 3, b.data(), 4 * 3 * sizeof(__fp16));
        check_fp16_exact("concat_f16_axis0", out, ref);
    }
    // 2D concat along axis 1: shapes {3,2} + {3,4} -> {3,6}
    {
        std::vector<__fp16> a = gen_fp16(3 * 2, 0xbbbb1003u, -1.0, 1.0);
        std::vector<__fp16> b = gen_fp16(3 * 4, 0xbbbb1004u, -1.0, 1.0);
        std::vector<__fp16> out(3 * 6);
        const size_t s1[2] = {3, 2}, s2[2] = {3, 4}, os[2] = {3, 6};
        cactus_concat_f16(a.data(), b.data(), out.data(), s1, s2, os, 2, 1);
        std::vector<__fp16> ref(3 * 6);
        for (size_t r = 0; r < 3; ++r) {
            std::memcpy(ref.data() + r * 6, a.data() + r * 2, 2 * sizeof(__fp16));
            std::memcpy(ref.data() + r * 6 + 2, b.data() + r * 4, 4 * sizeof(__fp16));
        }
        check_fp16_exact("concat_f16_axis1", out, ref);
    }
}

// ---------------------------------------------------------------------------
// softmax / reductions / sample
// ---------------------------------------------------------------------------
void test_softmax() {
    const size_t batch = 2, seq = 3, vocab = 8;
    std::vector<__fp16> in = gen_fp16(batch * seq * vocab, 0xcccc0001u, -2.0, 2.0);
    std::vector<double> ind = to_double(in);
    std::vector<__fp16> out(in.size());
    cactus_softmax_f16(in.data(), out.data(), batch, seq, vocab);

    std::vector<double> ref(in.size(), 0.0);
    for (size_t r = 0; r < batch * seq; ++r) {
        double mx = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < vocab; ++i) mx = std::max(mx, ind[r * vocab + i]);
        double sum = 0.0;
        for (size_t i = 0; i < vocab; ++i) sum += std::exp(ind[r * vocab + i] - mx);
        for (size_t i = 0; i < vocab; ++i) ref[r * vocab + i] = std::exp(ind[r * vocab + i] - mx) / sum;
    }
    check_fp16("softmax_f16", out, ref, 1e-3, 2e-2);
}

void test_reductions() {
    const size_t outer = 3, axis = 4, inner = 2;
    std::vector<__fp16> in = gen_fp16(outer * axis * inner, 0xcccc0002u, -1.0, 1.0);
    std::vector<double> ind = to_double(in);

    std::vector<__fp16> out(outer * inner);
    cactus_sum_axis_f16(in.data(), out.data(), outer, axis, inner);
    {
        std::vector<double> ref(outer * inner, 0.0);
        for (size_t o = 0; o < outer; ++o)
            for (size_t i = 0; i < inner; ++i)
                for (size_t a = 0; a < axis; ++a)
                    ref[o * inner + i] += ind[o * axis * inner + a * inner + i];
        check_fp16("sum_axis_f16", out, ref, 1e-3, 1e-2);
    }
    cactus_mean_axis_f16(in.data(), out.data(), outer, axis, inner);
    {
        std::vector<double> ref(outer * inner, 0.0);
        for (size_t o = 0; o < outer; ++o)
            for (size_t i = 0; i < inner; ++i) {
                double s = 0.0;
                for (size_t a = 0; a < axis; ++a) s += ind[o * axis * inner + a * inner + i];
                ref[o * inner + i] = s / axis;
            }
        check_fp16("mean_axis_f16", out, ref, 1e-3, 1e-2);
    }
}

void test_sample() {
    const size_t vocab = 64;
    std::vector<__fp16> logits = gen_fp16(vocab, 0xdddd0001u, -2.0, 2.0);
    std::vector<double> ld = to_double(logits);

    { // temperature == 0 -> argmax (deterministic)
        uint32_t out = 0xFFFFFFFFu;
        cactus_sample_f16_ex(logits.data(), &out, vocab, 0.0f, 0.9f, 0.0f, 1.0f, 0, 12345u, nullptr, nullptr, 0);
        const uint32_t ref = static_cast<uint32_t>(std::max_element(ld.begin(), ld.end()) - ld.begin());
        const bool ok = out == ref;
        report("sample_f16_ex_argmax", ok, ok ? 0.0 : 1.0, ok ? 0.0 : 1.0);
    }

    { // temperature=0.7, top_k=10, top_p=0.9 -> output in top-10
        const size_t top_k = 10;
        std::vector<size_t> order(vocab);
        for (size_t i = 0; i < vocab; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ld[a] > ld[b]; });

        uint32_t out = 0xFFFFFFFFu;
        cactus_sample_f16_ex(logits.data(), &out, vocab, 0.7f, 0.9f, 0.0f, 1.0f, top_k, 12345u, nullptr, nullptr, 0);

        bool in_top10 = false;
        for (size_t i = 0; i < top_k; ++i) if (order[i] == out) in_top10 = true;
        report("sample_f16_ex_topk", in_top10, in_top10 ? 0.0 : 1.0, in_top10 ? 0.0 : 1.0);
    }
}

} // namespace

int main() {
    std::cout << "cactus_x86: running CPU kernel correctness tests (clang++ __fp16)\n";

    test_matmul();
    test_rms_norm();
    test_quant_matmul_interleaved();
    test_quant_matmul_hadamard();
    test_quant_matmul_orthogonal();
    test_quant_matmul_orthogonal_interleaved();
    test_dequantize_hadamard_embedding();
    test_dequantize_orthogonal_embedding();
    test_attention();
    test_attention_hybrid();
    test_kv_quantize_roundtrip();
    test_casts();
    test_elementwise();
    test_scalar_op();
    test_broadcast();
    test_activations();
    test_transpose();
    test_concat();
    test_softmax();
    test_reductions();
    test_sample();

    if (g_failed) {
        std::cerr << "cactus_x86: FAILED\n";
        return EXIT_FAILURE;
    }
    std::cout << "cactus_x86: ALL PASS\n";
    return EXIT_SUCCESS;
}
