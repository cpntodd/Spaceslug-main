// M6b-1: x86 CPU implementation of the Cactus C-API subset (needle-relevant).
//
// Correct-first, simple, portable (scalar loops only, no NEON/ARM). All math is
// accumulated in fp32; __fp16 is used purely as the I/O storage format. This
// file is the CPU fallback + the reference for the M6c Vulkan GPU port, so
// clarity is prioritized over speed.
//
// The public functions use C++ linkage (matching the upstream cactus_kernels.h
// mangled symbol ABI) at global scope; the implementation helpers live in
// namespace vulkan_runtime::cactus_x86.

#include "cactus/cactus_x86.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace vulkan_runtime {
namespace cactus_x86 {

// --- bit-packed index extraction (LSB-first), mirroring upstream tq_extract_idx
inline uint32_t extract_idx_lsb(const uint8_t* packed, uint32_t k, uint32_t bits) {
    switch (bits) {
        case 4: return (packed[k / 2] >> ((k & 1u) * 4u)) & 0xFu;
        case 2: return (packed[k / 4] >> ((k & 3u) * 2u)) & 0x3u;
        case 1: return (packed[k / 8] >> (k & 7u)) & 0x1u;
        case 3: {
            const uint32_t bit_pos = k * 3u;
            const uint32_t byte_idx = bit_pos / 8u;
            const uint32_t bit_idx = bit_pos & 7u;
            uint32_t val = static_cast<uint32_t>(packed[byte_idx]) >> bit_idx;
            if (bit_idx > 5u) {
                val |= static_cast<uint32_t>(packed[byte_idx + 1u]) << (8u - bit_idx);
            }
            return val & 0x7u;
        }
        default: return 0;
    }
}

// INTERLEAVED_4ROW (4-bit) panel extraction, mirroring upstream
// tq_extract_interleaved_4row_4bit. `panel` points at the start of a
// 4-row panel (panel_bytes = 4 * packed_group_bytes(4, gs)).
inline uint32_t extract_idx_interleaved_4row(const uint8_t* panel, uint32_t row_in_panel, uint32_t k) {
    const uint32_t chunk = k / 8u;
    const uint32_t sub = k & 7u;
    const uint8_t packed_byte = panel[chunk * 16u + row_in_panel * 4u + (sub & 3u)];
    return (sub & 4u) ? static_cast<uint32_t>(packed_byte >> 4)
                      : static_cast<uint32_t>(packed_byte & 0x0Fu);
}

// Normalized (1/sqrt(n)) Walsh-Hadamard transform on a power-of-two vector.
inline void fwht_normalized(float* x, uint32_t n) {
    for (uint32_t h = 1; h < n; h <<= 1) {
        for (uint32_t i = 0; i < n; i += (h << 1)) {
            for (uint32_t j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(n));
    for (uint32_t i = 0; i < n; ++i) x[i] *= inv;
}

// Per-element input-scale reciprocal (input_scale_recip, else 1/input_scale,
// else 1) — mirrors upstream cactus_quant_input_scale_recip1.
inline float input_scale_recip1(const CactusQuantMatrix& W, uint32_t offset) {
    if (W.input_scale_recip != nullptr) return static_cast<float>(W.input_scale_recip[offset]);
    if (W.input_scale != nullptr) return 1.0f / static_cast<float>(W.input_scale[offset]);
    return 1.0f;
}

// Transforms one activation group into the Hadamard "code basis" (fp32).
// Mirrors upstream cactus_quant_transform_hadamard_group:
//   work[k] = x[k] * input_scale_recip * left_signs[k]
//   work = FWHT_normalized(work)
//   work[k] *= right_signs[k]
//   code_basis[j] = work[permutation[j]]   (if permutation present)
void transform_hadamard_group(const CactusQuantMatrix& W, const __fp16* x_group,
                              uint32_t group, float* code_basis) {
    const uint32_t gs = W.group_size;
    float work[256];
    const uint32_t base = group * gs;

    for (uint32_t k = 0; k < gs; ++k) {
        const float sign = W.left_signs ? static_cast<float>(W.left_signs[k]) : 1.0f;
        work[k] = static_cast<float>(x_group[k]) * input_scale_recip1(W, base + k) * sign;
    }

    fwht_normalized(work, gs);

    for (uint32_t k = 0; k < gs; ++k) {
        const float sign = W.right_signs ? static_cast<float>(W.right_signs[k]) : 1.0f;
        work[k] *= sign;
    }

    if (W.permutation != nullptr) {
        float tmp[256];
        for (uint32_t j = 0; j < gs; ++j) tmp[j] = work[W.permutation[j]];
        for (uint32_t j = 0; j < gs; ++j) code_basis[j] = tmp[j];
    } else {
        for (uint32_t j = 0; j < gs; ++j) code_basis[j] = work[j];
    }
}

// Quantize one group of fp16 -> int8 with per-group scale = max_abs / 127
// (mirrors upstream quantize_group_fp16_to_int8).
float quantize_group_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count) {
    float max_abs = 0.0f;
    for (size_t k = 0; k < count; ++k) {
        max_abs = std::max(max_abs, std::fabs(static_cast<float>(src[k])));
    }
    float scale = max_abs / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    const float inv_scale = 1.0f / scale;
    for (size_t k = 0; k < count; ++k) {
        const float val = static_cast<float>(src[k]) * inv_scale;
        const int32_t q = static_cast<int32_t>(std::round(val));
        dst[k] = static_cast<int8_t>(std::max(-128, std::min(127, q)));
    }
    return scale;
}

// Reference semantics for cactus_quant_matmul on a (non-orthogonal) matrix:
// for each output row n, decode the weight group into the Hadamard basis
// (codebook lookup * norm), and dot with the transformed activation.
void quant_matmul_hadamard(const CactusQuantMatrix& W, const __fp16* A, uint32_t M, __fp16* C) {
    const uint32_t gs = W.group_size;
    const uint32_t bits = W.bits;
    const uint32_t num_groups = W.num_groups;
    const bool interleaved = (W.flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0;
    const uint32_t pgb = (gs * bits + 7u) / 8u; // packed_group_bytes(bits, gs)
    const size_t panel_bytes = 4u * pgb;

    for (uint32_t m = 0; m < M; ++m) {
        std::vector<float> basis(static_cast<size_t>(W.K));
        for (uint32_t g = 0; g < num_groups; ++g) {
            transform_hadamard_group(W, A + static_cast<size_t>(m) * W.K + g * gs, g,
                                     basis.data() + g * gs);
        }
        for (uint32_t n = 0; n < W.N; ++n) {
            float acc = 0.0f;
            for (uint32_t g = 0; g < num_groups; ++g) {
                // INTERLEAVED_4ROW norms are stored PANEL-MAJOR in the weight
                // file (cq.py: norms.reshape(n//4,4,groups).transpose(0,2,1) ->
                // index (nb*ng+g)*4+ni), matching upstream matmul.cpp
                // tq_preexpand_weights_interleaved. Non-interleaved stays
                // row-major [n*ng+g].
                const float norm = interleaved
                    ? static_cast<float>(W.norms[((n / 4u) * num_groups + g) * 4u + (n & 3u)])
                    : static_cast<float>(W.norms[static_cast<size_t>(n) * num_groups + g]);
                const float* b = basis.data() + static_cast<size_t>(g) * gs;
                if (interleaved) {
                    const uint8_t* panel =
                        W.packed_indices + ((n / 4u) * num_groups + g) * panel_bytes;
                    const uint32_t row_in_panel = n & 3u;
                    for (uint32_t k = 0; k < gs; ++k) {
                        const uint32_t idx = extract_idx_interleaved_4row(panel, row_in_panel, k);
                        acc += b[k] * norm * static_cast<float>(W.codebook[idx]);
                    }
                } else {
                    const uint8_t* packed =
                        W.packed_indices + (static_cast<size_t>(n) * num_groups + g) * pgb;
                    for (uint32_t k = 0; k < gs; ++k) {
                        const uint32_t idx = extract_idx_lsb(packed, k, bits);
                        acc += b[k] * norm * static_cast<float>(W.codebook[idx]);
                    }
                }
            }
            C[static_cast<size_t>(m) * W.N + n] = static_cast<__fp16>(acc);
        }
    }
}

// Reference semantics for cactus_quant_matmul on an ORTHOGONAL matrix
// (single group spanning K; mirror upstream cactus_quant_orthogonal_matmul):
//   A_rot[m][i] = sum_k A[m][k] * input_scale_recip[k] * rotation[k][i]
//   C[m][n]     = norm[n] * sum_i codebook[idx(n,i)] * A_rot[m][i]
// INTERLEAVED_4ROW lm_head weights are 4-row interleaved panels (decoded via
// extract_idx_interleaved_4row) with panel-major norms, matching upstream.
void quant_matmul_orthogonal(const CactusQuantMatrix& W, const __fp16* A, uint32_t M, __fp16* C) {
    const uint32_t K = W.K;
    const uint32_t bits = W.bits;
    const uint32_t pgb = (K * bits + 7u) / 8u; // packed_group_bytes(bits, K)
    const bool interleaved = (W.flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0;
    const size_t panel_bytes = 4u * pgb;

    std::vector<float> A_rot(static_cast<size_t>(M) * K, 0.0f);
    for (uint32_t m = 0; m < M; ++m) {
        const __fp16* a_row = A + static_cast<size_t>(m) * K;
        float* ar = A_rot.data() + static_cast<size_t>(m) * K;
        for (uint32_t k = 0; k < K; ++k) {
            float a_val = static_cast<float>(a_row[k]);
            if (W.input_scale_recip) a_val *= static_cast<float>(W.input_scale_recip[k]);
            const __fp16* R_row = W.rotation + static_cast<size_t>(k) * K;
            for (uint32_t i = 0; i < K; ++i) {
                ar[i] += a_val * static_cast<float>(R_row[i]);
            }
        }
    }

    for (uint32_t m = 0; m < M; ++m) {
        const float* ar = A_rot.data() + static_cast<size_t>(m) * K;
        for (uint32_t n = 0; n < W.N; ++n) {
            float acc = 0.0f;
            float norm_n;
            if (interleaved) {
                // Orthogonal INTERLEAVED_4ROW lm_head (single group, gs == K):
                // weights are 4-row interleaved panels, norms stored panel-major
                // `norms[(n/4)*4 + (n&3)]` (== norms[n] for ng==1, but written
                // panel-major to mirror upstream cactus_quant_orthogonal_matmul).
                const uint8_t* panel = W.packed_indices + (n / 4u) * panel_bytes;
                norm_n = static_cast<float>(W.norms[(n / 4u) * 4u + (n & 3u)]);
                for (uint32_t i = 0; i < K; ++i) {
                    const uint32_t idx = extract_idx_interleaved_4row(panel, n & 3u, i);
                    acc += static_cast<float>(W.codebook[idx]) * ar[i];
                }
            } else {
                const uint8_t* packed = W.packed_indices + static_cast<size_t>(n) * pgb;
                norm_n = static_cast<float>(W.norms[n]);
                for (uint32_t i = 0; i < K; ++i) {
                    const uint32_t idx = extract_idx_lsb(packed, i, bits);
                    acc += static_cast<float>(W.codebook[idx]) * ar[i];
                }
            }
            C[static_cast<size_t>(m) * W.N + n] = static_cast<__fp16>(acc * norm_n);
        }
    }
}

// --- attention helpers ----------------------------------------------------

// Masked-softmax attention over a full fp32 KV (naive, correct). Used by both
// the fp16 and hybrid attention paths after dequantization.
// `k_f32`/`v_f32` are already in fp32, layout [kv_seq_len][num_kv_heads][dim].
void attention_naive(const __fp16* queries, const float* k_f32, const float* v_f32, __fp16* output,
                     size_t batch_size, size_t seq_len, size_t kv_seq_len, size_t num_q_heads,
                     size_t num_kv_heads, size_t head_dim, size_t v_head_dim, float scale,
                     const __fp16* mask, size_t position_offset, size_t window_size,
                     bool is_causal, bool mask_is_additive, bool mask_per_head, float logit_cap) {
    const size_t gqa = num_q_heads / num_kv_heads;

    const size_t q_batch_stride = seq_len * num_q_heads * head_dim;
    const size_t kv_batch_stride = kv_seq_len * num_kv_heads;
    const size_t o_batch_stride = seq_len * num_q_heads * v_head_dim;
    const size_t q_seq_stride = num_q_heads * head_dim;
    const size_t kv_seq_stride = num_kv_heads;
    const size_t o_seq_stride = num_q_heads * v_head_dim;
    const size_t mask_batch_stride =
        mask ? (mask_per_head ? num_q_heads * seq_len * kv_seq_len : seq_len * kv_seq_len) : 0;

    std::vector<float> scores(kv_seq_len);
    std::vector<float> o_acc(v_head_dim);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t qh = 0; qh < num_q_heads; ++qh) {
            const size_t kh = qh / gqa;
            for (size_t q = 0; q < seq_len; ++q) {
                const __fp16* qv = queries + b * q_batch_stride + q * q_seq_stride + qh * head_dim;
                const size_t abs_q = position_offset + q;

                size_t kv_start = 0;
                size_t kv_end = kv_seq_len;
                if (window_size > 0 && window_size < kv_seq_len) {
                    if (abs_q > window_size) kv_start = abs_q - window_size;
                    if (is_causal) kv_end = std::min(kv_end, abs_q + 1);
                } else if (is_causal) {
                    kv_end = std::min(kv_end, abs_q + 1);
                }

                const __fp16* head_mask =
                    mask ? mask + b * mask_batch_stride + (mask_per_head ? qh * seq_len * kv_seq_len : 0)
                         : nullptr;

                float max_score = -std::numeric_limits<float>::infinity();
                for (size_t kv = 0; kv < kv_seq_len; ++kv) {
                    float s = -std::numeric_limits<float>::infinity();
                    if (kv >= kv_start && kv < kv_end) {
                        s = 0.0f;
                        const float* kv_row = k_f32 + b * kv_batch_stride * head_dim +
                                              kv * kv_seq_stride * head_dim + kh * head_dim;
                        for (size_t d = 0; d < head_dim; ++d) {
                            s += static_cast<float>(qv[d]) * kv_row[d];
                        }
                        s *= scale;

                        if (head_mask) {
                            const float m = static_cast<float>(head_mask[q * kv_seq_len + kv]);
                            if (mask_is_additive) {
                                if (std::isfinite(m)) s += m; else s = -std::numeric_limits<float>::infinity();
                            } else if (m == 0.0f) {
                                s = -std::numeric_limits<float>::infinity();
                            }
                        }
                        if (logit_cap > 0.0f && std::isfinite(s)) {
                            s = logit_cap * std::tanh(s / logit_cap);
                        }
                    }
                    scores[kv] = s;
                    max_score = std::max(max_score, s);
                }

                std::fill(o_acc.begin(), o_acc.end(), 0.0f);
                float sum = 0.0f;
                if (std::isfinite(max_score)) {
                    for (size_t kv = 0; kv < kv_seq_len; ++kv) {
                        if (scores[kv] == -std::numeric_limits<float>::infinity()) continue;
                        const float w = std::exp(scores[kv] - max_score);
                        sum += w;
                        const float* vv = v_f32 + b * kv_batch_stride * v_head_dim +
                                          kv * kv_seq_stride * v_head_dim + kh * v_head_dim;
                        for (size_t d = 0; d < v_head_dim; ++d) {
                            o_acc[d] += w * vv[d];
                        }
                    }
                }

                __fp16* ov = output + b * o_batch_stride + q * o_seq_stride + qh * v_head_dim;
                if (sum > 0.0f) {
                    const float inv = 1.0f / sum;
                    for (size_t d = 0; d < v_head_dim; ++d) {
                        ov[d] = static_cast<__fp16>(o_acc[d] * inv);
                    }
                } else {
                    for (size_t d = 0; d < v_head_dim; ++d) ov[d] = static_cast<__fp16>(0.0f);
                }
            }
        }
    }
}

} // namespace cactus_x86
} // namespace vulkan_runtime

using namespace vulkan_runtime::cactus_x86;

// ---------------------------------------------------------------------------
// Public C-ABI implementations (C++ linkage — see cactus_x86.h).
// ---------------------------------------------------------------------------

void cactus_matmul_f16(const __fp16* a, const __fp16* b_transposed, __fp16* c, size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            const __fp16* ar = a + m * K;
            const __fp16* br = b_transposed + n * K;
            for (size_t k = 0; k < K; ++k) {
                acc += static_cast<float>(ar[k]) * static_cast<float>(br[k]);
            }
            c[m * N + n] = static_cast<__fp16>(acc);
        }
    }
}

void cactus_rms_norm_f16(const __fp16* input, const __fp16* weight, __fp16* output,
                         size_t batch_size, size_t dims, float eps) {
    for (size_t b = 0; b < batch_size; ++b) {
        const __fp16* in = input + b * dims;
        __fp16* out = output + b * dims;
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dims; ++i) {
            const float v = static_cast<float>(in[i]);
            sum_sq += v * v;
        }
        const float rms = std::sqrt(sum_sq / static_cast<float>(dims) + eps);
        const float inv = 1.0f / rms;
        for (size_t i = 0; i < dims; ++i) {
            out[i] = static_cast<__fp16>(static_cast<float>(in[i]) * inv * static_cast<float>(weight[i]));
        }
    }
}

uint32_t cactus_quant_packed_group_bytes(uint32_t bits, uint32_t group_size) {
    if (bits == 0 || bits > 4) return 0;
    return (group_size * bits + 7) / 8;
}

void cactus_quant_matmul(const CactusQuantMatrix* W, const __fp16* A, uint32_t M, __fp16* C) {
    if (!W || !A || !C || M == 0) return;
    if (W->bits == 0 || W->bits > 4) return;
    if (W->K == 0 || W->N == 0 || W->group_size == 0 || W->num_groups == 0) return;
    if (W->codebook == nullptr || W->norms == nullptr || W->packed_indices == nullptr) return;

    if ((W->flags & CACTUS_QUANT_FLAG_ORTHOGONAL) != 0) {
        if (W->rotation == nullptr) return;
        quant_matmul_orthogonal(*W, A, M, C);
        return;
    }
    if (W->K != W->group_size * W->num_groups) return;
    quant_matmul_hadamard(*W, A, M, C);
}

void cactus_quant_matmul_pair(const CactusQuantMatrix* W0, const CactusQuantMatrix* W1,
                              const __fp16* A, uint32_t M, __fp16* C0, __fp16* C1) {
    // The upstream sharing optimization (one activation transform for two
    // interleaved CQ4 matrices) is a perf concern only; correctness falls back
    // to two independent matmuls.
    cactus_quant_matmul(W0, A, M, C0);
    cactus_quant_matmul(W1, A, M, C1);
}

void cactus_quant_matmul_triple(const CactusQuantMatrix* W0, const CactusQuantMatrix* W1,
                                const CactusQuantMatrix* W2, const __fp16* A, uint32_t M,
                                __fp16* C0, __fp16* C1, __fp16* C2) {
    cactus_quant_matmul(W0, A, M, C0);
    cactus_quant_matmul(W1, A, M, C1);
    cactus_quant_matmul(W2, A, M, C2);
}

void cactus_quant_dequantize_hadamard_embedding_row(
    uint32_t bits, uint32_t hidden_dim, uint32_t group_size, uint32_t num_groups, size_t row,
    const uint8_t* packed_base, const __fp16* codebook, const __fp16* norms,
    const __fp16* input_scale_recip, const int8_t* left_signs, const int8_t* right_signs,
    const uint32_t* permutation, __fp16* out_row) {
    if (!packed_base || !codebook || !norms || !out_row || bits == 0 || bits > 4) return;
    if (hidden_dim == 0 || group_size == 0 || num_groups == 0) return;
    if ((group_size & (group_size - 1)) != 0) return; // power of two
    if (group_size > 256) return;

    const uint32_t packed_group_bytes = (group_size * bits + 7) / 8;
    std::vector<float> rotated(group_size);
    for (uint32_t g = 0; g < num_groups; ++g) {
        std::fill(rotated.begin(), rotated.end(), 0.0f);
        const uint8_t* packed =
            packed_base + (row * num_groups + g) * packed_group_bytes;
        for (uint32_t k = 0; k < group_size; ++k) {
            const uint32_t idx = extract_idx_lsb(packed, k, bits);
            const uint32_t dst = permutation ? permutation[k] : k;
            const float rs = right_signs ? static_cast<float>(right_signs[dst]) : 1.0f;
            rotated[dst] = static_cast<float>(codebook[idx]) * rs;
        }
        fwht_normalized(rotated.data(), group_size);
        const float norm = static_cast<float>(norms[row * num_groups + g]);
        for (uint32_t k = 0; k < group_size; ++k) {
            const uint32_t col = g * group_size + k;
            const float ls = left_signs ? static_cast<float>(left_signs[k]) : 1.0f;
            const float sc = input_scale_recip ? static_cast<float>(input_scale_recip[col]) : 1.0f;
            out_row[col] = static_cast<__fp16>(rotated[k] * ls * norm * sc);
        }
    }
    for (uint32_t col = num_groups * group_size; col < hidden_dim; ++col) {
        out_row[col] = static_cast<__fp16>(0);
    }
}

void cactus_quant_dequantize_orthogonal_embedding_row(
    uint32_t bits, uint32_t K, size_t row, const uint8_t* packed_base, const __fp16* codebook,
    const __fp16* norms, const __fp16* input_scale_recip, const __fp16* rotation, uint32_t flags,
    __fp16* out_row) {
    if (!packed_base || !codebook || !norms || !rotation || !out_row || bits == 0 || bits > 4) return;
    if (K == 0) return;

    const uint32_t packed_group_bytes = (K * bits + 7) / 8;
    const bool interleaved = (flags & CACTUS_QUANT_FLAG_INTERLEAVED_4ROW) != 0;
    if (interleaved && (bits != 4 || (K % 8) != 0)) return;

    const uint8_t* packed =
        interleaved ? nullptr : packed_base + row * packed_group_bytes;
    const float norm = interleaved
        ? static_cast<float>(norms[(row / 4) * 4 + (row & 3u)])
        : static_cast<float>(norms[row]);

    std::vector<float> dq(K);
    for (uint32_t i = 0; i < K; ++i) {
        uint32_t idx = 0;
        if (interleaved) {
            const size_t panel_bytes = 4u * packed_group_bytes;
            const uint8_t* panel = packed_base + (row / 4) * panel_bytes;
            idx = extract_idx_interleaved_4row(panel, row & 3u, i);
        } else {
            idx = extract_idx_lsb(packed, i, bits);
        }
        dq[i] = static_cast<float>(codebook[idx]);
    }

    // out_row[j] = (sum_i rotation[j*K+i] * dq[i]) * norm * input_scale_recip[j]
    for (uint32_t j = 0; j < K; ++j) {
        const __fp16* R_row = rotation + static_cast<size_t>(j) * K;
        float acc = 0.0f;
        for (uint32_t i = 0; i < K; ++i) {
            acc += dq[i] * static_cast<float>(R_row[i]);
        }
        const float sc = input_scale_recip ? static_cast<float>(input_scale_recip[j]) : 1.0f;
        out_row[j] = static_cast<__fp16>(acc * norm * sc);
    }
}

void cactus_attention_f16(const __fp16* queries, const __fp16* keys, const __fp16* values,
                          __fp16* output, size_t batch_size, size_t seq_len, size_t kv_seq_len,
                          size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale,
                          const __fp16* mask, size_t position_offset, size_t window_size,
                          bool is_causal, bool mask_is_additive, bool mask_per_head,
                          size_t v_head_dim, float logit_cap) {
    if (v_head_dim == 0) v_head_dim = head_dim;

    // Expand K/V into fp32 contiguous buffers: [kv_seq_len][num_kv_heads][dim].
    std::vector<float> k_f32(batch_size * kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> v_f32(batch_size * kv_seq_len * num_kv_heads * v_head_dim);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t kv = 0; kv < kv_seq_len; ++kv) {
            for (size_t h = 0; h < num_kv_heads; ++h) {
                const __fp16* ks = keys + b * kv_seq_len * num_kv_heads * head_dim +
                                   kv * num_kv_heads * head_dim + h * head_dim;
                float* kd = k_f32.data() + (b * kv_seq_len * num_kv_heads + kv * num_kv_heads + h) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) kd[d] = static_cast<float>(ks[d]);

                const __fp16* vs = values + b * kv_seq_len * num_kv_heads * v_head_dim +
                                   kv * num_kv_heads * v_head_dim + h * v_head_dim;
                float* vd = v_f32.data() + (b * kv_seq_len * num_kv_heads + kv * num_kv_heads + h) * v_head_dim;
                for (size_t d = 0; d < v_head_dim; ++d) vd[d] = static_cast<float>(vs[d]);
            }
        }
    }

    attention_naive(queries, k_f32.data(), v_f32.data(), output, batch_size, seq_len, kv_seq_len,
                    num_q_heads, num_kv_heads, head_dim, v_head_dim, scale, mask, position_offset,
                    window_size, is_causal, mask_is_additive, mask_per_head, logit_cap);
}

void cactus_attention_hybrid_int8_fp16(
    const __fp16* queries, const int8_t* keys_cached, const int8_t* values_cached,
    const float* k_scales, const float* v_scales, const __fp16* keys_new, const __fp16* values_new,
    __fp16* output, size_t batch_size, size_t seq_len, size_t cache_len, size_t new_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale, size_t position_offset,
    bool is_causal, size_t window_size, size_t group_size, size_t v_head_dim) {
    if (v_head_dim == 0) v_head_dim = head_dim;

    const size_t kv_seq_len = cache_len + new_len;
    const size_t num_groups = (head_dim + group_size - 1) / group_size;

    // Full fp32 KV: cached (dequantized int8) then new (fp16 -> fp32).
    std::vector<float> k_f32(batch_size * kv_seq_len * num_kv_heads * head_dim);
    std::vector<float> v_f32(batch_size * kv_seq_len * num_kv_heads * v_head_dim);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t kv = 0; kv < kv_seq_len; ++kv) {
            const bool cached = kv < cache_len;
            const size_t src_kv = cached ? kv : kv - cache_len;
            for (size_t h = 0; h < num_kv_heads; ++h) {
                const size_t row = (b * kv_seq_len + kv) * num_kv_heads + h;
                float* kd = k_f32.data() + row * head_dim;
                if (cached) {
                    const int8_t* ks = keys_cached +
                        (b * cache_len * num_kv_heads + src_kv * num_kv_heads + h) * head_dim;
                    const float* ksc = k_scales + (src_kv * num_kv_heads + h) * num_groups;
                    for (size_t g = 0; g < num_groups; ++g) {
                        const size_t gstart = g * group_size;
                        const size_t gcount = std::min(group_size, head_dim - gstart);
                        for (size_t d = 0; d < gcount; ++d) {
                            kd[gstart + d] = static_cast<float>(ks[gstart + d]) * ksc[g];
                        }
                    }
                } else {
                    const __fp16* ks = keys_new +
                        (b * new_len * num_kv_heads + src_kv * num_kv_heads + h) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) kd[d] = static_cast<float>(ks[d]);
                }

                const size_t vrow = (b * kv_seq_len + kv) * num_kv_heads + h;
                float* vd = v_f32.data() + vrow * v_head_dim;
                if (cached) {
                    const int8_t* vs = values_cached +
                        (b * cache_len * num_kv_heads + src_kv * num_kv_heads + h) * v_head_dim;
                    const float* vsc = v_scales + (src_kv * num_kv_heads + h) * num_groups;
                    for (size_t g = 0; g < num_groups; ++g) {
                        const size_t gstart = g * group_size;
                        const size_t gcount = std::min(group_size, v_head_dim - gstart);
                        for (size_t d = 0; d < gcount; ++d) {
                            vd[gstart + d] = static_cast<float>(vs[gstart + d]) * vsc[g];
                        }
                    }
                } else {
                    const __fp16* vs = values_new +
                        (b * new_len * num_kv_heads + src_kv * num_kv_heads + h) * v_head_dim;
                    for (size_t d = 0; d < v_head_dim; ++d) vd[d] = static_cast<float>(vs[d]);
                }
            }
        }
    }

    attention_naive(queries, k_f32.data(), v_f32.data(), output, batch_size, seq_len, kv_seq_len,
                    num_q_heads, num_kv_heads, head_dim, v_head_dim, scale, nullptr, position_offset,
                    window_size, is_causal, false, false, 0.0f);
}

void cactus_quantize_kv_fp16_to_int8(const __fp16* src, int8_t* dst, float* scales,
                                     size_t seq_len, size_t kv_heads, size_t head_dim,
                                     size_t group_size) {
    const size_t num_groups = (head_dim + group_size - 1) / group_size;
    for (size_t idx = 0; idx < seq_len * kv_heads; ++idx) {
        const __fp16* head_src = src + idx * head_dim;
        int8_t* head_dst = dst + idx * head_dim;
        float* head_scales = scales + idx * num_groups;
        for (size_t g = 0; g < num_groups; ++g) {
            const size_t gstart = g * group_size;
            const size_t gcount = std::min(group_size, head_dim - gstart);
            head_scales[g] = quantize_group_fp16_to_int8(head_src + gstart, head_dst + gstart, gcount);
        }
    }
}

void cactus_int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<__fp16>(static_cast<float>(src[i]) * scale);
    }
}

void cactus_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale) {
    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < count; ++i) {
        const float q = static_cast<float>(src[i]) * inv_scale;
        dst[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, q))));
    }
}

void cactus_fp16_to_fp32(const __fp16* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(src[i]);
}

void cactus_fp32_to_fp16(const float* src, __fp16* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<__fp16>(src[i]);
}

float cactus_fp16_max_abs(const __fp16* src, size_t count) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        max_abs = std::max(max_abs, std::fabs(static_cast<float>(src[i])));
    }
    return max_abs;
}

void cactus_add_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        output[i] = static_cast<__fp16>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    }
}

void cactus_subtract_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        output[i] = static_cast<__fp16>(static_cast<float>(a[i]) - static_cast<float>(b[i]));
    }
}

void cactus_multiply_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        output[i] = static_cast<__fp16>(static_cast<float>(a[i]) * static_cast<float>(b[i]));
    }
}

void cactus_divide_f16(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        output[i] = static_cast<__fp16>(static_cast<float>(a[i]) / static_cast<float>(b[i]));
    }
}

void cactus_add_scaled_f16(const __fp16* base, const __fp16* src, __fp16* output,
                           size_t num_elements, float scale) {
    for (size_t i = 0; i < num_elements; ++i) {
        output[i] = static_cast<__fp16>(static_cast<float>(base[i]) + static_cast<float>(src[i]) * scale);
    }
}

void cactus_scalar_op_f16(const __fp16* input, __fp16* output, size_t num_elements,
                          float scalar_value, ScalarOpType op_type) {
    const float s = scalar_value;
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]);
        float r = 0.0f;
        switch (op_type) {
            case ScalarOpType::ADD:      r = x + s; break;
            case ScalarOpType::SUBTRACT: r = x - s; break;
            case ScalarOpType::MULTIPLY: r = x * s; break;
            case ScalarOpType::DIVIDE:   r = x / s; break;
            case ScalarOpType::ABS:      r = std::fabs(x); break;
            case ScalarOpType::EXP:      r = std::exp(x); break;
            case ScalarOpType::POW:      r = std::pow(x, s); break;
            case ScalarOpType::SQRT:     r = std::sqrt(x); break;
            case ScalarOpType::COS:      r = std::cos(x); break;
            case ScalarOpType::SIN:      r = std::sin(x); break;
            case ScalarOpType::LOG:      r = std::log(x); break;
            default:                     r = x; break;
        }
        output[i] = static_cast<__fp16>(r);
    }
}

namespace {
// Generic N-D broadcast: output is contiguous row-major; a/b are indexed via
// their strides (stride 0 = broadcast that dim). Mirrors upstream
// broadcast_op_optimized's general path.
void broadcast_binop(const __fp16* a, const __fp16* b, __fp16* output, const size_t* a_strides,
                     const size_t* b_strides, const size_t* output_shape, size_t ndim, char op) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) total *= output_shape[i];
    if (total == 0) return;

    std::vector<size_t> coords(ndim, 0);
    for (size_t lin = 0; lin < total; ++lin) {
        size_t ai = 0, bi = 0;
        for (size_t d = 0; d < ndim; ++d) {
            ai += coords[d] * a_strides[d];
            bi += coords[d] * b_strides[d];
        }
        const float x = static_cast<float>(a[ai]);
        const float y = static_cast<float>(b[bi]);
        float r = 0.0f;
        switch (op) {
            case '+': r = x + y; break;
            case '-': r = x - y; break;
            case '*': r = x * y; break;
            case '/': r = x / y; break;
        }
        output[lin] = static_cast<__fp16>(r);

        // increment coords (row-major, least-significant last)
        for (size_t d = ndim; d-- > 0;) {
            if (++coords[d] < output_shape[d]) break;
            coords[d] = 0;
        }
    }
}
} // namespace

void cactus_add_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                              const size_t* a_strides, const size_t* b_strides,
                              const size_t* output_shape, size_t ndim) {
    broadcast_binop(a, b, output, a_strides, b_strides, output_shape, ndim, '+');
}
void cactus_subtract_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    broadcast_binop(a, b, output, a_strides, b_strides, output_shape, ndim, '-');
}
void cactus_multiply_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                   const size_t* a_strides, const size_t* b_strides,
                                   const size_t* output_shape, size_t ndim) {
    broadcast_binop(a, b, output, a_strides, b_strides, output_shape, ndim, '*');
}
void cactus_divide_broadcast_f16(const __fp16* a, const __fp16* b, __fp16* output,
                                 const size_t* a_strides, const size_t* b_strides,
                                 const size_t* output_shape, size_t ndim) {
    broadcast_binop(a, b, output, a_strides, b_strides, output_shape, ndim, '/');
}

void cactus_clamp_f16(const __fp16* input, __fp16* output, size_t num_elements, float lo, float hi) {
    const float lo_f = lo;
    const float hi_f = hi;
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<__fp16>(std::max(lo_f, std::min(hi_f, x)));
    }
}

void cactus_sigmoid_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<__fp16>(1.0f / (1.0f + std::exp(-x)));
    }
}

void cactus_softcap_f16(const __fp16* input, __fp16* output, size_t num_elements, float cap,
                        float input_scale) {
    if (!(cap > 0.0f)) return;
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]) * input_scale;
        output[i] = static_cast<__fp16>(cap * std::tanh(x / cap));
    }
}

void cactus_transpose_2d_f16(const __fp16* source, __fp16* destination, size_t num_rows,
                             size_t num_cols, size_t start_row, size_t end_row) {
    for (size_t r = start_row; r < end_row; ++r) {
        for (size_t c = 0; c < num_cols; ++c) {
            destination[c * num_rows + r] = source[r * num_cols + c];
        }
    }
}

void cactus_transpose_f16(const __fp16* source, __fp16* destination, const size_t* shape,
                          const size_t* permutation, size_t ndim, size_t start_idx, size_t end_idx) {
    if (ndim == 2 && permutation[0] == 1 && permutation[1] == 0) {
        // Upstream's 2D fast path transposes the full matrix (ignores the
        // start_idx/end_idx element range, which only applies to the general
        // N-D path below).
        const size_t num_rows = shape[0];
        const size_t num_cols = shape[1];
        cactus_transpose_2d_f16(source, destination, num_rows, num_cols, 0, num_rows);
        return;
    }

    // General N-D transpose, mirroring upstream's index math.
    for (size_t idx = start_idx; idx < end_idx; ++idx) {
        size_t src_idx = 0;
        size_t tmp_idx = idx;
        for (size_t i = 0; i < ndim; ++i) {
            const size_t p = permutation[ndim - 1 - i];
            const size_t coord = tmp_idx % shape[p];
            tmp_idx /= shape[p];
            size_t stride = 1;
            for (size_t j = p + 1; j < ndim; ++j) stride *= shape[j];
            src_idx += coord * stride;
        }
        destination[idx] = source[src_idx];
    }
}

void cactus_concat_f16(const __fp16* input1, const __fp16* input2, __fp16* output,
                       const size_t* shape1, const size_t* shape2, const size_t* output_shape,
                       size_t ndims, int axis) {
    if (axis < 0) axis += static_cast<int>(ndims);
    const size_t ax = static_cast<size_t>(axis);

    size_t outer = 1;
    for (size_t i = 0; i < ax; ++i) outer *= output_shape[i];
    size_t inner = 1;
    for (size_t i = ax + 1; i < ndims; ++i) inner *= output_shape[i];

    const size_t a1 = shape1[ax];
    const size_t a2 = shape2[ax];

    size_t outer1 = 1, outer2 = 1;
    for (size_t i = 0; i < ax; ++i) { outer1 *= shape1[i]; outer2 *= shape2[i]; }
    if (outer1 == 0) outer1 = 1;
    if (outer2 == 0) outer2 = 1;

    const size_t s1 = a1 * inner;
    const size_t s2 = a2 * inner;
    const size_t sout = (a1 + a2) * inner;

    for (size_t o = 0; o < outer; ++o) {
        const __fp16* in1 = input1 + (o % outer1) * s1;
        const __fp16* in2 = input2 + (o % outer2) * s2;
        __fp16* out = output + o * sout;
        std::memcpy(out, in1, s1 * sizeof(__fp16));
        std::memcpy(out + s1, in2, s2 * sizeof(__fp16));
    }
}

void cactus_softmax_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t seq_len,
                        size_t vocab_size) {
    const size_t rows = batch_size * seq_len;
    for (size_t r = 0; r < rows; ++r) {
        const __fp16* in = input + r * vocab_size;
        __fp16* out = output + r * vocab_size;

        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < vocab_size; ++i) {
            max_val = std::max(max_val, static_cast<float>(in[i]));
        }
        if (!std::isfinite(max_val)) {
            for (size_t i = 0; i < vocab_size; ++i) out[i] = static_cast<__fp16>(0.0f);
            continue;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            const float e = std::exp(static_cast<float>(in[i]) - max_val);
            out[i] = static_cast<__fp16>(e);
            sum += e;
        }
        const float inv = 1.0f / sum;
        for (size_t i = 0; i < vocab_size; ++i) {
            out[i] = static_cast<__fp16>(static_cast<float>(out[i]) * inv);
        }
    }
}

void cactus_sum_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size,
                         size_t inner_size) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t in = 0; in < inner_size; ++in) {
            float sum = 0.0f;
            for (size_t a = 0; a < axis_size; ++a) {
                sum += static_cast<float>(input[o * axis_size * inner_size + a * inner_size + in]);
            }
            output[o * inner_size + in] = static_cast<__fp16>(sum);
        }
    }
}

void cactus_mean_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size,
                          size_t inner_size) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t in = 0; in < inner_size; ++in) {
            float sum = 0.0f;
            for (size_t a = 0; a < axis_size; ++a) {
                sum += static_cast<float>(input[o * axis_size * inner_size + a * inner_size + in]);
            }
            output[o * inner_size + in] = static_cast<__fp16>(sum / static_cast<float>(axis_size));
        }
    }
}

void cactus_sample_f16_ex(const __fp16* logits, uint32_t* output, size_t vocab_size,
                          float temperature, float top_p, float min_p, float repetition_penalty,
                          size_t top_k, size_t random_seed, const float* bias_values,
                          const uint32_t* bias_indices, size_t bias_count) {
    if (vocab_size == 0) {
        output[0] = 0;
        return;
    }

    std::vector<float> l(vocab_size);
    cactus_fp16_to_fp32(logits, l.data(), vocab_size);

    const bool has_bias = bias_values && bias_indices && bias_count > 0;
    if (has_bias) {
        for (size_t i = 0; i < bias_count; ++i) {
            const uint32_t idx = bias_indices[i];
            if (idx < vocab_size) l[idx] += bias_values[i];
        }
    }

    if (temperature == 0.0f) {
        output[0] = static_cast<uint32_t>(std::max_element(l.begin(), l.end()) - l.begin());
        return;
    }
    if (temperature > 0) {
        for (size_t i = 0; i < vocab_size; ++i) l[i] /= temperature;
    }
    (void)repetition_penalty;

    if (top_k > 0 && top_k < vocab_size) {
        std::vector<std::pair<float, size_t>> pairs;
        pairs.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) pairs.emplace_back(l[i], i);
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        const float kth = pairs[top_k - 1].first;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (l[i] < kth) l[i] = -std::numeric_limits<float>::infinity();
        }
    }

    if (min_p > 0.0f) {
        float max_logit = *std::max_element(l.begin(), l.end());
        if (!std::isinf(max_logit)) {
            std::vector<float> tp(vocab_size, 0.0f);
            float sum = 0.0f;
            for (size_t i = 0; i < vocab_size; ++i) {
                if (!std::isinf(l[i])) {
                    tp[i] = std::exp(l[i] - max_logit);
                    sum += tp[i];
                }
            }
            if (sum > 0.0f) {
                for (size_t i = 0; i < vocab_size; ++i) tp[i] /= sum;
                const float max_prob = *std::max_element(tp.begin(), tp.end());
                const float threshold = max_prob * min_p;
                for (size_t i = 0; i < vocab_size; ++i) {
                    if (tp[i] < threshold) l[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, size_t>> sorted;
        sorted.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            if (!std::isinf(l[i])) sorted.emplace_back(l[i], i);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        const float max_logit = sorted.empty() ? 0.0f : sorted[0].first;
        float sum = 0.0f;
        std::vector<float> tp(sorted.size());
        for (size_t i = 0; i < sorted.size(); ++i) {
            tp[i] = std::exp(sorted[i].first - max_logit);
            sum += tp[i];
        }
        for (size_t i = 0; i < tp.size(); ++i) tp[i] /= sum;

        std::vector<bool> remove(sorted.size(), false);
        float cum = 0.0f;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += tp[i];
            if (cum > top_p) remove[i] = true;
        }
        for (size_t i = 1; i < remove.size(); ++i) remove[i] = remove[i - 1] || remove[i];
        if (!remove.empty()) remove[0] = false;
        for (size_t i = 0; i < sorted.size(); ++i) {
            if (remove[i]) l[sorted[i].second] = -std::numeric_limits<float>::infinity();
        }
    }

    float max_logit = *std::max_element(l.begin(), l.end());
    if (std::isinf(max_logit)) {
        output[0] = 0;
        return;
    }

    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (std::isinf(l[i])) {
            probs[i] = 0.0f;
        } else {
            probs[i] = std::exp(l[i] - max_logit);
            sum += probs[i];
        }
    }
    if (sum == 0.0f) {
        output[0] = 0;
        return;
    }
    for (size_t i = 0; i < vocab_size; ++i) probs[i] /= sum;

    const uint32_t actual_seed = (random_seed == 0) ? std::random_device{}() : static_cast<uint32_t>(random_seed);
    std::mt19937 gen(actual_seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float sample = dist(gen);

    float cumulative = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative += probs[i];
        if (cumulative >= sample) {
            output[0] = static_cast<uint32_t>(i);
            return;
        }
    }
    for (size_t i = vocab_size; i > 0; --i) {
        if (probs[i - 1] > 0.0f) {
            output[0] = static_cast<uint32_t>(i - 1);
            return;
        }
    }
    output[0] = 0;
}

// ---------------------------------------------------------------------------
// M6b-2 additions.
//
// REAL (needle hot path): cactus_rope_f16, cactus_add_f16_clipped.
// REAL (trivial / needle-unused but cheap to get right): activations, norms,
// reductions, casts, quant_orthogonal (delegates to cactus_quant_matmul).
// SAFE STUBS (needle-unused, linker-only): conv/image/fft/mel/spectrogram/lstm/
// deltanet/altup/bilstm/gaussian_topk/stft — write zeros where the output size
// is derivable, otherwise no-op (never executed on the needle path; no OOB).
// ---------------------------------------------------------------------------

void cactus_add_f16_clipped(const __fp16* a, const __fp16* b, __fp16* output, size_t num_elements) {
    constexpr float FP16_MAX = 65500.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        const float v = static_cast<float>(a[i]) + static_cast<float>(b[i]);
        output[i] = static_cast<__fp16>(std::max(-FP16_MAX, std::min(FP16_MAX, v)));
    }
}

// Non-interleaved rotary-position embedding (matches upstream cactus_rope_f16).
// Layout [batch][seq][head][head_dim]; position = start_pos + seq_idx.
void cactus_rope_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t seq_len,
                     size_t num_heads, size_t head_dim, size_t start_pos, float theta) {
    const size_t half_dim = head_dim / 2;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t q = 0; q < seq_len; ++q) {
            const float pos = static_cast<float>(start_pos + q);
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t off = ((b * seq_len + q) * num_heads + h) * head_dim;
                for (size_t i = 0; i < half_dim; ++i) {
                    const float freq =
                        1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
                    const float angle = pos * freq;
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const float x0 = static_cast<float>(input[off + i]);
                    const float x1 = static_cast<float>(input[off + i + half_dim]);
                    output[off + i] = static_cast<__fp16>(x0 * c - x1 * s);
                    output[off + i + half_dim] = static_cast<__fp16>(x1 * c + x0 * s);
                }
            }
        }
    }
}

// GPT-J rope: interleaved [x0,x1] pairs over the first rot_dim elements; the
// remaining head_dim - rot_dim elements are copied through unchanged.
void cactus_gpt_j_rope_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t seq_len,
                           size_t num_heads, size_t head_dim, size_t rot_dim, size_t start_pos,
                           float theta) {
    const size_t half_rot = rot_dim / 2;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t q = 0; q < seq_len; ++q) {
            const float pos = static_cast<float>(start_pos + q);
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t off = ((b * seq_len + q) * num_heads + h) * head_dim;
                for (size_t i = 0; i < half_rot; ++i) {
                    const float freq =
                        1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(rot_dim));
                    const float angle = pos * freq;
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const float x0 = static_cast<float>(input[off + 2 * i]);
                    const float x1 = static_cast<float>(input[off + 2 * i + 1]);
                    output[off + 2 * i] = static_cast<__fp16>(x0 * c - x1 * s);
                    output[off + 2 * i + 1] = static_cast<__fp16>(x1 * c + x0 * s);
                }
                for (size_t d = rot_dim; d < head_dim; ++d) output[off + d] = input[off + d];
            }
        }
    }
}

void cactus_relu_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<__fp16>(x > 0.0f ? x : 0.0f);
    }
}

void cactus_leaky_relu_f16(const __fp16* input, __fp16* output, size_t num_elements,
                           float negative_slope) {
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<__fp16>(x > 0.0f ? x : negative_slope * x);
    }
}

void cactus_silu_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]);
        const float s = 1.0f / (1.0f + std::exp(-x));
        output[i] = static_cast<__fp16>(x * s);
    }
}

void cactus_gelu_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    // tanh approximation (matches common "fast GELU").
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<__fp16>(0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x))));
    }
}

void cactus_gelu_f16_erf(const __fp16* input, __fp16* output, size_t num_elements) {
    const float inv_sqrt2 = 0.70710678118654752440f;
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<__fp16>(0.5f * x * (1.0f + std::erf(x * inv_sqrt2)));
    }
}

void cactus_tanh_f16(const __fp16* input, __fp16* output, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        output[i] = static_cast<__fp16>(std::tanh(static_cast<float>(input[i])));
    }
}

void cactus_gelu_scaled_multiply_f16(const __fp16* gate, const __fp16* up, __fp16* output,
                                     size_t num_elements, float gate_scale, float product_scale) {
    const float k = std::sqrt(2.0f / 3.14159265358979323846f);
    for (size_t i = 0; i < num_elements; ++i) {
        const float x = static_cast<float>(gate[i]) * gate_scale;
        const float g = 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
        output[i] = static_cast<__fp16>(g * static_cast<float>(up[i]) * product_scale);
    }
}

void cactus_layer_norm_f16(const __fp16* input, const __fp16* weight, const __fp16* bias,
                           __fp16* output, size_t batch_size, size_t dims, float eps) {
    for (size_t b = 0; b < batch_size; ++b) {
        const __fp16* in = input + b * dims;
        __fp16* out = output + b * dims;
        double mean = 0.0;
        for (size_t i = 0; i < dims; ++i) mean += static_cast<float>(in[i]);
        mean /= static_cast<double>(dims);
        double var = 0.0;
        for (size_t i = 0; i < dims; ++i) {
            const double d = static_cast<float>(in[i]) - mean;
            var += d * d;
        }
        var /= static_cast<double>(dims);
        const float inv = static_cast<float>(1.0 / std::sqrt(var + static_cast<double>(eps)));
        for (size_t i = 0; i < dims; ++i) {
            const float v = (static_cast<float>(in[i]) - static_cast<float>(mean)) * inv;
            const float w = weight ? static_cast<float>(weight[i]) : 1.0f;
            const float bv = bias ? static_cast<float>(bias[i]) : 0.0f;
            out[i] = static_cast<__fp16>(v * w + bv);
        }
    }
}

void cactus_glu_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t split_size,
                    size_t inner_size) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t s = 0; s < split_size; ++s) {
            for (size_t i = 0; i < inner_size; ++i) {
                const size_t base = o * (2 * split_size) * inner_size;
                const float a = static_cast<float>(input[base + s * inner_size + i]);
                const float b = static_cast<float>(input[base + (split_size + s) * inner_size + i]);
                output[(o * split_size + s) * inner_size + i] =
                    static_cast<__fp16>(a / (1.0f + std::exp(-b)));
            }
        }
    }
}

void cactus_glu_f32(const float* input, float* output, size_t outer_size, size_t split_size,
                    size_t inner_size) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t s = 0; s < split_size; ++s) {
            for (size_t i = 0; i < inner_size; ++i) {
                const size_t base = o * (2 * split_size) * inner_size;
                const float a = input[base + s * inner_size + i];
                const float b = input[base + (split_size + s) * inner_size + i];
                output[(o * split_size + s) * inner_size + i] = a / (1.0f + std::exp(-b));
            }
        }
    }
}

void cactus_int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale) {
    for (size_t i = 0; i < count; ++i) dst[i] = static_cast<float>(src[i]) * scale;
}

void cactus_fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale) {
    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < count; ++i) {
        const float q = src[i] * inv_scale;
        dst[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, q))));
    }
}

double cactus_sum_all_f16(const __fp16* data, size_t num_elements) {
    double sum = 0.0;
    for (size_t i = 0; i < num_elements; ++i) sum += static_cast<float>(data[i]);
    return sum;
}

double cactus_mean_all_f16(const __fp16* data, size_t num_elements) {
    if (num_elements == 0) return 0.0;
    return cactus_sum_all_f16(data, num_elements) / static_cast<double>(num_elements);
}

double cactus_variance_all_f16(const __fp16* data, size_t num_elements) {
    if (num_elements == 0) return 0.0;
    const double mean = cactus_mean_all_f16(data, num_elements);
    double var = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        const double d = static_cast<float>(data[i]) - mean;
        var += d * d;
    }
    return var / static_cast<double>(num_elements);
}

float cactus_min_all_f16(const __fp16* data, size_t num_elements) {
    if (num_elements == 0) return 0.0f;
    float m = static_cast<float>(data[0]);
    for (size_t i = 1; i < num_elements; ++i) m = std::min(m, static_cast<float>(data[i]));
    return m;
}

float cactus_max_all_f16(const __fp16* data, size_t num_elements) {
    if (num_elements == 0) return 0.0f;
    float m = static_cast<float>(data[0]);
    for (size_t i = 1; i < num_elements; ++i) m = std::max(m, static_cast<float>(data[i]));
    return m;
}

void cactus_min_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size,
                         size_t inner_size) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t in = 0; in < inner_size; ++in) {
            float m = std::numeric_limits<float>::infinity();
            for (size_t a = 0; a < axis_size; ++a) {
                m = std::min(m, static_cast<float>(input[o * axis_size * inner_size + a * inner_size + in]));
            }
            output[o * inner_size + in] = static_cast<__fp16>(m);
        }
    }
}

void cactus_max_axis_f16(const __fp16* input, __fp16* output, size_t outer_size, size_t axis_size,
                         size_t inner_size) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t in = 0; in < inner_size; ++in) {
            float m = -std::numeric_limits<float>::infinity();
            for (size_t a = 0; a < axis_size; ++a) {
                m = std::max(m, static_cast<float>(input[o * axis_size * inner_size + a * inner_size + in]));
            }
            output[o * inner_size + in] = static_cast<__fp16>(m);
        }
    }
}

void cactus_variance_axis_f16(const __fp16* input, __fp16* output, size_t outer_size,
                              size_t axis_size, size_t inner_size) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t in = 0; in < inner_size; ++in) {
            double mean = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                mean += static_cast<float>(input[o * axis_size * inner_size + a * inner_size + in]);
            }
            mean /= static_cast<double>(axis_size);
            double var = 0.0;
            for (size_t a = 0; a < axis_size; ++a) {
                const double d = static_cast<float>(input[o * axis_size * inner_size + a * inner_size + in]) - mean;
                var += d * d;
            }
            output[o * inner_size + in] = static_cast<__fp16>(var / static_cast<double>(axis_size));
        }
    }
}

void cactus_quant_orthogonal_matmul(const CactusQuantMatrix* W, const __fp16* A, uint32_t M,
                                    __fp16* C) {
    // cactus_quant_matmul already routes the ORTHOGONAL flag to the orthogonal
    // reference path; orthogonal matmul is just that path.
    cactus_quant_matmul(W, A, M, C);
}

void cactus_sample_f32_ex(const float* logits, uint32_t* output, size_t vocab_size,
                          float temperature, float top_p, float min_p, float repetition_penalty,
                          size_t top_k, size_t random_seed, const float* bias_values,
                          const uint32_t* bias_indices, size_t bias_count) {
    // fp32 logits are narrowed to fp16 and delegated to the (already correct)
    // fp16 sampler; fp32 sampling is not on the needle path (needle is fp16).
    std::vector<__fp16> h(vocab_size);
    cactus_fp32_to_fp16(logits, h.data(), vocab_size);
    cactus_sample_f16_ex(h.data(), output, vocab_size, temperature, top_p, min_p, repetition_penalty,
                         top_k, random_seed, bias_values, bias_indices, bias_count);
}

void cactus_batchnorm_f16(const __fp16* input, const float* weight, const float* bias,
                          const float* running_mean, const float* running_var, __fp16* output,
                          size_t outer_size, size_t channels, size_t inner_size, float epsilon) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t c = 0; c < channels; ++c) {
            const float inv = 1.0f / std::sqrt(running_var[c] + epsilon);
            const float w = weight ? weight[c] : 1.0f;
            const float b = bias ? bias[c] : 0.0f;
            const size_t base = (o * channels + c) * inner_size;
            for (size_t i = 0; i < inner_size; ++i) {
                const float x = static_cast<float>(input[base + i]);
                output[base + i] = static_cast<__fp16>((x - running_mean[c]) * inv * w + b);
            }
        }
    }
}

void cactus_batchnorm_f32(const float* input, const float* weight, const float* bias,
                          const float* running_mean, const float* running_var, float* output,
                          size_t outer_size, size_t channels, size_t inner_size, float epsilon) {
    for (size_t o = 0; o < outer_size; ++o) {
        for (size_t c = 0; c < channels; ++c) {
            const float inv = 1.0f / std::sqrt(running_var[c] + epsilon);
            const float w = weight ? weight[c] : 1.0f;
            const float b = bias ? bias[c] : 0.0f;
            const size_t base = (o * channels + c) * inner_size;
            for (size_t i = 0; i < inner_size; ++i) {
                output[base + i] = (input[base + i] - running_mean[c]) * inv * w + b;
            }
        }
    }
}

void cactus_maxpool1d_f16(const __fp16* input, __fp16* output, size_t batch_size, size_t channels,
                          size_t input_length, size_t kernel_size, size_t stride) {
    const size_t out_length = (input_length - kernel_size) / stride + 1;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            const __fp16* in = input + (b * channels + c) * input_length;
            __fp16* out = output + (b * channels + c) * out_length;
            for (size_t oi = 0; oi < out_length; ++oi) {
                float m = -std::numeric_limits<float>::infinity();
                for (size_t k = 0; k < kernel_size; ++k) {
                    m = std::max(m, static_cast<float>(in[oi * stride + k]));
                }
                out[oi] = static_cast<__fp16>(m);
            }
        }
    }
}

void cactus_bilinear_interpolation_f16(const __fp16* input, __fp16* output, size_t src_height,
                                       size_t src_width, size_t embed_dim, size_t dst_height,
                                       size_t dst_width, bool align_corners) {
    // SAFE STUB (needle-unused): nearest-neighbour copy is not correct bilinear,
    // but this op is only referenced for image models. To avoid a silent garbage
    // output, zero-fill. Not executed on the needle path.
    (void)input; (void)align_corners;
    (void)src_height; (void)src_width; (void)dst_height; (void)dst_width;
    std::fill(output, output + dst_height * dst_width * embed_dim, static_cast<__fp16>(0.0f));
}

void cactus_cat_f16(const __fp16** inputs, __fp16* output, const size_t** input_shapes,
                    const size_t* output_shape, size_t num_inputs, size_t rank, int axis) {
    // SAFE STUB (needle-unused): N-way concat. Zero-fill the output.
    (void)inputs; (void)input_shapes; (void)num_inputs; (void)axis;
    size_t total = 1;
    for (size_t i = 0; i < rank; ++i) total *= output_shape[i];
    std::fill(output, output + total, static_cast<__fp16>(0.0f));
}

// ---- SAFE STUBS: zero-fill where size is derivable, else no-op. ------------

void cactus_altup_predict_f16(const __fp16* coefs, const __fp16* const* streams, __fp16* output,
                              size_t n, size_t seq_len, size_t hidden_dim) {
    (void)coefs; (void)streams; (void)n;
    std::fill(output, output + seq_len * hidden_dim, static_cast<__fp16>(0.0f));
}

void cactus_altup_correct_f16(const __fp16* coefs, const __fp16* innovation,
                              const __fp16* const* predictions, __fp16* output, size_t n,
                              size_t seq_len, size_t hidden_dim) {
    (void)coefs; (void)innovation; (void)predictions; (void)n;
    std::fill(output, output + seq_len * hidden_dim, static_cast<__fp16>(0.0f));
}

void cactus_bilstm_sequence_f16(const __fp16* input, const __fp16* weight_ih_fwd,
                                const __fp16* weight_hh_fwd, const __fp16* bias_ih_fwd,
                                const __fp16* bias_hh_fwd, const __fp16* weight_ih_bwd,
                                const __fp16* weight_hh_bwd, const __fp16* bias_ih_bwd,
                                const __fp16* bias_hh_bwd, __fp16* output, size_t batch_size,
                                size_t seq_len, size_t input_size, size_t hidden_size) {
    (void)input; (void)weight_ih_fwd; (void)weight_hh_fwd; (void)bias_ih_fwd; (void)bias_hh_fwd;
    (void)weight_ih_bwd; (void)weight_hh_bwd; (void)bias_ih_bwd; (void)bias_hh_bwd; (void)input_size;
    std::fill(output, output + batch_size * seq_len * (2 * hidden_size), static_cast<__fp16>(0.0f));
}

void cactus_lstm_cell_f16(const __fp16* x_input, const __fp16* h_prev, const __fp16* c_prev,
                          const __fp16* weight_ih, const __fp16* weight_hh, const __fp16* bias_ih,
                          const __fp16* bias_hh, __fp16* h_new, __fp16* c_new, size_t batch_size,
                          size_t input_size, size_t hidden_size) {
    (void)x_input; (void)h_prev; (void)c_prev; (void)weight_ih; (void)weight_hh; (void)bias_ih;
    (void)bias_hh; (void)input_size;
    std::fill(h_new, h_new + batch_size * hidden_size, static_cast<__fp16>(0.0f));
    std::fill(c_new, c_new + batch_size * hidden_size, static_cast<__fp16>(0.0f));
}

void cactus_gated_deltanet_decode_f16(const __fp16* q_data, const __fp16* k_data,
                                      const __fp16* v_data, const __fp16* g_data, const __fp16* b_data,
                                      const __fp16* s_data, __fp16* out, size_t B, size_t Hq,
                                      size_t Hv, size_t K, size_t V, float scale) {
    (void)q_data; (void)k_data; (void)v_data; (void)g_data; (void)b_data; (void)s_data;
    (void)Hq; (void)Hv; (void)K; (void)V; (void)scale;
    std::fill(out, out + B * Hq * V, static_cast<__fp16>(0.0f));
}

void cactus_gated_deltanet_prefill_f16(const __fp16* q_data, const __fp16* k_data,
                                       const __fp16* v_data, const __fp16* g_data,
                                       const __fp16* b_data, const __fp16* s_data, __fp16* out,
                                       size_t B, size_t T, size_t Hq, size_t Hv, size_t K,
                                       size_t V, size_t requested_chunk_size, float scale) {
    (void)q_data; (void)k_data; (void)v_data; (void)g_data; (void)b_data; (void)s_data;
    (void)Hq; (void)Hv; (void)K; (void)V; (void)requested_chunk_size; (void)scale;
    std::fill(out, out + B * T * Hq * V, static_cast<__fp16>(0.0f));
}

void cactus_gaussian_topk_f16(const __fp16* input, __fp16* output, size_t rows, size_t cols,
                              float ppf) {
    (void)input; (void)ppf;
    std::fill(output, output + rows * cols, static_cast<__fp16>(0.0f));
}

void cactus_conv1d_causal_depthwise_f16(const __fp16* input, const __fp16* weight, __fp16* output,
                                        size_t N, size_t L, size_t C, size_t K, size_t dilation) {
    (void)input; (void)weight; (void)dilation;
    std::fill(output, output + N * L * C * K, static_cast<__fp16>(0.0f));
}

void cactus_conv1d_causal_depthwise_channel_first_f16(const __fp16* input, const __fp16* weight,
                                                      __fp16* output, size_t N, size_t C, size_t L,
                                                      size_t K, size_t dilation) {
    (void)input; (void)weight; (void)dilation;
    std::fill(output, output + N * C * L * K, static_cast<__fp16>(0.0f));
}

void cactus_conv1d_f16_k3(const __fp16* input, const __fp16* weight, __fp16* output, size_t N,
                          size_t L, size_t C_in, size_t C_out, size_t stride) {
    (void)input; (void)weight; (void)stride;
    std::fill(output, output + N * L * C_out, static_cast<__fp16>(0.0f));
    (void)C_in;
}

void cactus_conv1d_f16(const __fp16* input, const __fp16* weight, const __fp16* bias, __fp16* output,
                       size_t N, size_t L, size_t C_in, size_t C_out, size_t K, size_t stride) {
    (void)input; (void)weight; (void)bias; (void)K; (void)stride; (void)C_in;
    std::fill(output, output + N * L * C_out, static_cast<__fp16>(0.0f));
}

void cactus_conv1d_f16_k7s3_oc8(const __fp16* input, const __fp16* Wpack, const __fp16* bias,
                                __fp16* output, size_t N, size_t L, size_t C_in, size_t C_out) {
    (void)input; (void)Wpack; (void)bias; (void)C_in;
    std::fill(output, output + N * L * C_out, static_cast<__fp16>(0.0f));
}

void cactus_conv1d_same_depthwise_f16_k9(const __fp16* input, const __fp16* weight,
                                         const __fp16* bias, __fp16* output, size_t N, size_t L,
                                         size_t C) {
    (void)input; (void)weight; (void)bias;
    std::fill(output, output + N * L * C, static_cast<__fp16>(0.0f));
}

void cactus_conv1d_pointwise_f16_gemm(const __fp16* input, const __fp16* weight, const __fp16* bias,
                                      __fp16* output, size_t N, size_t L, size_t C_in,
                                      size_t C_out) {
    (void)input; (void)weight; (void)bias; (void)C_in;
    std::fill(output, output + N * L * C_out, static_cast<__fp16>(0.0f));
}

void cactus_conv2d_f16_k3s1p1_nchw(const __fp16* input, const __fp16* weight, const __fp16* bias,
                                   __fp16* output, size_t N, size_t C_in, size_t H, size_t W,
                                   size_t C_out) {
    (void)input; (void)weight; (void)bias; (void)C_in;
    std::fill(output, output + N * C_out * H * W, static_cast<__fp16>(0.0f));
}

void cactus_conv2d_f16_k3s2p1_nchw(const __fp16* input, const __fp16* weight, const __fp16* bias,
                                   __fp16* output, size_t N, size_t C_in, size_t H, size_t W,
                                   size_t C_out) {
    (void)input; (void)weight; (void)bias; (void)C_in;
    std::fill(output, output + N * C_out * H * W, static_cast<__fp16>(0.0f));
}

void cactus_conv2d_depthwise_f16_k3s2p1_nchw(const __fp16* input, const __fp16* weight,
                                             const __fp16* bias, __fp16* output, size_t N, size_t C,
                                             size_t H, size_t W) {
    (void)input; (void)weight; (void)bias;
    std::fill(output, output + N * C * H * W, static_cast<__fp16>(0.0f));
}

void cactus_conv2d_pointwise_f16_1x1_nchw_gemm(const __fp16* input, const __fp16* weight,
                                               const __fp16* bias, __fp16* output, size_t N,
                                               size_t C_in, size_t H, size_t W, size_t C_out) {
    (void)input; (void)weight; (void)bias; (void)C_in;
    std::fill(output, output + N * C_out * H * W, static_cast<__fp16>(0.0f));
}

void cactus_stft_f16(const __fp16* input, const __fp16* weight, __fp16* output, size_t N, size_t L,
                     size_t C_in, size_t C_out, size_t K, size_t stride, size_t num_fft_bins) {
    (void)input; (void)weight; (void)C_in; (void)C_out; (void)K; (void)stride;
    const size_t out_frames = L / stride;
    std::fill(output, output + N * out_frames * num_fft_bins, static_cast<__fp16>(0.0f));
}

void cactus_rfft_f32_1d(const float* input, float* output, size_t n, const char* norm) {
    (void)input; (void)norm;
    std::fill(output, output + n, 0.0f);
}

void cactus_irfft_f32_1d(const float* input, float* output, size_t n, const char* norm) {
    (void)input; (void)norm;
    std::fill(output, output + n, 0.0f);
}

void cactus_generate_mel_filter_bank(float* mel_filters, int num_frequency_bins, int num_mel_filters,
                                     float min_frequency, float max_frequency, int sampling_rate,
                                     const char* norm, const char* mel_scale,
                                     bool triangularize_in_mel_space) {
    (void)min_frequency; (void)max_frequency; (void)sampling_rate; (void)norm; (void)mel_scale;
    (void)triangularize_in_mel_space;
    std::fill(mel_filters, mel_filters + static_cast<size_t>(num_frequency_bins) * num_mel_filters, 0.0f);
}

void cactus_compute_spectrogram_f32(
    const float* waveform, size_t waveform_length, const float* window, size_t window_length,
    size_t frame_length, size_t hop_length, const size_t* fft_length, float* spectrogram,
    float power, bool center, const char* pad_mode, bool onesided, float dither,
    const float* preemphasis, const float* mel_filters, size_t mel_filters_size, float mel_floor,
    const char* log_mel, float reference, float min_value, const float* db_range,
    bool remove_dc_offset) {
    // SAFE STUB (needle-unused): output size is a function of fft_length/mel
    // filters and frame count; zero-fill a conservative span derived from args.
    (void)waveform; (void)window; (void)window_length; (void)power; (void)center; (void)pad_mode;
    (void)onesided; (void)dither; (void)preemphasis; (void)log_mel; (void)reference; (void)min_value;
    (void)db_range; (void)remove_dc_offset; (void)hop_length; (void)mel_floor;
    const size_t fft = (fft_length && *fft_length) ? *fft_length : frame_length;
    const size_t bins = onesided ? (fft / 2 + 1) : fft;
    const size_t frames = (frame_length && waveform_length >= frame_length)
        ? (waveform_length - frame_length) / hop_length + 1 : 0;
    const size_t out_elems = mel_filters_size ? (frames * mel_filters_size) : (frames * bins);
    (void)mel_filters;
    if (out_elems) std::fill(spectrogram, spectrogram + out_elems, 0.0f);
}

int cactus_image_info(const char* path, int* width, int* height, int* channels) {
    (void)path;
    if (width) *width = 0;
    if (height) *height = 0;
    if (channels) *channels = 0;
    return 0;
}

void cactus_image_resize_float(const float* input, int src_w, int src_h, float* output, int dst_w,
                               int dst_h, int channels) {
    (void)input; (void)src_w; (void)src_h;
    std::fill(output, output + static_cast<size_t>(dst_w) * dst_h * channels, 0.0f);
}

void cactus_image_normalize(const float* input, float* output, int width, int height, int channels,
                            float rescale_factor, const float* mean, const float* std_dev) {
    (void)mean; (void)std_dev;
    for (size_t i = 0; i < static_cast<size_t>(width) * height * channels; ++i) {
        output[i] = input[i] * rescale_factor;
    }
}

void cactus_image_to_patches(const float* image, float* patches, int width, int height, int channels,
                             int patch_size) {
    (void)image; (void)width; (void)height; (void)channels;
    const size_t num_patches = static_cast<size_t>(width / patch_size) * (height / patch_size);
    std::fill(patches, patches + num_patches * patch_size * patch_size * channels, 0.0f);
}
