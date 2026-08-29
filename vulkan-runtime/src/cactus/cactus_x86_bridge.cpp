// M6c-1: clang++-compiled bridge over the cactus_x86 CPU reference.
//
// See cactus_x86_bridge.h for the rationale. This TU is compiled with clang++
// (it includes cactus_x86.h, which uses __fp16) and defines every bridge
// function with C linkage. Each function reinterpret-casts its uint16_t*
// (fp16-bit) arguments to __fp16* and delegates to the corresponding
// cactus_x86 function, so the CPU reference exercised by the GPU tests is the
// M6b-1 code verbatim.

#include "cactus/cactus_x86.h"

#include "cactus/cactus_x86_bridge.h"

namespace {

inline const __fp16* f16(const uint16_t* p) {
    return reinterpret_cast<const __fp16*>(p);
}
inline __fp16* f16_mut(uint16_t* p) {
    return reinterpret_cast<__fp16*>(p);
}

} // namespace

extern "C" {

void cactus_bridge_fp16_to_fp32(const uint16_t* src, float* dst, size_t count) {
    cactus_fp16_to_fp32(f16(src), dst, count);
}

void cactus_bridge_fp32_to_fp16(const float* src, uint16_t* dst, size_t count) {
    cactus_fp32_to_fp16(src, f16_mut(dst), count);
}

void cactus_bridge_matmul_f16(const uint16_t* a, const uint16_t* b_transposed,
                              uint16_t* c, size_t M, size_t K, size_t N) {
    cactus_matmul_f16(f16(a), f16(b_transposed), f16_mut(c), M, K, N);
}

void cactus_bridge_rms_norm_f16(const uint16_t* input, const uint16_t* weight,
                                uint16_t* output, size_t batch_size, size_t dims,
                                float eps) {
    cactus_rms_norm_f16(f16(input), f16(weight), f16_mut(output), batch_size, dims, eps);
}

void cactus_bridge_scalar_op_f16(const uint16_t* input, uint16_t* output, size_t n,
                                 float scalar_value, int op_type) {
    cactus_scalar_op_f16(f16(input), f16_mut(output), n, scalar_value,
                         static_cast<ScalarOpType>(op_type));
}

void cactus_bridge_sigmoid_f16(const uint16_t* input, uint16_t* output, size_t n) {
    cactus_sigmoid_f16(f16(input), f16_mut(output), n);
}

void cactus_bridge_clamp_f16(const uint16_t* input, uint16_t* output, size_t n,
                             float lo, float hi) {
    cactus_clamp_f16(f16(input), f16_mut(output), n, lo, hi);
}

void cactus_bridge_relu_f16(const uint16_t* input, uint16_t* output, size_t n) {
    cactus_relu_f16(f16(input), f16_mut(output), n);
}

void cactus_bridge_tanh_f16(const uint16_t* input, uint16_t* output, size_t n) {
    cactus_tanh_f16(f16(input), f16_mut(output), n);
}

void cactus_bridge_silu_f16(const uint16_t* input, uint16_t* output, size_t n) {
    cactus_silu_f16(f16(input), f16_mut(output), n);
}

void cactus_bridge_softcap_f16(const uint16_t* input, uint16_t* output, size_t n,
                               float cap, float input_scale) {
    cactus_softcap_f16(f16(input), f16_mut(output), n, cap, input_scale);
}

void cactus_bridge_quantize_kv_fp16_to_int8(const uint16_t* src, int8_t* dst,
                                            float* scales, size_t seq_len,
                                            size_t kv_heads, size_t head_dim,
                                            size_t group_size) {
    cactus_quantize_kv_fp16_to_int8(f16(src), dst, scales, seq_len, kv_heads,
                                    head_dim, group_size);
}

void cactus_bridge_attention_f16(
    const uint16_t* queries, const uint16_t* keys, const uint16_t* values,
    uint16_t* output, size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale,
    const uint16_t* mask, size_t position_offset, size_t window_size,
    bool is_causal, bool mask_is_additive, bool mask_per_head,
    size_t v_head_dim, float logit_cap) {
    cactus_attention_f16(f16(queries), f16(keys), f16(values), f16_mut(output),
                         batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads,
                         head_dim, scale, f16(mask), position_offset, window_size,
                         is_causal, mask_is_additive, mask_per_head, v_head_dim,
                         logit_cap);
}

void cactus_bridge_attention_hybrid_int8_fp16(
    const uint16_t* queries, const int8_t* keys_cached, const int8_t* values_cached,
    const float* k_scales, const float* v_scales, const uint16_t* keys_new,
    const uint16_t* values_new, uint16_t* output, size_t batch_size, size_t seq_len,
    size_t cache_len, size_t new_len, size_t num_q_heads, size_t num_kv_heads,
    size_t head_dim, float scale, size_t position_offset, bool is_causal,
    size_t window_size, size_t group_size, size_t v_head_dim) {
    cactus_attention_hybrid_int8_fp16(
        f16(queries), keys_cached, values_cached, k_scales, v_scales, f16(keys_new),
        f16(values_new), f16_mut(output), batch_size, seq_len, cache_len, new_len,
        num_q_heads, num_kv_heads, head_dim, scale, position_offset, is_causal,
        window_size, group_size, v_head_dim);
}

void cactus_bridge_quant_matmul(const CactusQuantMatrixBridge* W, const uint16_t* A,
                                uint32_t M, uint16_t* C) {
    CactusQuantMatrix w{};
    w.bits = W->bits;
    w.K = W->K;
    w.N = W->N;
    w.group_size = W->group_size;
    w.num_groups = W->num_groups;
    w.flags = W->flags;
    w.codebook = f16(W->codebook);
    w.input_scale = f16(W->input_scale);
    w.input_scale_recip = f16(W->input_scale_recip);
    w.norms = f16(W->norms);
    w.packed_indices = W->packed_indices;
    w.left_signs = W->left_signs;
    w.right_signs = W->right_signs;
    w.permutation = W->permutation;
    w.rotation = f16(W->rotation);
    w.expanded = W->expanded;
    w.norm_f32 = W->norm_f32;
    cactus_quant_matmul(&w, f16(A), M, f16_mut(C));
}

void cactus_bridge_dequantize_orthogonal_embedding_row(
    uint32_t bits, uint32_t K, size_t row, const uint8_t* packed_base,
    const uint16_t* codebook, const uint16_t* norms, const uint16_t* input_scale_recip,
    const uint16_t* rotation, uint32_t flags, uint16_t* out_row) {
    cactus_quant_dequantize_orthogonal_embedding_row(
        bits, K, row, packed_base, f16(codebook), f16(norms), f16(input_scale_recip),
        f16(rotation), flags, f16_mut(out_row));
}

} // extern "C"
