// M6c-1: gcc-compatible bridge over the clang-only cactus_x86 CPU reference.
//
// The GPU kernel tests (tests/test_*.cpp) are compiled with the project-default
// g++ and therefore CANNOT include "cactus/cactus_x86.h" (which uses the ARM
// ACLE `__fp16` storage type and #errors on non-clang). The bridge exposes the
// exact same functions with `__fp16` replaced by `uint16_t` (IEEE 754 binary16
// bit patterns — bit-identical to __fp16 storage) and the CactusQuantMatrix
// fp16 pointers replaced by `const uint16_t*`.
//
// All bridge functions use C linkage (extern "C") so there is zero name-mangling
// risk between the gcc-compiled tests and the clang-compiled bridge TU. The
// bridge is compiled with clang++ (see CMakeLists.txt) and internally
// reinterpret-casts the uint16_t* pointers to __fp16* before delegating to the
// real cactus_x86 functions, so the CPU reference the GPU kernels are compared
// against is literally the same code as the M6b-1 library.

#ifndef CACTUS_X86_BRIDGE_H
#define CACTUS_X86_BRIDGE_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ScalarOpType integer values (mirror cactus_x86.h enum order).
enum CactusBridgeScalarOp {
    CACTUS_BRIDGE_OP_ADD = 0,
    CACTUS_BRIDGE_OP_SUBTRACT = 1,
    CACTUS_BRIDGE_OP_MULTIPLY = 2,
    CACTUS_BRIDGE_OP_DIVIDE = 3,
    CACTUS_BRIDGE_OP_ABS = 4,
    CACTUS_BRIDGE_OP_EXP = 5,
    CACTUS_BRIDGE_OP_POW = 6,
    CACTUS_BRIDGE_OP_SQRT = 7,
    CACTUS_BRIDGE_OP_COS = 8,
    CACTUS_BRIDGE_OP_SIN = 9,
    CACTUS_BRIDGE_OP_LOG = 10,
};

// CactusQuantFlags (mirror cactus_x86.h).
enum CactusBridgeQuantFlags {
    CACTUS_BRIDGE_QUANT_FLAG_ORTHOGONAL = 1u << 2,
    CACTUS_BRIDGE_QUANT_FLAG_INTERLEAVED_4ROW = 1u << 3,
};

// POD twin of CactusQuantMatrix with fp16 pointers as uint16_t*.
struct CactusQuantMatrixBridge {
    uint32_t bits;
    uint32_t K;
    uint32_t N;
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t flags;
    const uint16_t* codebook;        // fp16
    const uint16_t* input_scale;     // fp16
    const uint16_t* input_scale_recip; // fp16
    const uint16_t* norms;           // fp16
    const uint8_t* packed_indices;
    const int8_t* left_signs;
    const int8_t* right_signs;
    const uint32_t* permutation;
    const uint16_t* rotation;        // fp16
    const int8_t* expanded;
    const float* norm_f32;
};

// --- casts / data movement -------------------------------------------------
void cactus_bridge_fp16_to_fp32(const uint16_t* src, float* dst, size_t count);
void cactus_bridge_fp32_to_fp16(const float* src, uint16_t* dst, size_t count);

// --- matmul / norm ----------------------------------------------------------
void cactus_bridge_matmul_f16(const uint16_t* a, const uint16_t* b_transposed,
                              uint16_t* c, size_t M, size_t K, size_t N);
void cactus_bridge_rms_norm_f16(const uint16_t* input, const uint16_t* weight,
                                uint16_t* output, size_t batch_size, size_t dims,
                                float eps);

// --- elementwise / unary ----------------------------------------------------
void cactus_bridge_scalar_op_f16(const uint16_t* input, uint16_t* output,
                                 size_t n, float scalar_value, int op_type);
void cactus_bridge_sigmoid_f16(const uint16_t* input, uint16_t* output, size_t n);
void cactus_bridge_clamp_f16(const uint16_t* input, uint16_t* output, size_t n,
                             float lo, float hi);
void cactus_bridge_relu_f16(const uint16_t* input, uint16_t* output, size_t n);
void cactus_bridge_tanh_f16(const uint16_t* input, uint16_t* output, size_t n);
void cactus_bridge_silu_f16(const uint16_t* input, uint16_t* output, size_t n);
void cactus_bridge_softcap_f16(const uint16_t* input, uint16_t* output, size_t n,
                               float cap, float input_scale);

// --- KV quantize ------------------------------------------------------------
void cactus_bridge_quantize_kv_fp16_to_int8(const uint16_t* src, int8_t* dst,
                                            float* scales, size_t seq_len,
                                            size_t kv_heads, size_t head_dim,
                                            size_t group_size);

// --- attention ---------------------------------------------------------------
void cactus_bridge_attention_f16(
    const uint16_t* queries, const uint16_t* keys, const uint16_t* values,
    uint16_t* output, size_t batch_size, size_t seq_len, size_t kv_seq_len,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, float scale,
    const uint16_t* mask, size_t position_offset, size_t window_size,
    bool is_causal, bool mask_is_additive, bool mask_per_head,
    size_t v_head_dim, float logit_cap);

void cactus_bridge_attention_hybrid_int8_fp16(
    const uint16_t* queries, const int8_t* keys_cached, const int8_t* values_cached,
    const float* k_scales, const float* v_scales, const uint16_t* keys_new,
    const uint16_t* values_new, uint16_t* output, size_t batch_size, size_t seq_len,
    size_t cache_len, size_t new_len, size_t num_q_heads, size_t num_kv_heads,
    size_t head_dim, float scale, size_t position_offset, bool is_causal,
    size_t window_size, size_t group_size, size_t v_head_dim);

// --- CQ4 ---------------------------------------------------------------------
void cactus_bridge_quant_matmul(const CactusQuantMatrixBridge* W, const uint16_t* A,
                                uint32_t M, uint16_t* C);

void cactus_bridge_dequantize_orthogonal_embedding_row(
    uint32_t bits, uint32_t K, size_t row, const uint8_t* packed_base,
    const uint16_t* codebook, const uint16_t* norms, const uint16_t* input_scale_recip,
    const uint16_t* rotation, uint32_t flags, uint16_t* out_row);

#ifdef __cplusplus
}
#endif

#endif // CACTUS_X86_BRIDGE_H
