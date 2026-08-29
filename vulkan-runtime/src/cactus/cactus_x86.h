// M6b-1: self-contained x86 CPU kernel library implementing the Cactus C-API
// subset required by the needle model. This mirrors the upstream
// `cactus-kernels/cactus_kernels.h` public ABI *exactly* (parameter names,
// types, order, default arguments, struct field order) so it can be
// drop-in linked against the cactus engine (M6b-2) and serve as the CPU
// reference for the Vulkan GPU port (M6c).
//
// Unlike upstream (which is ARM-NEON only), this header has NO `<arm_neon.h>`
// dependency. Everything is scalar/portable; the implementation accumulates in
// fp32 and uses fp16/bf16 only as storage.
//
// __fp16 note: the upstream ABI uses the ARM ACLE `__fp16` storage type
// (IEEE 754 binary16, 2 bytes). On x86_64, clang accepts `__fp16` (arithmetic
// promotes to float) but gcc rejects it ("`__fp16` was not declared in this
// scope"). The cactus_x86 library + test are therefore compiled with clang++
// (see CMakeLists.txt). If a gcc build is ever needed, add a `_Float16` type
// alias here (bit-compatible for storage) instead of the `#error`.
// clang-format off
#ifndef CACTUS_X86_H
#define CACTUS_X86_H

#include <cstddef>
#include <cstdint>

#if !defined(__clang__)
#error "cactus_x86.h uses __fp16 and must be compiled with clang++ on x86_64 (gcc rejects __fp16)."
#endif

// ---------------------------------------------------------------------------
// Public ABI (global scope, mirroring cactus_kernels.h). The structs/enums
// must live at global scope with upstream's exact names so that a function
// such as cactus_quant_matmul(const CactusQuantMatrix*, ...) has identical
// C linkage + type identity to upstream's symbol.
// ---------------------------------------------------------------------------

enum class ScalarOpType {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    ABS,
    EXP,
    POW,
    SQRT,
    COS,
    SIN,
    LOG
};

enum CactusQuantFlags : uint32_t {
    CACTUS_QUANT_FLAG_ORTHOGONAL = 1u << 2,
    CACTUS_QUANT_FLAG_INTERLEAVED_4ROW = 1u << 3,
};

constexpr size_t KV_QUANT_GROUP_SIZE = 32;

struct CactusQuantMatrix {
    uint32_t bits;
    uint32_t K;
    uint32_t N;
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t flags;
    const __fp16* codebook;
    const __fp16* input_scale;
    const __fp16* input_scale_recip;
    const __fp16* norms;
    const uint8_t* packed_indices;
    const int8_t* left_signs;
    const int8_t* right_signs;
    const uint32_t* permutation;
    const __fp16* rotation;
    const int8_t* expanded;
    const float* norm_f32;
};

// NOTE: no `extern "C"` here. The upstream cactus_kernels.h declares the C-API
// with C++ linkage (mangled names); the engine/graph reference the mangled
// symbols, so cactus_x86 must define them with C++ linkage to link (M6b-2).

void cactus_matmul_f16(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N);

void cactus_rms_norm_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t batch_size,
    size_t dims,
    float eps);

uint32_t cactus_quant_packed_group_bytes(uint32_t bits, uint32_t group_size);

void cactus_quant_matmul(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C);

void cactus_quant_matmul_pair(
    const CactusQuantMatrix* W0,
    const CactusQuantMatrix* W1,
    const __fp16* A,
    uint32_t M,
    __fp16* C0,
    __fp16* C1);

void cactus_quant_matmul_triple(
    const CactusQuantMatrix* W0,
    const CactusQuantMatrix* W1,
    const CactusQuantMatrix* W2,
    const __fp16* A,
    uint32_t M,
    __fp16* C0,
    __fp16* C1,
    __fp16* C2);

void cactus_quant_dequantize_hadamard_embedding_row(
    uint32_t bits,
    uint32_t hidden_dim,
    uint32_t group_size,
    uint32_t num_groups,
    size_t row,
    const uint8_t* packed_base,
    const __fp16* codebook,
    const __fp16* norms,
    const __fp16* input_scale_recip,
    const int8_t* left_signs,
    const int8_t* right_signs,
    const uint32_t* permutation,
    __fp16* out_row);

void cactus_quant_dequantize_orthogonal_embedding_row(
    uint32_t bits,
    uint32_t K,
    size_t row,
    const uint8_t* packed_base,
    const __fp16* codebook,
    const __fp16* norms,
    const __fp16* input_scale_recip,
    const __fp16* rotation,
    uint32_t flags,
    __fp16* out_row);

void cactus_attention_f16(
    const __fp16* queries,
    const __fp16* keys,
    const __fp16* values,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t kv_seq_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float scale,
    const __fp16* mask,
    size_t position_offset = 0,
    size_t window_size = 0,
    bool is_causal = true,
    bool mask_is_additive = false,
    bool mask_per_head = false,
    size_t v_head_dim = 0,
    float logit_cap = 0.0f);

void cactus_attention_hybrid_int8_fp16(
    const __fp16* queries,
    const int8_t* keys_cached,
    const int8_t* values_cached,
    const float* k_scales,
    const float* v_scales,
    const __fp16* keys_new,
    const __fp16* values_new,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t cache_len,
    size_t new_len,
    size_t num_q_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float scale,
    size_t position_offset = 0,
    bool is_causal = true,
    size_t window_size = 0,
    size_t group_size = KV_QUANT_GROUP_SIZE,
    size_t v_head_dim = 0);

void cactus_quantize_kv_fp16_to_int8(
    const __fp16* src,
    int8_t* dst,
    float* scales,
    size_t seq_len,
    size_t kv_heads,
    size_t head_dim,
    size_t group_size = KV_QUANT_GROUP_SIZE);

void cactus_int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale = 1.0f);
void cactus_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale = 1.0f);
void cactus_fp16_to_fp32(const __fp16* src, float* dst, size_t count);
void cactus_fp32_to_fp16(const float* src, __fp16* dst, size_t count);
float cactus_fp16_max_abs(const __fp16* src, size_t count);

void cactus_add_f16(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    size_t num_elements);

void cactus_subtract_f16(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    size_t num_elements);

void cactus_multiply_f16(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    size_t num_elements);

void cactus_divide_f16(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    size_t num_elements);

void cactus_add_scaled_f16(
    const __fp16* base,
    const __fp16* src,
    __fp16* output,
    size_t num_elements,
    float scale);

void cactus_scalar_op_f16(
    const __fp16* input,
    __fp16* output,
    size_t num_elements,
    float scalar_value,
    ScalarOpType op_type);

void cactus_add_broadcast_f16(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    const size_t* a_strides,
    const size_t* b_strides,
    const size_t* output_shape,
    size_t ndim);

void cactus_subtract_broadcast_f16(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    const size_t* a_strides,
    const size_t* b_strides,
    const size_t* output_shape,
    size_t ndim);

void cactus_multiply_broadcast_f16(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    const size_t* a_strides,
    const size_t* b_strides,
    const size_t* output_shape,
    size_t ndim);

void cactus_divide_broadcast_f16(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    const size_t* a_strides,
    const size_t* b_strides,
    const size_t* output_shape,
    size_t ndim);

void cactus_clamp_f16(
    const __fp16* input,
    __fp16* output,
    size_t num_elements,
    float lo,
    float hi);

void cactus_sigmoid_f16(
    const __fp16* input,
    __fp16* output,
    size_t num_elements);

void cactus_softcap_f16(
    const __fp16* input,
    __fp16* output,
    size_t num_elements,
    float cap,
    float input_scale);

void cactus_transpose_2d_f16(
    const __fp16* source,
    __fp16* destination,
    size_t num_rows,
    size_t num_cols,
    size_t start_row,
    size_t end_row);

void cactus_transpose_f16(
    const __fp16* source,
    __fp16* destination,
    const size_t* shape,
    const size_t* permutation,
    size_t ndim,
    size_t start_idx,
    size_t end_idx);

void cactus_concat_f16(
    const __fp16* input1,
    const __fp16* input2,
    __fp16* output,
    const size_t* shape1,
    const size_t* shape2,
    const size_t* output_shape,
    size_t ndims,
    int axis);

void cactus_softmax_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t vocab_size);

void cactus_sum_axis_f16(
    const __fp16* input,
    __fp16* output,
    size_t outer_size,
    size_t axis_size,
    size_t inner_size);

void cactus_mean_axis_f16(
    const __fp16* input,
    __fp16* output,
    size_t outer_size,
    size_t axis_size,
    size_t inner_size);

void cactus_sample_f16_ex(
    const __fp16* logits,
    uint32_t* output,
    size_t vocab_size,
    float temperature,
    float top_p,
    float min_p,
    float repetition_penalty,
    size_t top_k,
    size_t random_seed,
    const float* bias_values = nullptr,
    const uint32_t* bias_indices = nullptr,
    size_t bias_count = 0);

// ---------------------------------------------------------------------------
// M6b-2 additions. The first two are on the needle hot path (REAL); the
// activation / reduction / norm / trivial-conv ops are REAL too (cheap to get
// right). The remainder are SAFE STUBS for ops the graph layer references but
// needle never executes — they satisfy the linker, write zeros where the
// output size is derivable, and are otherwise no-ops (no OOB).
// ---------------------------------------------------------------------------

void cactus_add_f16_clipped(
    const __fp16* a,
    const __fp16* b,
    __fp16* output,
    size_t num_elements);

void cactus_rope_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t start_pos,
    float theta);

void cactus_gpt_j_rope_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    size_t rot_dim,
    size_t start_pos,
    float theta);

void cactus_relu_f16(const __fp16* input, __fp16* output, size_t num_elements);
void cactus_leaky_relu_f16(const __fp16* input, __fp16* output, size_t num_elements, float negative_slope);
void cactus_silu_f16(const __fp16* input, __fp16* output, size_t num_elements);
void cactus_gelu_f16(const __fp16* input, __fp16* output, size_t num_elements);
void cactus_gelu_f16_erf(const __fp16* input, __fp16* output, size_t num_elements);
void cactus_tanh_f16(const __fp16* input, __fp16* output, size_t num_elements);

void cactus_gelu_scaled_multiply_f16(
    const __fp16* gate,
    const __fp16* up,
    __fp16* output,
    size_t num_elements,
    float gate_scale,
    float product_scale);

void cactus_layer_norm_f16(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t batch_size,
    size_t dims,
    float eps);

void cactus_glu_f16(
    const __fp16* input,
    __fp16* output,
    size_t outer_size,
    size_t split_size,
    size_t inner_size);

void cactus_glu_f32(
    const float* input,
    float* output,
    size_t outer_size,
    size_t split_size,
    size_t inner_size);

void cactus_int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale = 1.0f);
void cactus_fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale = 1.0f);

double cactus_sum_all_f16(const __fp16* data, size_t num_elements);
double cactus_mean_all_f16(const __fp16* data, size_t num_elements);
double cactus_variance_all_f16(const __fp16* data, size_t num_elements);
float cactus_min_all_f16(const __fp16* data, size_t num_elements);
float cactus_max_all_f16(const __fp16* data, size_t num_elements);

void cactus_min_axis_f16(
    const __fp16* input,
    __fp16* output,
    size_t outer_size,
    size_t axis_size,
    size_t inner_size);

void cactus_max_axis_f16(
    const __fp16* input,
    __fp16* output,
    size_t outer_size,
    size_t axis_size,
    size_t inner_size);

void cactus_variance_axis_f16(
    const __fp16* input,
    __fp16* output,
    size_t outer_size,
    size_t axis_size,
    size_t inner_size);

void cactus_quant_orthogonal_matmul(
    const CactusQuantMatrix* W,
    const __fp16* A,
    uint32_t M,
    __fp16* C);

void cactus_sample_f32_ex(
    const float* logits,
    uint32_t* output,
    size_t vocab_size,
    float temperature,
    float top_p,
    float min_p,
    float repetition_penalty,
    size_t top_k,
    size_t random_seed,
    const float* bias_values = nullptr,
    const uint32_t* bias_indices = nullptr,
    size_t bias_count = 0);

void cactus_batchnorm_f16(
    const __fp16* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    __fp16* output,
    size_t outer_size,
    size_t channels,
    size_t inner_size,
    float epsilon);

void cactus_batchnorm_f32(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    size_t outer_size,
    size_t channels,
    size_t inner_size,
    float epsilon);

void cactus_bilinear_interpolation_f16(
    const __fp16* input,
    __fp16* output,
    size_t src_height,
    size_t src_width,
    size_t embed_dim,
    size_t dst_height,
    size_t dst_width,
    bool align_corners = true);

void cactus_maxpool1d_f16(
    const __fp16* input,
    __fp16* output,
    size_t batch_size,
    size_t channels,
    size_t input_length,
    size_t kernel_size,
    size_t stride);

void cactus_cat_f16(
    const __fp16** inputs,
    __fp16* output,
    const size_t** input_shapes,
    const size_t* output_shape,
    size_t num_inputs,
    size_t rank,
    int axis);

// ---- SAFE STUBS (needle-unused, linker-only) ------------------------------

void cactus_altup_predict_f16(
    const __fp16* coefs,
    const __fp16* const* streams,
    __fp16* output,
    size_t n,
    size_t seq_len,
    size_t hidden_dim);

void cactus_altup_correct_f16(
    const __fp16* coefs,
    const __fp16* innovation,
    const __fp16* const* predictions,
    __fp16* output,
    size_t n,
    size_t seq_len,
    size_t hidden_dim);

void cactus_bilstm_sequence_f16(
    const __fp16* input,
    const __fp16* weight_ih_fwd,
    const __fp16* weight_hh_fwd,
    const __fp16* bias_ih_fwd,
    const __fp16* bias_hh_fwd,
    const __fp16* weight_ih_bwd,
    const __fp16* weight_hh_bwd,
    const __fp16* bias_ih_bwd,
    const __fp16* bias_hh_bwd,
    __fp16* output,
    size_t batch_size,
    size_t seq_len,
    size_t input_size,
    size_t hidden_size);

void cactus_lstm_cell_f16(
    const __fp16* x_input,
    const __fp16* h_prev,
    const __fp16* c_prev,
    const __fp16* weight_ih,
    const __fp16* weight_hh,
    const __fp16* bias_ih,
    const __fp16* bias_hh,
    __fp16* h_new,
    __fp16* c_new,
    size_t batch_size,
    size_t input_size,
    size_t hidden_size);

void cactus_gated_deltanet_decode_f16(
    const __fp16* q_data,
    const __fp16* k_data,
    const __fp16* v_data,
    const __fp16* g_data,
    const __fp16* b_data,
    const __fp16* s_data,
    __fp16* out,
    size_t B,
    size_t Hq,
    size_t Hv,
    size_t K,
    size_t V,
    float scale);

void cactus_gated_deltanet_prefill_f16(
    const __fp16* q_data,
    const __fp16* k_data,
    const __fp16* v_data,
    const __fp16* g_data,
    const __fp16* b_data,
    const __fp16* s_data,
    __fp16* out,
    size_t B,
    size_t T,
    size_t Hq,
    size_t Hv,
    size_t K,
    size_t V,
    size_t requested_chunk_size,
    float scale);

void cactus_gaussian_topk_f16(
    const __fp16* input,
    __fp16* output,
    size_t rows,
    size_t cols,
    float ppf);

void cactus_conv1d_causal_depthwise_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C,
    size_t K,
    size_t dilation);

void cactus_conv1d_causal_depthwise_channel_first_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N,
    size_t C,
    size_t L,
    size_t K,
    size_t dilation);

void cactus_conv1d_f16_k3(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out,
    size_t stride);

void cactus_conv1d_f16(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out,
    size_t K,
    size_t stride);

void cactus_conv1d_f16_k7s3_oc8(
    const __fp16* input,
    const __fp16* Wpack,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out);

void cactus_conv1d_same_depthwise_f16_k9(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C);

void cactus_conv1d_pointwise_f16_gemm(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out);

void cactus_conv2d_f16_k3s1p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in,
    size_t H,
    size_t W,
    size_t C_out);

void cactus_conv2d_f16_k3s2p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in,
    size_t H,
    size_t W,
    size_t C_out);

void cactus_conv2d_depthwise_f16_k3s2p1_nchw(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C,
    size_t H,
    size_t W);

void cactus_conv2d_pointwise_f16_1x1_nchw_gemm(
    const __fp16* input,
    const __fp16* weight,
    const __fp16* bias,
    __fp16* output,
    size_t N,
    size_t C_in,
    size_t H,
    size_t W,
    size_t C_out);

void cactus_stft_f16(
    const __fp16* input,
    const __fp16* weight,
    __fp16* output,
    size_t N,
    size_t L,
    size_t C_in,
    size_t C_out,
    size_t K,
    size_t stride,
    size_t num_fft_bins);

void cactus_rfft_f32_1d(const float* input, float* output, size_t n, const char* norm);
void cactus_irfft_f32_1d(const float* input, float* output, size_t n, const char* norm);

void cactus_generate_mel_filter_bank(
    float* mel_filters, int num_frequency_bins, int num_mel_filters,
    float min_frequency, float max_frequency, int sampling_rate,
    const char* norm, const char* mel_scale, bool triangularize_in_mel_space);

void cactus_compute_spectrogram_f32(
    const float* waveform, size_t waveform_length,
    const float* window, size_t window_length,
    size_t frame_length, size_t hop_length, const size_t* fft_length,
    float* spectrogram, float power,
    bool center, const char* pad_mode, bool onesided,
    float dither, const float* preemphasis,
    const float* mel_filters, size_t mel_filters_size,
    float mel_floor, const char* log_mel,
    float reference, float min_value, const float* db_range,
    bool remove_dc_offset);

int cactus_image_info(const char* path, int* width, int* height, int* channels);

void cactus_image_resize_float(
    const float* input, int src_w, int src_h,
    float* output, int dst_w, int dst_h, int channels);

void cactus_image_normalize(
    const float* input, float* output,
    int width, int height, int channels,
    float rescale_factor, const float* mean, const float* std_dev);

void cactus_image_to_patches(
    const float* image, float* patches,
    int width, int height, int channels, int patch_size);

#endif // CACTUS_X86_H
