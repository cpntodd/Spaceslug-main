#pragma once

#include "api/dataset_batch_buffer.h"
#include "core/vk_setup.h"
#include "exec/engine.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace vulkan_runtime::tiny {
// Capability metadata is deliberately scoped: training uses normal submissions;
// only the fixed-shape forward-only method retains a command buffer.
inline constexpr char capability[] =
    "tiny_forward_persistent_bounded_resource_graph_training_lora_rank4_rank8_sgd_adamw";
inline constexpr char command_buffer_capability[] =
    "production_fixed_shape_forward_loss_retained_command_buffer_resubmit";
inline constexpr bool retained_backward_optimizer_supported = false;
inline constexpr char retained_backward_optimizer_capability[] =
    "retained_forward_loss_only_backward_optimizer_unsupported";
inline constexpr bool arbitrary_shape_full_base_supported = false;
inline constexpr char arbitrary_shape_full_base_capability[] =
    "full_base_fixed_profile_only_arbitrary_shape_unsupported";
// Base training is an explicit ownership boundary. QKV AdamW consumes the
// graph's already-produced gradients and never reruns or duplicates SGD.
inline constexpr char base_train_capability[] = "base_train_group_lm_head_output_qkv_embeddings_owned_fp32_fixed_window_sgd_rows_"
                                                "le_128_lm_head_adamw_output_adamw_qkv_adamw_from_gradients_embeddings_sparse_sgd_positions_table_sgd_no_ffn_norm_dataset_retained_no_standalone_bridge";
inline constexpr char full_base_training_capability[] =
    "full_base_training_unsupported_gamma_ffn_schema_v4_cabi_dataset_retained_arbitrary_shape_gated";
inline constexpr char bounded_full_graph_training_capability[] =
    "tiny_fixed_profile_bounded_full_graph_adamw_true_rmsnorm_ffn_position_cabi_vvl_clean";
// Graph embedding training consumes graph-owned dstate and token buffers in the
// same normal submission as the fixed-window backward chain.
inline constexpr char graph_embedding_training_capability[] =
    "tiny_graph_embedding_dstate_gradient_sgd_graph_owned_tokens_fixed_window_rows_le_128_one_submit_cpu_parity";
inline constexpr int graph_embedding_training_status = 0;
enum class BaseTrainGroup : std::uint32_t {
    None = 0,
    LmHead = 1,
    Output = 2,
    QKV = 3,
    Embeddings = 4,
    Positions = 5,
};
inline constexpr bool base_train_group_supported(BaseTrainGroup group) noexcept {
    return group == BaseTrainGroup::LmHead || group == BaseTrainGroup::Output || group == BaseTrainGroup::QKV ||
           group == BaseTrainGroup::Embeddings || group == BaseTrainGroup::Positions;
}
inline constexpr std::uint32_t LoraRank4 = 4;
inline constexpr std::uint32_t LoraRank8 = 8;
inline constexpr bool lora_rank_supported(std::uint32_t rank) noexcept {
    return rank == LoraRank4 || rank == LoraRank8;
}
inline constexpr std::uint32_t H = 64;
inline constexpr std::uint32_t V = 259;
inline constexpr std::uint32_t Vp = 320;
inline constexpr std::uint32_t Tcap = 128;
inline constexpr std::uint32_t LoraRank = 4;
inline constexpr std::uint32_t bounded_full_graph_max_rows = Tcap;
inline constexpr bool bounded_full_graph_requires_trailing_mask = true;
inline constexpr bool bounded_full_graph_uses_retained_commands = false;

// Tiny is intentionally a fixed-shape runtime. These descriptors are the
// complete supported profile set; dimensions are not interchangeable.
struct ProfileDescriptor {
    char const* name;
    std::uint32_t hidden;
    std::uint32_t vocab;
    std::uint32_t padded_vocab;
    std::uint32_t token_capacity;
    std::uint32_t lora_rank;
};
inline constexpr ProfileDescriptor Profiles[] = {
    {"tiny_h64_v259_vp320_t128_rank4", H, V, Vp, Tcap, LoraRank4},
    {"tiny_h64_v259_vp320_t128_rank8", H, V, Vp, Tcap, LoraRank8},
};
inline constexpr std::size_t ProfileCount = sizeof(Profiles) / sizeof(Profiles[0]);
inline constexpr bool profile_supported(std::uint32_t hidden,
                                        std::uint32_t vocab,
                                        std::uint32_t padded_vocab,
                                        std::uint32_t token_capacity,
                                        std::uint32_t rank) noexcept {
    for (auto const& profile : Profiles)
        if (profile.hidden == hidden && profile.vocab == vocab && profile.padded_vocab == padded_vocab &&
            profile.token_capacity == token_capacity && profile.lora_rank == rank)
            return true;
    return false;
}

struct BaseCheckpoint {
    std::uint32_t version{4};
    ProfileDescriptor profile{};
    std::uint32_t group_mask{0};
    std::vector<float> embeddings;
    std::vector<float> positions;
    std::vector<float> positions_m;
    std::vector<float> positions_v;
    // Reserved schema-v4 trainable normalization and FFN payloads.
    std::vector<float> gamma, gamma_m, gamma_v;
    std::vector<float> ffn_w1, ffn_b1, ffn_w2, ffn_b2;
    std::vector<float> ffn_w1_m, ffn_b1_m, ffn_w2_m, ffn_b2_m;
    std::vector<float> ffn_w1_v, ffn_b1_v, ffn_w2_v, ffn_b2_v;
    std::vector<float> lm_head;
    std::vector<float> output;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> lm_head_m;
    std::vector<float> lm_head_v;
    std::vector<float> output_m;
    std::vector<float> output_v;
    std::vector<float> query_m;
    std::vector<float> key_m;
    std::vector<float> value_m;
    std::vector<float> query_v;
    std::vector<float> key_v;
    std::vector<float> value_v;
    std::uint64_t adamw_step{0};
};

inline constexpr std::uint32_t BaseCheckpointLmHead = 1u << 0;
inline constexpr std::uint32_t BaseCheckpointOutput = 1u << 1;
inline constexpr std::uint32_t BaseCheckpointQKV = 1u << 2;
inline constexpr std::uint32_t BaseCheckpointEmbeddings = 1u << 3;
inline constexpr std::uint32_t BaseCheckpointPositions = 1u << 4;
inline constexpr std::uint32_t BaseCheckpointNormalization = 1u << 5;
inline constexpr std::uint32_t BaseCheckpointFfn = 1u << 6;

class ForwardResourceGraph {
  public:
    int readback_base_checkpoint(BaseCheckpoint&) noexcept;
    int update_base_checkpoint(BaseCheckpoint const&) noexcept;
    // Schema-v4 carries gamma and FFN state, but these groups remain fail-closed
    // until their true-RMSNorm/full-chain semantics are proven end to end.
    int import_base_train_embeddings(float const* weight);
    int readback_base_train_embeddings(float* weight);
    // Reserved graph-owned trainable normalization/FFN constructor path.
    // Until schema-v4 state and ABI are complete, callers must use the legacy constructor.
    static constexpr bool trainable_full_base_constructor_supported = false;
    int import_base_train_positions(float const* positions);
    int readback_base_train_positions(float* positions);
    ProfileDescriptor profile_descriptor() const noexcept { return Profiles[loraRank_ == LoraRank8 ? 1 : 0]; }

    ForwardResourceGraph(core::VulkanContext const&,
                         float const* embeddings,
                         float const* positions,
                         float const* query,
                         float const* key,
                         float const* value,
                         float const* output,
                         float const* lm_head,
                         std::uint32_t rank = LoraRank4);
    ForwardResourceGraph(core::VulkanContext const&,
                         float const* embeddings,
                         float const* positions,
                         float const* lm_head,
                         std::uint32_t rank = LoraRank4);
    // Safe reduced-precision evaluation boundary: frozen inputs arrive as
    // IEEE-754 binary16 storage, are widened to FP32 on the host, and then use
    // the unchanged FP32 forward graph. This does not enable FP16 arithmetic.
    ForwardResourceGraph(core::VulkanContext const&,
                         std::uint16_t const* embeddings,
                         std::uint16_t const* positions,
                         std::uint16_t const* query,
                         std::uint16_t const* key,
                         std::uint16_t const* value,
                         std::uint16_t const* output,
                         std::uint16_t const* lm_head,
                         std::uint32_t rank = LoraRank4);
    ~ForwardResourceGraph();
    ForwardResourceGraph(ForwardResourceGraph const&) = delete;
    ForwardResourceGraph& operator=(ForwardResourceGraph const&) = delete;
    void forward(std::uint32_t const* tokens, std::uint32_t length, float* logits, bool final_only = false);
    // Production-safe retained subset: exactly Tcap rows, full logits. Inputs are
    // copied into a mutable host staging allocation; the complete forward command
    // (copy, barrier, dispatch, barrier, readback) is recorded once. Training,
    // backward, LoRA, and variable-length forward remain bounded normal-submit
    // paths, not retained command-buffer paths.
    void forward_fixed_retained(std::uint32_t const* tokens, float* logits);
    // Production-bounded retained forward + masked causal loss. Exactly Tcap
    // rows are executed; tokens, targets, and masks are copied into device
    // buffers by the retained command, so no descriptor or command re-record is
    // needed between calls.
    void forward_loss_fixed_retained(std::uint32_t const* tokens,
                                     std::uint32_t const* targets,
                                     std::uint32_t const* masks,
                                     float* logits,
                                     float* row_losses);
    // Fixed-window forward + causal loss with GPU reduction. Only two scalars
    // (sum of included row losses and included-row count) are copied back.
    void forward_loss_fixed_metrics(std::uint32_t const* tokens,
                                    std::uint32_t const* targets,
                                    std::uint32_t const* masks,
                                    float* loss,
                                    std::uint32_t* count);
    // Read post-output-projection activations from the most recent forward.
    void readback_projected(float* projected, std::uint32_t rows);
    bool fixed_forward_retained() const noexcept { return fixedForwardRecorded_; }
    void token_step(std::uint32_t token, std::uint32_t position, float* logits);
    int token_step_training(std::uint32_t token,
                            std::uint32_t position,
                            std::uint32_t target,
                            std::uint32_t mask,
                            float* loss,
                            float* dlogits,
                            float* dprojected);
    int token_step_training_backward(std::uint32_t token,
                                     std::uint32_t position,
                                     std::uint32_t target,
                                     std::uint32_t mask,
                                     float const* doutput,
                                     float* loss,
                                     float* dlogits,
                                     float* dprojected,
                                     float* dquery,
                                     float* dkey,
                                     float* dvalue,
                                     float* dcontext,
                                     float* dstates);
    // Runs the fixed Tiny forward/loss/LM-head/attention/projection backward
    // chain for rows <= Tcap and returns graph-owned per-token dstate rows.
    // This is readback-only: embeddings and all graph weights are unchanged.
    int readback_graph_dstate(std::uint32_t const* tokens,
                              std::uint32_t const* targets,
                              std::uint32_t const* masks,
                              std::uint32_t rows,
                              float* dstates);
    // Runs the complete fixed-window graph backward chain, then embedding
    // gradient and SGD, in one normal submission. Tokens and dstate stay
    // graph-owned; only the bounded token window is supplied by the caller.
    int train_embeddings_sgd(std::uint32_t const* tokens,
                             std::uint32_t const* targets,
                             std::uint32_t const* masks,
                             std::uint32_t rows,
                             float learning_rate) noexcept;
    // Graph-owned positional table [Tcap,H], updated from masked graph dstate.
    // This is bounded FP32 SGD only; AdamW, retained, dataset, and full-base
    // positional training remain unsupported.
    int readback_positions(float* positions);
    int train_positions_sgd(std::uint32_t const* tokens,
                            std::uint32_t const* targets,
                            std::uint32_t const* masks,
                            std::uint32_t rows,
                            float learning_rate);
    int readback_position_gradient(float* gradient, std::size_t count) noexcept;
    // Legacy API name retained for ABI compatibility; this bounded call now runs
    // the integrated true-RMSNorm/FFN/position optimizer chain. It is not a
    // general full-base training contract.
    int train_positions_adamw(std::uint32_t const* tokens,
                              std::uint32_t const* targets,
                              std::uint32_t const* masks,
                              std::uint32_t rows,
                              float learning_rate,
                              float beta1,
                              float beta2,
                              float epsilon,
                              float weight_decay) noexcept;
    int readback_base_train_positions_adamw_state(float* positions,
                                                  float* m,
                                                  float* v,
                                                  std::uint64_t* step);
    int update_base_train_positions_adamw_state(float const* positions,
                                                float const* m,
                                                float const* v,
                                                std::uint64_t step);
    // Trainable RMSNorm gamma remains gated until schema-v4 and ABI parity pass.
    static constexpr bool trainable_normalization_supported = false;
    static constexpr bool trainable_ffn_supported = false;
    static constexpr char const* trainable_ffn_capability() noexcept { return "ffn_graph_state_allocated_forward_backward_w1_adamw_only"; }
    static constexpr int trainable_ffn_unsupported = -6;
    int readback_ffn_state(float*, std::size_t, std::uint64_t*) noexcept;
    int update_ffn_state(float const*, std::size_t, std::uint64_t) noexcept;
    // Internal bounded primitive: applies graph-owned FFN weights to the latest projected rows.
    // This is not full-base training and does not update parameters.
    int run_ffn_forward(std::uint32_t rows) noexcept;
    int readback_ffn_output_staged(float* output, std::size_t count) noexcept;
    int seed_ffn_input_staged(float const* input, std::size_t count) noexcept;
    int seed_ffn_output_gradient_staged(float const* dy, std::size_t count) noexcept;
    int run_ffn_backward_staged(std::uint32_t rows) noexcept;
    int readback_ffn_dx_staged(float* dx, std::size_t count) noexcept;
    int readback_ffn_gradients_staged(float* w1, float* b1, float* w2, float* b2) noexcept;
    // Executes only the staged true-RMSNorm forward; no loss or optimizer side effects.
    // Kept internal until full-chain parity and checkpoint/C ABI gates pass.
    int run_rmsnorm_forward_staged(std::uint32_t rows, bool final_only = false) noexcept;
    int run_rmsnorm_state_only_staged(std::uint32_t rows) noexcept;
    int staged_rmsnorm_forward_is_valid(std::uint32_t rows) const noexcept;
    // Staged diagnostics accept only trailing-valid rows; holes would make
    // causal state/loss semantics ambiguous and are rejected fail-closed.
    int validate_rmsnorm_staged_rows(std::uint32_t const* mask, std::uint32_t rows) const noexcept;
    int run_rmsnorm_staged_state_chain(std::uint32_t rows) noexcept;
    int seed_rmsnorm_staged_dy(float const* dy, std::size_t count) noexcept;
    int seed_rmsnorm_staged_gamma(float const* gamma, std::size_t count) noexcept;
    int seed_rmsnorm_staged_mask(std::uint32_t const* mask, std::size_t count) noexcept;
    int seed_rmsnorm_staged_mask_for_rows(std::uint32_t const* mask, std::uint32_t rows) noexcept;
    int readback_rmsnorm_state_staged(float* raw, std::size_t raw_count, float* inv_rms, std::size_t inv_count) noexcept;
    int readback_rmsnorm_states_staged(float* states, std::size_t count) noexcept;
    int run_rmsnorm_backward_staged(std::uint32_t rows) noexcept;
    int readback_rmsnorm_dx_staged(float* dx, std::size_t count) noexcept;
    int run_rmsnorm_dgamma_staged(std::uint32_t rows) noexcept;
    int readback_rmsnorm_dgamma_staged(float* dgamma, std::size_t count) noexcept;
    int staged_rmsnorm_dgamma_is_valid(std::uint32_t rows) const noexcept;
    int run_rmsnorm_gamma_adamw_staged(float learning_rate, float beta1, float beta2, float epsilon, float weight_decay) noexcept;
    int readback_rmsnorm_gamma_state_staged(float* gamma, float* m, float* v, std::uint64_t* step) noexcept;
    int readback_rmsnorm_staged_diagnostics(float* dx, float* dgamma, std::uint32_t rows) noexcept;
    int run_ffn_backward(std::uint32_t rows) noexcept;
    int run_rmsnorm_gamma_gradient(std::uint32_t rows) noexcept;
    int run_ffn_parameter_gradients(std::uint32_t rows) noexcept;
    int run_ffn_w1_adamw(float, float, float, float, float) noexcept;
    int run_ffn_adamw_all(float, float, float, float, float) noexcept;
    int run_ffn_adamw_staged(float, float, float, float, float) noexcept;
    int train_ffn_forward_loss(std::uint32_t const*, std::uint32_t const*, std::uint32_t const*, std::uint32_t) noexcept;
    static constexpr int trainable_normalization_unsupported = -5;
    // Resource slots are reserved now; allocation and ABI exposure remain gated.
    int readback_trainable_gamma(float*) const noexcept;
    int update_trainable_gamma(float const*) noexcept;
    int train_gamma_adamw(float, float, float, float, float) noexcept { return trainable_normalization_unsupported; }
    int readback_gamma_state(float*, float*, float*, std::uint64_t*) noexcept;
    int update_gamma_state(float const*, float const*, float const*, std::uint64_t) noexcept;
    // Diagnostic readback for bounded combined-chain CPU parity. It returns the
    // most recent gamma/FFN gradients and the FFN output gradient; it does not
    // update parameters or widen the fail-closed full-base capability.
    int readback_combined_gradients(float* gamma_gradient,
                                    float* ffn_gradient,
                                    std::size_t ffn_count,
                                    float* ffn_output_gradient,
                                    float* activations,
                                    float* scaled_states,
                                    float* dstate,
                                    std::uint32_t rows) noexcept;
    int begin_lora_accumulation();
    int token_step_training_backward_accumulate(std::uint32_t token,
                                                std::uint32_t position,
                                                std::uint32_t target,
                                                std::uint32_t mask,
                                                float const* doutput,
                                                float* loss,
                                                float* dlogits,
                                                float* dprojected,
                                                float* dquery,
                                                float* dkey,
                                                float* dvalue,
                                                float* dcontext,
                                                float* dstates);
    int token_windows_training_backward_accumulate(std::uint32_t const* tokens,
                                                   std::uint32_t const* targets,
                                                   std::uint32_t const* mask,
                                                   std::uint32_t windows,
                                                   std::uint32_t window_length,
                                                   float* losses);
    int finalize_lora_sgd(float learning_rate, float normalizer);
    int begin_lora_adamw();
    int accumulate_lora_adamw(std::uint32_t token,
                              std::uint32_t position,
                              std::uint32_t target,
                              std::uint32_t mask,
                              float const* doutput,
                              float* loss,
                              float* dlogits,
                              float* dprojected,
                              float* dquery,
                              float* dkey,
                              float* dvalue,
                              float* dcontext,
                              float* dstates);
    int finalize_lora_adamw(float learning_rate,
                            float beta1,
                            float beta2,
                            float epsilon,
                            float weight_decay,
                            float normalizer);
    int readback_lora_adamw_state(float* adapters, float* m, float* v, std::uint64_t* step);
    int update_lora_adamw_state(float const* adapters, float const* m, float const* v, std::uint64_t step);
    int readback_lora_gradients(float* dquery_a,
                                float* dquery_b,
                                float* dkey_a,
                                float* dkey_b,
                                float* dvalue_a,
                                float* dvalue_b,
                                float* doutput_a,
                                float* doutput_b);
    int readback_lora_adapters(float* query_a,
                               float* query_b,
                               float* key_a,
                               float* key_b,
                               float* value_a,
                               float* value_b,
                               float* output_a,
                               float* output_b);
    int update_lora_adapters(float const* query_a,
                             float const* query_b,
                             float const* key_a,
                             float const* key_b,
                             float const* value_a,
                             float const* value_b,
                             float const* output_a,
                             float const* output_b);
    char const* capability_name() const noexcept { return capability; }
    char const* base_train_capability_name() const noexcept { return base_train_capability; }
    bool base_train_group_enabled(BaseTrainGroup group) const noexcept { return base_train_group_supported(group); }
    int import_base_train_lm_head(float const* weight);
    int import_base_train_output(float const* weight);
    int readback_base_train_output(float* weight);
    int readback_base_train_output_adamw_state(float* weight, float* m, float* v, std::uint64_t* step);
    int update_base_train_output_adamw_state(float const* weight, float const* m, float const* v, std::uint64_t step);
    int readback_base_train_lm_head_adamw_state(float* weight, float* m, float* v, std::uint64_t* step);
    int update_base_train_lm_head_adamw_state(float const* weight, float const* m, float const* v, std::uint64_t step);
    int readback_base_train_lm_head(float* weight);
    // Graph-owned fixed-window LM-head training. The graph owns both the
    // FP32 weight and gradient; each call is one normal, synchronously waited
    // forward/loss/gradient/SGD submission and retains no training commands.
    int
    train_lm_head_sgd(std::uint32_t const*, std::uint32_t const*, std::uint32_t const*, std::uint32_t, float) noexcept;
    int train_lm_head_adamw(std::uint32_t const*,
                            std::uint32_t const*,
                            std::uint32_t const*,
                            std::uint32_t,
                            float,
                            float,
                            float,
                            float,
                            float) noexcept;
    // Graph-owned output projection W[H,H], updated from the actual Tiny
    // pre-output context activations_ and loss upstream dprojected_.
    int train_output_adamw(std::uint32_t const*,
                           std::uint32_t const*,
                           std::uint32_t const*,
                           std::uint32_t,
                           float,
                           float,
                           float,
                           float,
                           float) noexcept;
    int
    train_output_sgd(std::uint32_t const*, std::uint32_t const*, std::uint32_t const*, std::uint32_t, float) noexcept;
    int import_base_train_qkv(float const* query, float const* key, float const* value);
    int readback_base_train_qkv(float* query, float* key, float* value);
    int readback_base_train_qkv_gradients(float* query, float* key, float* value);
    int readback_base_train_qkv_adamw_state(float* query,
                                            float* key,
                                            float* value,
                                            float* query_m,
                                            float* key_m,
                                            float* value_m,
                                            float* query_v,
                                            float* key_v,
                                            float* value_v,
                                            std::uint64_t* step);
    int update_base_train_qkv_adamw_state(float const* query,
                                          float const* key,
                                          float const* value,
                                          float const* query_m,
                                          float const* key_m,
                                          float const* value_m,
                                          float const* query_v,
                                          float const* key_v,
                                          float const* value_v,
                                          std::uint64_t step);
    // Applies AdamW to the gradients produced by a prior graph training accumulation.
    // It never runs train_qkv_sgd or recomputes gradients.
    int train_qkv_adamw_from_gradients(float learning_rate,
                                       float beta1,
                                       float beta2,
                                       float epsilon,
                                       float weight_decay) noexcept;
    int train_qkv_sgd(std::uint32_t const*, std::uint32_t const*, std::uint32_t const*, std::uint32_t, float) noexcept;
    std::uint64_t last_submission() const noexcept { return lastSubmission_; }

    // Bounded dataset staging subset. This retains fixed windows for callers
    // before training submissions; it does not claim full graph integration.
    static constexpr char const* dataset_capability() noexcept { return dataset::BatchBuffer::capability(); }
    std::unique_ptr<dataset::BatchBuffer> create_dataset_batch(std::uint32_t windows,
                                                               std::uint32_t window_length) const;
    static constexpr int dataset_training_unsupported = -3;
    static constexpr int dataset_training_full_unsupported = -4;
    static constexpr bool dataset_training_full_supported = false;

    // Explicit integration boundary: BatchBuffer owns opaque device-local
    // storage, while the training graph owns separate buffers and per-token
    // backward state. Do not silently fall back to a host window loop.
    int train_dataset_batch(dataset::BatchBuffer const&, float learning_rate, float normalizer) noexcept;
    // Reserved for the complete device-resident forward/loss/backward/LoRA path.
    int train_dataset_batch_full(dataset::BatchBuffer const&, float learning_rate, float normalizer) noexcept;
    // Read-only model loss/count for each device-resident window. Results are
    // returned as [loss_sum, included_count] pairs; weights are never mutated.
    std::vector<float> evaluate_dataset_batch(dataset::BatchBuffer const&);

  private:
    struct Buffer;
    void initialize(float const*, float const*, float const*, float const*, float const*, float const*, float const*);
    core::VulkanContext const& context_;
    exec::ExecEngine engine_;
    std::unique_ptr<Buffer> embeddings_, positions_, query_, key_, value_, output_, lmHead_;
    std::unique_ptr<Buffer> tokens_, targets_, mask_, states_, activations_, gammaInput_, rmsRaw_, rmsInv_, rmsDx_, projected_, qRows_, kRows_, vRows_,
        logits_, dlogits_;
    std::unique_ptr<Buffer> dprojected_, doutput_, dcontext_, dstates_, ffnOutput_, ffnDx_, dquery_, dkey_, dvalue_, rowLoss_, metrics_,
        embeddingGradient_, positionGradient_, positionM_, positionV_, gamma_, gammaGradient_, gammaM_, gammaV_, ffnW1_, ffnB1_, ffnW2_, ffnB2_, ffnW1Gradient_, ffnB1Gradient_, ffnW2Gradient_, ffnB2Gradient_, ffnW1M_, ffnB1M_, ffnW2M_, ffnB2M_, ffnW1V_, ffnB1V_, ffnW2V_, ffnB2V_, lmHeadGradient_, lmHeadM_, lmHeadV_, outputGradient_, outputM_, outputV_, qkvGradientQ_, qkvGradientK_,
        qkvGradientV_, qkvM_, qkvV_, readback_;
    std::unique_ptr<Buffer> loraA_, loraB_, loraDA_, loraDB_, loraMA_, loraVA_, loraMB_, loraVB_;
    std::uint32_t loraRank_{LoraRank4};
    std::uint64_t adamwStep_{0};
    std::array<std::vector<float>, 8> loraReadback_{};
    std::array<std::vector<float>, 8> loraAdapterReadback_{};
    vk::ShaderModule shader_{}, rmsForwardShader_{}, rmsBackwardStateShader_{}, rmsDgammaStateShader_{}, rmsFixedForwardShader_{}, lmHeadForwardShader_{}, qkvGradientShader_{}, rmsNormGradientShader_{}, gammaAdamwShader_{}, scaleDgammaShader_{}, ffnForwardShader_{}, ffnBackwardShader_{}, ffnGradientShader_{}, ffnAdamwShader_{}, ffnB1AdamwShader_{}, ffnW2AdamwShader_{}, ffnB2AdamwShader_{};
    vk::DescriptorSetLayout descriptorLayout_{}, rmsForwardLayout_{}, rmsBackwardStateLayout_{}, rmsDgammaStateLayout_{}, rmsFixedForwardLayout_{};
    vk::PipelineLayout pipelineLayout_{}, rmsForwardPipelineLayout_{}, rmsBackwardStatePipelineLayout_{}, rmsDgammaStatePipelineLayout_{}, rmsFixedForwardPipelineLayout_{};
    vk::Pipeline pipeline_{}, rmsForwardPipeline_{}, rmsBackwardStatePipeline_{}, rmsDgammaStatePipeline_{}, rmsFixedForwardPipeline_{};
    vk::DescriptorPool descriptorPool_{}, rmsForwardPool_{}, rmsBackwardStatePool_{}, rmsDgammaStatePool_{}, rmsFixedForwardPool_{};
    vk::DescriptorSet descriptorSet_{}, rmsForwardSet_{}, rmsBackwardStateSet_{}, rmsDgammaStateSet_{}, rmsFixedForwardSet_{};
    vk::ShaderModule lossShader_{}, metricsShader_{}, lmBackwardShader_{}, lmHeadGradientShader_{}, lmHeadSgdShader_{},
        lmHeadAdamwShader_{}, outputGradientShader_{}, outputSgdShader_{}, outputAdamwShader_{},
        projectionBackwardShader_{}, attentionBackwardShader_{}, qkvSgdShader_{}, qkvAdamwShader_{}, embeddingGradientShader_{},
        embeddingSgdShader_{}, positionGradientShader_{}, positionSgdShader_{}, positionAdamwShader_{}, loraShader_{},
        loraSgdShader_{}, loraClearShader_{}, loraFinalizeShader_{}, loraAdamwShader_{};
    vk::DescriptorSetLayout gammaAdamwLayout_{}, lmHeadForwardLayout_{}, ffnForwardLayout_{}, ffnBackwardLayout_{}, ffnGradientLayout_{}, rmsNormGradientLayout_{}, scaleDgammaLayout_{}, ffnAdamwLayout_{}, ffnB1AdamwLayout_{}, ffnW2AdamwLayout_{}, ffnB2AdamwLayout_{}, lossLayout_{}, metricsLayout_{}, lmBackwardLayout_{}, lmHeadGradientLayout_{},
        lmHeadSgdLayout_{}, lmHeadAdamwLayout_{}, outputGradientLayout_{}, outputSgdLayout_{}, outputAdamwLayout_{},
        projectionBackwardLayout_{}, attentionBackwardLayout_{}, qkvGradientLayout_{}, qkvSgdLayout_{},
        qkvAdamwLayout_{}, embeddingGradientLayout_{}, embeddingSgdLayout_{}, positionGradientLayout_{}, positionSgdLayout_{}, positionAdamwLayout_{}, loraLayout_{}, loraSgdLayout_{},
        loraClearLayout_{}, loraFinalizeLayout_{},
        loraAdamwLayout_{};
    vk::PipelineLayout gammaAdamwPipelineLayout_{}, lmHeadForwardPipelineLayout_{}, ffnForwardPipelineLayout_{}, ffnBackwardPipelineLayout_{}, ffnGradientPipelineLayout_{}, rmsNormGradientPipelineLayout_{}, scaleDgammaPipelineLayout_{}, ffnAdamwPipelineLayout_{}, ffnB1AdamwPipelineLayout_{}, ffnW2AdamwPipelineLayout_{}, ffnB2AdamwPipelineLayout_{}, lossPipelineLayout_{}, metricsPipelineLayout_{}, lmBackwardPipelineLayout_{},
        lmHeadGradientPipelineLayout_{}, lmHeadSgdPipelineLayout_{}, lmHeadAdamwPipelineLayout_{},
        outputGradientPipelineLayout_{}, outputSgdPipelineLayout_{}, outputAdamwPipelineLayout_{},
        projectionBackwardPipelineLayout_{}, attentionBackwardPipelineLayout_{}, qkvGradientPipelineLayout_{},
        qkvSgdPipelineLayout_{}, qkvAdamwPipelineLayout_{}, embeddingGradientPipelineLayout_{}, embeddingSgdPipelineLayout_{}, positionGradientPipelineLayout_{}, positionSgdPipelineLayout_{}, positionAdamwPipelineLayout_{},
        loraPipelineLayout_{}, loraSgdPipelineLayout_{},
        loraClearPipelineLayout_{}, loraFinalizePipelineLayout_{}, loraAdamwPipelineLayout_{};
    vk::Pipeline gammaAdamwPipeline_{}, lmHeadForwardPipeline_{}, ffnForwardPipeline_{}, ffnBackwardPipeline_{}, ffnGradientPipeline_{}, rmsNormGradientPipeline_{}, scaleDgammaPipeline_{}, ffnAdamwPipeline_{}, ffnB1AdamwPipeline_{}, ffnW2AdamwPipeline_{}, ffnB2AdamwPipeline_{}, lossPipeline_{}, metricsPipeline_{}, lmBackwardPipeline_{}, lmHeadGradientPipeline_{},
        lmHeadSgdPipeline_{}, lmHeadAdamwPipeline_{}, outputGradientPipeline_{}, outputSgdPipeline_{},
        outputAdamwPipeline_{}, projectionBackwardPipeline_{}, attentionBackwardPipeline_{}, qkvGradientPipeline_{},
        qkvSgdPipeline_{}, qkvAdamwPipeline_{}, embeddingGradientPipeline_{}, embeddingSgdPipeline_{}, positionGradientPipeline_{}, positionSgdPipeline_{}, positionAdamwPipeline_{}, loraPipeline_{},
        loraSgdPipeline_{}, loraClearPipeline_{},
        loraFinalizePipeline_{}, loraAdamwPipeline_{};
    vk::DescriptorPool gammaAdamwPool_{}, lmHeadForwardPool_{}, ffnForwardPool_{}, ffnBackwardPool_{}, ffnGradientPool_{}, rmsNormGradientPool_{}, scaleDgammaPool_{}, ffnAdamwPool_{}, ffnB1AdamwPool_{}, ffnW2AdamwPool_{}, ffnB2AdamwPool_{}, lossPool_{}, metricsPool_{}, lmBackwardPool_{}, lmHeadGradientPool_{}, lmHeadSgdPool_{},
        lmHeadAdamwPool_{}, outputGradientPool_{}, outputSgdPool_{}, outputAdamwPool_{}, projectionBackwardPool_{},
        attentionBackwardPool_{}, qkvGradientPool_{}, qkvSgdPool_{}, qkvAdamwPool_{}, embeddingGradientPool_{}, embeddingSgdPool_{}, positionGradientPool_{}, positionSgdPool_{}, positionAdamwPool_{},
        loraPool_{}, loraSgdPool_{},
        loraClearPool_{}, loraFinalizePool_{}, loraAdamwPool_{};
    vk::DescriptorSet gammaAdamwSet_{}, lmHeadForwardSet_{}, ffnForwardSet_{}, ffnBackwardSet_{}, ffnGradientSet_{}, rmsNormGradientSet_{}, scaleDgammaSet_{}, ffnAdamwSet_{}, ffnB1AdamwSet_{}, ffnW2AdamwSet_{}, ffnB2AdamwSet_{}, lossSet_{}, metricsSet_{}, lmBackwardSet_{}, lmHeadGradientSet_{}, lmHeadSgdSet_{},
        outputGradientSet_{}, outputSgdSet_{}, projectionBackwardSet_{}, projectionBackwardQkvSet_{}, qkvGradientSet_{},
        qkvSgdSet_{}, embeddingGradientSet_{}, embeddingSgdSet_{}, positionGradientSet_{}, positionSgdSet_{}, positionAdamwSet_{}, attentionBackwardSet_{}, attentionBackwardQSet_{}, attentionBackwardKSet_{},
        attentionBackwardVSet_{}, lmHeadAdamwSet_{}, outputAdamwSet_{}, qkvAdamwSet_{}, loraSet_{}, loraSgdSet_{},
        loraClearSet_{}, loraFinalizeSet_{}, loraAdamwSet_{};
    vk::DescriptorSet projectionBackwardKVSet_{}, projectionBackwardVVSet_;
    std::unique_ptr<Buffer> trainingStaging_;
    std::uint64_t lastSubmission_{0};
    bool accumulating_{false};
    bool fixedForwardRecorded_{false};
    bool fixedForwardLossRecorded_{false};
    static constexpr vk::DeviceSize fixedForwardStagingBytes = Tcap * sizeof(std::uint32_t);
};
} // namespace vulkan_runtime::tiny

#ifdef __cplusplus
extern "C" {
#endif
typedef struct spaceslug_tiny_forward_graph spaceslug_tiny_forward_graph;
const char* spaceslug_tiny_forward_capability(void);
const char* spaceslug_tiny_forward_command_buffer_capability(void);
enum {
    SPACESLUG_TINY_BASE_TRAIN_GROUP_NONE = 0,
    SPACESLUG_TINY_BASE_TRAIN_GROUP_LM_HEAD = 1,
    SPACESLUG_TINY_BASE_TRAIN_GROUP_OUTPUT = 2,
    SPACESLUG_TINY_BASE_TRAIN_GROUP_QKV = 3,
    SPACESLUG_TINY_BASE_TRAIN_GROUP_EMBEDDINGS = 4,
    SPACESLUG_TINY_BASE_TRAIN_GROUP_POSITIONS = 5,
    SPACESLUG_TINY_BASE_TRAIN_GROUP_NORMALIZATION = 6,
    SPACESLUG_TINY_BASE_TRAIN_GROUP_FFN = 7,
};
const char* spaceslug_tiny_forward_base_train_capability(void);
const char* spaceslug_tiny_forward_full_base_training_capability(void);
const char* spaceslug_tiny_forward_bounded_full_graph_training_capability(void);
const char* spaceslug_tiny_forward_retained_backward_optimizer_capability(void);
int spaceslug_tiny_forward_retained_backward_optimizer_supported(void);
const char* spaceslug_tiny_forward_arbitrary_shape_full_base_capability(void);
int spaceslug_tiny_forward_arbitrary_shape_full_base_supported(void);
int spaceslug_tiny_forward_readback_gamma_state(spaceslug_tiny_forward_graph*, float*, float*, float*, uint64_t*);
int spaceslug_tiny_forward_update_gamma_state(spaceslug_tiny_forward_graph*, float const*, float const*, float const*, uint64_t);
int spaceslug_tiny_forward_readback_combined_gradients(spaceslug_tiny_forward_graph*, float*, float*, size_t, float*, float*, float*, float*, uint32_t);
int spaceslug_tiny_forward_readback_ffn_state(spaceslug_tiny_forward_graph*, float*, size_t, uint64_t*);
int spaceslug_tiny_forward_update_ffn_state(spaceslug_tiny_forward_graph*, float const*, size_t, uint64_t);
const char* spaceslug_tiny_forward_ffn_capability(void);
int spaceslug_tiny_forward_base_train_group_supported(uint32_t group);
const char* spaceslug_tiny_forward_graph_embedding_training_capability(void);
int spaceslug_tiny_forward_graph_embedding_training_status(void);
int spaceslug_tiny_forward_train_embeddings_sgd(spaceslug_tiny_forward_graph*,
                                                const uint32_t*,
                                                const uint32_t*,
                                                const uint32_t*,
                                                uint32_t,
                                                float);
typedef struct spaceslug_tiny_base_checkpoint spaceslug_tiny_base_checkpoint;
spaceslug_tiny_base_checkpoint* spaceslug_tiny_base_checkpoint_create(void);
void spaceslug_tiny_base_checkpoint_destroy(spaceslug_tiny_base_checkpoint*);
int spaceslug_tiny_forward_readback_base_checkpoint(spaceslug_tiny_forward_graph*, spaceslug_tiny_base_checkpoint*);
int spaceslug_tiny_forward_update_base_checkpoint(spaceslug_tiny_forward_graph*, spaceslug_tiny_base_checkpoint const*);
uint32_t spaceslug_tiny_base_checkpoint_group_mask(const spaceslug_tiny_base_checkpoint*);
uint64_t spaceslug_tiny_base_checkpoint_adamw_step(const spaceslug_tiny_base_checkpoint*);
uint32_t spaceslug_tiny_base_checkpoint_profile_rank(const spaceslug_tiny_base_checkpoint*);
uint32_t spaceslug_tiny_base_checkpoint_float_count(const spaceslug_tiny_base_checkpoint*, uint32_t group);
float const* spaceslug_tiny_base_checkpoint_weights(const spaceslug_tiny_base_checkpoint*, uint32_t group);
uint32_t spaceslug_tiny_base_checkpoint_positions_float_count(const spaceslug_tiny_base_checkpoint*);
float const* spaceslug_tiny_base_checkpoint_positions(const spaceslug_tiny_base_checkpoint*);
uint32_t spaceslug_tiny_base_checkpoint_embeddings_float_count(const spaceslug_tiny_base_checkpoint*);
float const* spaceslug_tiny_base_checkpoint_embeddings(const spaceslug_tiny_base_checkpoint*);
float const* spaceslug_tiny_base_checkpoint_qkv_weights(const spaceslug_tiny_base_checkpoint*, uint32_t projection);
float const* spaceslug_tiny_base_checkpoint_adamw_m(const spaceslug_tiny_base_checkpoint*, uint32_t group);
float const* spaceslug_tiny_base_checkpoint_adamw_v(const spaceslug_tiny_base_checkpoint*, uint32_t group);
uint32_t spaceslug_tiny_base_checkpoint_state_float_count(const spaceslug_tiny_base_checkpoint*, uint32_t group);
float const* spaceslug_tiny_base_checkpoint_normalization_state(const spaceslug_tiny_base_checkpoint*, uint32_t state);
float const* spaceslug_tiny_base_checkpoint_ffn_state(const spaceslug_tiny_base_checkpoint*, uint32_t component, uint32_t state);
int spaceslug_tiny_forward_import_base_train_lm_head(spaceslug_tiny_forward_graph*, float const* weight);
int spaceslug_tiny_forward_import_base_train_output(spaceslug_tiny_forward_graph*, float const* weight);
int spaceslug_tiny_forward_readback_base_train_output(spaceslug_tiny_forward_graph*, float* weight);
int spaceslug_tiny_forward_readback_base_train_output_adamw_state(spaceslug_tiny_forward_graph*,
                                                                  float*,
                                                                  float*,
                                                                  float*,
                                                                  uint64_t*);
int spaceslug_tiny_forward_update_base_train_output_adamw_state(spaceslug_tiny_forward_graph*,
                                                                float const*,
                                                                float const*,
                                                                float const*,
                                                                uint64_t);
int spaceslug_tiny_forward_readback_base_train_lm_head(spaceslug_tiny_forward_graph*, float* weight);
int spaceslug_tiny_forward_readback_base_train_positions_adamw_state(spaceslug_tiny_forward_graph*, float*, float*, float*, uint64_t*);
int spaceslug_tiny_forward_update_base_train_positions_adamw_state(spaceslug_tiny_forward_graph*, float const*, float const*, float const*, uint64_t);
int spaceslug_tiny_forward_readback_base_train_lm_head_adamw_state(spaceslug_tiny_forward_graph*,
                                                                   float*,
                                                                   float*,
                                                                   float*,
                                                                   uint64_t*);
int spaceslug_tiny_forward_update_base_train_lm_head_adamw_state(spaceslug_tiny_forward_graph*,
                                                                 float const*,
                                                                 float const*,
                                                                 float const*,
                                                                 uint64_t);
int spaceslug_tiny_forward_train_lm_head_sgd(spaceslug_tiny_forward_graph*,
                                             const uint32_t*,
                                             const uint32_t*,
                                             const uint32_t*,
                                             uint32_t,
                                             float);
int spaceslug_tiny_forward_train_output_sgd(spaceslug_tiny_forward_graph*,
                                            const uint32_t*,
                                            const uint32_t*,
                                            const uint32_t*,
                                            uint32_t,
                                            float);
int spaceslug_tiny_forward_train_output_adamw(spaceslug_tiny_forward_graph*,
                                              const uint32_t*,
                                              const uint32_t*,
                                              const uint32_t*,
                                              uint32_t,
                                              float,
                                              float,
                                              float,
                                              float,
                                              float);
int spaceslug_tiny_forward_import_base_train_qkv(spaceslug_tiny_forward_graph*,
                                                 float const*,
                                                 float const*,
                                                 float const*);
int spaceslug_tiny_forward_readback_base_train_qkv(spaceslug_tiny_forward_graph*, float*, float*, float*);
int spaceslug_tiny_forward_readback_base_train_qkv_gradients(spaceslug_tiny_forward_graph*, float*, float*, float*);
int spaceslug_tiny_forward_readback_base_train_qkv_adamw_state(spaceslug_tiny_forward_graph*,
                                                               float*,
                                                               float*,
                                                               float*,
                                                               float*,
                                                               float*,
                                                               float*,
                                                               float*,
                                                               float*,
                                                               float*,
                                                               uint64_t*);
int spaceslug_tiny_forward_update_base_train_qkv_adamw_state(spaceslug_tiny_forward_graph*,
                                                             float const*,
                                                             float const*,
                                                             float const*,
                                                             float const*,
                                                             float const*,
                                                             float const*,
                                                             float const*,
                                                             float const*,
                                                             float const*,
                                                             uint64_t);
int spaceslug_tiny_forward_train_qkv_adamw_from_gradients(spaceslug_tiny_forward_graph*,
                                                          float,
                                                          float,
                                                          float,
                                                          float,
                                                          float);
int spaceslug_tiny_forward_readback_positions(spaceslug_tiny_forward_graph*, float* positions);
int spaceslug_tiny_forward_readback_position_gradient(spaceslug_tiny_forward_graph*, float*, size_t);
int spaceslug_tiny_forward_import_base_train_embeddings(spaceslug_tiny_forward_graph*, float const*);
int spaceslug_tiny_forward_readback_base_train_embeddings(spaceslug_tiny_forward_graph*, float*);
int spaceslug_tiny_forward_import_base_train_positions(spaceslug_tiny_forward_graph*, float const*);
int spaceslug_tiny_forward_readback_base_train_positions(spaceslug_tiny_forward_graph*, float*);
int spaceslug_tiny_forward_train_positions_sgd(spaceslug_tiny_forward_graph*,
                                                const uint32_t*,
                                                const uint32_t*,
                                                const uint32_t*,
                                                uint32_t,
                                                float);
int spaceslug_tiny_forward_train_positions_adamw(spaceslug_tiny_forward_graph*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, float, float, float, float, float);
// Explicit name for the bounded graph-owned gamma/FFN/position AdamW chain.
int spaceslug_tiny_forward_train_bounded_full_graph_adamw(spaceslug_tiny_forward_graph*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t, float, float, float, float, float);
int spaceslug_tiny_forward_train_qkv_sgd(spaceslug_tiny_forward_graph*,
                                         const uint32_t*,
                                         const uint32_t*,
                                         const uint32_t*,
                                         uint32_t,
                                         float);
int spaceslug_tiny_forward_train_lm_head_adamw(spaceslug_tiny_forward_graph*,
                                               const uint32_t*,
                                               const uint32_t*,
                                               const uint32_t*,
                                               uint32_t,
                                               float,
                                               float,
                                               float,
                                               float,
                                               float);
typedef struct spaceslug_tiny_profile_descriptor {
    char const* name;
    uint32_t hidden;
    uint32_t vocab;
    uint32_t padded_vocab;
    uint32_t token_capacity;
    uint32_t lora_rank;
} spaceslug_tiny_profile_descriptor;
typedef enum spaceslug_tiny_profile_status {
    SPACESLUG_TINY_PROFILE_SUPPORTED = 0,
    SPACESLUG_TINY_PROFILE_UNSUPPORTED = 1,
    SPACESLUG_TINY_PROFILE_INVALID_ARGUMENT = 2
} spaceslug_tiny_profile_status;
uint32_t spaceslug_tiny_profile_count(void);
int spaceslug_tiny_profile_query(uint32_t index, spaceslug_tiny_profile_descriptor* out);
int spaceslug_tiny_profile_validate(uint32_t hidden,
                                    uint32_t vocab,
                                    uint32_t padded_vocab,
                                    uint32_t token_capacity,
                                    uint32_t rank);
int spaceslug_tiny_forward_lora_rank_supported(uint32_t rank);
int spaceslug_tiny_forward_train_dataset_batch(spaceslug_tiny_forward_graph*, void*, float, float);
int spaceslug_tiny_forward_train_dataset_batch_full(spaceslug_tiny_forward_graph*, void*, float, float);
void* spaceslug_tiny_forward_create_dataset_batch(spaceslug_tiny_forward_graph*, uint32_t, uint32_t);
void spaceslug_tiny_forward_destroy_dataset_batch(void*);
int spaceslug_tiny_forward_upload_dataset_batch(void*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*);
const char* spaceslug_tiny_forward_dataset_training_capability(void);
spaceslug_tiny_forward_graph* spaceslug_tiny_forward_create(float const*, float const*, float const*);
spaceslug_tiny_forward_graph*
spaceslug_tiny_forward_create_rank(float const*, float const*, float const*, uint32_t rank);
spaceslug_tiny_forward_graph* spaceslug_tiny_forward_create_full(float const*,
                                                                 float const*,
                                                                 float const*,
                                                                 float const*,
                                                                 float const*,
                                                                 float const*,
                                                                 float const*);
int spaceslug_tiny_forward(const spaceslug_tiny_forward_graph*, const uint32_t*, uint32_t, float*, uint32_t);
int spaceslug_tiny_forward_fixed_retained(spaceslug_tiny_forward_graph*, const uint32_t*, float*);
int spaceslug_tiny_forward_loss_fixed_retained(spaceslug_tiny_forward_graph*,
                                               const uint32_t*,
                                               const uint32_t*,
                                               const uint32_t*,
                                               float*,
                                               float*);
int spaceslug_tiny_forward_loss_fixed_metrics(spaceslug_tiny_forward_graph*,
                                              const uint32_t*,
                                              const uint32_t*,
                                              const uint32_t*,
                                              float*,
                                              uint32_t*);
int spaceslug_tiny_forward_token_step(spaceslug_tiny_forward_graph*, uint32_t, uint32_t, float*);
int spaceslug_tiny_forward_token_step_training(spaceslug_tiny_forward_graph*,
                                               uint32_t,
                                               uint32_t,
                                               uint32_t,
                                               uint32_t,
                                               float*,
                                               float*,
                                               float*);
int spaceslug_tiny_forward_begin_lora_accumulation(spaceslug_tiny_forward_graph*);
int spaceslug_tiny_forward_token_step_training_backward_accumulate(spaceslug_tiny_forward_graph*,
                                                                   uint32_t,
                                                                   uint32_t,
                                                                   uint32_t,
                                                                   uint32_t,
                                                                   float const*,
                                                                   float*,
                                                                   float*,
                                                                   float*,
                                                                   float*,
                                                                   float*,
                                                                   float*,
                                                                   float*,
                                                                   float*);
int spaceslug_tiny_forward_token_windows_training_backward_accumulate(spaceslug_tiny_forward_graph*,
                                                                      const uint32_t*,
                                                                      const uint32_t*,
                                                                      const uint32_t*,
                                                                      uint32_t,
                                                                      uint32_t,
                                                                      float*);
int spaceslug_tiny_forward_finalize_lora_sgd(spaceslug_tiny_forward_graph*, float, float);
int spaceslug_tiny_forward_begin_lora_adamw(spaceslug_tiny_forward_graph*);
int spaceslug_tiny_forward_finalize_lora_adamw(spaceslug_tiny_forward_graph*, float, float, float, float, float, float);
int spaceslug_tiny_forward_readback_lora_adamw_state(spaceslug_tiny_forward_graph*, float*, float*, float*, uint64_t*);
int spaceslug_tiny_forward_update_lora_adamw_state(spaceslug_tiny_forward_graph*,
                                                   float const*,
                                                   float const*,
                                                   float const*,
                                                   uint64_t);
int spaceslug_tiny_forward_token_step_training_backward(spaceslug_tiny_forward_graph*,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        uint32_t,
                                                        float const*,
                                                        float*,
                                                        float*,
                                                        float*,
                                                        float*,
                                                        float*,
                                                        float*,
                                                        float*,
                                                        float*);
int spaceslug_tiny_forward_readback_graph_dstate(spaceslug_tiny_forward_graph*,
                                                  uint32_t const*,
                                                  uint32_t const*,
                                                  uint32_t const*,
                                                  uint32_t,
                                                  float*);
int spaceslug_tiny_forward_readback_lora_adapters(spaceslug_tiny_forward_graph*,
                                                  float*,
                                                  float*,
                                                  float*,
                                                  float*,
                                                  float*,
                                                  float*,
                                                  float*,
                                                  float*);
int spaceslug_tiny_forward_update_lora_adapters(spaceslug_tiny_forward_graph*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*,
                                                float const*);
void spaceslug_tiny_forward_destroy(spaceslug_tiny_forward_graph*);
#ifdef __cplusplus
}
#endif
