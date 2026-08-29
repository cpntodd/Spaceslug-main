#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spaceslug_lora_session spaceslug_lora_session;

// Creates a persistent fixed-width LoRA trainer. Initial A/B are copied into
// device-resident buffers; X and dY are supplied to each step call. Shapes are
// X/dY[M,64], A[64,R], B[R,64], row-major fp32.
// Returns 0 on success, 1 invalid arguments, 2 Vulkan/runtime failure.
int spaceslug_lora_session_create(uint32_t m, uint32_t rank, float learning_rate,
                                  float const* a, float const* b,
                                  spaceslug_lora_session** out_session);

// Runs delta -> gradients -> SGD. X and dY must contain M*64 fp32 values. A/B
// remain device-resident; if y is non-null, it receives the pre-update delta.
// The call is synchronous: on return all device writes from this step are done.
int spaceslug_lora_session_step(spaceslug_lora_session* session,
                                float const* x, float const* dy, float* y);

// Token-derived graph boundary: the persistent tensor session does not yet own
// embeddings, positions, attention, logits, or loss buffers.
// Returns 3 (unsupported) without modifying the session.
int spaceslug_lora_session_token_step(spaceslug_lora_session* session,
                                      uint32_t const* tokens, uint32_t const* targets,
                                      uint32_t const* mask, uint32_t rows,
                                      float* loss);

// Copies persistent device-resident A/B back to host for checkpointing.
int spaceslug_lora_session_readback(spaceslug_lora_session* session, float* a, float* b);

// Destroys a session. NULL is accepted as a no-op.
int spaceslug_lora_session_destroy(spaceslug_lora_session* session);

#ifdef __cplusplus
} // extern "C"
#endif
