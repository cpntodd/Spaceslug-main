#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spaceslug_tiny_base_training spaceslug_tiny_base_training;

// Returns a stable capability string. This standalone API trains only an FP32
// LM head from caller-supplied projected activations and dlogits through normal
// ExecEngine submissions, with SGD or persistent AdamW state; it is not wired
// into Tiny forward activations, retains no training command buffer, and does
// not consume BatchBuffer resources.
const char* spaceslug_tiny_base_training_capability(void);

// Creates a trainable [hidden,padded_vocab] FP32 LM head. vocab must be <=
// padded_vocab and tcap is the fixed deterministic accumulation capacity.
spaceslug_tiny_base_training* spaceslug_tiny_base_training_create(
    float const* weight, std::uint32_t hidden, std::uint32_t vocab,
    std::uint32_t padded_vocab, std::uint32_t tcap);
void spaceslug_tiny_base_training_destroy(spaceslug_tiny_base_training* handle);

// Computes dW[h,v] = sum_t activation[t,h] * dlogits[t,v], using exactly tcap
// loop iterations (rows beyond rows are zero), then applies SGD in-place.
int spaceslug_tiny_base_training_step(spaceslug_tiny_base_training* handle,
                                      float const* activation,
                                      float const* dlogits,
                                      std::uint32_t rows,
                                      float learning_rate);

// Persistent AdamW update with bias correction, epsilon, and decoupled decay.
int spaceslug_tiny_base_training_adamw_step(spaceslug_tiny_base_training* handle,
                                            float const* activation,
                                            float const* dlogits,
                                            std::uint32_t rows,
                                            float learning_rate, float beta1,
                                            float beta2, float epsilon,
                                            float weight_decay);
int spaceslug_tiny_base_training_readback_adamw_state(
    spaceslug_tiny_base_training* handle, float* weight, float* gradient,
    float* m, float* v, std::uint64_t* step);
int spaceslug_tiny_base_training_update_adamw_state(
    spaceslug_tiny_base_training* handle, float const* weight, float const* m,
    float const* v, std::uint64_t step);

// Copies the current device weight and last gradient to host arrays.
int spaceslug_tiny_base_training_readback(spaceslug_tiny_base_training* handle,
                                          float* weight,
                                          float* gradient);
// Replaces the device weight from a host [hidden,padded_vocab] FP32 array.
int spaceslug_tiny_base_training_update(spaceslug_tiny_base_training* handle,
                                        float const* weight);

#ifdef __cplusplus
}
#endif
