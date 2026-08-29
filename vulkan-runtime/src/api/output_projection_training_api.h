#pragma once
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct spaceslug_output_projection_training spaceslug_output_projection_training;
const char* spaceslug_output_projection_training_capability(void);
// Standalone FP32 row-major [hidden,hidden] output projection. Inputs are
// caller-supplied projected activations and upstream gradients, both [tcap,H].
spaceslug_output_projection_training* spaceslug_output_projection_training_create(float const* weight, std::uint32_t hidden, std::uint32_t tcap);
void spaceslug_output_projection_training_destroy(spaceslug_output_projection_training* handle);
int spaceslug_output_projection_training_step(spaceslug_output_projection_training* handle, float const* activation, float const* upstream, std::uint32_t rows, float learning_rate);
// Persistent AdamW update with bias correction, epsilon, and decoupled decay.
int spaceslug_output_projection_training_adamw_step(
    spaceslug_output_projection_training* handle, float const* activation,
    float const* upstream, std::uint32_t rows, float learning_rate,
    float beta1, float beta2, float epsilon, float weight_decay);
int spaceslug_output_projection_training_readback_adamw_state(
    spaceslug_output_projection_training* handle, float* weight, float* gradient,
    float* m, float* v, std::uint64_t* step);
int spaceslug_output_projection_training_update_adamw_state(
    spaceslug_output_projection_training* handle, float const* weight,
    float const* m, float const* v, std::uint64_t step);
int spaceslug_output_projection_training_readback(spaceslug_output_projection_training* handle, float* weight, float* gradient);
int spaceslug_output_projection_training_update(spaceslug_output_projection_training* handle, float const* weight);
#ifdef __cplusplus
}
#endif

