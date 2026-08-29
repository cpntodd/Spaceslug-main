#pragma once
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct spaceslug_qkv_projection_training spaceslug_qkv_projection_training;
const char* spaceslug_qkv_projection_training_capability(void);
// Standalone FP32 row-major [H,H] Q/K/V projections. Caller supplies states X and
// dquery/dkey/dvalue, each [tcap,H]. No Tiny graph integration is performed.
spaceslug_qkv_projection_training* spaceslug_qkv_projection_training_create(
    float const* query_weight, float const* key_weight, float const* value_weight,
    std::uint32_t hidden, std::uint32_t tcap);
void spaceslug_qkv_projection_training_destroy(spaceslug_qkv_projection_training* handle);
int spaceslug_qkv_projection_training_step(spaceslug_qkv_projection_training* handle,
    float const* states, float const* dquery, float const* dkey, float const* dvalue,
    std::uint32_t rows, float learning_rate);
int spaceslug_qkv_projection_training_readback(spaceslug_qkv_projection_training* handle,
    float* query_weight, float* key_weight, float* value_weight,
    float* query_gradient, float* key_gradient, float* value_gradient);
#ifdef __cplusplus
}
#endif
