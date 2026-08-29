#pragma once
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct spaceslug_embedding_training spaceslug_embedding_training;
const char* spaceslug_embedding_training_capability(void);
spaceslug_embedding_training* spaceslug_embedding_training_create(float const* weight, std::uint32_t vocab, std::uint32_t hidden);
void spaceslug_embedding_training_destroy(spaceslug_embedding_training* handle);
int spaceslug_embedding_training_step(spaceslug_embedding_training* handle, std::uint32_t const* token_ids, float const* dstate, std::uint8_t const* mask, std::uint32_t rows, float learning_rate);
int spaceslug_embedding_training_readback(spaceslug_embedding_training* handle, float* weight, float* gradient);
int spaceslug_embedding_training_update(spaceslug_embedding_training* handle, float const* weight);
#ifdef __cplusplus
}
#endif
