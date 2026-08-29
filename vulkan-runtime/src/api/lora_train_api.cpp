#include "api/lora_train_api.h"
#include "api/lora_delta_api.h"
#include "api/lora_gradients_api.h"
#include "api/lora_sgd_api.h"
#include <cstdint>
#include <vector>

extern "C" int spaceslug_lora_train_step(float const* x, float const* dy, float* a, float* b,
                                         float* y, float learning_rate, std::uint32_t m, std::uint32_t rank) {
    if (!x || !dy || !a || !b || !y || learning_rate <= 0.0f || m == 0 || m > 128 || rank == 0 || rank > 8) return 1;
    std::vector<float> da(std::size_t(64) * rank);
    std::vector<float> db(std::size_t(rank) * 64);
    if (spaceslug_lora_delta(x, a, b, y, m, rank) != 0) return 2;
    if (spaceslug_lora_gradients(x, dy, a, b, da.data(), db.data(), m, rank) != 0) return 2;
    if (spaceslug_lora_sgd(a, b, da.data(), db.data(), learning_rate, rank) != 0) return 2;
    return 0;
}
