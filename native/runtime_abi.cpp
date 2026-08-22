#include "runtime_abi.h"
#include <stddef.h>
int spaceslug_vector_add(const float* left, const float* right, float* output, size_t count) {
    if (!left || !right || !output) return 1;
    for (size_t i = 0; i < count; ++i) output[i] = left[i] + right[i];
    return 0;
}
