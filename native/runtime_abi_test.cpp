#include "runtime_abi.h"
#include <cstdio>
int main() {
    float a[3] = {0.25f, 1.5f, -2.0f};
    float b[3] = {0.75f, 2.5f, 3.0f};
    float out[3] = {};
    if (spaceslug_vector_add(a, b, out, 3) != 0 || out[0] != 1.0f || out[1] != 4.0f || out[2] != 1.0f) return 1;
    std::puts("runtime_abi vector_add: PASS");
    return 0;
}
