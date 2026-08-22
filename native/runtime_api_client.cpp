#include "runtime_abi.h"
#include <cstdio>
#include <vector>

int main() {
    std::vector<float> a(1u << 20, 1.0F), b(1u << 20, 2.0F), output(a.size());
    if (spaceslug_vector_add(a.data(), b.data(), output.data(), output.size()) != 0) return 1;
    if (output.front() != 3.0F || output.back() != 3.0F) return 1;
    std::puts("host runtime API: PASS");
    return 0;
}
