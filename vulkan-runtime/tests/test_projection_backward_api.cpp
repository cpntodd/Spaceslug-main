#include "api/projection_backward_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
int main() {
    constexpr std::uint32_t rows = 3, in = 4, out = 5;
    std::vector<float> dy(rows * out), w(in * out), dx(rows * in), ref(rows * in, 0);
    for (std::size_t i = 0; i < dy.size(); ++i)
        dy[i] = std::sin(i * .07f);
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = std::cos(i * .11f);
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t i = 0; i < in; ++i)
            for (std::uint32_t o = 0; o < out; ++o)
                ref[r * in + i] += dy[r * out + o] * w[i * out + o];
    if (spaceslug_projection_backward(dy.data(), w.data(), dx.data(), rows, in, out) != 0)
        return 1;
    float e = 0;
    for (std::size_t i = 0; i < dx.size(); ++i)
        e = std::max(e, std::abs(dx[i] - ref[i]));
    if (e > 3e-5f)
        return 1;
    std::cout << "Projection backward: PASS max_abs_error=" << e << '\n';
}
