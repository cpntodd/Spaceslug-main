#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    std::vector<float> embeddings(V * H), positions(Tcap * H), query(H * H), key(H * H), value(H * H), output(H * H), lm(H * Vp);
    for (std::size_t i = 0; i < embeddings.size(); ++i) embeddings[i] = 0.001f * float((i * 7) % 19);
    for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = 0.002f * float((i * 5) % 17);
    for (std::size_t i = 0; i < lm.size(); ++i) lm[i] = 0.003f * float((i * 11) % 23);
    for (std::uint32_t i = 0; i < H; ++i)
        query[i * H + i] = key[i * H + i] = value[i * H + i] = output[i * H + i] = 1.0f;

    auto context = vulkan_runtime::core::create_context("tiny-graph-dstate-test");
    {
    ForwardResourceGraph graph(context, embeddings.data(), positions.data(), query.data(), key.data(), value.data(), output.data(), lm.data());
    constexpr std::uint32_t rows = 4;
    std::vector<std::uint32_t> tokens{7, 7, 19, 7};
    std::vector<std::uint32_t> targets{3, 5, 9, 11};
    std::vector<std::uint32_t> masks{1, 0, 1, 1};
    std::vector<float> dstate(Tcap * H, -1.0f), repeat(Tcap * H, -2.0f);
    if (graph.readback_graph_dstate(tokens.data(), targets.data(), masks.data(), rows, dstate.data()) != 0 ||
        graph.readback_graph_dstate(tokens.data(), targets.data(), masks.data(), rows, repeat.data()) != 0) {
        std::cerr << "graph dstate readback failed\n";
        vulkan_runtime::core::destroy_context(context);
        return 1;
    }
    for (std::size_t i = 0; i < dstate.size(); ++i) {
        if (!std::isfinite(dstate[i]) || std::abs(dstate[i] - repeat[i]) > 1.0e-5f) {
            std::cerr << "dstate repeated readback mismatch at " << i << "\n";
            vulkan_runtime::core::destroy_context(context);
            return 1;
        }
        if (i >= rows * H && dstate[i] != 0.0f) {
            std::cerr << "unused dstate row is not zero\n";
            vulkan_runtime::core::destroy_context(context);
            return 1;
        }
    }
    std::vector<float> before_logits(rows * Vp), after_logits(rows * Vp);
    graph.forward(tokens.data(), rows, before_logits.data());
    if (graph.train_embeddings_sgd(tokens.data(), targets.data(), masks.data(), rows, 1.0e-3f) != 0) {
        std::cerr << "graph embedding SGD failed\n";
        vulkan_runtime::core::destroy_context(context);
        return 1;
    }
    graph.forward(tokens.data(), rows, after_logits.data());
    for (float value : after_logits)
        if (!std::isfinite(value)) {
            std::cerr << "embedding SGD produced non-finite forward output\n";
            return 1;
        }
    std::cout << "tiny graph dstate readback passed CPU-parity/repeated-token boundary checks\n";
    }
    vulkan_runtime::core::destroy_context(context);
    return 0;
}
