// Tiny integrated training versus retained forward/loss profile.
//
// This is intentionally a host wall-clock benchmark: the public Tiny API waits
// for each synchronous operation, so each sample includes staging map/flush,
// command recording/submission, GPU execution, and the required wait/readback.
// It is a profile artifact, not a performance gate.
#include "api/tiny_forward_persistent.h"
#include "bench/bench_common.h"
#include "core/vk_setup.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kRows = vulkan_runtime::tiny::Tcap;
constexpr std::uint32_t kRuns = 9;
constexpr std::uint32_t kWarmup = 128;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct Inputs {
    std::vector<float> embeddings, positions, query, key, value, output, lm_head;
    std::vector<std::uint32_t> tokens, targets, masks;
    Inputs()
        : embeddings(vulkan_runtime::tiny::V * vulkan_runtime::tiny::H),
          positions(vulkan_runtime::tiny::Tcap * vulkan_runtime::tiny::H),
          query(vulkan_runtime::tiny::H * vulkan_runtime::tiny::H),
          key(query.size()),
          value(query.size()),
          output(query.size()),
          lm_head(vulkan_runtime::tiny::H * vulkan_runtime::tiny::Vp),
          tokens(kRows),
          targets(kRows),
          masks(kRows, 1) {
        for (std::size_t i = 0; i < embeddings.size(); ++i) embeddings[i] = 0.001f * float(i % 17);
        for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = 0.002f * float(i % 13);
        for (std::size_t i = 0; i < lm_head.size(); ++i) lm_head[i] = 0.003f * float(i % 11);
        for (std::uint32_t i = 0; i < vulkan_runtime::tiny::H; ++i)
            query[i * vulkan_runtime::tiny::H + i] = key[i * vulkan_runtime::tiny::H + i] =
                value[i * vulkan_runtime::tiny::H + i] = output[i * vulkan_runtime::tiny::H + i] = 1.0f;
        for (std::uint32_t i = 0; i < kRows; ++i) {
            tokens[i] = (i * 11u) % vulkan_runtime::tiny::V;
            targets[i] = (i * 5u + 3u) % vulkan_runtime::tiny::V;
        }
    }
};
} // namespace

int main() {
    using namespace vulkan_runtime;
    try {
        auto ctx = core::create_context("vulkan-runtime-tiny-training-bench");
        if (!bench::is_discrete_gpu(ctx)) {
            std::cout << "bench skipped: not a discrete GPU\n";
            core::destroy_context(ctx);
            return EXIT_SUCCESS;
        }
        auto props = ctx.physicalDevice.getProperties();
        std::cout << "device: " << props.deviceName << "\n";
        Inputs in;
        auto graph = std::make_unique<tiny::ForwardResourceGraph>(ctx,
                                                                   in.embeddings.data(),
                                                                   in.positions.data(),
                                                                   in.query.data(),
                                                                   in.key.data(),
                                                                   in.value.data(),
                                                                   in.output.data(),
                                                                   in.lm_head.data());
        std::vector<float> loss(kRows * tiny::Vp), row_loss(kRows);
        float metric_loss = 0.0f;
        std::uint32_t metric_count = 0;

        // Warm both paths enough to move the RX580 out of its idle clock state.
        for (std::uint32_t i = 0; i < kWarmup; ++i) {
            graph->train_lm_head_sgd(in.tokens.data(), in.targets.data(), in.masks.data(), kRows, 1.0e-8f);
            graph->forward_loss_fixed_metrics(in.tokens.data(), in.targets.data(), in.masks.data(), &metric_loss, &metric_count);
        }

        std::vector<double> training_ms, retained_metrics_ms, retained_logits_ms;
        training_ms.reserve(kRuns);
        retained_metrics_ms.reserve(kRuns);
        retained_logits_ms.reserve(kRuns);
        for (std::uint32_t i = 0; i < kRuns; ++i) {
            auto a = Clock::now();
            graph->train_lm_head_sgd(in.tokens.data(), in.targets.data(), in.masks.data(), kRows, 1.0e-8f);
            auto b = Clock::now();
            training_ms.push_back(ms(a, b));

            a = Clock::now();
            graph->forward_loss_fixed_metrics(in.tokens.data(), in.targets.data(), in.masks.data(), &metric_loss, &metric_count);
            b = Clock::now();
            retained_metrics_ms.push_back(ms(a, b));

            a = Clock::now();
            graph->forward_loss_fixed_retained(in.tokens.data(), in.targets.data(), in.masks.data(), loss.data(), row_loss.data());
            b = Clock::now();
            retained_logits_ms.push_back(ms(a, b));
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "profile rows=" << kRows << " warmup=" << kWarmup << " samples=" << kRuns << " median_ms\n"
                  << "normal_train_lm_head_sgd=" << median(training_ms) << "\n"
                  << "retained_forward_loss_metrics=" << median(retained_metrics_ms) << "\n"
                  << "retained_forward_loss_logits=" << median(retained_logits_ms) << "\n"
                  << "normal_over_retained_metrics=" << median(training_ms) / median(retained_metrics_ms) << "\n"
                  << "normal_over_retained_logits=" << median(training_ms) / median(retained_logits_ms) << "\n"
                  << "metric_count=" << metric_count << " loss=" << metric_loss << "\n";
        graph.reset();
        core::destroy_context(ctx);
        return EXIT_SUCCESS;
    } catch (std::exception const& e) {
        std::cerr << "bench_tiny_training: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
