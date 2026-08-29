#include "api/dataset_batch_buffer.h"
#include "core/vk_setup.h"
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    try {
        auto context = vulkan_runtime::core::create_context("dataset-batch-buffer-test");
        constexpr std::uint32_t windows = 3, tokens = 5;
        {
        vulkan_runtime::dataset::BatchBuffer batch(context, windows, tokens);
        std::vector<std::uint32_t> t(windows * tokens), y(windows * tokens), m(windows * tokens, 1), c(windows);
        for (std::uint32_t round = 0; round < 6; ++round) {
            for (std::size_t i = 0; i < t.size(); ++i) { t[i] = std::uint32_t(i + round); y[i] = std::uint32_t(i % 4); m[i] = (i + round) % 3 != 0; }
            for (std::uint32_t i = 0; i < windows; ++i) c[i] = 100 + round * 10 + i;
            batch.upload(t, y, m, c);
            auto out = batch.process_readback();
            auto metrics = batch.metrics_readback();
            if (metrics != out) { std::cerr << "metrics mismatch\n"; return 1; }
            for (std::size_t i = 0; i < metrics.size(); i += 2) {
                if (!std::isfinite(metrics[i]) || !std::isfinite(metrics[i + 1])) {
                    std::cerr << "non-finite metrics\n";
                    return 1;
                }
            }
            for (std::uint32_t w = 0; w < windows; ++w) {
                float expected = 0.0f;
                for (std::uint32_t i = 0; i < tokens; ++i) { float e = float(t[w * tokens + i]) - float(y[w * tokens + i]); expected += float(m[w * tokens + i]) * e * e; }
                if (std::abs(out[w * 2] - expected) > 1e-4f || out[w * 2 + 1] != float(c[w])) { std::cerr << "window mismatch\n"; return 1; }
            }
        }
        if (batch.last_submission() != 18 || std::string(batch.capability()).find("lm_head_sgd_device_windows") == std::string::npos) { std::cerr << "submission/capability mismatch " << batch.last_submission() << " " << batch.capability() << "\n"; return 1; }
        context.device.waitIdle();
        }
        vulkan_runtime::core::destroy_context(context);
        std::cout << "dataset batch retained all fixed windows on device across repeated processing\n";
        return 0;
    } catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}
