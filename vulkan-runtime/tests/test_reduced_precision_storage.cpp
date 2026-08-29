#include "api/reduced_precision_storage.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using vulkan_runtime::storage::fp16_to_fp32;
    using vulkan_runtime::storage::fp32_to_fp16;

    std::vector<float> source = {-0.0f, 0.0f, 1.0f, -2.5f, 65504.0f, 1.0e-4f,
                                 std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN()};
    std::vector<std::uint16_t> encoded(source.size());
    std::vector<float> decoded(source.size());
    fp32_to_fp16(source.data(), encoded.data(), source.size());
    fp16_to_fp32(encoded.data(), decoded.data(), decoded.size());

    for (std::size_t i = 0; i < source.size(); ++i) {
        if (std::isnan(source[i])) {
            if (!std::isnan(decoded[i])) {
                std::cerr << "NaN did not survive FP16 storage conversion\n";
                return 1;
            }
        } else if (std::isinf(source[i])) {
            if (!std::isinf(decoded[i]) || std::signbit(source[i]) != std::signbit(decoded[i])) {
                std::cerr << "infinity did not survive FP16 storage conversion\n";
                return 1;
            }
        } else if (std::abs(decoded[i] - source[i]) > std::max(1.0e-7f, std::abs(source[i]) * 1.0e-3f)) {
            std::cerr << "FP16 storage conversion exceeded expected rounding\n";
            return 1;
        }
    }

    auto context = vulkan_runtime::core::create_context("reduced_precision_storage");
    auto capabilities = vulkan_runtime::storage::query_capabilities(context);
    if (!capabilities.packedFp16Storage || !capabilities.fp32ConversionAndAccumulation ||
        capabilities.fp16ArithmeticEnabled) {
        std::cerr << "invalid reduced-precision capability contract\n";
        vulkan_runtime::core::destroy_context(context);
        return 1;
    }
    std::cout << "reduced_precision_storage: "
              << context.physicalDevice.getProperties().deviceName << " storage=fp16 math=fp32\n";
    vulkan_runtime::core::destroy_context(context);
    return 0;
}
