#include "api/reduced_precision_storage.h"

#include <bit>
namespace vulkan_runtime::storage {
namespace {
constexpr char kCapability[] = "fp16_storage_only_fp32_conversion_accumulation";

std::uint32_t round_shift(std::uint32_t value, int shift) noexcept {
    std::uint32_t const base = value >> shift;
    std::uint32_t const remainder = value & ((1u << shift) - 1u);
    std::uint32_t const halfway = 1u << (shift - 1);
    return base + ((remainder > halfway || (remainder == halfway && (base & 1u))) ? 1u : 0u);
}
} // namespace

Capabilities query_capabilities(core::VulkanContext const& context) noexcept {
    Capabilities result;
    vk::PhysicalDeviceShaderFloat16Int8Features float16Features;
    vk::PhysicalDeviceFeatures2 features2;
    features2.setPNext(&float16Features);
    context.physicalDevice.getFeatures2(&features2);
    result.physicalFp16Arithmetic = float16Features.shaderFloat16 == VK_TRUE;
    result.fp16ArithmeticEnabled = false; // create_context deliberately does not enable shaderFloat16.
    result.storageBuffer16BitAccess = false; // packed uint32 path needs no optional 16-bit storage feature.
    return result;
}

std::uint16_t fp32_to_fp16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10)
            return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        int shift = 14 - exponent;
        return static_cast<std::uint16_t>(sign | round_shift(mantissa, shift));
    }
    if (exponent >= 31) {
        if ((bits & 0x7fffffffU) >= 0x7f800000U)
            return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa >> 13u));
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    std::uint32_t rounded = round_shift(mantissa, 13);
    if (rounded == 0x400u) {
        rounded = 0;
        ++exponent;
        if (exponent >= 31)
            return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10u) | rounded);
}

float fp16_to_fp32(std::uint16_t bits) noexcept {
    std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000u) << 16u;
    std::int32_t exponent = static_cast<std::int32_t>((bits >> 10u) & 0x1fu);
    std::uint32_t mantissa = bits & 0x3ffu;
    std::uint32_t out;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x3ffu;
            out = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        out = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    return std::bit_cast<float>(out);
}

void fp32_to_fp16(float const* input, std::uint16_t* output, std::size_t count) noexcept {
    if (!input || !output)
        return;
    for (std::size_t i = 0; i < count; ++i)
        output[i] = fp32_to_fp16(input[i]);
}

void fp16_to_fp32(std::uint16_t const* input, float* output, std::size_t count) noexcept {
    if (!input || !output)
        return;
    for (std::size_t i = 0; i < count; ++i)
        output[i] = fp16_to_fp32(input[i]);
}

} // namespace vulkan_runtime::storage

extern "C" const char* spaceslug_storage_capability(void) {
    return "fp16_storage_only_fp32_conversion_accumulation";
}

extern "C" spaceslug_storage_capabilities spaceslug_storage_capabilities_query(void) {
    spaceslug_storage_capabilities result{};
    result.packed_fp16_storage = 1;
    result.fp32_conversion_and_accumulation = 1;
    return result;
}
