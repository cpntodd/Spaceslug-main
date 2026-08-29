#pragma once

#include "core/vk_setup.h"

#include <cstddef>
#include <cstdint>

namespace vulkan_runtime::storage {

// Reduced precision is a storage format only. All runtime math remains FP32.
// Values are represented as IEEE-754 binary16 bits; callers may pack two values
// into one uint32 for device buffers without enabling shaderFloat16.
struct Capabilities {
    bool packedFp16Storage{true};
    bool storageBuffer16BitAccess{false};
    bool physicalFp16Arithmetic{false};
    bool fp16ArithmeticEnabled{false};
    bool fp32ConversionAndAccumulation{true};
};

Capabilities query_capabilities(core::VulkanContext const& context) noexcept;

std::uint16_t fp32_to_fp16(float value) noexcept;
float fp16_to_fp32(std::uint16_t bits) noexcept;
void fp32_to_fp16(float const* input, std::uint16_t* output, std::size_t count) noexcept;
void fp16_to_fp32(std::uint16_t const* input, float* output, std::size_t count) noexcept;

} // namespace vulkan_runtime::storage

#ifdef __cplusplus
extern "C" {
#endif

typedef struct spaceslug_storage_capabilities {
    int packed_fp16_storage;
    int storage_buffer_16bit_access;
    int physical_fp16_arithmetic;
    int fp16_arithmetic_enabled;
    int fp32_conversion_and_accumulation;
} spaceslug_storage_capabilities;

spaceslug_storage_capabilities spaceslug_storage_capabilities_query(void);
const char* spaceslug_storage_capability(void);

#ifdef __cplusplus
}
#endif
