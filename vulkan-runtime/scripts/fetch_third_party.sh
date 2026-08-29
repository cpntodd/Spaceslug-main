#!/usr/bin/env bash
# Regenerate the vendored third-party headers.
#
# These files are git-ignored (see the repo root .gitignore) and must be
# present before configuring the build. Run this script from the repo root:
#
#     ./scripts/fetch_third_party.sh
#
# It downloads:
#   - VulkanMemoryAllocator (VMA)  v3.4.0   (MIT)
#   - Vulkan-Hpp                  v1.4.309  (Apache-2.0 OR MIT)
#
# IMPORTANT: the Vulkan-Hpp tag MUST match the installed Vulkan SDK header
# version (VK_HEADER_VERSION). See third_party/README.md.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VMA_URL="https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/master/include/vk_mem_alloc.h"
VULKAN_HPP_TAG="v1.4.309"
VULKAN_HPP_BASE="https://raw.githubusercontent.com/KhronosGroup/Vulkan-Hpp/${VULKAN_HPP_TAG}/vulkan"

# The full set of Vulkan-Hpp split headers (mirrors upstream include/vulkan/).
VULKAN_HPP_FILES=(
    vulkan
    vulkan_enums
    vulkan_extension_inspection
    vulkan_format_traits
    vulkan_funcs
    vulkan_handles
    vulkan_hash
    vulkan_hpp_macros
    vulkan_raii
    vulkan_shared
    vulkan_static_assertions
    vulkan_structs
    vulkan_to_string
    vulkan_video
)

mkdir -p "${repo_root}/third_party/vma"
mkdir -p "${repo_root}/third_party/vulkan-hpp/vulkan"

echo "==> VMA"
curl -fsSL "${VMA_URL}" -o "${repo_root}/third_party/vma/vk_mem_alloc.h"

echo "==> Vulkan-Hpp ${VULKAN_HPP_TAG}"
for f in "${VULKAN_HPP_FILES[@]}"; do
    curl -fsSL "${VULKAN_HPP_BASE}/${f}.hpp" -o "${repo_root}/third_party/vulkan-hpp/vulkan/${f}.hpp"
done

echo "==> Done."
echo "    VMA:         $(grep -m1 'VMA_VERSION' "${repo_root}/third_party/vma/vk_mem_alloc.h")"
echo "    Vulkan-Hpp:  $(grep -m1 'VK_HEADER_VERSION ==' "${repo_root}/third_party/vulkan-hpp/vulkan/vulkan.hpp")"
