#!/usr/bin/env bash
set -euo pipefail

runtime_root="${1:-../vulkan-runtime}"
cmake --preset debug -S "$runtime_root" -B "$runtime_root/build/debug"
cmake --build "$runtime_root/build/debug" -j"${JOBS:-2}"

echo "Built $runtime_root/build/debug/libvulkan_runtime_api.so"
