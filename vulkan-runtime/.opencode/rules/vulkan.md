# vulkan-runtime project rules

C++20 Vulkan compute runtime for AMD RX580 (RADV/gfx803). Headless compute kernels for LLM inference (GEMM, dequant, RMSNorm, RoPE, attention). See `RESEARCH.md` in the repo root for the full report.

## Build

- CMake with presets. Always enable `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`; symlink `compile_commands.json` to the repo root so clangd resolves (agent-lsp + opencode built-in lsp).
- Shaders: GLSL `#version 450` → `glslc -O` at build time → SPIR-V → embedded. `spirv-val` every shader.
- Build: `cmake --build build` (or the debug preset). Create the build dir first if missing.

## Verify loop (per-milestone acceptance gate)

1. `cmake --build build` — clean build
2. Correctness vs CPU reference: `ctest --test-dir build --output-on-failure`
3. Headless/CI on software Vulkan: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ctest --test-dir build --output-on-failure`
4. Real perf on RADV: bench targets with timestamp queries, median of runs
5. Confirm device: `vulkaninfo | grep -i polaris` (expect `RADV POLARIS10`)

## Milestones (acceptance tests)

- **M0** scaffold: clean build; vulkaninfo shows RADV
- **M1** hello-compute: vector-add correct vs CPU on lavapipe AND RADV
- **M2** GEMM: correct vs CPU; ≥1.0 TFLOPS @ 1024³ (stretch: vkblas 2.54)
- **M3** tuning harness: reproducible medians, variant ranking
- **M4** kernel library: transpose / int4-dequant / RMSNorm / RoPE / attention correct vs CPU
- **M5** exec engine v2: N-slot ring + timeline semaphores; pipelined overlap, no stalls
- **M6** cactus integration: `cactus run Cactus-Compute/needle` with the gfx803 backend active
- **M7** training kernels: fp32 backward GEMM + AdamW with correct gradients on GPU

## Rules

- fp32 math (no fp16 compute on Polaris); fp16/bf16 storage only.
- Workgroups must be multiples of 64 (wave64); LDS ≤ 32 KB per workgroup.
- No coop matrices. Chunk long dispatches (compute-ring timeout).
- Keep weights device-resident (no resizable BAR); batch host transfers.
- VVL on in debug builds; never mask validation errors.
- Use the `vulkan-gfx803` skill for hardware gotchas; delegate kernel work to `vulkan-pro`.
