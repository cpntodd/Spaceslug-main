---
description: Recompile all GLSL shaders to SPIR-V and validate them
agent: vulkan-pro
---

Recompile the project's GLSL shaders to SPIR-V and validate them.

1. Rebuild shaders (usually via the build system: `cmake --build build` re-runs glslc; or the repo's shader build script in `shaders/`).
2. ALWAYS compile with optimization (`glslc -O`) — forgetting `-O` costs ~1000× on this GPU.
3. Validate every generated `.spv` with `spirv-val`.
4. Sanity-check a GEMM/attention kernel's inner loop: `spirv-dis <shader>.spv | grep -c Fma` (should be large; 0 means the optimizer stripped it or `-O` was missed).
5. Report the shader list, validation status, and any FMA-count anomalies.
