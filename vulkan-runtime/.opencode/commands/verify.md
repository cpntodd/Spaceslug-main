---
description: Build and run the milestone acceptance tests (correctness vs CPU + lavapipe)
agent: vulkan-pro
---

Build the project and run the acceptance gate for the current milestone (see `.opencode/rules/vulkan.md`).

1. Build: `cmake --build build` (create the build dir first if missing; ensure `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` and `compile_commands.json` symlinked to the repo root).
2. Run CPU-reference correctness tests: `ctest --test-dir build --output-on-failure`.
3. Run the same tests on lavapipe (software Vulkan): `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ctest --test-dir build --output-on-failure`.
4. For GEMM/perf milestones, also run the bench target (timestamp queries, median of runs).
5. Report: build result, test results (both drivers), perf numbers vs the milestone target, and any remaining failures. Do not mark the milestone complete until the gate passes on both drivers.
