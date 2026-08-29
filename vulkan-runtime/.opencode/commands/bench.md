---
description: Run the tuning harness / benchmark (timestamp queries, median-of-runs)
agent: vulkan-pro
---

Run the performance benchmark for the kernels (see `.opencode/rules/vulkan.md`).

1. Ensure the build is current: `cmake --build build`.
2. Run the bench target (typically `./build/bench/...` or a ctest bench label). Use timestamp queries; take the **median** of multiple runs (not the mean — scheduler noise).
3. Compare against milestone targets and the vkblas reference (2.54 TFLOPS fp32 GEMM on this GPU; ~62% of the 5.8–6.2 TFLOPS peak).
4. Report per-size TFLOPS numbers, kernel breakdown, and where time goes (memory vs ALU vs sync).
5. If a variant regressed, note it and revert to the last known-good configuration.
