# GPU contribution guide

**Status: Proposed contributor contract; current validated implementation: Vulkan on RX580/RADV/gfx803.**

Spaceslug welcomes GPU support, especially for older and capable hardware. New backends must be additive: preserve the CPU reference path, expose capability limits, and never silently change model semantics.

## What to contribute

- A new Vulkan device or vendor path.
- CUDA, ROCm, Metal, DirectML, or another backend adapter.
- A shader/kernel, memory layout, dispatch strategy, or profiling improvement.
- A correctness fixture, parity test, benchmark, or documentation correction.

## Minimum backend contract

A backend contribution should document:

1. Supported device, driver, API, architecture, precision, tensor shapes, and limits.
2. Ownership and synchronization of every input, output, activation, and gradient buffer.
3. Fallback behavior and how users can see that fallback occurred.
4. CPU-reference parity tests, numerical tolerances, and known failure cases.
5. Build and run commands, including software-renderer or emulation options where useful.
6. Performance measurements with workload, warm-up policy, timing method, and raw results.

## Validation ladder

1. Add or update a CPU/reference test.
2. Add the backend implementation behind an explicit capability gate.
3. Prove numerical parity on small deterministic fixtures.
4. Run negative tests for unsupported shapes and precision modes.
5. Publish hardware, driver, compiler, commit, and environment details.
6. Mark only the tested scope **Validated**; everything else remains **Experimental** or **Proposed**.

For the existing runtime, read [`vulkan-runtime/README.md`](../vulkan-runtime/README.md) and the [gfx803 research notes](../vulkan-runtime/docs/AMD_VULKAN_GFX803_RESEARCH.md).

## Hardware matrix

| Target | Status | Evidence |
|---|---|---|
| AMD RX580 / RADV / gfx803 | Validated for stated runtime gates | Runtime tests and linked reports |
| lavapipe | Validation/correctness coverage | Runtime test suite |
| Other AMD GPUs | Experimental or Proposed per feature | Add device-specific evidence |
| NVIDIA, Intel, integrated GPUs | Community target | Add a backend and evidence before claiming support |

## Pull request checklist

- [ ] CPU/reference behavior exists or is covered first.
- [ ] Capability gates are explicit.
- [ ] Unsupported paths fail clearly.
- [ ] Numerical parity and negative tests are included.
- [ ] Hardware/driver/compiler details are recorded.
- [ ] Benchmark and failure results are published.
- [ ] README/docs status language matches the evidence.
