# Devlog 0003 — Backend integration baseline

- **Status:** Validated (runtime baseline and `vector_add` host slice only)
- **Date:** 2026-08-23
- **Spaceslug-main revision:** record the commit containing this acceptance update
- **vulkan-runtime revision:** `a195bc6d50bb16528fe8970d74254a855264a35c` (`a195bc6`)

## Objective

Define the first stable boundary between the Spaceslug-main host and the separate Spaceslug Vulkan runtime.

## Runtime baseline

The pinned runtime is `cpntodd/Spaceslug` at immutable revision `a195bc6d50bb16528fe8970d74254a855264a35c` (`a195bc6`, `feat: export Vulkan vector add API`). The runtime contains existing validated-operation targets including vector addition and SGEMM and exports `libvulkan_runtime_api.so` for the first host operation. On 2026-08-23, `cmake --build build/debug -j2` succeeded; all 25 CTest targets passed on the default RADV path and all 25 passed with lavapipe. This validates the runtime baseline and the narrowly tested host `vector_add` slice only.

## API decision

The first adapter will be coarse-grained and plan-based. It reports capabilities, runtime revision, device, tensors, structured errors, explicit fallback, and operation metrics. It does not expose Vulkan handles or embed GUI/model/training policy.

Vector addition is the first smoke operation because it has a small CPU reference and an existing runtime correctness target.

## Measurements

- Runtime build: `cmake --build build/debug -j2` passed.
- Runtime CTest: 25/25 passed on RADV and 25/25 passed with lavapipe.
- Runtime device: AMD Radeon RX 580 Series (RADV POLARIS10).
- Host acceptance: `PYTHONPATH=python python3 -m unittest -v` passed all four tests, including vector-add parity on RADV and lavapipe.
- Native C++ smoke adapter: the existing structured JSON executable remains available for the same operation.

The Python host path loads the runtime-owned `libvulkan_runtime_api.so`, invokes `spaceslug_vector_add`, and compares returned first/last values to the independent CPU reference. The runtime operation itself reports its deterministic CPU-reference comparison (`N=1048576 PASS`). Transfer byte metrics, arbitrary-shape API support, and other backend operations remain future work.

## Next gate

Keep the `vector_add` ABI versioned and add one new runtime operation only with an independent CPU parity test on RADV and lavapipe. Do not infer host support for the runtime's remaining validated kernels from this narrow smoke result.
