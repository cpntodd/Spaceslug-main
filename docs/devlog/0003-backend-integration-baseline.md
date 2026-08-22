# Devlog 0003 — Backend integration baseline

- **Status:** Proposed
- **Date:** 2026-08-23
- **Spaceslug-main revision:** record the commit containing this entry
- **vulkan-runtime revision:** `3e2b6f0`

## Objective

Define the first stable boundary between the Spaceslug-main host and the separate Spaceslug Vulkan runtime.

## Runtime baseline

The pinned runtime is `cpntodd/Spaceslug` on `main` at `3e2b6f0` (`M7a: packed fp16 CQ4 chain + norm-layout correctness fix + bench`). The runtime contains existing validated-operation targets including vector addition and SGEMM. A fresh configure/build completed successfully. Device smoke reports AMD Radeon RX 580 Series (RADV POLARIS10), Vulkan 1.4.305, with validation enabled. All 25 CTest targets passed on the default RADV path and with lavapipe. This validates the runtime baseline only; no host adapter exists yet.

## API decision

The first adapter will be coarse-grained and plan-based. It reports capabilities, runtime revision, device, tensors, structured errors, explicit fallback, and operation metrics. It does not expose Vulkan handles or embed GUI/model/training policy.

Vector addition is the first smoke operation because it has a small CPU reference and an existing runtime correctness target.

## Measurements

No host adapter or new parity run exists yet. There are no performance or loss results to report.

## Next gate

Build and test the pinned runtime on RADV and lavapipe where supported, then invoke a host smoke adapter and compare its result with a CPU reference. The integration remains Proposed until that evidence is recorded.
