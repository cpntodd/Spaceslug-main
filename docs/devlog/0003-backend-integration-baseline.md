# Devlog 0003 — Backend integration baseline

- **Status:** Proposed
- **Date:** 2026-08-23
- **Spaceslug-main revision:** record the commit containing this entry
- **vulkan-runtime revision:** `3e2b6f0`

## Objective

Define the first stable boundary between the Spaceslug-main host and the separate Spaceslug Vulkan runtime.

## Runtime baseline

The pinned runtime is `cpntodd/Spaceslug` on `main` at `3e2b6f0` (`M7a: packed fp16 CQ4 chain + norm-layout correctness fix + bench`). The runtime contains existing validated-operation targets including vector addition and SGEMM. A fresh configure/build completed successfully. Device smoke reports AMD Radeon RX 580 Series (RADV POLARIS10), Vulkan 1.4.305, with validation enabled. All 25 CTest targets passed on the default RADV path and with lavapipe. This validates the runtime baseline; the host adapter remains Experimental.

## API decision

The first adapter will be coarse-grained and plan-based. It reports capabilities, runtime revision, device, tensors, structured errors, explicit fallback, and operation metrics. It does not expose Vulkan handles or embed GUI/model/training policy.

Vector addition is the first smoke operation because it has a small CPU reference and an existing runtime correctness target.

## Measurements

- Runtime configure/build: passed.
- Runtime CTest: 25/25 passed on RADV and 25/25 passed with lavapipe.
- Runtime device: AMD Radeon RX 580 Series (RADV POLARIS10), Vulkan 1.4.305.
- Host smoke test: 1/1 passed using Python's standard-library `unittest`.
- Host Python smoke test: passed on RADV and lavapipe; no CPU fallback reported.
- Native C++ smoke adapter: passed on RADV and lavapipe; structured JSON returned.

Both adapters invoke the runtime's existing vector-add executable, whose deterministic CPU-reference comparison reports `N=1048576 PASS`. The host also has an independent reference helper; direct runtime tensor-buffer exchange and transfer metrics remain future work.

## Next gate

Expose runtime tensor buffers through a stable native ABI or binding and compare host-computed vectors against returned values. The current executable-wrapper adapter is Experimental.
