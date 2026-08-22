# Devlog 0005 — Native tensor exchange

- **Status:** Experimental
- **Date:** 2026-08-23
- **Spaceslug-main revisions:** `3f567b2`, `52a138f`
- **vulkan-runtime revision:** `3e2b6f0`

## Objective

Advance the initial executable-level smoke boundary to a structured native host tensor exchange for the vector-add proof.

## Result

`native/smoke_adapter.cpp` now computes a deterministic float32 host tensor result and reports it as structured JSON alongside the pinned runtime's validated vector-add pass:

```json
{"dtype":"float32","shape":[3],"output":[1,4,1]}
```

The runtime remains the execution oracle in this milestone; the adapter does not yet export Vulkan buffers or invoke runtime kernels through a stable C ABI. The adapter accepts custom CSV float32 inputs; `0.5,2.0` plus `-0.5,3.0` produced `[0,5]` with runtime pass evidence.

## Evidence

- Native C++20 compile passed with `-Wall -Wextra -Wpedantic`.
- RADV adapter result: `status=ok`, RX 580/RADV POLARIS10, runtime `N=1048576 PASS`.
- lavapipe adapter result: `status=ok`, llvmpipe, runtime `N=1048576 PASS`.
- Spaceslug-main host suite: 4 tests passed.

## Limitations and next step

Tensor values are currently exchanged within the native host adapter; direct runtime tensor-buffer exchange remains the next boundary. No performance claim is made from this smoke path.
