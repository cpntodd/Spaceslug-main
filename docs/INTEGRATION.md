# Spaceslug runtime integration

## Repository boundary

`spaceslug-main` is the host engine and model laboratory. `vulkan-runtime` (the GitHub repository `cpntodd/Spaceslug`) remains the separate C++20 headless Vulkan backend for RX580/RADV/gfx803.

The host must not duplicate Vulkan device management, shader loading, descriptor management, or kernel correctness logic. The runtime must not own tokenizers, datasets, model-family policy, GUI state, or experiment policy.

## Pinned baseline

The first integration baseline is:

```text
repository: cpntodd/Spaceslug
local path: ../vulkan-runtime
branch: main
commit: 3e2b6f0
commit subject: M7a: packed fp16 CQ4 chain + norm-layout correctness fix + bench
```

This revision contains validated runtime targets including `vector_add`, `sgemm`, kernel-library tests, execution-engine tests, and CQ4 integration tests. The pinned runtime baseline was freshly configured and built successfully. Device smoke reports AMD Radeon RX 580 Series (RADV POLARIS10), Vulkan 1.4.305, with validation enabled. Its full 25-test CTest suite passed on the default RADV path and with lavapipe; this is runtime evidence, not yet host-adapter evidence.

The runtime's authoritative verification is documented in its `AGENTS.md` and `.opencode/rules/vulkan.md`: configure/build, CTest on RADV, CTest with lavapipe, and benchmark execution on a discrete GPU.

## Initial host API

The first host adapter is deliberately narrow and coarse-grained. It must not expose Vulkan handles or require one Python call per kernel dispatch.

Conceptual types:

```text
BackendCapabilities
BackendSession
TensorDescriptor
ExecutionPlan
ExecutionResult
ExecutionMetrics
BackendError
```

Required behavior:

- open and close a backend session;
- report backend name, runtime revision, device identity, supported operations, precision, memory budget, and fallback capabilities;
- accept a validated execution plan or a named smoke operation;
- return structured outputs and metrics;
- report transfers, elapsed time, operation counts, and CPU fallback explicitly;
- return structured errors for unsupported operations, invalid tensors, device failures, and cancelled work;
- keep backend selection per session/request rather than process-global mutable state.

The initial smoke operation is vector addition because the runtime already has a small CPU-reference correctness test. The plan should describe input/output tensors, dtype, shape, operation name, and backend policy. The adapter may initially use a subprocess boundary while the API is stabilized; a native C ABI or Python binding can follow without changing the host contract.

## Result contract

A smoke result must include:

```json
{
  "status": "ok",
  "operation": "vector_add",
  "backend": "spaceslug",
  "runtime_revision": "3e2b6f0",
  "device": "...",
  "dtype": "float32",
  "shape": ["..."],
  "fallback": {"used": false, "operations": []},
  "metrics": {
    "elapsed_seconds": 0,
    "host_to_device_bytes": 0,
    "device_to_host_bytes": 0,
    "operation_count": 1
  },
  "output": "..."
}
```

Values are illustrative schema shape only until an implementation produces measurements. The adapter must never report a performance result without recording whether CPU fallback or software Vulkan was used.

## First acceptance gate

1. Build the pinned runtime revision.
2. Run the existing vector-add test against its CPU reference on RADV.
3. Run the same test with lavapipe when supported.
4. Invoke the host smoke adapter against the same operation.
5. Compare host output with a host CPU reference within a defined tolerance.
6. Record commands, environment, runtime commit, device, metrics, and limitations in an experiment record and devlog.

The first subprocess-backed host adapter and standard-library smoke test now exist. The host test invokes the pinned runtime's `smoke` and `vector_add` executables, verifies the RADV capability report, requires the runtime's CPU-reference comparison to pass, and surfaces the runtime revision and fallback state. This adapter remains **Experimental** because it does not yet expose a native ABI or independent host-side tensor comparison.

## Versioning and upgrades

A runtime upgrade requires a new host integration record. The host must pin a commit or immutable release, run the acceptance gate, and document API or numerical changes. A moving branch is not a reproducible runtime dependency.
