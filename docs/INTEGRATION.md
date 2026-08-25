# Spaceslug runtime integration

## Repository boundary

`spaceslug-main` is the host engine and model laboratory. `vulkan-runtime` (the GitHub repository `cpntodd/Spaceslug`) remains the separate C++20 headless Vulkan backend for RX580/RADV/gfx803.

The host must not duplicate Vulkan device management, shader loading, descriptor management, or kernel correctness logic. The runtime must not own tokenizers, datasets, model-family policy, GUI state, or experiment policy.

## Pinned baseline

The accepted first integration baseline is immutable, not a moving branch:

```text
repository: cpntodd/Spaceslug
local path: ../vulkan-runtime
branch at verification: main
commit: a195bc6d50bb16528fe8970d74254a855264a35c
short revision: a195bc6
commit subject: feat: export Vulkan vector add API
```

This revision contains validated runtime targets including `vector_add`, `sgemm`, kernel-library tests, execution-engine tests, and CQ4 integration tests, plus the runtime-owned `libvulkan_runtime_api.so` vector-add entry point. On 2026-08-23, `cmake --build build/debug -j2` completed and the full 25-test CTest suite passed both on the default RADV path (AMD Radeon RX 580 Series / RADV POLARIS10) and with lavapipe (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`). This is accepted runtime-baseline evidence; it does not make untested runtime operations host-validated.

### Acceptance status

| Scope | Status | Evidence |
|---|---|---|
| Runtime baseline at `a195bc6` | **Validated** | Build and 25/25 CTest pass on RADV and lavapipe. |
| Host `vector_add` native API path | **Validated** | Python acceptance test loads the runtime shared library, verifies returned output against the CPU reference, and passes on RADV and lavapipe. |
| Other host backend operations | **Proposed** | No host API or parity acceptance claim yet. |
| Integrated Tiny graph groups | **Conditional** | ctypes/trainer capability metadata reports graph-owned fixed-window SGD for **LM-head**, **output projection**, **combined QKV**, and **token embeddings** (`tiny_graph_integrated_lm_head_sgd`, `tiny_graph_integrated_output_sgd`, `tiny_graph_integrated_qkv_sgd`, `tiny_graph_integrated_embedding_sgd`) when their symbols are exported. Runtime integration is from `vulkan-runtime` `328b721`, with CPU-parity verification at `fd0fe95`; positions, FFN, and norm remain unsupported. |
| Integrated Tiny LM-head AdamW | **Conditional** | ctypes/trainer binds graph-owned LM-head AdamW plus optimizer-state readback/update when all symbols are exported (`tiny_graph_integrated_lm_head_adamw`); otherwise reports unsupported (`-4`). AdamW for integrated output projection and combined QKV is likewise capability-gated on the complete symbol set (else `-4`); QKV AdamW applies to existing QKV gradients without recompute or double update. Token windows remain host-staged; this is not dataset or retained-command training. |
| Device-resident dataset LM-head SGD | **Conditional** | Optional ABI consumes a `BatchBuffer` with at most 32 windows and window length <=128, and updates only the graph-owned LM head. Metadata is bounded batch metadata, explicitly not full-dataset training or all-parameter training. |
| Standalone FP32 training subsets | **Separate bounded API** | Caller-supplied activation/gradient tensors; never report as graph-integrated training. |
| Performance | **Not claimed** | Smoke timing is diagnostic only; no benchmark acceptance result is recorded. |

A later runtime change requires a new immutable pin and a repeat of the applicable acceptance gates.

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

The first host adapter and standard-library smoke test now exist. The Python adapter invokes the pinned runtime's `smoke` and `vector_add` executables, while `native/smoke_adapter.cpp` provides a native structured JSON boundary for the same operation. The native result includes an explicit float32 tensor exchange payload computed from host inputs supplied at the adapter boundary (default `[1,4,1]`), alongside runtime pass evidence. Custom CSV tensors are accepted for the smoke proof. The runtime now exports `spaceslug_vector_add` through `libvulkan_runtime_api.so` at runtime revision `a195bc6`. It accepts host float32 tensors, executes the existing embedded Vulkan vector-add path, copies device-produced output back to the caller, and rejects invalid pointers or shapes. `native/runtime_abi.h` remains the host-facing declaration shape. Both the executable adapter and shared API verify the runtime CPU-reference result and preserve explicit backend state. This integration remains **Experimental** pending a committed cross-repository host binding test.

## Versioning and upgrades

A runtime upgrade requires a new host integration record. The host must pin a commit or immutable release, run the acceptance gate, and document API or numerical changes. A moving branch is not a reproducible runtime dependency.
