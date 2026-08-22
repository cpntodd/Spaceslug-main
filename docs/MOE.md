# MoE execution plan

MoE is a first-class subsystem rather than a single opaque graph operator.

```text
Router
  ↓
Expert selection
  ↓
Expert union formation
  ↓
Residency and placement
  ↓
CPU/GPU dispatch
  ↓
Weighted aggregation
```

## Initial capabilities

- Standard top-k routing.
- Gated SwiGLU experts.
- Optional shared expert.
- Token-to-expert compaction.
- Deterministic expert IDs.
- CPU expert execution.
- Resident GPU expert execution.
- Per-layer routing and residency metrics.

## Placement model

The planner may place frozen expert tensors across:

- GPU VRAM;
- system RAM;
- local storage.

Placement must change speed only, never routing probabilities, selected expert IDs, or model semantics. Expert union batching should load or dispatch each unique expert once per layer where possible.

## RX580 initial policy

```text
Dense trunk: GPU only where validated
Router: CPU initially
Hot experts: Spaceslug Vulkan GPU
Cold experts: CPU/RAM
Storage streaming: later, after RAM/VRAM correctness
```

PCIe transfers must be measured as part of end-to-end latency. A GPU kernel is not an optimization if staging costs more than the saved computation.

## Training later

MoE training requires router gradients, load-balancing loss, expert capacity behavior, expert-gradient accumulation, and optimizer state. It is outside the first inference/LoRA acceptance and must use tiny oracle models before real checkpoints.

## Acceptance gates

- CPU router parity.
- Expert ID parity.
- Expert output parity.
- Shared expert parity.
- CPU-only versus mixed-placement output equivalence within defined tolerance.
- Cold/hot residency does not alter routing.
- Expert load and transfer metrics are recorded.
