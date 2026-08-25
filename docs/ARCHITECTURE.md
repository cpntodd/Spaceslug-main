# Spaceslug-main architecture

## Layer model

```text
GUI
  ↓ local HTTP/IPC service
Application API and job manager
  ↓
Model registry · trainer · evaluator · agent runner
  ↓
Canonical IR · artifact loader · execution planner
  ↓
CPU reference backend · Spaceslug Vulkan backend
  ↓
Spaceslug / vulkan-runtime
```

## Modules

- `api/` — versioned service and library API.
- `cli/` — headless equivalents of GUI operations.
- `compiler/` — importers, canonical IR, shape inference, lowering, and fusion.
- `runtime/` — execution, state, KV cache, batching, cancellation, and memory planning.
- `backends/` — CPU reference, Spaceslug Vulkan, and future backends.
- `moe/` — router, token compaction, expert union, residency, prefetch, placement, and aggregation.
- `training/` — autograd, LoRA, optimizer, checkpoints, datasets, and evaluation hooks.
- `formats/` — artifact, tensor-store, tokenizer, adapter, dataset, and experiment schemas.
- `datasets/` — source manifests, `.dts` bundle tooling, protected benchmark references, and dataset statistics.
- `experiments/` — reproducible run metadata, reports, and references to external large outputs.
- `docs/devlog/` — chronological, evidence-based development entries.
- `gui/` — desktop shell and web UI.
- `agent/` — tools, permissions, worktrees, test execution, and review workflow.

## Canonical IR requirements

Every tensor and operation must declare:

- shape and strides;
- dtype and accumulation dtype;
- quantization format and metadata;
- layout and orientation;
- device placement;
- aliasing and ownership;
- static/dynamic dimensions;
- stateful inputs/outputs;
- forward and backward availability;
- deterministic behavior;
- fallback policy.

The IR must not contain Vulkan descriptors, shader names, Colibri container assumptions, or Cactus backend-specific state.

## Execution modes

- `INFERENCE` — frozen forward execution.
- `PREFILL` — prompt processing.
- `DECODE` — persistent KV-cache token generation.
- `EVALUATION` — deterministic scoring and reference comparisons.
- `LORA_TRAIN` — frozen base plus trainable adapters.
- `FULL_TRAIN` — trainable base model; later research scope.

## Backend contract

Backends expose capabilities and explicit results for unsupported operations. Unsupported work may fall back to CPU only when the plan permits it and the fallback is recorded in metrics. The optional Tiny graph-owned base checkpoint ABI snapshots/restores LM-head, output, and QKV weights, and LM-head/output/QKV AdamW moments plus step. QKV AdamW is capability-gated on the complete runtime symbol set; dataset storage and retained training buffers never enter the checkpoint.

The Spaceslug backend must provide asynchronous submission/events, explicit tensor transfers, memory budget reporting, and operation-level profiling. Backend selection is per execution plan or request, not a process-global mutable setting.

## Design constraints inherited from Spaceslug

- RX580 workgroups are wave64-compatible.
- Workgroup LDS remains within the 32 KB limit.
- Vulkan math is fp32; fp16/bf16 are storage formats unless a future capability proves otherwise.
- Packed fp16 and CQ4 layouts must be versioned and tested.
- Vulkan validation and lavapipe correctness paths remain mandatory.

## Design constraints inherited from Cactus and Colibri

- Graph execution must support dynamic shapes and persistent state.
- Prefill and decode are separate plans.
- MoE routing semantics are independent of expert placement.
- Expert union formation avoids repeated loads.
- Residency can span VRAM, RAM, and storage without changing model semantics.
- CPU reference execution is the oracle for new operators and architectures.
