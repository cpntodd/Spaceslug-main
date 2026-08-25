# Staged roadmap — approved sequence

**Status:** Approved direction. **Only Phase 1 is authorized for implementation now.**
**Scope:** `spaceslug-main` with `vulkan-runtime`, RX580/RADV/gfx803, FP32 arithmetic.

This document is the durable record of the approved staged implementation sequence. It consolidates the decision sequence in [`RX580_TRAINING_ALTERNATIVES.md`](RX580_TRAINING_ALTERNATIVES.md) into six ordered phases, and it inherits the fixed Tiny boundaries from [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md). The older exploratory phase list in [`ROADMAP.md`](ROADMAP.md) remains for historical context; where the two differ on sequencing, this document is authoritative.

## Authorization boundary

- **Phase 1 is authorized now.** No later phase is authorized to begin until its predecessor's gate is met and the sequence is re-approved.
- Phases 2–6 are recorded here as **direction only**. They are not approval to write code, add native symbols, rename capabilities, or change the Tiny contract.
- Nothing in this document relaxes the fixed Tiny MVP contract or the "CPU authoritative outside the bounded GPU path" rule.
- Changing the sequence, the gate for any phase, or the Tiny boundaries requires an explicit, separately recorded approval.

## Fixed Tiny boundaries (unchanged)

The current Tiny contract is fixed and is not expanded, renamed, or generalized by this roadmap:

- Exactly two GPU training profiles are supported: `tiny_h64_v259_vp320_t128_rank4` and `tiny_h64_v259_vp320_t128_rank8`. Both use **fp32**, **H=64**, vocabulary **V=259**, padded vocabulary **Vp=320**, and sequence capacity **T=128**; only the LoRA rank differs (4 or 8). Unlisted shapes, dimensions, and ranks are rejected, never silently coerced.
- Graph-owned SGD is available for token embeddings, the LM-head, output projection, and combined Q/K/V for exactly `1 <= rows <= 128`. The deterministic sparse embedding path handles repeated token IDs without floating-point atomics.
- Graph-owned AdamW is available for the LM-head, output projection, and combined Q/K/V with persistent state and unified checkpoint/resume coverage. This does not make arbitrary-shape or all-dataset GPU training available.
- Positions, FFN, and normalization are unsupported in the fixed Tiny architecture. Dataset-resident training remains bounded by its documented group/path contract, and the retained-command path is fixed forward/loss only; it does not retain backward or optimizer updates.
- FP16 is a storage format only; arithmetic, accumulation, optimizer state, and training remain FP32.
- GPU execution is primary only for the explicitly supported native operation selected by a job. CPU is the transparent fallback outside that bounded path; standalone caller-supplied APIs remain separate from graph-owned training.

See [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md) and the Tiny status section of [`ROADMAP.md`](ROADMAP.md) for the full evidence-backed statement.

## Staged sequence

| Phase | Scope | Authorization |
|---|---|---|
| 1 | GUI + current Tiny | **Authorized now** |
| 2 | Optimizer abstraction + plain SGD | Proposed — not authorized |
| 3 | Tiny-Plus architecture | Proposed — not authorized |
| 4 | Factored optimizer (SM3/Adafactor) | Proposed — not authorized |
| 5 | Small / LargeSmall full-base | Proposed — not authorized |
| 6 | Experiments | Proposed — not authorized |

### Phase 1 — GUI + current Tiny (AUTHORIZED)

Finish the GUI shell and local service against the current Tiny/LoRA capability metadata and the fixed Tiny profile API. The GUI drives training, chat, evaluation, and bounded experiments for the fixed Tiny profiles without a CLI, and it surfaces the exact capability and fallback boundaries rather than hiding them.

**Gate:** the complete Tiny-model workflow is usable from the GUI without the CLI, with capability/fallback status visible to the user. The Tiny contract is unchanged.

### Phase 2 — Optimizer abstraction + plain SGD (proposed)

Generalize the optimizer interface into a common parameter-group optimizer contract without changing CPU authority or unsupported-operation semantics. Implement and benchmark plain FP32 SGD as the control/reference across the current bounded groups.

**Gate:** CPU/GPU gradient parity, deterministic checkpoint/resume, explicit memory-plan rejection, bounded dispatch timing, and a real-RX580 benchmark. See [`RX580_ALTERNATIVE_1_SGD.md`](RX580_ALTERNATIVE_1_SGD.md).

### Phase 3 — Tiny-Plus architecture (proposed)

Define a configurable profile larger than fixed Tiny but below the Small tier, and benchmark plain SGD on it to validate scale. Fixed Tiny is left unchanged: metadata alone cannot generalize the fixed profiles, so kernels and graph ownership must support each new shape. A 5–20M quality and stability gate precedes any larger experiment.

**Gate:** a configurable Tiny-Plus profile trains under plain SGD with documented peak-live memory and passes the 5–20M quality/stability gate.

### Phase 4 — Factored optimizer (proposed)

Implement exact CPU oracles for one factored optimizer — SM3 first (the simpler milestone), then Adafactor — and version per-tensor state as factored or unfactored. Select one for GPU implementation only after convergence and peak-live-memory measurements.

**Gate:** oracle tests against published equations, factored/unfactored edge cases, and a 5–20M quality gate; no 100–150M promise until basis construction, workspace, and bounded kernels are measured. See [`RX580_ALTERNATIVE_2_ADAFACTOR.md`](RX580_ALTERNATIVE_2_ADAFACTOR.md).

### Phase 5 — Small / LargeSmall full-base (proposed)

The next full-base scale tiers above Tiny-Plus and below Spaceslug-0.5B, in the gated ~100–150M full-base research range. Attempt only after real RX580 evidence for activations, workspaces, dispatch duration, and transfers. Spaceslug-0.5B remains validated quantized inference plus fixed-rank LoRA until broader paths exist.

**Gate:** real-RX580 activation, workspace, dispatch-duration, and transfer evidence before any Small/LargeSmall full-base experiment.

### Phase 6 — Experiments (proposed)

Separate, individually gated experiments: LOMO fused backward/update, ZeRO-Offload-style CPU placement, GaLore, Muon, Sophia, DoRA/rsLoRA, and compressed Adam. Each requires an exact CPU oracle, a peak-live memory model, and a bounded-kernel plan before benchmarking. Compressed Adam stays deferred until a measured bottleneck justifies its numerical and kernel complexity.

**Gate:** each experiment has an exact CPU oracle, a peak-live memory model, and a bounded-kernel plan before real-RX580 benchmarking. See [`RX580_ALTERNATIVE_3_COMPRESSED_ADAM.md`](RX580_ALTERNATIVE_3_COMPRESSED_ADAM.md) and [`RX580_ALTERNATIVE_4_LORA_ADALORA.md`](RX580_ALTERNATIVE_4_LORA_ADALORA.md).

## Related documents

- [`ROADMAP.md`](ROADMAP.md) — broader historical phase list and current Tiny graph optimizer status.
- [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md) — fixed Tiny boundaries and acceptance evidence.
- [`RX580_TRAINING_ALTERNATIVES.md`](RX580_TRAINING_ALTERNATIVES.md) — the review whose decision sequence this roadmap consolidates.
- [`GUI.md`](GUI.md) — GUI and local service plan for Phase 1.
