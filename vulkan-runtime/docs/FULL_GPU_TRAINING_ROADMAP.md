# Full GPU training completion roadmap

This document defines the engineering path from the verified bounded Tiny graph to a complete GPU training ABI. A capability is enabled only after a CPU reference, lavapipe, RADV/gfx803, synchronization, checkpoint, and negative-ABI gate pass. Until then, the public capability must continue to report the feature as unsupported.

## Dependency order

1. **Atomic active-chain integration** — resolve the placement contract first: the staged true RMSNorm currently operates on embedding-plus-position state, while the FFN contract requires the chosen pre-FFN/post-attention placement. Integrate raw/inverse-RMS/state buffers, FFN forward/backward/parameter gradients, and true `rmsDx` consumers into one fixed-profile loss step with explicit producer/consumer barriers and liveness.
2. **End-to-end active-chain parity** — compare the complete attention/projection → normalization → FFN → LM-head chain against a CPU double-precision reference, including duplicate tokens, maximum and short windows, trailing masks, all-masked rows, finite differences, and multi-step loss/convergence behavior. Keep every public group gate closed until this evidence exists.
3. **Synchronization and atomic update gate** — eliminate the known shared-staging write-after-read hazard and prove cross-submit visibility, staging reuse, retained/device interactions, and selected-group all-or-nothing mutation under validation on RADV and lavapipe.
4. **Dataset-integrated training** — extend `BatchBuffer` with a training descriptor/control contract. Consume all retained windows device-side, reset local positions per window, accumulate safely, normalize by included mask count, update selected groups, and return loss/count without a host window loop or hidden CPU fallback.
5. **Retained backward/optimizer** — use fixed upper-bound schedules with device control buffers or immutable operation-mode graphs; prove descriptor lifetime, command resubmission, cross-submit visibility, and no stale state. This follows the dataset path.
6. **General arbitrary-shape full-base ABI** — validate dimensions, strides, padded vocabulary, sequence limits, rank, and memory sizes; add bounds-checked kernels, watchdog-safe dispatch chunking, dynamic scratch sizing, shape-specific pipeline caches, and a shape matrix. Keep the fixed Tiny path as a specialization.

## Acceptance gates

Every new parameter group requires:

- CPU double-precision reference and finite-difference gradient check.
- GPU result parity on lavapipe and RADV POLARIS10 within documented FP32 tolerance.
- Masking, duplicate-token, short-window, maximum-window, and invalid-input tests.
- Optimizer step/bias-correction/weight-decay parity and checkpoint round-trip.
- C ABI argument validation, error conversion, and no hidden fallback.
- Capability metadata and docs updated only after all gates pass.

Complete training additionally requires deterministic multi-window resume parity, selected-group atomicity, device-resident dataset execution, watchdog-safe chunking, and an end-to-end loss-decrease test on a tiny learnable fixture. Inference remains independently gated by logits parity and shape-tail coverage.

## Explicit current boundaries

The current implementation contains schema-v4 graph-owned normalization and FFN state, active fixed-profile post-attention true-RMSNorm forward/backward/dgamma, FFN forward/backward/parameter gradients, AdamW recurrence, checkpoint round-trip, C ABI accessors, and bounded orchestration evidence. The bounded true-RMSNorm/FFN training path is exposed; device-resident full dataset training, retained backward/optimizer execution, general full-base construction, and arbitrary-shape full-base training remain disabled. No single successful kernel test is sufficient to enable the complete full-training ABI.

## Current milestone outcome

The staged normalization/FFN increment is implemented and verified on RADV and lavapipe: CPU-double parity, finite differences, masked behavior, composed backward, optimizer continuation, schema-v4 checkpoint/C ABI, and downstream position/embedding `rmsDx` consumer primitives pass with validation enabled. Bounded graph-owned orchestration also passes, but this is not evidence that the active attention/projection/loss chain trains those parameters end to end.

A lifecycle cleanup regression found during resumed validation was corrected: shader modules are now destroyed exactly once under their matching resource blocks and in reverse resource order. The complete integrated lifecycle still requires clean synchronization evidence from exercising the active chain end to end, plus end-to-end parity and convergence; current tested paths are VVL-clean.

## Next safe increment

Implement the active fixed-profile chain atomically at the contractually chosen normalization placement. Wire raw/inverse-RMS state and true `rmsDx` to position/embedding consumers, compose FFN forward/backward/parameter gradients with attention/projection and LM-head gradients, and add explicit barriers/liveness. Before opening any public capability, require CPU-double, finite-difference, duplicate-token, maximum-window, trailing-mask, checkpoint, C ABI, synchronization, and multi-step loss/convergence gates. Keep dataset-resident, retained, and arbitrary-shape capabilities closed until their dedicated contracts pass.
