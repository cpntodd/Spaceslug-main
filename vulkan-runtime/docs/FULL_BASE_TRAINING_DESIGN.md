# Full-base training integration design

The current `ForwardResourceGraph` owns the bounded attention/output/embedding/position resources. A complete full-base model must extend that ownership atomically rather than layering host-side loops over the existing APIs.

## Resource ownership

A schema-v4 graph owns `gamma[H]`, `gamma_m[H]`, `gamma_v[H]`, and FFN `W1[H,4H]`, `b1[4H]`, `W2[4H,H]`, `b2[H]` plus matching gradients and optimizer moments. Constructor validation must reject null pointers and checked-size mismatches. Initialization is device-resident after one upload.

## Atomic bounded step

For rows <= `Tcap`, one normal submission performs: masked embedding/position lookup, RMSNorm, attention, FFN forward, logits/loss, reverse attention/projection, RMSNorm/FFN backward, deterministic gradient reductions, then group-selected AdamW updates. A single shared one-based optimizer step advances only after all selected groups complete. Barriers must separate every producer/consumer pair; no CPU per-token loop or hidden fallback is permitted.

## Acceptance gates

Before exposing the constructor or C ABI: CPU-double and finite-difference tests for RMSNorm/FFN, mixed/all-masked rows, optimizer two-step parity with nonzero moments, checkpoint round-trip/resume, direct C ABI invocation, lavapipe and RADV parity, finite loss, and a measurable loss decrease on a deterministic toy batch.

Dataset-resident, retained backward/optimizer, and arbitrary-shape execution remain separate milestones. Their APIs must return explicit unsupported status until device-resident scratch, retained dependency synchronization, and arbitrary-shape descriptor/product validation are implemented and tested.
