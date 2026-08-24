# Devlog 0012 — Tiny TUI and CPU-to-GPU MVP boundary

- **Status:** Experimental foundation
- **Date:** 2026-08-23
- **vulkan-runtime baseline:** `a195bc6d50bb16528fe8970d74254a855264a35c`

## Implemented

- `spaceslug tui`: curses-based mouse-capable TUI shell.
- Headless TUI controller for tests and future GUI/service reuse.
- Filesystem picker with recursive discovery, extension filtering, exclusions, and file-size bound.
- Intended model profiles: Tiny, 0.1B, 0.5B, 1B, and 3.5B.
- Configurable steps and epochs in the TUI state.
- Live textual worm graph from recorded loss points.
- Projected training exposes an `on_step(step, loss)` callback so a future training service/TUI can update the graph without coupling rendering to optimization.
- The TUI controller now verifies a selected `.dts` bundle, enforces the CPU gate, runs the configured steps × epochs, and receives the live loss series.
- After CPU verification, the TUI can run the current Vulkan SGEMM parity gate and report RADV/lavapipe backend status; it still does not claim Tiny forward execution.
- The backend and TUI expose a resolved projected-attention forward plan, with explicit dimensions, operation order, and a `CPU logits vs RADV logits` parity gate. This is a contract/planning boundary, not a GPU implementation.
- `vulkan-runtime` now exports a native `spaceslug_sgemm` ABI and the Python host can call it with dimension validation and structured result metadata. This is the first callable GPU projection primitive.
- The host now exposes projected QKV orchestration over that ABI for padded production shapes, while explicitly returning `not-run` for shapes that do not satisfy the current SGEMM tile contract. Native output is returned and can be compared against an explicit CPU projection reference with a structured pass/fail report, tolerance, maximum relative error, and first failing index. The CPU projected-attention forward is now executable through the backend, and `forward_parity.py` records a baseline logits vector plus a future Vulkan logits comparison report. Full causal attention on Vulkan remains open; no GPU logits pass is claimed until an actual Vulkan forward produces the compared vector. A reusable CPU-first inference session and TUI CPU inference action now record deterministic logits/next-token output and explicit `gpu_execution: false` metadata. The same session/TUI can consume a future structured Vulkan result and display pass/fail logits parity with runtime device metadata. It can also execute the current GPU-forward plan and visibly report `not-run` with the reason until causal-attention Vulkan orchestration exists.
- `spaceslug tiny-cpu-verify`: explicit CPU-reference acceptance gate.
- Vulkan backend readiness report that names the current validated operation and refuses to claim Tiny GPU inference/training prematurely.
- TUI dashboard capability display now reports the detected runtime device, revision, software-Vulkan flag, and validated operations (`vector_add`, `sgemm`, `attention_kernel`). The TUI also exposes an attention-gate action (`[a]`) that runs the validated RX580 kernel contract and displays its parity result.

## Current GPU status

The custom runtime is host-exposed for validated fp32 `sgemm`, causal attention, backward, and update APIs, with a CPU-first gate. Tiny GPU LoRA training covers the fixed MVP contract (single-head H=64, V=259, padded T=128, rank=4): persistent embeddings/positions and LoRA-aware Q/K/V/output composition, padded causal loss and dLogits, LM-head/output/causal-attention/QKV backward, four-adapter gradients, GPU SGD, persistent activation buffers, reusable ExecEngine submissions, and adapter checkpoint/restore. Native parity tests pass on RADV and lavapipe; the fixed scope is fp32 H=64/V=259/Vp=320, T<=128, rank=4, frozen base weights, and SGD. Gradient accumulation, AdamW, other model shapes, and dataset training remain unsupported. CPU remains authoritative outside this path.

## Verification

Focused tests cover model profiles, filesystem selection, TUI rendering/actions, CPU verification, and GPU gate reporting. The existing runtime smoke test continues to pass on the configured Vulkan runtime baseline.

## Acceptance boundary

This is an interactive CPU-first foundation, not yet a GPU-trained Tiny model. GPU work may proceed only after the CPU gate passes and every Vulkan result carries explicit device, runtime revision, fallback, and parity evidence.
