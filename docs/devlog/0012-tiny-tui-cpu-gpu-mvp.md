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
- `spaceslug tiny-cpu-verify`: explicit CPU-reference acceptance gate.
- Vulkan backend readiness report that names the current validated operation and refuses to claim Tiny GPU inference/training prematurely.

## Current GPU status

The custom runtime remains validated for `vector_add` only. Tiny GPU inference and training are **not implemented yet**. The next declared operation is tensor/GEMM parity, followed by projected-attention forward parity, LoRA forward/backward parity, and optimizer/update parity. CPU remains authoritative.

## Verification

Focused tests cover model profiles, filesystem selection, TUI rendering/actions, CPU verification, and GPU gate reporting. The existing runtime smoke test continues to pass on the configured Vulkan runtime baseline.

## Acceptance boundary

This is an interactive CPU-first foundation, not yet a GPU-trained Tiny model. GPU work may proceed only after the CPU gate passes and every Vulkan result carries explicit device, runtime revision, fallback, and parity evidence.
