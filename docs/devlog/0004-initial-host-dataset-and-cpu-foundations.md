# Devlog 0004 — Initial host, dataset, and CPU foundations

- **Status:** Experimental
- **Date:** 2026-08-23
- **Spaceslug-main revisions:** `8a024f3`, `aa2116e`
- **vulkan-runtime revision:** `3e2b6f0`

## Objective

Establish the first executable host boundary, deterministic dataset bundle slice, and CPU reference training slice.

## Implemented

- Python host package with a subprocess-backed `BackendSession`.
- Structured capabilities and execution results for the runtime `vector_add` operation.
- Independent host-side vector-add reference helper.
- Deterministic `.dts` directory writer and verifier with canonical sorted JSONL, manifest metadata, SHA-256 checksums, split counts, and tamper detection.
- Tiny dependency-free CPU bigram model with teacher-forced next-token loss, gradient descent, and checkpoint save/reload.

## Evidence

```text
PYTHONPATH=python python3 -m unittest -v
3 tests passed
```

The pinned runtime separately passed its complete 25-test suite on both RADV and lavapipe. The host adapter currently invokes the runtime's existing CPU-reference-backed vector-add executable; it does not yet provide a native ABI or independent access to runtime tensor buffers.

## Limitations

- `.dts` compression, tokenized derived shards, JSON Schema enforcement, statistics, and benchmark-overlap checks remain future work.
- The CPU model is a bigram acceptance model, not the planned tiny dense transformer.
- The host adapter remains Experimental and subprocess-backed.
