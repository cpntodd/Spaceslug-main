# Devlog 0008 — Projected Spaceslug-Tiny training acceptance

- **Status:** Experimental
- **Date:** 2026-08-23
- **Spaceslug-main revision:** record the commit containing this entry
- **vulkan-runtime revision:** `a195bc6d50bb16528fe8970d74254a855264a35c`

## Objective

Exercise the complete pre-0.5B CPU-reference workflow: canonical `.dts` input, projected causal-attention training, AdamW checkpoint resume, checksummed model artifact reload, and deterministic greedy inference.

## Command

```text
PYTHONPATH=python python3 -m spaceslug.cli tiny-attention-train \
  BUNDLE CHECKPOINT ARTIFACT EXPERIMENT \
  --steps 2 --learning-rate 0.1 --batch-size 2

PYTHONPATH=python python3 -m spaceslug.cli tiny-attention-train \
  BUNDLE RESUMED_CHECKPOINT RESUMED_ARTIFACT RESUMED_EXPERIMENT \
  --resume CHECKPOINT --steps 3 --learning-rate 0.1 --batch-size 2
```

The fixture contained two target-only training records (`Q: → a`, `Q: → b`) and one validation record (`Q: → a`), with tokenizer `spaceslug-byte` revision `v1`.

## Evidence

- Initial CLI run: train loss `5.571223918 → 5.537153225`.
- Resumed CLI run: train loss `5.537153225 → 5.448220779`.
- Both runs emitted checkpoints, checksummed projected artifacts, and experiment records.
- Resumed artifact loaded successfully after manifest/file checksum and tokenizer-fingerprint verification.
- Loaded artifact produced deterministic greedy next-token ID `10` for prompt `Q: `.
- The projected checkpoint stores AdamW first/second moments and step state; unit acceptance separately proves resumed optimizer state equals uninterrupted training state.

## Limitations

This is a CPU-reference projected single-head attention model with a deliberately small hidden width and deterministic byte tokenizer. It is not yet a production-sized 10M–50M parameter model, Vulkan implementation, GUI workflow, or quality claim. The acceptance fixture is synthetic and does not establish useful language behavior.

## Next gate

Add a model-load/inference report to the machine-readable experiment output, enforce checkpoint/artifact/config identity on resume, and add a held-out test metric before declaring the Tiny workflow ready for broader training tests.
