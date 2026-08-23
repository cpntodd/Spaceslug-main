# Devlog 0007 — Spaceslug-Tiny training readiness

- **Status:** Experimental
- **Date:** 2026-08-23
- **Spaceslug-main revisions:** `0a87e75`, `db3c2ce`, and the commit containing this record
- **vulkan-runtime revision:** `a195bc6d50bb16528fe8970d74254a855264a35c`

## Objective

Create a small, reproducible CPU-reference Spaceslug-Tiny artifact and make a dataset-backed training run ready for testing before work on larger models.

## Implemented slice

- A deterministic UTF-8 byte tokenizer (`spaceslug-byte` v1) with explicit PAD/BOS/EOS tokens and a tokenizer fingerprint.
- An inspectable, checksummed `Spaceslug-Tiny` artifact directory containing tokenizer metadata, model metadata, and CPU-reference weights.
- Dataset-backed teacher-forced training from the canonical `.dts` train split, with a validation-loss report when a validation split is present.
- Deterministic JSON training checkpoints containing weights, AdamW state, configuration, dataset revision, tokenizer fingerprint, and loss metrics.
- `spaceslug tiny-train-dataset BUNDLE CHECKPOINT ARTIFACT` to produce a checkpoint and artifact without overwriting existing outputs.

## Acceptance evidence

```text
PYTHONPATH=python python3 -m unittest -v tests.test_tokenizer_artifact tests.test_tiny_training
```

The acceptance tests establish deterministic UTF-8 tokenization, artifact checksum rejection, loss reduction on a fixed `.dts` fixture, checkpoint save/load identity, and equality between resumed and uninterrupted AdamW training.

## Limitations

This is a CPU-reference bigram model, not yet the documented tiny dense transformer or a useful language model. It has no batching, target-only chat masking, gradient clipping, scheduler, memory planner, experiment directory, or CLI resume option. The produced artifact is suitable for repeatable training-path testing only; it is not a claim of Vulkan inference, GUI readiness, or 10M–50M-parameter model readiness.

## Next gate

Implement the smallest dense causal model with an independent numerical gradient acceptance test, then add bounded dataset training configuration, checkpoint resume via CLI, and a machine-readable experiment record before calling Spaceslug-Tiny ready for broader training tests.
