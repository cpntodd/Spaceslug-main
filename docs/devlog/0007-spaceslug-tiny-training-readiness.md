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
- A smallest dense causal CPU reference model with explicit embedding/output gradients. Centered finite differences validate representative embedding, output, and bias derivatives before optimization is relied on.
- Dataset-backed dense-model training, deterministic dense checkpoints, a checksummed dense artifact, and a schema-compatible `experiment.json` record. `spaceslug tiny-dense-train BUNDLE CHECKPOINT ARTIFACT EXPERIMENT` produces all four, and accepts `--resume` for deterministic checkpoint continuation.
- Bounded dense-training controls: `--gradient-clip` and `--memory-budget-bytes`. The memory planner rejects an over-budget plan before any update is applied.
- Deterministic padded causal batches with explicit target-only masks for prompt/completion records, plus a dependency-free causal self-attention semantic reference that proves future tokens cannot affect prior outputs.
- A trainable single-head attention-scale reference integrated with those masks. Its analytic masked-loss gradient matches a centered finite difference, and fixed-batch optimization reduces the target-only loss.
- An explicit projected causal-attention reference with separate Q/K/V/output tensor paths and analytic backward propagation. Centered finite differences validate representative gradients for every projection, and fixed-batch updates reduce the masked loss.
- Dataset-backed projected-attention training with target-only batches and deterministic checkpoint round-trip: `python/spaceslug/projected_attention_training.py`.
- Deterministic sinusoidal positional encoding and a checksummed projected-attention artifact manifest with explicit architecture and position settings.

## Acceptance evidence

```text
PYTHONPATH=python python3 -m unittest -v tests.test_tokenizer_artifact tests.test_tiny_training
```

The acceptance tests establish deterministic UTF-8 tokenization, artifact checksum rejection, loss reduction on fixed `.dts` fixtures, checkpoint save/load identity, equality between resumed and uninterrupted AdamW training, agreement between analytic dense-model gradients and centered finite differences, a schema-compatible experiment record tied to the dataset revision, target-only loss masking that excludes prompt/PAD tokens, strict causal attention semantics, and a trainable masked attention gradient.

## Limitations

The original dataset CLI produces the CPU-reference bigram artifact. The dense CLI now produces a dense artifact, checkpoint, and experiment record and can resume a deterministic SGD checkpoint; its bounded preflight estimates parameter-and-gradient memory only. The dense causal model is not an attention-based transformer. There is no batching, target-only chat masking, scheduler, optimizer state beyond SGD, or full activation/workspace memory planner. These outputs are suitable for repeatable training-path testing only; they are not a claim of Vulkan inference, GUI readiness, or 10M–50M-parameter model readiness.

## Next gate

Add a projected-attention artifact reader/checksum verifier and connect projected training to experiment records and CLI resume. Then add a second acceptance batch with validation metrics before declaring the attention-based Tiny reference ready for broader training tests.
