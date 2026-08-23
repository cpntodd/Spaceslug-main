# Devlog 0009 — Tiny resume and inference identity

- **Status:** Experimental
- **Date:** 2026-08-23
- **Spaceslug-main revision:** record the commit containing this entry
- **vulkan-runtime revision:** `a195bc6d50bb16528fe8970d74254a855264a35c`

## Changes

- Resume now fails closed when checkpoint dataset revision, tokenizer fingerprint, batch size, optimizer, or weight decay is incompatible with the requested run.
- Projected experiment metrics include the deterministic inference prompt (`Q: `) and greedy next-token ID produced by the trained model.
- Training metrics now include held-out `test_loss` when the `.dts` test split is non-empty.
- Acceptance coverage verifies incompatible dataset rejection and experiment inference metadata.

## Evidence

```text
PYTHONPATH=python python3 -m unittest -v tests.test_projected_attention_training
```

The focused suite passes four tests, including training loss reduction, validation reporting, checkpoint/artifact/experiment production, resumed-vs-uninterrupted AdamW parity, incompatible dataset rejection, and inference metric recording.

## Limitations

The reported inference is a deterministic CPU-reference smoke result, not a quality evaluation. The model remains a small synthetic projected-attention reference and has no GUI or Vulkan path.

## Next gate

The complete CLI acceptance report now includes artifact revision, checkpoint schema/path, held-out test loss, and inference output. The next gate is to add a committed small fixture/report reference and enforce that artifact loading reproduces the recorded inference result.
