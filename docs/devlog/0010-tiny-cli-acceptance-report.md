# Devlog 0010 — Tiny CLI acceptance report

- **Status:** Experimental
- **Date:** 2026-08-23
- **Spaceslug-main revision:** record the commit containing this entry
- **vulkan-runtime revision:** `a195bc6d50bb16528fe8970d74254a855264a35c`

## Acceptance output

A synthetic `.dts` fixture with two training records, one validation record, and one held-out test record was trained with:

```text
PYTHONPATH=python python3 -m spaceslug.cli tiny-attention-train \
  BUNDLE CHECKPOINT ARTIFACT EXPERIMENT \
  --steps 2 --learning-rate 0.1 --batch-size 2
```

The run reduced train loss from `5.571223918` to `5.537153225` and wrote a schema-version-2 checkpoint, a checksummed artifact, and `experiment.json`.

Machine-readable report fields were verified:

```json
{
  "artifact_revision": "sha256:8da6da4baf651d5d4aa02914a332748588b9328271b10635bca0ba51b3e1c32e",
  "checkpoint_identity": {"schema_version": 2},
  "inference": {"prompt": "Q: ", "next_token": 10},
  "test_loss": 5.531122666391473
}
```

The artifact was separately loaded with checksum and tokenizer verification in the prior acceptance gate; deterministic greedy inference returned token ID `10` for `Q: `.

## Limitations

The fixture is synthetic and the result is a correctness/reproducibility smoke, not a language-quality or performance claim. The model remains CPU-reference-only and small.

## Next gate

Commit a minimal reproducible fixture/report reference and enforce that loading the emitted artifact reproduces the experiment's recorded inference result.
