# Devlog 0011 — Spaceslug-Tiny training readiness gate

- **Status:** Experimental
- **Date:** 2026-08-23
- **Spaceslug-main revision:** record the commit containing this entry
- **vulkan-runtime revision:** `a195bc6d50bb16528fe8970d74254a855264a35c`

## Gate result

The committed fixture `tiny-acceptance-v1` now runs through the complete CPU-reference projected-attention workflow and produces a machine-readable experiment report containing:

- train, validation, and held-out test loss;
- held-out masked token accuracy;
- dataset and tokenizer identity;
- code/runtime/command provenance;
- checkpoint schema/path;
- checksummed artifact revision;
- deterministic loaded-artifact inference result.

The current fixture report is reproducible with final train loss `5.537153225016332`, validation loss `5.54318378364119`, test loss `5.531122666391473`, test token accuracy `0.0`, and inference token ID `10` for `Q: `. The zero accuracy is explicitly reported and is not treated as a quality pass.

## Acceptance command

```text
PYTHONPATH=python python3 -m unittest -v
```

The complete suite passes. `tests/test_tiny_cli_acceptance.py` is the end-to-end gate; it trains from the committed fixture, loads the artifact, and compares report identity and inference output.

## Readiness status

**Ready for bounded training-path testing; not ready for a useful-model or quality claim.** The artifact is a tiny CPU-reference projected-attention model, not a 10M–50M parameter production Tiny model. GUI integration, Vulkan execution, larger-data quality, performance, and model-quality acceptance remain open.

## Next gate

Add a bounded training budget/early-stop policy and a report-level acceptance status that distinguishes reproducibility success from quality success. Do not promote this synthetic fixture to a useful-model claim.
