# Experiments and reproducibility

Experiments are first-class project outputs. Publish the recipe and the result, not only the result. A failed run, unsupported shape, or regression is useful evidence when its conditions are visible.

## What to track

Keep lightweight manifests, reports, checksums, and source records in Git. Keep large datasets, checkpoints, caches, and generated run directories out of ordinary Git history; reference them by stable checksums and retrieval instructions.

Each report should include the exact code revision, purpose or hypothesis, command/configuration, hardware/backend/driver details, dataset provenance and checksums, metrics, failures, limitations, and review status.

Use the [experiment report template](experiments/EXPERIMENT_REPORT_TEMPLATE.md). Link reports from the relevant devlog, benchmark, model, or backend documentation.

## Status and evidence

- **Proposed:** expected outcome only; no run yet.
- **Experimental:** run completed, but scope or repeatability is incomplete.
- **Validated:** the stated claim reproduced under the stated conditions.
- **Deprecated:** superseded or known not to be the recommended path.

Do not generalize a hardware-specific result to all GPUs. Do not call an experiment validated without recording failures and limitations.

## Artifact hygiene

- Never publish private, restricted, or unlicensed source data.
- Preserve canonical records; derived/tokenized data gets a new revision.
- Record checksums for external datasets and large artifacts.
- Avoid committing credentials, local paths containing secrets, or machine-specific caches.
- Prefer small fixtures that let contributors reproduce the claim quickly.
