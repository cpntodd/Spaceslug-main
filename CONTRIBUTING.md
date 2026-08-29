# Contributing to Spaceslug

Welcome. Spaceslug is a community project for making local AI useful on older and capable hardware. Documentation, issue reports, reproducibility notes, model experiments, GPU profiling, and small test fixtures are all valuable contributions.

> **Status:** Experimental. Keep claims narrow, evidence visible, and unknowns explicit.

## Before you start

1. Read the [documentation index](docs/INDEX.md).
2. Check existing issues, decisions, roadmap, and devlog entries.
3. For GPU work, read the [GPU contribution guide](docs/GPU_CONTRIBUTING.md).
4. For models/adapters, read the [model contribution guide](docs/MODEL_CONTRIBUTING.md).
5. For measurements, use the [experiment protocol](docs/EXPERIMENTS.md) and [report template](docs/experiments/EXPERIMENT_REPORT_TEMPLATE.md).

## Contribution rules

- Keep the CPU/reference path authoritative for correctness.
- Add a reference test before adding or changing a backend implementation.
- Make capability limits, fallbacks, precision, ownership, and tensor layouts explicit.
- Record model, dataset, code, hardware, driver, precision, and artifact revisions.
- Never publish private, restricted, or unlicensed source data or artifacts.
- Never make a fallback silent in user-visible metrics.
- Keep changes small, reversible, and independently testable.
- Do not apply autonomous source changes directly to `main`; use a branch and review.
- Treat Cactus, Colibri, and Needle as references/import sources; do not silently fork their assumptions.

## Monorepo layout

- `python/` — host application and reference implementation.
- `vulkan-runtime/` — native Vulkan runtime, shaders, and backend tests.
- `docs/` — contracts, guides, decisions, devlogs, and evidence.
- `datasets/`, `manifests/`, `experiments/`, `benchmarks/` — provenance and measurement surfaces.

## Pull requests

Please include:

- What changed and why.
- Tests and exact commands run.
- Hardware/backend/environment details when relevant.
- Documentation and status-label updates.
- Known failures, limitations, and follow-up work.

A successful process exit is not proof of model correctness. Reviewers should be able to reproduce the stated claim and see what was not tested.

## License status

The project license is currently pending. Until a license is committed, do not assume code, documentation, logo artwork, datasets, or model artifacts are licensed for reuse.
