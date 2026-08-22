# Contributing

Spaceslug-main is currently a documentation-first greenfield repository. Do not implement engine code until the contracts in `docs/` have been reviewed.

## Rules

- Keep `vulkan-runtime` as a separate product and repository.
- Treat `cactus-main`, `colibri-main`, and `needle-main` as references or import sources; do not silently fork their assumptions.
- Add a CPU/reference test before adding a backend implementation.
- Record model, dataset, code, hardware, precision, and artifact revisions in experiments.
- Never make a fallback silent in user-visible metrics.
- Never apply autonomous source changes directly to the main branch.
- Keep changes small, reversible, and independently testable.

## Documentation expectations

Architecture decisions belong in `docs/`. Artifact schema changes require versioning notes and compatibility behavior. Model changes require provenance and evaluation updates.

## Data and measurement rules

- Preserve canonical dataset records; derived or lossy transformations create a new `.dts` revision.
- Never publish private or licensed source data without documented permission and provenance.
- Do not train on protected benchmark revisions; record dataset lineage and overlap checks.
- Keep large datasets and run outputs outside ordinary Git history, referenced by checksums and retrieval instructions.
- Devlogs must distinguish proposed, experimental, and validated work and must not invent measurements.

## Planned checks

The future repository should provide formatting, schema validation, dataset bundle creation and verification, unit tests, CPU reference tests, Vulkan parity tests, and reproducible experiment checks from the CLI and GUI service.
