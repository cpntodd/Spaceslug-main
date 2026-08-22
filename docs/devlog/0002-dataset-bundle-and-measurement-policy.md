# Devlog 0002 — Dataset bundle and measurement policy

- **Status:** Proposed
- **Date:** 2026-08-23
- **Spaceslug-main revision:** record the commit when this entry is committed
- **vulkan-runtime revision:** not applicable; no runtime code changed

## Objective

Define a lossless canonical dataset representation, efficient derived training data, lineage tracking, and reproducible progress reporting.

## Decision

The proposed `.dts` Spaceslug Dataset Bundle keeps canonical UTF-8 JSONL records with lossless Zstandard compression. Token IDs, masks, packed sequences, filtered records, and synthetic examples are derived artifacts with explicit parent revisions and transformation metadata.

The repository adds dedicated locations for dataset manifests, protected benchmark definitions, experiment metadata, and chronological devlogs. Large data and run outputs are external artifacts referenced by checksum rather than committed directly to Git.

## Measurements

No dataset writer or training measurement exists yet. This entry records contracts only; there are no loss or throughput results to report.

## Next acceptance

Implement deterministic `.dts` creation and verification, then prove byte-identical reload, pinned-tokenizer reproducibility, statistics generation, and benchmark-overlap protection.
