# Dataset bundles and `.dts`

Spaceslug datasets are versioned, inspectable, and reproducible. A dataset used for training or evaluation is identified by its content hash and lineage, not by a mutable filename.

## Goals

- Preserve a lossless canonical record representation.
- Provide an efficient derived representation for tokenized training.
- Make transformations, filtering, deduplication, and splits auditable.
- Prevent benchmark contamination and silent dataset drift.
- Allow other users to inspect, verify, and reproduce measurements.

## `.dts` bundle

`.dts` means **Spaceslug Dataset Bundle**. The format is a directory or a `.tar.zst` archive with this logical layout:

```text
example.dts/
├── manifest.json
├── records/
│   ├── train.jsonl.zst
│   ├── validation.jsonl.zst
│   └── test.jsonl.zst
├── tokens/                 # optional derived representation
│   ├── train.u32
│   ├── validation.u32
│   └── test.u32
├── indexes/                # optional memory-mapped indexes
├── source/
│   ├── sources.jsonl
│   └── licenses.jsonl
├── preprocessing/
│   ├── config.json
│   └── stats.json
└── checksums/sha256.json
```

The canonical records are UTF-8 JSON Lines compressed with Zstandard. Compression is lossless. Tokenized files and filtered/constructed records are derived artifacts and must identify their parent dataset and transformation revision.

The `.dts` extension identifies the bundle contract; it does not hide the contents behind an opaque binary format. A reader must be able to inspect the manifest and verify every shard without a model runtime.

## Record contract

Each normalized record must have a stable `record_id`, a record `kind`, and content appropriate to that kind. Chat and instruction records should preserve messages, roles, tool metadata, and source provenance. Raw source text must not be silently replaced by summaries or truncated content; truncation is a named derived transformation.

## Lossless and lossy processing

- **Lossless canonical data:** normalized UTF-8 records, source references, licenses, and metadata.
- **Lossless compression:** Zstandard for transport and storage; decompression must reproduce bytes exactly.
- **Derived training data:** token IDs, masks, packed sequences, filtered records, and synthetic examples.
- **Lossy transformations:** truncation, summarization, redaction, or lossy normalization create a new revision with a parent revision, configuration, tool version, and counts of changed records.

A source dataset is never overwritten by a derived dataset.

## Provenance and identity

The manifest and every shard are content-addressed with SHA-256. Dataset identity includes:

- source locations and source revisions;
- license and usage metadata;
- tokenizer and chat-template revisions;
- preprocessing code revision and configuration;
- split algorithm, seed, and membership;
- parent dataset revision, when derived;
- record and token counts;
- privacy/redaction status;
- per-file checksums.

A dataset loader must fail closed for unsupported schema versions and must report lineage overlap with protected benchmark datasets.

## Benchmark protection

Datasets under `benchmarks/` are immutable evaluation inputs. Training loaders must reject known benchmark revisions or report an explicit contamination error. Benchmark revisions must not be silently regenerated under the same identifier.

## Storage policy

Small schemas, manifests, examples, statistics, and experiment reports belong in Git. Large `.dts` bundles belong in Git LFS, release assets, or a content-addressed artifact store. The repository must retain the manifest, checksum, retrieval location, and verification command for every published bundle.

## Initial acceptance

The first implementation must:

1. create a deterministic bundle from normalized records;
2. verify the manifest and all checksums;
3. reload records byte-for-byte;
4. reproduce token IDs with a pinned tokenizer;
5. print split, length, license, and provenance statistics;
6. reject or flag benchmark overlap.

The format is **Proposed** until this acceptance test is implemented and passing.

See [`schemas/dataset-manifest.schema.json`](../schemas/dataset-manifest.schema.json).

> Dataset quality and training loss are separate measurements. Every loss result must reference an exact dataset revision, model revision, tokenizer revision, code revision, and configuration.

> Dates and measurements in this repository must be taken from actual runs. Planned values are labeled as planned and are not presented as results.
