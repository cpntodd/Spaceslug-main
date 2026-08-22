# Reproducibility and measurement

Spaceslug reports progress through immutable inputs, machine-readable records, and commands that other users can run. A successful process exit is not evidence of correctness by itself.

## Required identity fields

Every training, evaluation, dataset, and backend result should identify:

- Spaceslug-main Git commit;
- `Spaceslug`/`vulkan-runtime` Git commit;
- model and artifact revision;
- dataset and split revision;
- tokenizer and chat-template revision;
- configuration and random seeds;
- hardware, driver, Vulkan implementation, and backend;
- precision, quantization, and explicit fallbacks;
- command or GUI configuration;
- metrics, tolerances, warnings, and known limitations.

## Experiment directory

A completed run should produce a directory under `experiments/runs/` (ignored from Git for large outputs) and a small report or metadata reference under `experiments/` when appropriate:

```text
experiments/<run-id>/
├── experiment.json
├── config.json
├── metrics.json
├── logs/
├── checkpoints/          # normally external or ignored
└── report.md
```

The committed experiment record must remain sufficient to locate or retrieve large outputs by checksum.

## Comparisons

Comparisons must keep constant the metric definition, evaluation split, tokenizer behavior, prompt/template, and stopping rules. If any of these change, the result is a new comparison series rather than a directly comparable continuation.

Loss must report at least the reduction definition, token count, masking policy, and validation dataset revision. Throughput must report warmup, run count, statistic (normally median), transfer behavior, and whether CPU fallback occurred.

## Reproduction checklist

```text
1. checkout the recorded Spaceslug-main commit
2. checkout or fetch the recorded vulkan-runtime commit
3. retrieve each dataset/model artifact by checksum
4. verify manifests and shard checksums
5. install the recorded toolchain/dependencies
6. run the recorded command
7. compare metrics using the recorded tolerances
```

Hardware-specific performance is labeled as such. Correctness results should include the CPU reference and, where supported, both RADV and lavapipe paths.

## Evidence policy

- **Proposed:** design only; no implementation evidence.
- **Experimental:** implementation exists; acceptance evidence is incomplete.
- **Validated:** acceptance test, environment, artifacts, and reproducibility evidence are recorded.
- **Deprecated:** retained for compatibility but not recommended.

Devlogs summarize changes and evidence; schemas and machine-readable experiment records are authoritative.
