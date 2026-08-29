# Spaceslug documentation

> **Transparency rule:** Every page should say what is implemented, what is measured, what is inferred, and what remains unknown.

## Start here

- [User Guide — Phase 1](USER_GUIDE_PHASE1.md)
- [Product direction](PRODUCT.md)
- [Architecture](ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)
- [Staged roadmap](STAGED_ROADMAP.md)

## For users

- [GUI](GUI.md)
- [Datasets](DATASETS.md)
- [Artifact format](ARTIFACT_FORMAT.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Training](TRAINING.md)

## For GPU contributors

- [GPU contribution guide](GPU_CONTRIBUTING.md)
- [Tiny GPU training status](TINY_GPU_TRAINING_STATUS.md)
- [Vulkan runtime](../vulkan-runtime/)
- [RX580 training alternatives](RX580_TRAINING_ALTERNATIVES.md)
- [AMD Vulkan/gfx803 research](../vulkan-runtime/docs/AMD_VULKAN_GFX803_RESEARCH.md)

## For model contributors

- [Model contribution guide](MODEL_CONTRIBUTING.md)
- [Integration](INTEGRATION.md)
- [Mixture-of-experts notes](MOE.md)
- [Self-improvement boundaries](SELF_IMPROVEMENT.md)

## Experiments and evidence

- [Experiment protocol](EXPERIMENTS.md)
- [Experiment report template](experiments/EXPERIMENT_REPORT_TEMPLATE.md)
- [Benchmarks](../benchmarks/README.md)
- [Devlog](DEVLOG.md)
- [Decision record](DECISIONS.md)

## Community

- [Contributing](../CONTRIBUTING.md)
- [Issue tracker](https://github.com/cpntodd/Spaceslug-main/issues)
- [Discussions](https://github.com/cpntodd/Spaceslug-main/discussions)

## Status labels

Use these labels consistently:

- **Proposed:** an idea or contract under discussion.
- **Experimental:** code exists, but evidence or coverage is incomplete.
- **Validated:** tests, environment details, and reproducible evidence are linked.
- **Deprecated:** kept for compatibility, but no longer recommended.

A “Validated” label must never mean “works everywhere.” It means the stated scope and environment were tested.
