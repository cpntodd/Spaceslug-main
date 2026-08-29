<p align="center">
  <img src="Spaceslug Logo.png" alt="Spaceslug logo" width="220">
</p>

<h1 align="center">Spaceslug</h1>

<p align="center"><strong>Bringing local AI to older systems.</strong></p>

<p align="center">
  A transparent, community-built model lab and runtime for making capable local AI useful on hardware that still has life in it.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11 or newer">
  <img src="https://img.shields.io/badge/Platform-Linux-333333?logo=linux&logoColor=white" alt="Linux">
  <img src="https://img.shields.io/badge/Hardware-RX580%20%2B%20RADV%20validated-E95420" alt="RX580 and RADV validated">
  <img src="https://img.shields.io/badge/License-pending-lightgrey" alt="License pending">
</p>

> **Project status: Experimental.** Spaceslug is an active research and engineering project. Claims in this README are intentionally bounded by published tests, reports, and hardware evidence. If a capability is not marked **Validated**, expect rough edges and help-wanted signs.

## Why Spaceslug?

Older GPUs are not disposable. They are affordable, widely available, and often capable enough for useful local inference, adaptation, and experimentation when software respects their constraints.

Spaceslug is building a friendly path from **dataset → model artifact → measured run → inspectable result**. We want new users to be able to try local AI, and contributors to be able to add GPU backends, kernels, models, tokenizers, and experiments without reverse-engineering hidden assumptions.

## What is here today?

- A Python host application with TUI, desktop, dataset, training, inference, and local API surfaces.
- A CPU-first reference path that remains the authority for integrated training and correctness.
- A Vulkan runtime in [`vulkan-runtime/`](vulkan-runtime/) with validated RX580/RADV/gfx803 gates and lavapipe coverage.
- An experimental, bounded GPU LoRA path for Spaceslug-Tiny.
- Dataset bundles, artifact contracts, reproducibility guidance, devlogs, and experiment records.
- A deliberately transparent status vocabulary: **Proposed**, **Experimental**, **Validated**, and **Deprecated**.

Current GPU support is not universal. RX580/RADV/gfx803 is the evidence-backed target today; other GPUs and APIs are welcome as community work, but must earn their status through reproducible parity and acceptance evidence.

## Quick start

```bash
cd spaceslug-main
python3 -m venv .venv
. .venv/bin/activate
PYTHONPATH=python python3 -m spaceslug.cli tui
```

Useful smoke checks:

```bash
PYTHONPATH=python python3 -m spaceslug.cli desktop
PYTHONPATH=python python3 -m spaceslug.cli tiny-infer 1 2 3
PYTHONPATH=python python3 -m spaceslug.cli tiny-attention-gate --tokens 128 --hidden-size 64
PYTHONPATH=python python3 -m unittest -v
```

Start with the [user guide](docs/USER_GUIDE_PHASE1.md) if this is your first visit.

## Navigate the lab

| I want to… | Start here |
|---|---|
| Run Spaceslug for the first time | [`docs/INDEX.md`](docs/INDEX.md) |
| Understand the architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Add or validate GPU support | [`docs/GPU_CONTRIBUTING.md`](docs/GPU_CONTRIBUTING.md) |
| Add a model or adapter | [`docs/MODEL_CONTRIBUTING.md`](docs/MODEL_CONTRIBUTING.md) |
| Publish a reproducible experiment | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and [`docs/experiments/EXPERIMENT_REPORT_TEMPLATE.md`](docs/experiments/EXPERIMENT_REPORT_TEMPLATE.md) |
| See what is decided and what is not | [`docs/DECISIONS.md`](docs/DECISIONS.md), [`docs/ROADMAP.md`](docs/ROADMAP.md), and [`docs/DEVLOG.md`](docs/DEVLOG.md) |
| Contribute code or documentation | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## A linked workspace, one community

This repository is the monorepo. Its products are kept in clear directories:

- `python/` — host orchestration, CLI, GUI, training, and inference.
- `vulkan-runtime/` — Vulkan compute runtime, shaders, native API, and backend tests.
- `docs/` — contracts, user guides, design notes, devlogs, and status evidence.
- `datasets/`, `manifests/`, `experiments/`, `benchmarks/` — provenance and measurement surfaces.

References to Cactus, Colibri, and Needle are research/import references, not hidden dependencies or silently forked code.

## Community mission

We are especially interested in contributions that make local AI more accessible:

- GPU support for older AMD, NVIDIA, Intel, and integrated hardware.
- Vulkan, ROCm, CUDA, Metal, DirectML, or other carefully scoped backend adapters.
- New model families, tokenizers, quantization paths, and LoRA/adapters.
- CPU reference implementations and small fixtures that make GPU work verifiable.
- Documentation, translations, benchmark reports, failure reports, and friendly onboarding.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Please publish limitations and failed experiments too: negative evidence is part of the project.

## Principles

1. CPU reference first; acceleration second.
2. Explicit ownership, layout, precision, fallback, and capability contracts.
3. Reproducible artifacts and experiments over impressive claims.
4. Small, useful models before frontier-scale promises.
5. Human-reviewed changes and bounded self-improvement.
6. No silent semantic changes when placement or precision changes.

## License

**License pending.** Until a license is added to this repository, all rights remain reserved by the copyright holder. Do not assume that code, documentation, logo artwork, datasets, or model artifacts are licensed for reuse. Contributions and redistribution terms will be clarified before public release.

## Status vocabulary

- **Proposed** — documented direction; not implemented.
- **Experimental** — implementation exists but lacks complete acceptance evidence.
- **Validated** — acceptance tests and reproducibility evidence are published.
- **Deprecated** — retained for compatibility but not recommended.

If you spot an inaccurate status, missing evidence, or an easier path for new contributors, please open an issue.

<p align="center"><em>Keep useful hardware useful. Keep the evidence visible. Build together.</em></p>
