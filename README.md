# Spaceslug-main

Spaceslug-main is the GUI-first model laboratory and engine built around the separate `vulkan-runtime` project (Product 1: Spaceslug). It combines Cactus-inspired graph compilation and model conversion with Colibri-inspired MoE scheduling, expert residency, and tiered storage.

The first product milestone is a dedicated small Spaceslug model that can be trained, evaluated, chatted with, and used under supervision for narrow coding-agent tasks on constrained hardware.

## Products

- **Spaceslug** — `/vulkan-runtime`: RX580/RADV/gfx803 Vulkan compute runtime and validated kernels.
- **Spaceslug-main** — this repository: canonical model IR, artifacts, runtime orchestration, GUI, training, evaluation, experiments, and agent workflows.

## MVP success criterion

A user can open the GUI, train or adapt a small Spaceslug model, chat with it through the Spaceslug Vulkan backend, run a measured experiment, and inspect or approve the resulting improvement.

## Current status

**Experimental CPU-first MVP with a bounded GPU LoRA training path.** The repository includes the mouse-capable curses TUI, deterministic Spaceslug-Tiny CPU training/inference, checkpoint/artifact verification, live loss callbacks, and explicit CPU acceptance gates. The custom Vulkan runtime is host-connected for validated RX580/RADV SGEMM and attention-kernel gates, with lavapipe correctness coverage and structured parity reports.

Tiny projected-attention GPU LoRA training is implemented for the fixed MVP contract (single-head H=64, V=259, padded T=128, rank=4): LoRA-aware Q/K/V/output forward composition, padded-vocabulary causal loss and dLogits, LM-head/output/causal-attention/QKV backward, four-adapter dA/dB gradients, GPU SGD, persistent device-resident activations, reusable ExecEngine submissions, and adapter checkpoint/restore. The native persistent graph is CPU-parity tested on RADV and lavapipe. Scope is fixed to fp32 H=64/V=259/Vp=320/T<=128, rank=4, frozen base weights, and SGD. Gradient accumulation, AdamW, other model shapes, and dataset-scale training are unsupported; CPU remains authoritative outside this experimental path.

### Quick start

```bash
cd spaceslug-main
PYTHONPATH=python python3 -m spaceslug.cli tui
PYTHONPATH=python python3 -m spaceslug.cli tiny-infer 1 2 3
PYTHONPATH=python python3 -m spaceslug.cli tiny-attention-gate --tokens 128 --hidden-size 64
PYTHONPATH=python python3 -m spaceslug.cli tiny-attention-gate --tokens 128 --hidden-size 64 --software-vulkan
PYTHONPATH=python python3 -m unittest -v
```

The TUI keyboard actions include dataset/model/training navigation, CPU verification/training, CPU inference, Vulkan GEMM, the validated attention gate, and GPU-chain status inspection. See [`docs/devlog/0012-tiny-tui-cpu-gpu-mvp.md`](docs/devlog/0012-tiny-tui-cpu-gpu-mvp.md).

The proposed Spaceslug Dataset Bundle (`.dts`) preserves lossless canonical records, supports derived tokenized training data, and records lineage and checksums. See [`docs/DATASETS.md`](docs/DATASETS.md), [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and [`docs/DEVLOG.md`](docs/DEVLOG.md).

## Design principles

1. CPU reference first; Vulkan acceleration second.
2. Explicit tensor layout, ownership, precision, quantization, and fallback contracts.
3. Reproducible artifacts and experiments.
4. Small trainable model before ambitious 0.5B pretraining.
5. Human approval for code changes and bounded self-improvement.
6. Model semantics must not change silently when placement or precision changes.

Read [`docs/PRODUCT.md`](docs/PRODUCT.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and [`docs/ROADMAP.md`](docs/ROADMAP.md) first.

## Repository relationships

| Repository | Role |
|---|---|
| `../vulkan-runtime` | Spaceslug Vulkan backend, RX580/gfx803 kernels |
| `../cactus-main` | Reference for graph, conversion, quantization, and model execution |
| `../colibri-main` | Reference for MoE routing, expert residency, and tiered inference |
| `../needle-main` | Reference for LoRA data, training, and adapter export workflows |

## Non-goals for the first release

- Frontier-scale training on the RX580.
- Unrestricted autonomous source modification.
- Silent replacement of CPU reference behavior.
- Supporting every Hugging Face architecture.
- Treating a successful process exit as proof of model correctness.

## License

To be decided before implementation and publication.

## Status vocabulary

- **Proposed** — documented direction, not implemented.
- **Experimental** — implementation exists but lacks complete acceptance evidence.
- **Validated** — acceptance tests and reproducibility evidence exist.
- **Deprecated** — retained for compatibility but not recommended.
