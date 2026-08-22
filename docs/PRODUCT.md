# Product definition

## Vision

Spaceslug-main is a local, inspectable model-development workstation for constrained hardware. It should support the complete loop from data and training through inference, evaluation, supervised coding assistance, and bounded automated improvement.

## Two-product boundary

### Spaceslug

The existing `vulkan-runtime` repository remains the hardware product:

- Vulkan device, memory, queue, and synchronization management.
- RX580/RADV/gfx803-specific kernels and constraints.
- Quantized inference primitives.
- Future backward and optimizer primitives.
- CPU-reference parity and GPU benchmarks.

It must not own tokenizers, model-family policy, datasets, GUI state, or experiment policy.

### Spaceslug-main

This repository owns:

- Canonical model IR and backend-independent plans.
- Model import, conversion, and artifact packaging.
- CPU reference execution.
- Spaceslug backend integration.
- Persistent decode and request state.
- MoE routing, expert residency, and placement policy.
- LoRA adapters and training orchestration.
- Evaluation and experiment tracking.
- GUI and local service.
- Supervised coding-agent and self-improvement workflows.

## MVP user journey

1. Open the local GUI.
2. Create or import a small model.
3. Select a dataset and a bounded training configuration.
4. Train a tiny model or LoRA adapter.
5. Evaluate loss, chat behavior, and coding tasks.
6. Chat through CPU or Spaceslug Vulkan.
7. Start a bounded experiment.
8. Review metrics and logs.
9. Approve or reject the retained model, adapter, or patch.

## Model tiers

### Spaceslug-Tiny

Approximately 10M–50M parameters. The first full-training and autograd target. It must train quickly enough for repeated correctness tests and autoresearch-style experiments.

### Spaceslug-125M

The first useful local assistant target. It should support instruction tuning, basic tool use, and narrow coding tasks.

### Spaceslug-0.5B

The dedicated product model target. It should be optimized for local chat, repository explanation, tool invocation, and constrained coding assistance. Initial delivery may use externally produced pretraining plus local LoRA/adaptation; full RX580 pretraining is not an MVP acceptance requirement.

## Product claims we may make

Only claim a capability after an acceptance test records:

- exact model/artifact revision;
- code revision;
- dataset revision;
- hardware and backend;
- precision and quantization;
- command or GUI configuration;
- metrics and tolerances;
- known fallbacks and limitations.

## Safety and control

Self-improvement is bounded experimentation, not unrestricted autonomy. Agents use isolated worktrees, fixed evaluation suites, resource budgets, and human approval before source changes are applied.
