# Bounded self-improvement and autoresearch

Self-improvement means measured, reproducible experimentation. It does not mean unrestricted autonomous modification.

## Experiment loop

```text
Select target
  → isolated branch/worktree
  → one proposed change
  → fixed training/evaluation budget
  → run tests and benchmarks
  → compare primary and safety metrics
  → keep or reject
  → human review of retained result
```

## First experiment variables

- learning rate;
- warmup and scheduler;
- batch and sequence length;
- LoRA rank and target modules;
- dataset mixture;
- weight decay and clipping;
- training-step budget;
- quantization and deployment settings;
- prompts and tool-use templates.

Architecture, compiler, shader, and kernel changes come later and require stricter gates.

## Required experiment record

```json
{
  "experiment_id": "...",
  "parent_model": "...",
  "code_revision": "...",
  "dataset_revision": "...",
  "config": {},
  "hardware": {},
  "budget": {"seconds": 3600, "steps": 1000},
  "metrics": {"train_loss": 0, "validation_loss": 0, "coding": 0},
  "resource": {"throughput": 0, "peak_memory": 0},
  "status": "kept"
}
```

## Agent permission levels

1. Read-only repository inspection.
2. Read and test execution.
3. Isolated patch proposal.
4. Isolated patch plus benchmark.
5. Human-approved application.

The agent may not alter evaluation criteria, disable tests, push directly to main, run without a budget, or apply source changes without approval.

## Improvement categories

- **Model:** weights, adapters, data, and training configuration.
- **Agent:** prompts, tools, workflows, and task decomposition.
- **Engine:** compiler, scheduler, and runtime changes.
- **Kernel:** Vulkan shader and synchronization changes.

Kernel and engine changes require CPU-reference parity, Vulkan validation, regression tests, and benchmark evidence.

## Success condition

An experiment is valuable only when it records an improvement against a fixed baseline without violating correctness, memory, performance, or safety gates.
