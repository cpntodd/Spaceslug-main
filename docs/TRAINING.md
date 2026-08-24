# Training and LoRA plan

## Strategy

Training is a first-class execution mode, not inference with an afterthought. The order is:

1. CPU reference training for Spaceslug-Tiny.
2. External adapter import and deployment.
3. CPU LoRA training.
4. Spaceslug Vulkan LoRA forward.
5. Vulkan backward and optimizer primitives.
6. Mixed CPU/GPU LoRA training.
7. Full training of small models.
8. MoE training research.

## LoRA forward

For a frozen projection:

```text
Y = X Wᵀ + scale · ((X Aᵀ) Bᵀ)
```

`W` remains frozen. `A` and `B` are trainable and use explicit accumulation and optimizer dtypes. Quantized frozen weights may be dequantized for the reference path; adapter parameters are not silently quantized during training.

## Required training components

- Dataset loader and target-only loss masking.
- Teacher-forced causal language-model loss.
- Autograd or explicit backward plans.
- Gradient accumulation.
- Gradient clipping.
- AdamW optimizer.
- Checkpoint and resume.
- Deterministic seeds and RNG state.
- Validation split and regression metrics.
- Memory-budget planner.
- Optional activation checkpointing.

## First training acceptance

A tiny one-layer model must:

1. Run a fixed batch through CPU forward and loss.
2. Match a Python reference gradient.
3. Apply one optimizer update.
4. Reduce loss on the fixed batch.
5. Save and reload an identical checkpoint.
6. Produce the same result from a resumed run.

Only after this passes should Vulkan backward work begin.

## Optional fixed retained forward-loss binding

The native `spaceslug_tiny_forward_loss_fixed_retained` ABI is optional and is exposed by the Python backend as `execute_tiny_fixed_retained_loss` and by `PersistentTinyTrainer.fixed_forward_loss`. It is a distinct capability from forward-only retained execution: tokens, targets, and mask each must contain exactly 128 values, and the operation returns 128 row losses plus the fixed `128 × 259` logits buffer. When the runtime does not export the symbol, the binding reports `not-run` rather than falling back or claiming production training support.

## Vulkan LoRA acceptance

For the same tiny model and batch, CPU and Vulkan must agree on:

- forward activations;
- loss;
- adapter gradients;
- accumulated gradients;
- AdamW state;
- updated adapter parameters;
- next-step loss.

Tolerances must be defined per dtype and reduction order. Exact token identity is not a sufficient training metric.

## Native FP32 LM-head-only base-training boundary

The repository exposes a bounded native FP32 LM-head-only path via `native_fp32_lm_head_capability()` and `native_fp32_lm_head_training_plan()`. The runtime ABI computes LM-head gradients and applies SGD using caller-supplied projected Tiny activations and dlogits; the only trainable group is `lm_head`. The backbone groups (embeddings, attention QKV/output, feed-forward, normalization, and positional embeddings) remain unsupported for native base training.

This is **implemented standalone**, not integrated into Tiny forward activations, Tiny loss, or dataset training. The native binding does not produce the activations or dlogits itself; CPU remains authoritative for the integrated training path. The plan must not be interpreted as full-base training: `full_base_training` is false and full-base backward/update are explicitly listed as unsupported. Tests assert these boundaries.

## Spaceslug-0.5B policy

The dedicated 0.5B model is an MVP deployment target. Its initial pretraining may be external or staged/offloaded. The RX580 MVP requirement is quantized inference and practical LoRA adaptation, not a claim that frontier-quality 0.5B pretraining is fast locally.

## Memory accounting

The planner must separately budget:

- frozen weights;
- trainable parameters;
- optimizer state;
- activations saved for backward;
- gradients;
- KV/cache state;
- temporary workspace;
- checkpoints.

Execution fails before starting when the plan exceeds the selected budget.
