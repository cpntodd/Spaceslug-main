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

## Optional graph dstate readback binding

The optional `spaceslug_tiny_forward_readback_graph_dstate` ABI is exposed by `BackendSession.readback_tiny_graph_dstate`. It accepts equal token/target/mask windows with `1 <= rows <= 128`, returns graph-produced FP32 `dstate` rows with hidden size 64 plus zero-padded capacity rows through 128, and explicitly does not update embeddings. Capability/status metadata is reported as `tiny_graph_dstate_readback_*`; absent symbols remain unsupported without fallback.

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

## Native FP32 standalone base-training subsets

The native runtime exposes three bounded FP32 standalone base-training subsets: **LM-head**, **output projection**, and **combined QKV**. Each uses **SGD** and caller-supplied activations/states plus upstream gradients. Separately, the persistent Tiny graph exports graph-owned fixed-window SGD for the LM-head, output projection, and combined QKV through `train_tiny_graph_lm_head_sgd`, `train_tiny_graph_output_sgd`, and `train_tiny_graph_qkv_sgd`; all validate 1..128 rows. The capability metadata is available through `native_fp32_lm_head_capability()` and `native_fp32_lm_head_training_plan()`.

These are **implemented standalone** native ABIs: none is integrated into Tiny forward activations, the Tiny command graph, Tiny loss, or dataset training. They are independent composition primitives, not a complete transformer trainer. The runtime graph separately owns the Tiny LM-head parameter and exposes FP32 import/readback. When the runtime exports the complete AdamW symbol set, ctypes binds graph-owned LM-head AdamW training plus `m`/`v`/step readback and update; the same conditional binding now applies to graph-owned output projection AdamW (`tiny_graph_integrated_output_adamw`) with its 64x64 state. If required symbols are absent, the binding reports **`-4`**. AdamW remains unsupported for the graph-owned combined QKV group, whose SGD metadata explicitly reports `adamw_return_code: -4`. This remains a fixed 1..128 host-staged token window: it is not dataset-device training, retained-command training, or full-base training. Graph ownership and integrated AdamW must not be confused with the standalone API. The remaining backbone groups (embeddings, feed-forward, normalization, and positional embeddings) remain unsupported, and `full_base_training` is false; do not interpret these three subsets as full base training. CPU remains authoritative for the integrated training path, and tests assert these boundaries.

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
