# Alternative 4 — Fixed-rank LoRA, rsLoRA, DoRA, and AdaLoRA research

> **Correction note (2026-08-25):** this revision makes fixed-rank LoRA the only immediate recommendation, treats rsLoRA as an optional rank-stabilizing rule, corrects DoRA to train magnitude plus LoRA direction, and removes any implication that NF4/QLoRA or AdaLoRA is already supported.

**Recommendation:** **Productize the existing fixed-rank FP32 LoRA path first. Add rsLoRA, DoRA, NF4-style frozen storage, and AdaLoRA only as separately capability-gated experiments with CPU references and real RX580 measurements.**

## Fixed-rank LoRA proposal

For a frozen projection:

```text
Y = X W^T + gamma_r * ((X A^T) B^T)
```

`W` is frozen; `A` and `B` are trainable. The artifact must record rank, target module, `alpha`, exact `gamma_r` rule, initialization, optimizer, and base-model identity. Conventional LoRA commonly uses `gamma_r = alpha/r`. rsLoRA proposes `gamma_r = alpha/sqrt(r)` to prevent gradient collapse as rank grows under the paper's assumptions.

rsLoRA is a rank-stabilizing alternative, not a universal correctness requirement and not a guarantee of equal effective learning rate, update magnitude, or quality. Keep both rules explicit in profile and checkpoint metadata; benchmark them rather than silently changing existing rank-4/rank-8 semantics.

## Current repository fit

This is the only alternative with a bounded GPU training implementation. The accepted runtime contract is exact:

- FP32 arithmetic;
- frozen base weights;
- `H=64`, `V=259`, `Vp=320`, `T=128`;
- rank 4 or rank 8 profiles only;
- bounded ordinary backward/update submissions;
- current LoRA gradients, accumulation, GPU SGD, and checkpoint/restore paths.

This is not arbitrary-shape adapter training, retained full training, 0.5B adaptation, NF4, DoRA, AdaLoRA, full dataset integration, or full-base training. Any broader method requires new capability metadata and acceptance tests.

## Memory assessment

For one base matrix `[d_out,d_in]` and rank `r`, LoRA trainable parameters are approximately:

```text
r * (d_in + d_out)
```

Their FP32 weights, gradients, and optimizer state scale with this adapter size rather than the frozen base. Total VRAM still includes frozen base storage, saved activations, dequantization/workspace, adapter accumulation, optimizer state, Vulkan alignment/staging, and checkpoints.

A quantized 0.5B frozen base may fit in principle, but useful **training** is conditional on architecture-specific Q4 dequantization during forward/backward activation flow, workspace, context length, and the adapter path. The current Tiny CQ4/Q4 capabilities do not establish an arbitrary 0.5B training graph.

## rsLoRA experiment

Add rsLoRA only by versioning the scaling rule and testing:

- forward/backward parity for `alpha/r` and `alpha/sqrt(r)`;
- rank 4 versus rank 8 gradient statistics;
- checkpoint identity and import/export compatibility;
- whether higher rank improves held-out quality enough to justify more activation/GEMM cost.

Do not retroactively reinterpret existing adapters under the new rule.

## DoRA experiment

DoRA decomposes the pretrained weight into a magnitude vector and a normalized direction. It trains the magnitude vector and uses LoRA to parameterize updates to the direction. It does **not** adapt direction only.

DoRA may improve quality in the paper's tested settings, but it adds trainable magnitude parameters, normalization/reparameterization passes, gradients, checkpoint data, and bandwidth. Although merged inference can avoid additional inference overhead, training is not cost-free. Required gates include:

- CPU formula/gradient oracle for magnitude and direction;
- stable norm/epsilon behavior for near-zero columns/rows;
- FP32 Vulkan parity for normalization and backward;
- added peak VRAM and dispatch measurements;
- held-out comparison against fixed-rank LoRA at matched rank and budget.

## NF4/QLoRA-style frozen storage experiment

QLoRA uses NF4 frozen-base storage, double quantization, and paged optimizer techniques in an ecosystem that commonly computes in BF16. None of those capabilities should be inferred here. A Spaceslug variant would use FP32 compute and would need:

- an NF4 codebook/packing format distinct from existing CQ4 unless explicitly proven equivalent;
- scale and double-quant metadata contracts;
- blockwise FP32 dequantization integrated into each supported forward/backward consumer;
- bounded dequant workspace and dispatches;
- adapter gradient parity and checkpoint/base-identity validation;
- architecture-specific support rather than “arbitrary QLoRA.”

Until those exist, call the current path “LoRA over a supported frozen base,” not QLoRA.

## AdaLoRA experiment

AdaLoRA allocates a global rank budget dynamically using importance scores and a scheduled pruning process. Fixed maximum-rank physical buffers plus masks are a proposed Vulkan implementation technique, not the paper algorithm itself. They avoid descriptor reallocation but may retain maximum-rank memory and thus erase expected memory savings.

A correct implementation needs importance-score state, budget schedule, prune/mask semantics, optimizer-state masking or compaction, regrowth policy if supported, stable descriptor ownership, and checkpoint/resume across allocation events. Fixed-rank LoRA must remain the fallback/control.

## Other optional methods

- **VeRA** shares frozen random low-rank bases and trains small scaling vectors; lower trainable state but different expressiveness.
- **LoRA+** uses different learning rates for `A` and `B`; inexpensive to test once optimizer parameter groups exist.
- **ReLoRA** periodically merges adapters during full-model training; it changes the trainable-base and optimizer-reset contract and is not ordinary adapter tuning.
- **GaLore** projects full gradients; it belongs to full-base research, not LoRA productization.

## Main risks

- A frozen adapter cannot replace pretraining or repair an unsuitable base.
- Scaling-rule changes can make checkpoints incompatible.
- DoRA and AdaLoRA add more state and kernels than fixed LoRA.
- Maximum-rank masked AdaLoRA buffers may not save peak VRAM.
- NF4 is not the existing Q4 format by default and may be slow under FP32-only compute.
- Current fixed Tiny profiles cannot be generalized by metadata alone; kernels and graph ownership must also support each shape.
- Small datasets can overfit; held-out evaluation is mandatory.

## Validation gates

- Existing CPU/GPU LoRA parity: forward, loss, gradients, accumulation, update, next-step loss, and resume.
- Exact base-model identity and scaling-rule metadata in every adapter artifact.
- Real RX580 peak VRAM, timestamped bounded dispatches, thermal/watchdog behavior, and tokens/second.
- rsLoRA versus conventional scaling at matched ranks and learning-rate sweeps.
- Separate DoRA gradient/norm/checkpoint gates.
- Separate NF4 pack/dequant/parity and workspace gates.
- AdaLoRA importance, budget, pruning, masked-state, and resume parity.
- Held-out task quality and regression against the unadapted base and fixed-rank LoRA control.

## Recommendation

**Approve only fixed-rank LoRA for immediate GUI/product work.** Preserve the exact rank-4/rank-8 Tiny profiles and expose their capability boundaries. Next, add parameter-group optimizer support and compare conventional scaling with rsLoRA. Evaluate DoRA only if fixed-rank LoRA quality is insufficient. Pursue NF4/QLoRA-style storage only after an architecture-specific dequantization contract exists. Leave AdaLoRA until stable fixed-rank profiles and checkpoint semantics are complete.

## References

- Current implementation boundary: [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md).
- Hu et al., *LoRA*, 2021: https://arxiv.org/abs/2106.09685
- Zhang et al., *AdaLoRA*, 2023: https://arxiv.org/abs/2303.10512
- Kalajdzievski, *rsLoRA*, 2023: https://arxiv.org/abs/2312.03732
- Liu et al., *DoRA*, 2024: https://arxiv.org/abs/2402.09353
- Dettmers et al., *QLoRA*, 2023: https://arxiv.org/abs/2305.14314
- Kopiczko et al., *VeRA*, 2023: https://arxiv.org/abs/2310.11454
- Hayou et al., *LoRA+*, 2024: https://arxiv.org/abs/2402.12354
- Lialin et al., *ReLoRA*, 2023: https://arxiv.org/abs/2307.05695

Only the bounded fixed-rank LoRA contract is currently implemented; the other methods are research proposals.
