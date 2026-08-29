# RX580 training alternatives: review index

**Status:** Proposed engineering review — corrected 2026-08-25 after independent algorithm and gfx803 audits; see §Audit corrections and better solutions.
**Scope:** Single AMD Radeon RX 580, 8 GB VRAM, RADV/gfx803, FP32 arithmetic
**Repository:** `spaceslug-main` with `vulkan-runtime`
**Decision question:** Which training approach should follow the current Tiny GPU LoRA MVP?

## Executive summary

The four alternatives are materially different:

1. **FP32 SGD / momentum SGD** — simplest full-parameter baseline, lowest implementation risk, but slow convergence and difficult scaling.
2. **Factored Adafactor-style optimizer** — lower optimizer-state memory for matrix parameters, but 100–150M remains only a gated research target because activations and workspaces are not solved.
3. **Compressed 8-bit Adam-like state** — preserves Adam-family behavior while reducing state storage, but adds quantization error, conversion bandwidth, and a substantial validation burden.
4. **LoRA/AdaLoRA adapter training** — freezes the base and trains small adapters; the most practical and best aligned with the repository's existing GPU path, but it is adaptation rather than full pretraining.

## Recommendation

Adopt **Alternative 4 (fixed-rank LoRA) as the immediate product path**, retain **Alternative 1 (plain SGD) as the CPU/GPU control**, and investigate **Adafactor or SM3** only after the common optimizer and memory-planner contracts exist. Treat Muon, GaLore, CPU offload, compressed Adam, DoRA, NF4, and dynamic-rank AdaLoRA as separate experiments rather than assumed improvements. Keep **Alternative 3 (compressed Adam)** deferred until a measured bottleneck justifies its numerical and kernel complexity.

A 0.5B model should remain an inference plus adapter target. The existing roadmap explicitly treats full 0.5B RX580 pretraining as deferred research, and the current runtime does not implement complete all-parameter dataset training.

## Evidence boundary

Repository facts are drawn from [`ROADMAP.md`](ROADMAP.md), [`TRAINING.md`](TRAINING.md), [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md), and [`README.md`](../README.md). Those documents report passing correctness tests, not RX580 throughput, long-run stability, useful model quality, or full-training feasibility. Hardware assumptions follow the gfx803 constraints: FP32 compute, no tensor cores/cooperative matrices, 32 KB maximum LDS per workgroup, and watchdog-sensitive dispatches.

## Comparison matrix

| Alternative | Full-base capability | VRAM pressure | Existing fit | Primary recommendation |
|---|---|---:|---|---|
| FP32 SGD | Yes in principle | Low–medium | Standalone and graph SGD subsets exist | Build as control/reference |
| Factored Adafactor | Yes in principle | Medium, shape-dependent | Optimizer abstraction required | Research next for small/full models |
| 8-bit Adam-like | Yes in principle | Medium | No current implementation | Defer until measured need |
| LoRA/AdaLoRA | No; frozen base | Low | Bounded GPU LoRA already exists | Immediate practical path |

The matrix deliberately keeps optimizer choice, execution placement, and trainable scope as separate axes (see below). Combinations are possible only where their update semantics, memory ownership, and transfer schedule are explicitly defined and validated; no combination is presumed superior.

## Audit corrections and better solutions

Independent algorithm and gfx803 audits found material problems in the earlier revision. The reports now apply these corrections:

- **Separate optimizer, placement, and trainable scope.** These are independent design axes:

| Axis | Examples |
|---|---|
| Optimizer state | none (SGD); one full buffer (momentum/Lion/Muon); factored (Adafactor/SM3); two full or compressed moments (Adam) |
| Placement | resident; activation-checkpointed/recomputed; fused backward/update (LOMO); explicit CPU offload (ZeRO-Offload-style) |
| Trainable scope | full base; projected full gradient (GaLore); frozen base plus adapters (LoRA/DoRA/AdaLoRA) |

- **Correct 8-bit attribution.** Dettmers et al. quantize optimizer states, while retaining standard-precision weights and gradients; quantized gradients are a separate, more aggressive proposal and are not attributed to that paper.
- **Correct Adafactor.** Factored versus unfactored describes second-moment storage. Momentum, relative-step scheduling, and update clipping are configurable algorithm choices; update clipping applies to the preconditioned update. The factored estimate must reconstruct the normalized row/column outer product rather than merely store `R+C` values.
- **Correct LOMO and offload.** LOMO fuses backward gradient production with immediate parameter update to avoid a full gradient tensor and optimizer state; it is not CPU offload. ZeRO-Offload-style placement is a separate option with explicit PCIe traffic and host-resident state.
- **Correct adapter claims.** rsLoRA is a rank-stabilizing scaling rule, not a guarantee of equal effective learning rate. DoRA trains a magnitude vector and a LoRA-parameterized direction. NF4/QLoRA is not an existing runtime capability and would need its own dequantization, metadata, and parity path.
- **Use lower-bound memory tables carefully.** Bytes-per-parameter figures exclude activations, accumulation buffers, temporary reductions, dequantization/SVD/Newton–Schulz workspace, Vulkan allocation alignment, staging, and checkpoints unless explicitly stated.

### Better solutions retained as experiments

- **LOMO** is a strong low-gradient-memory experiment for plain SGD-like updates, but its immediate update order, clipping, accumulation, and recomputation semantics require an exact CPU oracle.
- **ZeRO-Offload-style CPU placement** may free VRAM, but it must report bytes transferred per step, pinned staging, overlap, writeback, and end-to-end slowdown; it does not solve saved activations.
- **GaLore, Muon, Sophia, SM3, DoRA, and rsLoRA** remain research candidates with documented algorithms and validation gates, not automatically superior defaults on gfx803.

## Common acceptance gates

Every alternative must pass CPU/GPU gradient parity, deterministic checkpoint/resume, explicit memory-plan rejection, bounded dispatch timing, and a real-RX580 benchmark. Software Vulkan correctness is necessary but does not establish hardware performance. Metrics must include loss trajectory, validation loss, tokens/second, peak VRAM, host transfer volume, checkpoint size/time, and failure/recovery behavior.

## Reports

- [`RX580_ALTERNATIVE_1_SGD.md`](RX580_ALTERNATIVE_1_SGD.md)
- [`RX580_ALTERNATIVE_2_ADAFACTOR.md`](RX580_ALTERNATIVE_2_ADAFACTOR.md)
- [`RX580_ALTERNATIVE_3_COMPRESSED_ADAM.md`](RX580_ALTERNATIVE_3_COMPRESSED_ADAM.md)
- [`RX580_ALTERNATIVE_4_LORA_ADALORA.md`](RX580_ALTERNATIVE_4_LORA_ADALORA.md)

## Decision sequence

1. Finish the GUI contract against the current Tiny/LoRA capability metadata.
2. Generalize the optimizer interface without changing the CPU authority or unsupported-operation semantics.
3. Implement and benchmark plain SGD on a configurable Tiny-Plus profile.
4. Implement exact CPU oracles for one factored optimizer (SM3 or Adafactor) and LOMO-style fused backward/update as separate experiments.
5. Benchmark explicit CPU offload, GaLore, and Muon only after each has a peak-live memory model and bounded-kernel plan.
6. Decide whether compressed Adam solves a demonstrated, still-open memory problem.
7. Attempt 100–150M full-base experiments only after real RX580 evidence for activations, workspaces, dispatch duration, and transfers; keep 0.5B to validated quantized inference and fixed-rank LoRA until broader paths exist.

These reports are feasibility reviews, not claims that any deferred alternative is implemented.

## References

- Shazeer, *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*, 2018: https://arxiv.org/abs/1804.04235
- Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, 2022: https://arxiv.org/abs/2110.02861
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021: https://arxiv.org/abs/2106.09685
- Zhang et al., *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, 2023: https://arxiv.org/abs/2303.10512
- Chen et al., *Symbolic Discovery of Optimization Algorithms* (Lion), 2023: https://arxiv.org/abs/2302.06675
- Keller Jordan, *Muon: An optimizer for hidden layers in neural networks* (reference implementation; no paper is cited here): https://github.com/KellerJordan/Muon
- Lv et al., *Full Parameter Fine-tuning for Large Language Models with Limited Resources* (LOMO), 2023: https://arxiv.org/abs/2306.09782
- Ren et al., *ZeRO-Offload: Democratizing Billion-Scale Model Training*, 2021: https://arxiv.org/abs/2101.06840
- Zhao et al., *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection*, 2024: https://arxiv.org/abs/2403.03507
- Liu et al., *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training*, 2023: https://arxiv.org/abs/2305.14342
- Anil et al., *Memory-Efficient Adaptive Optimization* (SM3), 2019: https://arxiv.org/abs/1901.11150
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, 2024: https://arxiv.org/abs/2402.09353
- Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (rsLoRA), 2023: https://arxiv.org/abs/2312.03732
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023: https://arxiv.org/abs/2305.14314

The external papers are design references; their reported results do not transfer automatically to gfx803 or this runtime.
