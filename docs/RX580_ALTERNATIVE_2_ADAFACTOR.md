# Alternative 2 — Factored Adafactor, SM3, and projected-gradient research

> **Correction note (2026-08-25):** this revision corrects the Adafactor algorithm, removes the false “factorized versus full variant” split, corrects SM3's tensor-cover rule, restates Adafactor's sum-based reconstruction, and recommends randomized/power-iteration basis construction for GaLore instead of dense SVD.

**Recommendation:** **Implement one exact factored optimizer CPU oracle after SGD. SM3 is the simpler first oracle; Adafactor is the stronger established matrix candidate. Do not select GaLore or promise 100–150M until its basis construction, workspace, and bounded kernels are measured.**

## Adafactor proposal

For a matrix gradient `G_t` with squared entries, Adafactor maintains exponentially decayed row and column statistics rather than a full second-moment matrix. Conceptually:

```text
R_t <- b2d * R_(t-1) + (1 - b2d) * row_sums(G_t^2 + epsilon1)
C_t <- b2d * C_(t-1) + (1 - b2d) * col_sums(G_t^2 + epsilon1)
Vhat_t <- outer(R_t, C_t) / sum(R_t)
U_t <- G_t / sqrt(Vhat_t)
Uhat_t <- U_t / max(1, RMS(U_t) / clip_threshold)
alpha_t <- max(epsilon2, RMS(w_(t-1))) * relative_step
w_t <- w_(t-1) - alpha_t * Uhat_t
```

The sketch follows the paper's row/column sum convention and its `outer(R,C)/sum(R)` reconstruction. Epsilon placement, decay schedule, bias handling, and regularization constants must still be frozen exactly in the CPU oracle; the sketch is explanatory, not an ABI. The normalized outer-product reconstruction is essential. Simply storing `R+C` values does not define Adafactor.

“Factored” versus “unfactored” describes second-moment storage for different tensor shapes. Momentum/first moment (`beta1`), parameter-relative step size, warmup, weight decay, and update clipping are configurable choices, not a separate “full Adafactor” algorithm. The memory-minimal paper configuration drops first-moment momentum and uses update clipping.

## Current repository fit

The repository already separates optimizer state from model ownership and requires explicit memory planning. Its current GPU optimizer capability remains bounded to selected Tiny parameter groups. There is no current Adafactor, SM3, GaLore, FFN/norm full-training, arbitrary-shape, or complete dataset-training implementation.

## Memory assessment

For a dense `[R,C]` matrix in FP32:

- full second moment: `4RC` bytes;
- factored row and column statistics: approximately `4(R+C)` bytes;
- optional full first-moment momentum: an additional `4RC` bytes;
- full resident gradient: `4RC` bytes unless LOMO-like fusion or another schedule avoids it.

The factored-state figure excludes staged reduction partials, synchronization buffers, update scratch, accumulation buffers, allocator alignment, checkpoints, and activations. A correct implementation must reconstruct each element on the fly; materializing `Vhat` as a full matrix would silently erase part of the memory benefit.

Factoring benefits large two-dimensional matrices. Vectors, small/skinny matrices, and tensors whose factored approximation behaves poorly need explicit unfactored or alternative policies. The choice must be based on total live bytes and validation behavior, not merely `R+C < RC`.

## Vulkan execution design

1. Reduce squared gradients into deterministic staged row/column partials; do not use a naive per-element global atomic reduction.
2. Synchronize reductions before updating running statistics.
3. Fuse normalized outer-product reconstruction, preconditioning, update-RMS reduction, clipping, decay, and weight update where this lowers peak live memory without creating watchdog-length dispatches.
4. Chunk large matrices and include all scratch in the memory plan.
5. Version per-tensor state as factored/unfactored, shape, epsilon/decay schedule, clipping, momentum option, and step.
6. Reject checkpoint resumes when any optimizer-state rule changes.

## SM3 as a simpler factored-state oracle

SM3 maintains accumulator statistics over a chosen cover of tensor dimensions. Each element's preconditioner is derived from the elementwise minimum of the covering accumulators, and the cover statistics are updated with the paper's decay/max rules. Describing SM3 as merely “row/column running maxima” is incomplete; the tensor cover, elementwise minimum, decay, and update order are part of the algorithm.

SM3 may be easier to implement than Adafactor because it avoids Adafactor's normalized rank-one reconstruction, but convergence on this model family is unproven. It is a useful oracle milestone, not an assumed winner.

## GaLore as a separate full-training candidate

GaLore projects full gradients into a low-rank subspace and stores optimizer state in projected coordinates. For an `[m,n]` matrix and rank `r`, projected state can scale approximately with `r(m+n)` rather than `mn`, depending on orientation and optimizer.

The periodic basis refresh is not a plain GEMM. A dense full SVD does not map onto the existing tiled GEMM kernels and would need a separate, expensive implementation. Prefer randomized subspace iteration or power-iteration variants, which are expressed as FP32 matrix products and can reuse the GEMM path. The refresh still requires persistent basis metadata, projection/reconstruction, temporary workspaces, synchronization, and checkpoint semantics. A naive full-matrix basis workspace can exceed the saved optimizer state on 8 GB hardware. Before GPU work, the project needs:

- an exact CPU oracle and basis-update cadence;
- a bounded basis-construction algorithm appropriate for gfx803;
- peak-live workspace accounting;
- parity across basis refresh and resume boundaries;
- direct convergence comparison with Adafactor and SGD.

## Main risks

- Factored statistics approximate correlations and may not behave like AdamW on all tensors.
- Incorrect normalization, epsilon placement, clipping target, or update order creates a different optimizer.
- Row/column reductions add bandwidth, barriers, and scratch.
- Optional first-moment momentum removes much of the memory advantage.
- Deterministic reductions may be slower but are needed for auditable parity.
- GaLore basis refresh can dominate time and memory.
- Optimizer savings do not solve saved activations or full backward coverage.

## Validation gates

- CPU oracle tests against published/reference equations for factored and unfactored tensors.
- Edge cases: zero gradients, very small RMS, skinny matrices, vectors, sparse embeddings, and non-finite inputs.
- CPU/GPU parity for statistics, reconstructed preconditioner, clipped update, weight decay, and checkpoint state.
- Peak-live VRAM including reduction/SVD scratch.
- Timestamped bounded dispatches on real RX580/RADV.
- Loss and validation trajectories against SGD and the existing CPU AdamW reference.
- For GaLore: parity before/after basis refresh and interruption/resume.
- A 5–20M quality and stability gate before any 100–150M experiment.

## Recommendation

**Approve CPU research for SM3 and Adafactor after plain SGD.** Choose one for GPU implementation only after convergence and peak-live memory measurements. Adafactor is the leading resident factored-state candidate; SM3 is the simpler verification milestone. Keep GaLore as a separate, later experiment. None of these methods alone supports a 0.5B local-pretraining claim.

## References

- Repository memory contract: [`TRAINING.md`](TRAINING.md).
- Shazeer & Stern, *Adafactor*, 2018: https://arxiv.org/abs/1804.04235
- Anil et al., *Memory-Efficient Adaptive Optimization* (SM3), 2019: https://arxiv.org/abs/1901.11150
- Zhao et al., *GaLore*, 2024: https://arxiv.org/abs/2403.03507

These papers define candidate algorithms; they do not establish Vulkan/gfx803 performance.
