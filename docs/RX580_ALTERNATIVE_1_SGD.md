# Alternative 1 — FP32 SGD, momentum, and low-gradient-memory execution

> **Correction note (2026-08-25):** this revision separates LOMO from CPU offload, removes an invalid Muon paper citation, makes the memory figures lower bounds, adds the update-order/recomputation requirements that layerwise execution previously omitted, and flags Muon's `M@M^T` transient workspace spike.

**Recommendation:** **Implement plain FP32 SGD as the reference optimizer and first configurable full-parameter experiment. Evaluate LOMO-style fused backward/update next. Treat Nesterov, Muon, and CPU offload as separate measured extensions, not assumed improvements.**

## Proposal

Start with decoupled-weight-decay SGD, making its semantics explicit:

```text
g <- accumulated_gradient / accumulation_count
w <- (1 - learning_rate * weight_decay) * w
w <- w - learning_rate * g
```

Coupled L2 regularization (`g <- g + lambda*w`) is a different algorithm and must have a different configuration/checkpoint identity. For classical or Nesterov momentum, define and test the exact recurrence; “Nesterov” is not just a flag on the plain update. One common form is:

```text
v_t <- momentum * v_(t-1) + g_t
w_t <- w_(t-1) - lr * (g_t + momentum * v_t)
```

Use the exact CPU recurrence as the contract because libraries implement equivalent-looking Nesterov forms with different buffer initialization and dampening behavior.

## Current repository fit

The repository already has bounded SGD paths: standalone LM-head, output projection, and combined QKV subsets; graph-owned fixed-window SGD for LM-head/output/QKV/embedding; and GPU SGD for Tiny LoRA adapters. This makes SGD the shortest route to a common optimizer ABI.

This evidence is narrow. It does **not** establish general full-model SGD: the current graph accepts rows `1..128`, supports exactly two Tiny GPU profiles (`H=64`, `V=259`, `Vp=320`, `T=128`, rank 4/8), and does not train positions, FFN, or normalization. Dataset-integrated all-parameter training and retained backward/update are unsupported.

## Memory accounting

These are persistent/lower-bound FP32 bytes per trainable parameter:

| Mode | Included storage | Lower bound |
|---|---|---:|
| Plain SGD | weights + one resident full gradient | `8P` bytes |
| Momentum/Nesterov | weights + gradient + velocity | `12P` bytes |
| AdamW comparison | weights + gradient + first + second moments | `16P` bytes |

The table excludes gradient-accumulation buffers, saved activations, recomputation scratch, reduction partials, allocator alignment, Vulkan staging, checkpoint snapshots, and overlapping command-ring slots. At 500M parameters, `8P` is approximately 4 GB decimal (3.73 GiB), but this does not make the model trainable in 8 GB because activations and workspaces remain.

## LOMO: the stronger low-memory SGD execution candidate

LOMO is not CPU offload. It fuses backward gradient production with immediate parameter update, reducing gradient storage from all parameters to approximately the largest current parameter tensor and avoiding adaptive optimizer state. This is attractive on 8 GB VRAM, but it changes execution semantics:

- parameters must not be updated before an earlier backward operation still needs their old values;
- accumulation across microbatches needs either retained per-parameter gradients, delayed updates, or an exact recomputation/two-pass schedule;
- global-norm clipping may require a first backward pass to compute the norm and a second pass to update, or a documented approximation;
- checkpoint interruption must define whether partially updated backward passes are recoverable.

A concrete reverse-layer schedule and CPU oracle are therefore mandatory. “Layerwise update” without this dependency schedule is incorrect.

## CPU offload is a separate placement option

ZeRO-Offload-style placement keeps selected optimizer state and update computation in host memory. It may reduce VRAM, but it requires per-step transfers and does not solve activations. For a 100M-parameter FP32 model, one full state buffer is 400 MB decimal; reading and writing it is at least 800 MB/step before gradient upload, weight writeback, metadata, or synchronization. Adam-like two-state traffic is larger. Any proposal must report:

- exactly which weights, gradients, and states are host- versus device-resident;
- bytes transferred per optimizer step;
- pinned/host-visible staging and overlap with bounded GPU work;
- weight writeback timing and stale-weight prevention;
- end-to-end tokens/second rather than kernel-only speed.

## Muon research note

Muon applies momentum followed by iterative orthogonalization to suitable 2-D hidden-layer updates. It has one persistent momentum buffer, but **not** the same total peak as momentum SGD: Newton–Schulz iterations form products such as `M @ M^T` (or `M^T @ M`), so a naive per-matrix workspace can transiently allocate a full `[R,C]` FP32 buffer and spike VRAM on 8 GB. These iterations also need shape/transpose-specific operations, barriers, and multiple dispatches. Embeddings, scalars, vectors, and usually output heads need another optimizer.

The current runtime GEMM tests do not prove a Muon implementation. A gfx803 path would need exact scaling, Newton–Schulz coefficients/iteration count, weight-decay semantics, CPU parity, bounded dispatches, and peak-live workspace measurements. Muon remains an experiment after SGD/LOMO, not the default.

## Engineering design

1. Add optimizer metadata for update rule, coupled/decoupled decay, state tensors, dtype, step, schedule, and checkpoint version.
2. Implement CPU references for SGD, momentum, and Nesterov before GPU kernels.
3. Reduce accumulated gradients through staged deterministic partial buffers; do not assume FP32 atomics.
4. Bound every reduction and update dispatch for the amdgpu watchdog.
5. Add activation-checkpoint/recompute ownership to the memory plan; optimizer savings alone are insufficient.
6. Implement LOMO as a separately capability-gated execution mode with an exact reverse-layer schedule.
7. Add CPU offload only after transfer-volume simulation and explicit placement metadata.

## Main risks

- Plain SGD may require more optimization steps and schedule tuning than adaptive optimizers.
- Momentum adds one full state buffer.
- Immediate layer updates can corrupt backpropagation if old weights are still needed.
- Global clipping and gradient accumulation conflict with naive LOMO fusion.
- CPU offload may become PCIe- or synchronization-bound.
- Muon can add more workspace and matrix passes than its persistent-state count suggests.
- Tiny loss reduction does not establish larger-model quality or RX580 long-run stability.

## Validation gates

- CPU/GPU parity for SGD, coupled L2, decoupled decay, momentum, and the selected Nesterov recurrence.
- Accumulated-gradient parity across microbatch counts.
- LOMO parity against ordinary delayed-update SGD on a graph where update order is provably safe.
- Peak-live bytes, not only persistent bytes.
- Timestamped, watchdog-safe dispatches on real RX580/RADV.
- For offload: measured bytes/step, overlap, synchronization, and tokens/second.
- Deterministic checkpoint/resume, including interruption boundaries.
- Explicit memory-plan rejection before allocation.

## Recommendation

**Approve plain SGD first, then prototype LOMO-style fused backward/update.** These reuse current concepts with the fewest new numerical assumptions. Add Nesterov only if it improves convergence per wall-clock time. Evaluate Muon and explicit CPU offload only after their workspace and transfer costs are modeled. No result should be extrapolated to 100–150M or 0.5B without real RX580 activation, workspace, and throughput evidence.

## References

- Repository boundaries: [`TRAINING.md`](TRAINING.md) and [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md).
- Lv et al., *LOMO*, 2023: https://arxiv.org/abs/2306.09782
- Ren et al., *ZeRO-Offload*, 2021: https://arxiv.org/abs/2101.06840
- Keller Jordan, Muon reference implementation (no arXiv paper claimed): https://github.com/KellerJordan/Muon

This report proposes future capabilities; only the bounded repository contracts above are currently implemented.
