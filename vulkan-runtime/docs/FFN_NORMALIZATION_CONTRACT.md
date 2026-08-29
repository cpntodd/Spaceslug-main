# Tiny FFN and normalization contract

This contract is the prerequisite for integrating the next trainable base groups. It is intentionally fixed-shape and does not widen the public capability until the listed tests pass.

## FFN

The first graph block is a pre-normalized residual MLP:

```text
x = norm(h)
h' = h + W2 * GELU(W1 * x + b1) + b2
```

- `H = 64`; intermediate width `I = 4H = 256`.
- `W1` is row-major `[H,I]`, `W2` is row-major `[I,H]`.
- `b1[I]` and `b2[H]` are FP32 graph-owned parameters.
- GELU uses the tanh approximation consistently in forward and backward.
- All math is FP32; storage is FP32 for the initial training path.
- The block is applied once per token after attention output and before the LM head.
- Masked rows contribute no loss or gradients and do not update FFN state.

Backward uses the exact local chain rule: residual receives `dout`, `d_norm = W2^T dout * gelu'(z)`, and `dW1/dW2/db1/db2` are reductions over included rows. Reductions are staged in bounded FP32 scratch; the initial implementation may use one workgroup per matrix tile, followed by a deterministic reduction pass.

## Normalization

Use pre-norm RMSNorm with trainable scale only:

```text
r = mean(x*x)
y = gamma * x / sqrt(r + epsilon)
```

- `gamma[H]` is graph-owned FP32; no beta in the first contract.
- `epsilon = 1e-5` is fixed and part of the checkpoint descriptor.
- RMSNorm is applied before the FFN and is included in the FFN residual block above.
- The backward formula is computed in FP32 with a two-pass reduction per token:
  `dx = gamma*inv_rms*dout - x*gamma*dot(x,dout)*inv_rms^3/H`.
- Masked rows produce zero `dx` and do not contribute to `dgamma`.

## Optimizer and checkpoint

FFN and gamma each get independent AdamW moments and use the graph's single global optimizer step. Weight decay is applied only to participating parameter lanes; inactive/padded lanes are never decayed. Checkpoint schema v4 adds FFN weights, biases, RMSNorm gamma, and all corresponding moments. A schema-v3 checkpoint remains readable only by the v3 graph path; it is not silently upgraded.

## Acceptance gates

Before capability exposure:

1. CPU double forward/backward reference and finite-difference checks for GELU, RMSNorm, both FFN matrices, and biases.
2. Masked, duplicate, one-row, and maximum-window tests.
3. AdamW two-step parity and schema-v4 continuation parity.
4. lavapipe and RADV/POLARIS10 parity with validation enabled.
5. C ABI validation and no-mutation failures.
6. End-to-end loss decrease on a tiny learnable fixture.

Until all gates pass, the public capability must continue to list `ffn-training` and `normalization-training` as unsupported.

## Sequencing

Implement RMSNorm forward/backward as a standalone graph stage first, then FFN forward, then FFN backward reductions, then separate SGD/AdamW updates. Only after those isolated gates pass should the stage be inserted into the bounded complete base submission.

Retained execution and dataset integration must consume this same contract rather than introducing a second architecture.

## Hardware constraints

Use 256-thread wave64 workgroups, no cooperative matrix operations, no FP16 arithmetic, and chunk matrix/reduction dispatches to stay below the RX580 compute watchdog threshold and the 32 KB LDS limit.

## Current status

The fixed-profile graph owns true-RMSNorm raw/inverse/state and backward buffers plus graph-owned FFN parameters, gradients, and AdamW moments. CPU, finite-difference, masked, partial/all-masked, composed backward, optimizer, schema-v4 checkpoint/C ABI, active-chain gradient, and bounded-orchestration evidence passes on RADV and lavapipe with validation enabled. This bounded post-attention RMSNorm/FFN path is exposed through the bounded full-graph ABI; general full-base construction remains gated.
