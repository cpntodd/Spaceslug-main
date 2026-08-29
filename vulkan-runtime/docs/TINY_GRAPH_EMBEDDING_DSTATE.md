# Tiny graph embedding `dstate` prerequisite status

## Decision

The graph-owned embedding SGD integration now consumes this prerequisite within the same normal submission. The graph computes masked dstate, reduces repeated token rows, and updates the graph-owned embedding buffer without a host readback bridge. The public dstate readback API remains available as a separate readback-only diagnostic path.

This milestone adds the safe prerequisite for a future graph-owned embedding update:
a bounded fixed-window backward submission produces, synchronizes, and exposes the
complete per-token `dstate[Tcap,H]`; the integrated training submission can then
consume those rows for the graph-owned sparse embedding gradient/update. The
separate readback API does not update embeddings or any graph-owned weights.

The graph embedding capability is now:

```
tiny_graph_embedding_dstate_gradient_sgd_graph_owned_tokens_fixed_window_rows_le_128_one_submit_cpu_parity
```

and its graph-embedding status is **`0`** (supported). This describes bounded
FP32 sparse SGD only; no graph-owned embedding AdamW state or dataset integration
is defined. The standalone `spaceslug_embedding_training_*` API remains separate
and consumes caller-supplied dstate; it is not a graph integration bridge.

## What exists today

`ForwardResourceGraph::readback_graph_dstate(tokens, targets, masks, rows, output)`
is the fixed-window public path for graph-owned batch dstate readback. It validates
`1 <= rows <= Tcap`, token/target ranges, and binary masks, runs the actual forward,
causal-loss, LM-head backward, output-projection backward, attention Q/K/V backward,
and projection Q/K/V chain, then synchronously waits and copies `Tcap*H` FP32 rows
to `output`. Rows at or beyond `rows` are zeroed by the fixed forward/loss inputs.
No embedding, QKV, LoRA, or other optimizer dispatch is recorded.

The C ABI spelling is
`spaceslug_tiny_forward_readback_graph_dstate(graph, tokens, targets, masks, rows, dstates)`.
The existing `ForwardResourceGraph::token_step_training_backward` remains available
for its single-token/upstream-output contract. The relevant chain is:

1. `token_step_training` runs forward, causal loss, LM-head backward, and produces
   `dprojected`.
2. The backward submission runs attention backward in modes Q/K/V, then the three
   projection-backward dispatches. The Q projection writes `dstates`; K and V
   accumulate into the same buffer.
3. A compute-to-transfer barrier makes shader writes visible to the four small
   gradient readbacks and the `Tcap*H` `dstates_` copy.
4. The host waits on `lastSubmission_`, invalidates the host staging allocation,
   and copies `dstates_` to the caller's `dstates` array.

`dstates_` is device-local storage-buffer memory with `TRANSFER_SRC` usage. The
three projection descriptor sets all reference it, so the accumulation order and
barriers are explicit. This is a valid CPU-visible parity source for the
single-token backward API, but it is coupled to caller-supplied `doutput` and
returns one backward result at a time.

## Why `train_qkv_sgd` is not a safe source

`train_qkv_sgd` records forward → loss → LM-head backward → projection backward
for the QKV parameter-gradient path → QKV SGD. Its public contract returns only a
status and updates graph-owned Q/K/V weights. It does not expose a dstate output,
rows-used metadata, or an embedding-gradient buffer. Adding an embedding update
or silently making this method populate a new output would require deciding the
upstream state for every row, preserving the complete projection/attention chain,
and defining whether QKV SGD and embedding SGD observe the same pre-update
weights. It would also change the meaning and synchronization contract of an
existing optimizer API.

In particular, a future implementation must not reuse the standalone embedding
API's caller-supplied `dstate`, read `dstates_` without the compute-to-transfer
barrier and host wait, or place a sparse embedding update after only the QKV
weight-gradient dispatch. Those choices can produce a numerically plausible but
stale or incomplete embedding gradient.

## Boundary for the next milestone

Only after this synchronized result is available should a future sparse
embedding-gradient shader consume it. That future update must use the same
token/mask semantics as `embedding_training_grad.comp` (repeated IDs reduced in
deterministic row order; invalid IDs ignored), and its own update dispatch needs a
separate shader-write → shader-read barrier before modifying embedding weights.
This milestone deliberately does not add that shader, gradient buffer, optimizer
state, or update dispatch. The existing standalone `embedding_training_*` shader/API
is not a substitute: it consumes caller-provided dstate and therefore cannot prove
that the graph backward chain and embedding update observe one normal submission.
The exact failed integration blocker is the missing graph-owned combined submission
contract: the existing graph backward path does not yet append the embedding
gradient/update dispatch with graph-owned buffers and explicit shader-write to
shader-read synchronization. It is not shader compilation or the sparse
repeated-token CPU reference. Until that contract exists, graph embedding status
must remain `-5`; the standalone API cannot be substituted.

## Verification boundary

`test_tiny_profile_api` checks the capability metadata without initializing
Vulkan. `test_tiny_graph_dstate` runs the new path on Vulkan, checks finite and
repeatable output for multiple rows including repeated token IDs and masked rows,
checks zero padding beyond `rows`, and verifies forward behavior is unchanged.
The test is run on both RADV/default Vulkan and lavapipe.

See also [TINY_BASE_TRAINING.md](TINY_BASE_TRAINING.md) and
[EMBEDDING_TRAINING.md](EMBEDDING_TRAINING.md).
