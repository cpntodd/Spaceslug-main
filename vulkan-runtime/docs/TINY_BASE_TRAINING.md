# Tiny FP32 base-training ownership boundary

## Current optimizer status

The graph-owned Tiny path integrates fixed-window AdamW for the **LM-head**,
**output projection**, and QKV. QKV AdamW is an explicit second-stage method:
`train_qkv_adamw_from_gradients` consumes gradients already produced by a prior
QKV graph training accumulation, without calling SGD or recomputing the graph.
This status does not claim full-parameter or dataset training.

`ForwardResourceGraph` owns the FP32 base-training groups `LmHead`, `Output`, `QKV`, and
`Embeddings`, and `Positions`. All five graph-integrated groups support fixed-window SGD for exactly
`1 <= rows <= 128`; the LM-head and output projection additionally support
graph-owned AdamW with persistent `m`/`v`/step state. QKV AdamW has persistent graph-owned state. Embeddings use graph-owned sparse SGD only; positions use masked fixed-table SGD and AdamW with persistent state. The bounded graph-owned true-RMSNorm/FFN/position AdamW chain is exposed through `spaceslug_tiny_forward_train_bounded_full_graph_adamw()` for the fixed profiles only; it does not imply general full-base support. Dataset-resident full training and retained backward/optimizer training remain unsupported.

The unified graph-owned base checkpoint API is schema version 4. It captures profile
metadata, a complete LM-head/output/QKV/embeddings/positions/normalization/FFN group
mask, all graph-owned weight groups, and AdamW state for the LM-head/output/QKV/
positions/normalization/FFN groups. Embeddings remain sparse SGD-only. QKV contributes
`m`/`v` payloads for each Q/K/V matrix; positions contribute table-shaped `m`/`v`
payloads; normalization contributes gamma `m`/`v`; and FFN contributes packed W1,
B1, W2, B2 `m`/`v` payloads. All optimizer groups share the checkpoint step. Readback and update use the same unified methods. The runtime parity suite verifies this round-trip on RADV and lavapipe. `BaseCheckpoint` is the structured C++ snapshot;
the opaque `spaceslug_tiny_base_checkpoint` C ABI handle provides synchronized
readback/update for language-neutral callers. A checkpoint can be read back from
one graph and applied to a newly recreated graph with the same profile. The
LM-head is the graph's FP32 row-major `[H,Vp]` parameter (`H=64`, `Vp=320`).
`import_base_train_lm_head` and `readback_base_train_lm_head` synchronously
transfer the complete padded FP32 state.
Tiny forward continues to use this same owned buffer; no constructor or forward
behavior changed.

This is an ownership/integration boundary, not full base-training integration. Graph
embedding training is supported as a fixed-window sparse SGD step: the graph computes
its own forward/loss/backward `dstate`, aggregates masked repeated token rows, and
updates the graph-owned FP32 embedding matrix in the same normal submission. The
standalone `spaceslug_tiny_base_training_*` API remains deliberately separate: it
accepts caller-supplied projected activations and dlogits and performs normal-submit
SGD or AdamW. It is not connected to `ForwardResourceGraph`, Tiny activations,
`BatchBuffer`, or retained training command buffers. Callers must not infer that
standalone API state is the graph-owned LM head.

## Graph-owned fixed-window SGD

`train_lm_head_sgd`, `train_output_sgd`, `train_qkv_sgd`, and `train_embeddings_sgd` are supported for
`1 <= rows <= Tcap` (with `Tcap=128`). Each uses one normal Engine submission
containing graph forward, causal loss/dlogits, the relevant graph-owned gradient,
and in-place FP32 SGD, with explicit barriers and a host wait before returning.
The graph owns the parameters and gradients; no training command buffer is retained.
Invalid rows, tokens, targets, masks, and non-positive learning rates are rejected.
`train_lm_head_adamw` additionally updates graph-owned `m`/`v`/step state. AdamW is supported for output and through the explicit QKV gradient-consumer method. The bounded full-graph ABI additionally updates graph-owned positions, gamma scale, and all FFN groups in one ordinary fixed-profile submission. Its primary attention path accepts only trailing-valid masks; dataset-resident full training and retained backward/optimizer training remain unsupported.

The former integration boundary is now narrowed to AdamW only. The existing forward shader
writes `activations_` as the attention context, while the LM-head gradient requires
the post-output-projection activation consumed by `lm_head`. Wiring the existing
weight-gradient shaders to `activations_` would therefore train against the wrong
values. A safe next step must add and retain that projected activation in the Tiny
graph, then chain graph-owned forward/loss `dlogits` and the weight-gradient/update
shaders in one bounded normal submission. No caller-supplied standalone activation
or dlogits path is accepted as an integration substitute.

The legacy base capability string remains
`base_train_group_lm_head_output_qkv_embeddings_owned_fp32_fixed_window_sgd_rows_le_128_lm_head_adamw_output_adamw_qkv_adamw_from_gradients_embeddings_sparse_sgd_positions_table_sgd_no_ffn_norm_dataset_retained_no_standalone_bridge`.
`None` is not a supported group; `LmHead`, `Output`, `QKV`, `Embeddings`, and `Positions` are the graph-integrated
base selections. Gamma and FFN are available only through the separately named bounded full-graph ABI and schema-v4
state/checkpoint APIs; they are not general selectable groups. The standalone API's padded-column and fixed-`tcap`
semantics remain documented below.
See [TINY_CAPABILITIES.md](TINY_CAPABILITIES.md) for the complete profile and
layout contract.

## Standalone API contract

The standalone session copies its initial matrix to device-local storage. Each step
uploads caller inputs, computes the deterministic gradient, applies SGD or AdamW,
and waits for completion. `readback` returns weight and last gradient; checkpoint
APIs transfer AdamW state. It does not use `BatchBuffer`.

## Verification

Run the focused ownership test, standalone base-training test, and full suite with
both RADV and lavapipe:

```sh
ctest --test-dir build/debug --output-on-failure
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build/debug --output-on-failure
```

Shaders remain FP32 math with wave64-compatible workgroups and are compiled with
`glslc -O` through the existing embedded shader toolchain.
