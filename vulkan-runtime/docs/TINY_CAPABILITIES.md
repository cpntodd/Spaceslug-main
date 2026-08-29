# Tiny profile and graph capability contract

This page is the authoritative capability response for the two supported Tiny
profiles. It distinguishes the validation-clean bounded graph-owned optimizer
from the still-gated general full-base trainer.

The bounded optimizer capability is:

```text
tiny_fixed_profile_bounded_full_graph_adamw_true_rmsnorm_ffn_position_cabi_vvl_clean
```

It is available through `spaceslug_tiny_forward_train_bounded_full_graph_adamw()`
and is limited to `H=64`, `V=259`, `Vp=320`, `Tcap=128`, ordinary submissions,
trailing-mask windows, and graph-owned FP32 state. The machine-readable
contract constants are `bounded_full_graph_max_rows`,
`bounded_full_graph_requires_trailing_mask`, and
`bounded_full_graph_uses_retained_commands`. It is not retained training,
dataset-resident training, arbitrary-shape training, or a general full-base
constructor. It is the fixed-profile post-attention true-RMSNorm + FFN path;
general full-base training remains separately gated. The general full-base capability remains fail-closed and
contains `unsupported`/`gated` until those boundaries are implemented.

## Supported profiles

The metadata API (`spaceslug_tiny_profile_*`) reports exactly these profiles:

| Name | Hidden `H` | Vocabulary `V` | Padded vocabulary `Vp` | Token capacity | LoRA rank |
|---|---:|---:|---:|---:|---:|
| `tiny_h64_v259_vp320_t128_rank4` | 64 | 259 | 320 | 128 | 4 |
| `tiny_h64_v259_vp320_t128_rank8` | 64 | 259 | 320 | 128 | 8 |

Other shapes and ranks are **unsupported**. The profile query is metadata-only
and does not initialize Vulkan.

## Architecture groups

The graph computes token embedding plus position lookup, Q/K/V projections,
causal attention, output projection, and the padded LM head. Its graph-owned
base-training API has this exact status:

| Group | Status | Existing layout / boundary |
|---|---|---|
| Token embeddings | **applicable: bounded SGD** | Graph-owned sparse SGD uses graph-produced `dstate`, masked repeated-token reduction, and one ordinary submission. No AdamW or checkpoint group is defined. |
| Positions | **applicable: bounded SGD and bounded AdamW** | Graph-owned table `[Tcap,H]`, row-major FP32. Masked graph `dstate` drives one-submit SGD or the bounded combined gamma/FFN/position AdamW chain for `1 <= rows <= 128`; synchronized readback is available for parity. No dataset or retained-backward integration. Schema-v4 checkpoints include the table and optimizer state. |
| Q/K/V projections | **applicable** | Graph-owned FP32 matrices `[H,H]`, row-major (`weight[input*H + output]`). Fixed-window SGD is integrated; AdamW is integrated through `train_qkv_adamw_from_gradients`, which consumes prior graph-produced Q/K/V gradients and maintains persistent graph-owned `m`/`v`/step state. |
| Output projection | **applicable** | Graph-owned FP32 matrix `[H,H]`, row-major. Fixed-window SGD and AdamW are integrated. |
| LM head | **applicable** | Graph-owned FP32 matrix `[H,Vp]`, row-major (`lm_head[hidden*Vp + vocab]`); only columns `[0,V)` are trained, padded columns are protected. Fixed-window SGD and AdamW are integrated. |
| FFN | **applicable: bounded fixed-profile graph-owned AdamW** | Graph-owned FP32 W1/W2/biases, gradients, moments, and FFN buffers are used by the bounded combined optimizer. The explicit C ABI entry point is `spaceslug_tiny_forward_train_bounded_full_graph_adamw()`. This does not imply a general full-base constructor, arbitrary shapes, retained backward/optimizer, or dataset-resident training. |
| Normalization parameters | **applicable: bounded true RMSNorm** | Graph-owned gamma participates in fixed-profile post-attention RMSNorm, with preserved raw/inverse-RMS state, masked dgamma, and AdamW state. The general normalization constructor remains unsupported. |

The base capability string is returned by
`spaceslug_tiny_forward_base_train_capability()`. The separate bounded
capability is returned by
`spaceslug_tiny_forward_bounded_full_graph_training_capability()`. The group
query `spaceslug_tiny_forward_base_train_group_supported()` returns true for
`LM_HEAD`, `OUTPUT`, `QKV`, `EMBEDDINGS`, and `POSITIONS`; `NONE`, normalization,
FFN, and other unassigned IDs are false. Gamma and FFN state are represented by
schema-v4 checkpoint/state APIs and the bounded full-graph ABI, but are not yet
selectable general base-group IDs.

Graph embedding has an explicit status API:
`spaceslug_tiny_forward_graph_embedding_training_capability()` returns
`tiny_graph_embedding_dstate_gradient_sgd_graph_owned_tokens_fixed_window_rows_le_128_one_submit_cpu_parity`,
and `spaceslug_tiny_forward_graph_embedding_training_status()` returns **`0`**.
The graph computes its own masked `dstate`, reduces repeated token rows, and
updates graph-owned embeddings in the same normal submission. It does not
change or wrap the standalone `spaceslug_embedding_training_*` API, whose capability remains
`standalone_fp32_embedding_training_V259_H64_deterministic_sparse_sgd_no_tiny_graph_integration`
and which remains supported only for caller-supplied FP32 `dstate` rows.

The dataset API narrows the boundary further: `train_dataset_batch()` updates
only the graph-owned LM head, while `train_dataset_batch_full()` validates its
bounded batch contract and returns the explicit unsupported status. Retained
command-buffer APIs are forward/loss execution only; they do not retain
training or optimizer updates. The bounded graph-owned gamma/FFN/position
AdamW chain uses ordinary submissions and is exposed as
`spaceslug_tiny_forward_train_bounded_full_graph_adamw()`; it is not implied by
the dataset or retained APIs.

## Remaining integration boundaries

Graph-owned embedding SGD is intentionally limited to the fixed Tiny profile,
masked token windows, FP32 sparse SGD, and one ordinary submission. It has no
embedding AdamW state or dataset/full-graph integration. The bounded combined
chain additionally owns positional AdamW state plus gamma-scale and FFN
weights, gradients, moments, and AdamW updates; gamma remains elementwise
scaling rather than complete RMSNorm. Schema-v4 checkpoints include embedding,
position, gamma, and FFN payloads and optimizer state. Retained training,
dataset-resident full training, arbitrary-shape training, and general full-base
training remain unsupported. See
[FULL_GPU_TRAINING_ROADMAP.md](FULL_GPU_TRAINING_ROADMAP.md) for the staged path
covering optimizer completion, FFN/norm architecture, dataset execution,
retained execution, and arbitrary-shape generalization.

## Embedding sparse-training layout

The graph-owned embedding update uses the existing forward buffer layout:

- FP32, contiguous row-major `[V,H]` = `[259,64]` for both profiles;
- element `(token, hidden)` is `embedding[token * H + hidden]`;
- token rows are contiguous, each row is `H * sizeof(float)` bytes (`256` bytes);
- row `token` starts at byte offset `token * H * sizeof(float)`;
- only token IDs `[0,V)` are real rows. The LM-head padding `[V,Vp)` is not an
  embedding range and must not be treated as sparse rows;
- a sparse update should identify rows by token ID and apply a complete
  `H`-element row update. No optimizer-state layout or sparse-update ABI is
  currently defined.

Positions are a separate `[Tcap,H]` buffer and must not be folded into an
embedding sparse-update format. Adding sparse embedding training would require
new gradient/update/checkpoint capability and focused tests; this document
intentionally does not implement that architecture group.

## Verification

The focused metadata test is host-only and checks both profile descriptors, the
capability/group response, and the explicit graph embedding unsupported status:

```sh
ctest --test-dir build/debug -R '^tiny_profile_api$' --output-on-failure
```

The test must also pass through the lavapipe workflow. No Vulkan device is
needed by this particular test.
