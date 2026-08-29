# Immutable command reuse boundary

## What is production-safe now

`ForwardResourceGraph::forward_fixed_retained()` and `ForwardResourceGraph::forward_loss_fixed_retained()` are fixed-shape Tiny paths. The former retains forward-only work; the latter retains a complete forward+masked-loss Vulkan primary command buffer containing:

1. copy of all `Tcap` token, target, and mask slots from mutable host staging into device buffers,
2. transfer-to-compute visibility barrier,
3. the full Tiny forward dispatch (`5 x Tcap`),
4. the fixed masked causal-loss dispatch (`Tcap x 1`),
5. compute-to-transfer visibility barrier, and
6. readback of all `Tcap * Vp` logits plus `Tcap` row losses.

The descriptor sets, pipelines, push constants, dispatch dimensions, barriers, and readback copies are all recorded once. Each call changes only host staging bytes; the retained command copies changed tokens, targets, and masks into device buffers before executing. The API intentionally requires exactly `Tcap` rows and returns every row; no variable sequence length, `final_only`, backward, optimizer, or adapter update mode is implied.

The capability string is `production_fixed_shape_forward_loss_retained_command_buffer_resubmit`. This is the production retained-command capability for fixed forward+loss; all variable-shape forward and backward/optimizer/LoRA operations remain bounded normal-submit paths.

## Exact remaining blocker

The general `forward()` and all training/backward/LoRA methods are **not** retained. They use the normal `ExecEngine::submit()` path because their command recordings contain runtime-dependent push constants, dispatch/readback sizes, and operation-specific command chains. Training also changes targets, masks, upstream gradients, optimizer controls, and may update device-resident adapter/moment buffers. A truthful full-graph retained API therefore needs a second fixed-shape command graph for each operation/control mode (or device-side control buffers plus a fixed upper-bound schedule), and must prove synchronization and output semantics for each mode.

Do not advertise `command_buffer_capability` as full ForwardResourceGraph retention: it covers the fixed forward-only capability; the forward+loss capability is reported separately and neither one includes backward, optimizer, or LoRA updates. The diagnostic `ImmutableCommandPrototype` remains separate and is not the production graph.

## Verification

The `tiny_forward_persistent` test runs the retained method twice with changed tokens, compares the second result with the ordinary forward path, and checks that the command was resubmitted. Run both:

```sh
ctest --test-dir build/debug --output-on-failure
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ctest --test-dir build/debug --output-on-failure
```

The same test is intended for RADV (including RX580/gfx803) and lavapipe. It does not claim performance; each test call waits for completion to make correctness and lifetime boundaries explicit.
