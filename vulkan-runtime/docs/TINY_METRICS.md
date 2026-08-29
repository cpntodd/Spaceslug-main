# Tiny bounded scalar metrics

`ForwardResourceGraph::forward_loss_fixed_metrics` and the C ABI
`spaceslug_tiny_forward_loss_fixed_metrics` run the fixed `Tcap=128` Tiny
forward pass followed by causal loss and a GPU reduction. The device result
buffer is exactly two scalars: FP32 sum of included row losses and a uint32
included-row count. The host copies back only those 8 bytes; logits and row
losses are not read back by this API.

This is an evaluation/telemetry boundary. It does not update weights, expose a
dataset, or claim full-graph training integration. QKV AdamW is not part of
this path and must not be run concurrently with it.

The focused CPU-parity test is `tiny_metrics_api` and must pass with both
RADV and lavapipe:

```sh
ctest --test-dir build/debug -R '^tiny_metrics_api$' --output-on-failure
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build/debug -R '^tiny_metrics_api$' --output-on-failure
```
