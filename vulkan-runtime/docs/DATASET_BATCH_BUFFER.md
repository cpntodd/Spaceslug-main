# Bounded device-resident dataset training

`vulkan_runtime::dataset::BatchBuffer` owns fixed-shape device-local token,
target, mask, control, and result buffers. `ForwardResourceGraph::train_dataset_batch`
consumes the token/target/mask windows directly from those buffers and records one
normal submission containing the Tiny FP32 forward graph, causal loss, LM-head
gradient accumulation, and one LM-head SGD update. The host does not invoke a
training call per window; it waits once and reads no parameters or gradients.

The supported boundary is intentionally explicit:

- fixed `window_count` (bounded to 32) and `window_tokens <= 128`;
- input is uploaded once with `BatchBuffer::upload`, then consumed device-side;
- only the graph-owned FP32 LM head is updated, using `learning_rate / normalizer`;
- controls are retained dataset metadata and are not optimizer parameters;
- scalar per-window validation results remain available through
  `process_readback()`; training does not read back logits, gradients, or weights;
- the `BatchBuffer` must outlive the synchronous training call (its `DeviceView`
  is non-owning and exposes no allocation handles);
- output, QKV, embeddings, positions, FFN, and normalization groups are not
  claimed as dataset-trained by this milestone.

This is bounded graph consumption, not full-dataset training or all-parameter training: only the graph-owned LM head is updated. Callers must split
larger datasets into explicit bounded batches and must not infer that the runtime
retains command buffers or processes unbounded windows. `BatchBuffer::capability()`
provides the same boundary string programmatically. The dataset and Tiny tests
run on both RADV and lavapipe in the documented CTest workflows.

`BatchBuffer::process()` remains a standalone validation convenience: it uploads,
processes all retained windows on GPU, and reads back only the masked loss/control
pairs.
