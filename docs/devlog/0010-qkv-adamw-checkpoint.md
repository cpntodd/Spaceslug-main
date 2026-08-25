# QKV AdamW checkpoint binding

Graph-owned base checkpoints now serialize and restore the combined QKV AdamW first and second moments (`qkv_m` and `qkv_v`) alongside QKV weights and the shared AdamW step. The binding uses the runtime's existing-gradient QKV AdamW ABI and does not recompute gradients, invoke QKV SGD, upload datasets, or retain training windows.

The checkpoint remains a graph-owned parameter/state snapshot: dataset and retained-training boundaries are explicitly false. Runtime capability remains conditional on the complete QKV train/readback/update symbol set.
