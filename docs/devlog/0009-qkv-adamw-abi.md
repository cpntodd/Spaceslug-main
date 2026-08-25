# Devlog 0009 — QKV AdamW ABI binding

The optional graph-owned QKV AdamW ABI is now exposed by Spaceslug when the runtime exports the complete train, readback-state, and update-state symbol set. The trainer calls `train_qkv_adamw_from_gradients` only after the existing QKV backward/gradient path; it does not recompute gradients or invoke QKV SGD.

QKV AdamW state remains graph-owned (`m`, `v`, and step) and is checkpoint metadata. Dataset storage/training and retained-forward boundaries remain false; this is not full-base training or dataset integration. Older runtimes remain supported through capability gating and return `-4` when the optional train symbol is absent.
