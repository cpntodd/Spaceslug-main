# FFN graph state design

The graph owns allocated bounded FFN state for the fixed profile (`H=64`, `intermediate=256`):

- `W1[H,4H]`, `b1[4H]`, `W2[4H,H]`, `b2[H]`;
- matching gradient buffers;
- matching AdamW first and second moments;
- `ffnOutput_[Tcap,H]` and `ffnDx_[Tcap,H]` activation buffers.

Bounded graph primitives include `run_ffn_forward(rows)`, `run_ffn_backward(rows)`, `run_ffn_parameter_gradients(rows)`, `run_ffn_w1_adamw(...)`, `run_ffn_adamw_all(...)`, and `train_ffn_forward_loss(...)`.

These are bounded primitives, not a complete full-base training contract. FFN state readback and restore now expose the fixed packed layout through fail-closed C++/C ABI entry points, with device-transfer validation, shared optimizer-step handling, finite-payload rejection, and schema-v4 checkpoint round-trip/continuation tests. FFN training remains staged/diagnostic; active loss-chain insertion, public FFN training, and the full-base constructor remain disabled.

Dataset-resident, retained, and arbitrary-shape full-base paths remain unsupported boundaries.
