# Schema-v4 payload design

Schema v4 extends the complete bounded base checkpoint with trainable normalization and FFN state. New group bits are `BaseCheckpointNormalization` and `BaseCheckpointFfn`.

The normalization payload is `gamma`, `gamma_m`, and `gamma_v`, each `[H]`. The FFN payload is `ffn_w1[H,4H]`, `ffn_b1[4H]`, `ffn_w2[4H,H]`, and `ffn_b2[H]`, with matching AdamW `m` and `v` arrays. The fixed-profile graph now owns these buffers and schema-v4 C++/C ABI readback/restore, finite-payload validation, shared-step handling, and nonzero-moment continuation tests. These groups remain diagnostic/staged for training; the public full-base constructor and active loss-chain integration are still gated.

A v4 checkpoint must require both new group bits and validate every exact product before upload. Partial v4 payloads must fail closed. Resume tests must include nonzero moments and a two-step continuation comparison.
