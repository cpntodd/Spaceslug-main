# FFN checkpoint layout

The schema-v4 FFN payload is a contiguous float sequence with four parameter groups, followed by matching AdamW `m` and `v` groups:

```text
parameters = W1[H,4H], b1[4H], W2[4H,H], b2[H]
moments_m  = W1_m[H,4H], b1_m[4H], W2_m[4H,H], b2_m[H]
moments_v  = W1_v[H,4H], b1_v[4H], W2_v[4H,H], b2_v[H]
```

Each group is row-major and FP32. The state element count is `3 * (H*4H + 4H + 4H*H + H)`. `tests/test_ffn_state_pack.cpp` reconstructs the complete flattened sequence from every group and checks exact equality with the source sequence. `tests/test_ffn_state_roundtrip.cpp` validates nonzero Vulkan graph transfer and optimizer step `17` with `max=0`.

Native schema-v4 checkpoint emission and restore now populate and validate all FFN groups. The C ABI reports the FFN parameter-plus-moments count and exposes the packed parameter pointer for host serialization. Host schema-v4 restore forwards the complete packed state to the native transfer ABI.

Remaining full-training gates are optimizer continuation across grouped graph dispatches, integration into the primary loss/backward orchestration, and end-to-end loss-decrease evidence.
