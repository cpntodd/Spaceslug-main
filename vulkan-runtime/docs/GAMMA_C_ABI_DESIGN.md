# Trainable gamma C ABI design

The C ABI declares and implements fail-closed gamma state accessors:

- `spaceslug_tiny_forward_readback_gamma_state`
- `spaceslug_tiny_forward_update_gamma_state`

It also exposes an explicit FFN capability query and FFN state accessors:

- `spaceslug_tiny_forward_ffn_capability`
- `spaceslug_tiny_forward_readback_ffn_state`
- `spaceslug_tiny_forward_update_ffn_state`

These interfaces return the explicit unsupported code for non-null graph calls until gamma/FFN readback, restore, and optimizer/checkpoint parity are complete. Null graph handles return `-1`. This prevents schema-v4 callers from treating initialized-but-unverified graph buffers as checkpoint state.

The eventual contract is fixed Tiny profile arrays plus a shared one-based `uint64_t` optimizer step, with strict null/size/profile validation and no host fallback.
