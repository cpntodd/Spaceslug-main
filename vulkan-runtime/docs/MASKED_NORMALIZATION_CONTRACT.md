# Masked normalization contract

The bounded normalization primitives now have explicit row-mask variants:

- `rmsnorm_backward_masked.comp` writes zero `dx` for masked rows and excludes them from row-statistic work.
- `rmsnorm_dgamma_masked.comp` excludes masked rows from deterministic `dgamma` accumulation.

Mask is a `uint` storage buffer with one element per row. A nonzero value includes the row. These variants use FP32 arithmetic, 256-thread workgroups, and no floating-point atomics. They are compile/validation milestones only until a dedicated masked CPU-double parity executable covers mixed masks, all-masked, repeated inputs, and boundary hidden sizes on lavapipe and RADV.

The unmasked verified ABI is unchanged. Graph integration, gamma optimizer state, and schema-v4 remain gated on these masked tests plus checkpoint/C ABI evidence.
