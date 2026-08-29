# Normalization execution design

True RMSNorm gamma is now bound in the bounded fixed-profile `tiny_forward_logits` training graph. The graph-owned gamma buffer participates in post-attention normalization with preserved raw and inverse-RMS state; the gamma buffer is initialized to ones, so the baseline output remains stable until a checkpoint or update changes gamma.

The graph also constructs a masked `rmsnorm_dgamma` pipeline over projected rows, upstream projected gradients, the graph row mask, and graph-owned `gammaGradient_`. The bounded `run_rmsnorm_gamma_gradient(rows)` primitive validates `rows <= Tcap`, dispatches through `ExecEngine`, and waits for completion. The bounded FFN forward/loss orchestration chains a dependent gamma-gradient dispatch behind a compute barrier.

Gamma parameters and AdamW moments have graph readback/restore, schema-v4 checkpoint integration, constructed-graph round-trip coverage, and are chained from the bounded primary training gradient. General full-base construction remains gated despite the bounded fixed-profile chain being available.

No floating-point atomics or subgroup assumptions are used, so the path remains compatible with RADV/gfx803 and lavapipe. Workgroups are 256 threads and arithmetic is FP32. Standalone masked backward and `dgamma` tests compare against CPU references.
