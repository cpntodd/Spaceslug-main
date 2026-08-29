# Trainable normalization contract

The bounded training block reserves one FP32 `gamma[H]` vector and two FP32 AdamW moment vectors `gamma_m[H]` and `gamma_v[H]`. RMSNorm uses epsilon `1e-5`; `dgamma[i]` is the deterministic sum of `x[row,i] * inv_rms[row] * dy[row,i]` over included rows.

`rmsnorm_gamma_adamw.comp` uses the graph-wide one-based AdamW step for bias correction, decoupled weight decay, and the existing positional AdamW hyperparameter semantics. A CPU two-step nonzero-prior-moment reference sanity test is registered as `rmsnorm_gamma_adamw`; this does not yet prove GPU parity.

Schema-v4 integration remains gated until graph-owned allocation, GPU optimizer execution, two-step CPU/GPU parity, checkpoint round-trip/resume, and direct C ABI tests exist. Masked rows are excluded from normalization gradient accumulation and downstream parameter updates.
