# Gamma graph integration plan

Graph-owned `gamma_`, `gammaGradient_`, `gammaM_`, and `gammaV_` allocations now exist and initialize to RMSNorm identity/zero state. The next implementation must mirror the existing positional AdamW setup path:

1. add `GammaAdamwPC` and register `rmsnorm_gamma_adamw.spv` with four descriptors;
2. bind gamma in the actual RMSNorm forward/backward graph rather than leaving the current hard-coded/frozen path;
3. add a graph method that consumes an already-produced `gammaGradient_` and performs one optimizer dispatch without recomputing gradients;
4. add device readback/update methods for gamma, moments, and shared step;
5. only then add gamma fields to schema-v4 emission/restore and C ABI.

The current fail-closed `train_gamma_adamw` entry point returns `-5`. It is intentionally not wired to a partial pipeline.
