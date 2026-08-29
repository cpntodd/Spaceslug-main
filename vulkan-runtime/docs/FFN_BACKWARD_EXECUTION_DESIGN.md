# Bounded FFN backward execution design

`shaders/ffn_gelu_backward.comp` is the first backward primitive for the contracted `H -> 4H -> H` GELU residual block. It computes the input gradient (`dx`) per output lane, retaining the residual gradient and applying the exact tanh-GELU derivative through `W2^T` and `W1^T`.

This is deliberately a compile/validation milestone only. Parameter gradients (`dW1`, `dW2`, `db1`, `db2`) require deterministic reduction passes rather than floating-point atomics. The production sequence will be:

1. forward writes pre-activation and intermediate activation buffers;
2. backward writes `dx` and per-row partial parameter gradients;
3. reduction kernels combine partials into graph-owned gradients;
4. isolated SGD and AdamW kernels update weights and moments;
5. only after CPU-double/finite-difference and GPU parity gates pass is the block inserted into full-base training.

The current shader uses bounded scalar loops for correctness and is not a performance claim. It uses FP32 math and 256-thread dispatch, with no cooperative matrices or unsupported atomics. Masked rows will be represented by an explicit mask binding before integration.
