# FFN parameter-gradient contract

`ffn_param_grads.comp` and `ffn_param_grads_masked.comp` compute deterministic gradients for the bounded residual FFN contract (`H -> 4H -> H`) without floating-point atomics. Each invocation owns one parameter and scans all rows: `dW1`, `db1`, `dW2`, and `db2`. The masked variant excludes rows whose `uint` mask value is zero. Both use the tanh GELU approximation and FP32 arithmetic.

`tests/test_ffn_param_grads_cpu.cpp` provides a mixed-mask CPU reference sanity gate. GPU parameter-gradient parity and finite-difference tests are still required; this CPU test does not by itself validate the shader. Production integration also requires an activation/recomputation policy, optimizer two-step parity, checkpoint state, and C ABI tests. Until those gates pass, FFN remains compile-validated only.
