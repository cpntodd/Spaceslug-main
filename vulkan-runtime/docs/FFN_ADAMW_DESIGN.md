# FFN AdamW design

`shaders/ffn_adamw.comp` defines a deterministic flattened-parameter AdamW primitive with persistent `W`, gradient, `m`, and `v` buffers. It uses one-based optimizer steps, bias correction, decoupled weight decay, FP32 arithmetic, and 256-thread bounded dispatch. The optimized embedded SPIR-V contains floating-point operations.

The graph constructs separate descriptor-backed pipelines for all four FFN groups and `run_ffn_adamw_all(...)` dispatches those groups in one submission using the same incremented optimizer step. `run_ffn_w1_adamw(...)` remains available as a scoped primitive.

`tests/test_ffn_adamw.cpp` covers two CPU reference steps from nonzero prior moments. `tests/test_ffn_adamw_gpu.cpp` covers direct GPU parity for one step with nonzero moments. A genuine two-step GPU continuation test is still required; the graph must also expose readback/restore before schema-v4 can be enabled.
