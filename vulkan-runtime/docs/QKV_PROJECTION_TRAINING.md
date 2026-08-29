# Standalone Q/K/V projection training

`spaceslug_qkv_projection_training_*` provides a standalone FP32 parameter-gradient
and synchronous SGD or persistent AdamW step for three independent row-major `[H,H]` matrices. AdamW maintains zero-initialized first/second moments, bias correction, decoupled weight decay, and a persisted step counter; all state can be checkpointed and restored. The
caller supplies transformer states `X` and `dquery`, `dkey`, and `dvalue`, each
`[tcap,H]`. For each projection `P`, the deterministic gradient is
`dW_P[h,o] = sum_t X[t,h] * dP[t,o]` in ascending `t`, followed by
`W_P -= learning_rate*dW_P`.

The combined Vulkan dispatch uses 256 invocations per workgroup, one owner per
`[projection,h,o]` element, FP32 arithmetic, and no atomics. Weights remain
device-resident between calls; `readback` returns all three updated weights and
last gradients. Matrices are explicitly row-major and are not head-major or
transposed.

This API is deliberately **standalone**. It is not integrated with the Tiny
command graph, attention backward pass, batching, or a full
transformer training loop. It accepts caller-supplied states and upstream
projection gradients so those systems can be composed externally. Separately, `ForwardResourceGraph` owns a bounded QKV FP32 SGD group driven by its actual Tiny backward `dquery`, `dkey`, and `dvalue` states. That graph path is normal-submit only, has no retained or dataset integration. QKV AdamW is available through `train_qkv_adamw_from_gradients`: call it only after a prior graph backward/gradient accumulation, and it consumes the existing QKV gradient buffers directly. It does not call `train_qkv_sgd`, recompute the graph, or mutate weights twice. Persistent QKV `m`/`v` state, step, weights, and CPU-visible checkpoint restore/readback are exposed through the matching C++ and C ABI methods.

The `qkv_projection_training` CTest performs CPU parity for all gradients and
SGD updates plus centered finite-difference checks. Run it on both default RADV
and lavapipe using the commands in `docs/TESTING.md`.
