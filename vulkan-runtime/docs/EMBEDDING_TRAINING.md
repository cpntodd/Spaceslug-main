# Standalone FP32 embedding training

`embedding_training_api.h` exposes a deliberately standalone Vulkan compute API for
FP32 row-major embedding weights `[V,H]`. The production layout is `[259,64]`;
smaller dimensions are accepted for focused tests. It is not integrated with the Tiny graph or Cactus execution graph. The graph
embedding capability remains unsupported (`-5`); this API is deliberately a
separate caller-owned path and cannot satisfy the graph-owned dstate prerequisite.

The caller provides up to 128 token ids, FP32 `dstate` rows `[rows,H]`, and an
optional byte mask. Invalid token ids are ignored. For every `(v,h)`, exactly one
shader invocation scans rows in ascending order and accumulates matching masked
rows. This avoids floating-point atomics and makes the sparse reduction
reproducible. A second dispatch applies `weight -= learning_rate * gradient`.

The API is synchronous: `step` completes both dispatches before returning;
`readback` returns the updated weights and last gradient, and `update` replaces
weights without changing the gradient.

## Verification

Build and run correctness on RADV:

```sh
cmake --preset debug && cmake --build build/debug
ctest --test-dir build/debug --output-on-failure -R embedding_training
```

Run the same test on lavapipe:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build/debug --output-on-failure -R embedding_training
```

The test covers repeated token ids, masks, invalid ids, CPU parity, and a finite
difference check. The shader is compiled with `glslc -O` and validated by the
existing shader toolchain.
