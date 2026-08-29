# Standalone FP32 output-projection training

This milestone adds the next safe base-training subset: a standalone row-major
FP32 output projection `[H,H]`. The caller supplies projected activations
`[tcap,H]` and upstream gradients `dY [tcap,H]`; the runtime computes
`dW[h,o] = sum_t activation[t,h] * dY[t,o]` in a deterministic ascending
`t` loop, then applies in-place SGD `W -= learning_rate*dW` or persistent AdamW with bias correction and decoupled weight decay. AdamW state is initialized to zero and can be synchronously read back/restored with the checkpoint APIs.

The `spaceslug_output_projection_training_*` C ABI keeps weights device-local
between steps and exposes synchronous `step`, `readback`, and `update` calls.
It is deliberately analogous to the standalone `tiny_base_training` LM-head
API, but has no vocabulary padding and no dependency on Tiny graph state.

This is **not** integration into the Tiny graph, and it is not full base
training: no transformer-layer backward pass, batching, or retained training graph is provided; AdamW state is limited to this standalone projection session.

## Verification

The `output_projection_training` test compares GPU gradient and SGD results to
a CPU FP32 reference and checks every gradient element with a centered finite
difference. Verify on both drivers:

```sh
ctest --test-dir build/debug -R output_projection_training --output-on-failure
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build/debug -R output_projection_training --output-on-failure
```

Both shaders are compiled with `glslc -O` and `spirv-val`; the 256-thread
workgroups are wave64-compatible on gfx803/RADV.
