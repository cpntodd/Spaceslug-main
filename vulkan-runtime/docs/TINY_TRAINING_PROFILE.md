# Tiny normal-training versus retained-path profile

## Scope and question

This artifact profiles the current normal integrated Tiny LM-head training submission against the existing retained forward/loss paths on the real RADV device. It does **not** implement retained training. The question is whether command recording in the normal path is large enough to justify adding a retained training command graph.

The benchmark uses the public `ForwardResourceGraph` APIs at the fixed supported profile (`H=64`, `V=259`, padded `Vp=320`, `Tcap=128`):

- `train_lm_head_sgd(..., rows=128, ...)`: normal path; staging upload, normal `ExecEngine::submit`, forward, causal loss, LM-head gradient, and SGD update, then wait.
- `forward_loss_fixed_metrics(...)`: existing retained forward + loss + GPU scalar reduction; only loss/count are copied back.
- `forward_loss_fixed_retained(...)`: existing retained forward + loss; full logits and row losses are copied back.

Timing is host wall time around each synchronous API call, so it includes staging map/flush, command recording, queue submission, GPU work, wait, and readback. It is intentionally non-invasive and is not a GPU timestamp-query breakdown. The benchmark warms up 128 iterations and reports the median of 9 samples, following the RX580 clock-ramp guidance.

## Device and commands

Device reported by `vulkaninfo --summary` and the benchmark:

```text
AMD Radeon RX 580 Series (RADV POLARIS10)
Mesa 25.0.7-2+deb13u1
```

Build and run:

```sh
cmake --preset debug
cmake --build build/debug --target bench_tiny_training -j2
./build/debug/bench_tiny_training
```

Correctness/build verification:

```sh
cmake --build build/debug -j2
ctest --test-dir build/debug --output-on-failure
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build/debug --output-on-failure
```

## Measured result

Captured from the command above on 2025-08-25:

```text
device: AMD Radeon RX 580 Series (RADV POLARIS10)
profile rows=128 warmup=128 samples=9 median_ms
normal_train_lm_head_sgd=64.466
retained_forward_loss_metrics=64.034
retained_forward_loss_logits=64.007
normal_over_retained_metrics=1.007
normal_over_retained_logits=1.007
metric_count=128 loss=711.274
```

The normal training call was only **1.007x** the retained comparisons, or about **0.43–0.46 ms (0.7%)** slower at this profile. Training necessarily performs additional gradient/update work, so this small total delta is not evidence that command recording dominates. The retained metrics and retained full-logits variants were themselves indistinguishable at the displayed precision despite different readback sizes; this reinforces that the synchronous host/API boundary and fixed small Tiny workload dominate this measurement more than command recording.

## Conclusion

Command recording is **not a meaningful bottleneck demonstrated by this profile on RADV**. Do not add retained Tiny training based on these results. Keep training on the normal-submit path: it has runtime-dependent controls and additional backward/update dispatches, while the retained implementation would need separate fixed-shape graphs and synchronization/output semantics. Revisit only with a GPU-timestamp or calibrated host breakdown showing a material recording fraction at a larger representative training workload (and with the required correctness/synchronization contract).

The benchmark source is `bench/bench_tiny_training.cpp`; it is a profiling tool only and does not alter runtime behavior or retain training commands.
