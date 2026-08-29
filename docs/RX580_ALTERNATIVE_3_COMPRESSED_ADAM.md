# Alternative 3 — Blockwise 8-bit Adam optimizer state

> **Correction note (2026-08-25):** this revision removes the false claim that the cited 8-bit optimizer quantizes gradients, removes the unsupported `6P` baseline, qualifies checkpoint savings, and separates Sophia and CPU offload from state quantization.

**Recommendation:** **Defer implementation until the memory planner demonstrates that FP32 SGD/factored state cannot meet a concrete target and Adam-like behavior is required. If pursued, quantize optimizer moments only and keep weights/gradients FP32.**

## Proposal

Store Adam-like first and second moments in blockwise 8-bit form with per-block quantization metadata:

```text
8-bit m/v block + scales
    -> dequantize current block to FP32
    -> compute FP32 Adam-like update
    -> requantize updated m/v block
    -> write weight and compressed state
```

The cited Dettmers et al. method quantizes **optimizer states**; standard-precision weights and gradients are retained. Quantizing gradients would be a separate, more aggressive algorithm with an additional numerical error source and is outside this recommendation unless independently designed, cited, and validated.

The exact algorithm must define dynamic quantization map, block size, scale representation, signed/range conventions for first and non-negative second moments, percentile/outlier handling if any, epsilon placement, bias correction, decay semantics, rounding, NaN/Inf behavior, and state initialization.

## Current repository and hardware fit

The repository has bounded FP32 AdamW state/update for selected Tiny graph groups, but no compressed-state implementation or parity evidence. gfx803 offers no useful native INT8 training arithmetic for this design. Compression is therefore a **storage/bandwidth** technique; dequantization, update math, and accumulation remain FP32.

Current bounded Tiny support does not imply all-parameter Adam: positions, FFN, normalization, arbitrary shapes, complete dataset training, and retained backward/update remain unsupported.

## Memory accounting

For `P` FP32 trainable parameters, lower-bound persistent/live components are:

| Component | Raw bytes |
|---|---:|
| FP32 weights | `4P` |
| one full FP32 gradient | `4P` |
| two 8-bit moments | `2P` |
| subtotal | `10P` plus quantization metadata |

For 500M parameters, the raw subtotal is approximately 5.0 GB decimal (4.66 GiB), before activations, accumulation, block scales, alignment, conversion scratch, staging, workspaces, and checkpoint snapshots. This remains too close to an 8 GB ceiling for a fully resident 0.5B trainer.

Raw optimizer-state bytes fall from `8P` to near `2P`; actual checkpoint reduction is smaller and applies only to optimizer state, because scales, metadata, alignment, headers, and FP32 weights remain. A quantized-state checkpoint resumes the quantized algorithm; it is not numerically equivalent to an FP32 AdamW checkpoint.

## Correct bounded-kernel design

A whole-tensor dequantization buffer would invalidate the memory objective. Process bounded blocks:

1. Load one block's 8-bit moments and metadata.
2. Dequantize into bounded FP32 register/LDS/staging storage.
3. Apply the exact FP32 moment and weight update.
4. Compute new block scales deterministically.
5. Requantize with specified rounding and write back.
6. Synchronize only where scale reduction requires it, and chunk dispatches for the watchdog.

The memory planner must track simultaneous gradient, weight, compressed state, FP32 block scratch, scale-reduction partials, command-ring overlap, and checkpoint staging. Small tensors may stay FP32 when scale metadata/alignment overwhelms the saving or quantization is unstable; embeddings may also need a stable/full-precision state policy.

## CPU offload and Sophia are different alternatives

- **CPU offload** keeps full-precision state in host RAM and transfers gradients/weights/state according to an explicit placement plan. It avoids quantization error but can be PCIe-bound. It must be compared separately, with bytes/step and overlap measured.
- **Sophia/SophiaG** maintains a momentum estimate and a periodically updated diagonal curvature/Hessian estimate, with clipped preconditioned updates. The paper reports step-efficiency improvements in its tested pretraining settings, not a universal “half the steps” guarantee. Its two persistent full-size statistics are not a low-state-memory replacement for compressed Adam unless they are separately compressed.

## Main risks

- Moment quantization can change convergence, especially for small second moments and outliers.
- Quantizer/scaling decisions are part of the optimizer and checkpoint ABI.
- Scale reductions and repeated conversion can erase bandwidth gains.
- Full-tensor temporary buffers can silently erase the VRAM benefit.
- Small tensors and embedding states may require FP32 fallback.
- Deterministic CPU/GPU rounding parity can be difficult.
- No current code or RX580 long-run evidence supports this method.

## Validation gates

- CPU reference vectors for quantize/dequantize, moment update, bias correction, decay, non-finite handling, and rounding.
- CPU/GPU parity at block boundaries and across multiple optimizer steps.
- Ablations for block size, state fallback thresholds, scale rule, and clipping.
- Long-run trajectory versus FP32 AdamW on Tiny and Tiny-Plus.
- Peak-live VRAM including all block/scale/checkpoint scratch.
- Timestamped bounded dispatches and end-to-end tokens/second on RX580/RADV.
- Deterministic resume of the **quantized** state algorithm.
- Comparison with factored state and explicit CPU offload on the same task.

## Recommendation

**Keep this as a conditional research fallback.** First establish plain SGD/LOMO and one factored optimizer. If a measured workload needs Adam-like behavior and optimizer state is the proven bottleneck, prototype blockwise moment compression on CPU and one GPU tensor, with an FP32 fallback threshold for small or unstable tensors. Do not call it “INT8 training” or claim FP32 AdamW compatibility.

## References

- Repository FP32 boundary: [`TINY_GPU_TRAINING_STATUS.md`](TINY_GPU_TRAINING_STATUS.md).
- Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, 2022: https://arxiv.org/abs/2110.02861
- Liu et al., *Sophia*, 2023: https://arxiv.org/abs/2305.14342
- Ren et al., *ZeRO-Offload*, 2021: https://arxiv.org/abs/2101.06840

The references support research candidates, not gfx803 compatibility or this runtime's implementation.
