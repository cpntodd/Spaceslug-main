# Tiny GPU training acceptance status

**Audit date:** 2025-08-24  
**Audited repositories:** `spaceslug-main` and `../vulkan-runtime`  
**Status vocabulary:** this document records implemented contracts and test evidence only. It is not a model-quality, throughput, or production-readiness claim.

## Audited revisions and evidence

| Repository | Revision | Relevant committed artifacts |
|---|---|---|
| `spaceslug-main` | `f614d939e35d2115bd9ff8cf7824c5d451aba092` (`f614d93`) | Python host/backend and orchestration under `python/spaceslug/`; native ABI/client under `native/`; GPU/CPU acceptance tests under `tests/` |
| `vulkan-runtime` | `79896e80a0208ce448877a5e654f008f8ef4e705` (`79896e8`) | Vulkan APIs under `src/api/`, embedded GLSL kernels under `shaders/`, and per-kernel tests under `tests/` |

Relevant capability history includes `f614d93` (Spaceslug fixed retained-forward binding), `9db129a` (Spaceslug rank-8 profile), `8e5ccb1` (persistent AdamW windows), `fa54319` (runtime retained-forward ABI), `b4238b5` (runtime rank-8 profile), and `c065209` (runtime FP16 storage utility). These commits are implementation history, not separate acceptance releases.

## Supported contract

### Host (`spaceslug-main`)

- Deterministic CPU reference training and inference for the checked-in Tiny reference paths, including dataset-backed projected attention, masked causal loss, analytic-gradient checks, AdamW checkpoint/resume, checksummed artifacts, held-out metrics, and deterministic inference reporting.
- A bounded host/native Vulkan path for Tiny projected-attention LoRA training. The implemented fixed contract is **fp32**, single-head **H=64**, vocabulary **V=259**, padded vocabulary **Vp=320**, sequence cap **T=128**, frozen base weights, and LoRA rank **4 or 8**.
- The GPU training graph covers the implemented forward composition, causal loss/dLogits, LM-head/output/causal-attention/QKV backward, four adapter gradient paths, accumulation, SGD or AdamW update paths exposed by the current APIs, device-resident activation/state buffers, adapter/state readback/update, and checkpoint/restore orchestration.
- The host exposes explicit capability and fallback/status metadata. CPU remains authoritative when a shape, rank, operation, device, or runtime prerequisite is outside the GPU contract.
- The fixed retained-command path is intentionally narrower: exactly `T=128` input tokens and all `T*Vp` logits. It is forward-only; changing token staging data and resubmitting the recorded graph is supported.

### Runtime (`vulkan-runtime`)

- CPU-reference-checked Vulkan implementations and APIs for the committed kernel/test inventory, including fp32 vector add, SGEMM variants, normalization/RoPE, Q4 GEMM, attention, causal loss, LM-head/projection/backward, LoRA gradient/update/session operations, and Tiny persistent forward/training components.
- The runtime's Tiny capability metadata names rank 4/rank 8 LoRA and SGD/AdamW support, but this does not mean every combination is one retained graph or a complete end-to-end product trainer.
- `ForwardResourceGraph::forward_fixed_retained()` retains one complete fixed-shape forward command buffer (copy, barriers, dispatch, readback) and resubmits it with new token bytes. Its capability is `production_fixed_shape_forward_only_retained_command_buffer_resubmit`.
- Optional FP16 conversion/packing is a **storage** utility. Arithmetic, accumulation, logits, loss, optimizer state, and the existing Tiny forward path remain fp32 by default.

## Tests and commands run

### Spaceslug host

`python3 -m pytest -q` was attempted but cannot run in this environment because the system Python has no `pytest` module (`No module named pytest`). The repository's dependency-free equivalent was run successfully:

```sh
PYTHONPATH=python python3 -m unittest -v
```

**Result: 106 tests passed in 12.919s.** This includes CPU/reference, host backend, native SGEMM/LoRA parity, persistent Tiny window/AdamW state, checkpoint/artifact, TUI, and explicit GPU-boundary tests. Passing host tests establish contract/parity behavior; they do not establish RX580 performance or useful model quality.

### Vulkan runtime

The existing debug build was rebuilt and the full suite was run on the default Vulkan driver:

```sh
cmake --build build/debug --parallel 2
ctest --test-dir build/debug --output-on-failure
```

**Result: 41/41 tests passed.** The suite includes the Tiny persistent and immutable-command tests, LoRA/causal-loss/backward APIs, reduced-precision storage, fp32/fp16-storage/CQ4 kernels, execution engine, and benchmarks (benchmark tests passed; no performance number is asserted here).

The same correctness suite was run on lavapipe:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build/debug --output-on-failure
```

**Result: 41/41 tests passed in 97.09s.** The logged run does not constitute an RX580 throughput measurement; lavapipe is software Vulkan.

## Explicit remaining boundaries

- This is not evidence of a useful, production-sized, or quality-accepted Tiny model. No quality threshold, corpus-scale training result, or committed trained production checkpoint is claimed here.
- This is not a claim of full-graph retained training. General forward and all training/backward/LoRA methods use bounded normal submissions; only the exact fixed-shape forward-only method is retained.
- GPU support is not arbitrary-shape Tiny training: the fixed dimensions, fp32 arithmetic, frozen base, rank 4/8, bounded sequence length, and current adapter/optimizer API contracts apply. Unsupported shapes/modes must fall back or fail explicitly.
- The host path is bounded orchestration and parity coverage, not a GUI-complete training service, distributed trainer, multi-device trainer, or unrestricted autonomous improvement loop.
- FP16 is not a gfx803 compute-throughput feature. Storage conversion may lose precision or erase any bandwidth benefit; no speedup is promised, and full Tiny FP16 forward integration is deferred.
- The test suite proves correctness against CPU references and software Vulkan where specified. It does not prove RADV/RX580 performance, long-run stability, memory scaling, watchdog behavior, or production deployment packaging.
- The repository still does not claim full-parameter training, MoE training, frontier-scale RX580 pretraining, arbitrary Hugging Face architecture support, or unrestricted autonomous source modification.

For the narrower retained-command semantics, see [`../vulkan-runtime/docs/IMMUTABLE_COMMAND_REUSE.md`](../../vulkan-runtime/docs/IMMUTABLE_COMMAND_REUSE.md). For storage precision boundaries, see [`../vulkan-runtime/REDUCED_PRECISION.md`](../../vulkan-runtime/REDUCED_PRECISION.md).

## Acceptance interpretation

The current evidence supports the label **implemented experimental bounded GPU LoRA training and fixed forward-command prototype, with passing host and runtime correctness gates**. It does **not** support labels such as “fully GPU-trained Tiny,” “production trainer,” “retained full graph,” “FP16 acceleration,” or “useful model.”
