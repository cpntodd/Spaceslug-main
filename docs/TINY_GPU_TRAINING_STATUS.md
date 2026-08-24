# Tiny GPU training acceptance status

**Audit date:** 2026-08-24
**Audited repositories:** `spaceslug-main` and `../vulkan-runtime`
**Status vocabulary:** this document records implemented contracts and test evidence only. It is not a model-quality, throughput, or production-readiness claim.

## Audited revisions and evidence

| Repository | Revision | Relevant committed artifacts |
|---|---|---|
| `spaceslug-main` | `4423acd` | Python host/backend and orchestration under `python/spaceslug/`; native ABI/client under `native/`; GPU/CPU acceptance tests under `tests/` |
| `vulkan-runtime` | `df65bb7` | Vulkan APIs under `src/api/`, embedded GLSL kernels under `shaders/`, and per-kernel tests under `tests/` |

The current runtime facts are: the integrated Tiny graph exposes fixed-window SGD for the LM-head, output projection, and combined QKV groups, plus LM-head AdamW with graph-owned `m`/`v`/step state; every integrated group accepts exactly `1 <= rows <= 128`; output-projection and combined-QKV AdamW are explicitly unsupported and return `-4`; embeddings, positions, FFN, and norm remain unsupported; dataset training and retained training remain unsupported; standalone FP32 subset APIs remain distinct and caller-supplied; the `dataset_batch_buffer` is only standalone staging and `ForwardResourceGraph::train_dataset_batch(...)` returns `-3`; the FP16 Tiny constructor is an evaluation/storage path that widens frozen binary16 inputs to the unchanged FP32 graph; ranks 4 and 8 are supported; fixed retained forward+loss is exposed through the native `spaceslug_tiny_forward_loss_fixed_retained` ABI and Python `execute_tiny_fixed_retained_loss` / `PersistentTinyTrainer.fixed_forward_loss` binding; and full training uses ordinary submissions. These are implementation contracts, not separate acceptance releases.

## Supported contract

### Host (`spaceslug-main`)

- Deterministic CPU reference training and inference for the checked-in Tiny reference paths, including dataset-backed projected attention, masked causal loss, analytic-gradient checks, AdamW checkpoint/resume, checksummed artifacts, held-out metrics, and deterministic inference reporting.
- A bounded host/native Vulkan path for Tiny projected-attention LoRA training. Exactly two named profiles are supported: **`tiny_h64_v259_vp320_t128_rank4`** and **`tiny_h64_v259_vp320_t128_rank8`**. Both use **fp32**, **H=64**, vocabulary **V=259**, padded vocabulary **Vp=320**, sequence capacity **T=128**, and frozen base weights; only the LoRA rank differs (4 or 8). Full training and backward operations are ordinary bounded submissions. Arbitrary dimensions and unlisted profile combinations are rejected, rather than treated as supported GPU configurations.
- The GPU training graph covers the implemented forward composition, causal loss/dLogits, LM-head/output/causal-attention/QKV backward, four adapter gradient paths, accumulation, graph-integrated SGD for the LM-head/output/QKV groups, LM-head AdamW state/update, device-resident activation/state buffers, adapter/state readback/update, and checkpoint/restore orchestration. Graph-integrated AdamW is explicitly unsupported for output and QKV and returns `-4`; standalone base-training APIs are not a bridge. This does not include dataset-batch-buffer integration: that standalone buffer is retained and GPU-processed, but it is not the training graph and `train_dataset_batch` returns `-3`.
- The host exposes explicit capability and fallback/status metadata. CPU remains authoritative when a shape, rank, operation, device, or runtime prerequisite is outside the GPU contract.
- The fixed retained-command path is intentionally narrower: exactly `T=128` input tokens and all `T*Vp` logits. Native fixed forward+loss is also retained for exactly 128 tokens, 128 targets, and 128 mask values, returning 128 row losses plus the fixed logits buffer; the Python binding is `BackendSession.execute_tiny_fixed_retained_loss` and `PersistentTinyTrainer.fixed_forward_loss`. Changing token/target/mask staging data and resubmitting the recorded graph is supported. Backward, LoRA gradient/adapter updates, and optimizer/update are not retained capabilities; they remain unsupported in the retained path and use ordinary bounded submissions only.

### Profile metadata API

The runtime C API is the authoritative profile enumeration and validation boundary:

```c
uint32_t spaceslug_tiny_profile_count(void);
int spaceslug_tiny_profile_query(uint32_t index,
                                 spaceslug_tiny_profile_descriptor *out);
int spaceslug_tiny_profile_validate(uint32_t hidden, uint32_t vocab,
                                    uint32_t padded_vocab,
                                    uint32_t token_capacity, uint32_t rank);
```

`spaceslug_tiny_profile_count()` currently returns **2**. Querying index `0` or `1` returns a descriptor for `tiny_h64_v259_vp320_t128_rank4` or `tiny_h64_v259_vp320_t128_rank8`, respectively (`H=64`, `V=259`, `Vp=320`, `T=128`, rank `4` or `8`). The query result is `SPACESLUG_TINY_PROFILE_SUPPORTED` (`0`). An out-of-range index returns `SPACESLUG_TINY_PROFILE_UNSUPPORTED` (`1`), and a null output pointer returns `SPACESLUG_TINY_PROFILE_INVALID_ARGUMENT` (`2`); a failed query leaves its output untouched.

`spaceslug_tiny_profile_validate()` returns `SUPPORTED` (`0`) only for those two exact tuples. A non-zero dimension or rank is an `INVALID_ARGUMENT` (`2`); positive but unlisted dimensions or ranks return `UNSUPPORTED` (`1`). Callers must enumerate or validate before dispatch. Rejection is explicit: unsupported profile requests must not be silently coerced into a nearby profile or reported as GPU support. This metadata API does not initialize Vulkan and can be tested on hosts without a Vulkan device or ICD. The Python bridge exposes the same boundary through `BackendSession.tiny_profiles()` and `BackendSession.validate_tiny_profile(...)`.


### Runtime (`vulkan-runtime`)

- CPU-reference-checked Vulkan implementations and APIs for the committed kernel/test inventory, including fp32 vector add, SGEMM variants, normalization/RoPE, Q4 GEMM, attention, causal loss, LM-head/projection/backward, LoRA gradient/update/session operations, and Tiny persistent forward/training components.
- The runtime's Tiny capability metadata names rank 4/rank 8 LoRA, integrated LM-head/output/QKV SGD for `rows <= 128`, and integrated LM-head AdamW support; output/QKV AdamW, embeddings, positions, FFN, and norm are explicitly unsupported. Dataset training and retained training are also unsupported. This does not mean every combination is one retained graph or a complete end-to-end product trainer.
- `ForwardResourceGraph::forward_fixed_retained()` retains one complete fixed-shape forward command buffer (copy, barriers, dispatch, readback) and resubmits it with new token bytes. Its capability is `production_fixed_shape_forward_only_retained_command_buffer_resubmit`. The optional `spaceslug_tiny_forward_loss_fixed_retained` path extends the retained graph through masked causal loss for exactly 128 tokens/targets/mask values; it does not retain backward, LoRA, or optimizer/update work.
- Optional FP16 conversion/packing is a **storage** utility. The FP16-storage Tiny evaluation constructor widens frozen inputs before using the unchanged FP32 graph; arithmetic, accumulation, logits, loss, optimizer state, and training remain FP32.

## Tests and commands run

### Spaceslug host

`python3 -m pytest -q` was attempted but cannot run in this environment because the system Python has no `pytest` module (`No module named pytest`). The repository's dependency-free equivalent was run successfully:

```sh
PYTHONPATH=python python3 -m unittest -v
```

**Result: 111 tests passed.** This includes CPU/reference, host backend, native SGEMM/LoRA parity, persistent Tiny window/AdamW state, retained forward+loss binding, checkpoint/artifact, TUI, and explicit GPU-boundary tests. Passing host tests establish contract/parity behavior; they do not establish RX580 performance or useful model quality.

### Vulkan runtime

The existing debug build was rebuilt and the full suite was run on the default Vulkan driver:

```sh
cmake --build build/debug --parallel 2
ctest --test-dir build/debug --output-on-failure
# Focused, device-independent profile metadata gate:
ctest --test-dir build/debug -R '^tiny_profile_api$' --output-on-failure
```

**Result: 43/43 tests passed.** The suite includes the Tiny profile metadata, persistent, and immutable-command tests, LoRA/causal-loss/backward APIs, reduced-precision storage, fp32/fp16-storage/CQ4 kernels, execution engine, and benchmarks (benchmark tests passed; no performance number is asserted here).

The same correctness suite was run on lavapipe:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build/debug --output-on-failure
```

**Result: 43/43 tests passed.** The logged run does not constitute an RX580 throughput measurement; lavapipe is software Vulkan.

## Explicit remaining boundaries

- This is not evidence of a useful, production-sized, or quality-accepted Tiny model. No quality threshold, corpus-scale training result, or committed trained production checkpoint is claimed here.
- This is not a claim of full-graph retained training. The exact fixed-shape forward+loss subset is retained, but backward, LoRA gradient/adapter update, and optimizer/update are not retained capabilities and remain unsupported in the retained path; general training methods use bounded normal submissions. A fixed full-training retained subset was not accepted because its synchronization, mutable optimizer controls, and rank-dependent staging were not verified. Integrated graph SGD/AdamW support is a normal-submission capability, not retained-command training; AdamW remains unsupported for output and QKV.
- GPU support is not arbitrary-shape Tiny training: only the named `tiny_h64_v259_vp320_t128_rank4` and `tiny_h64_v259_vp320_t128_rank8` profiles are supported. The fixed dimensions, fp32 arithmetic, frozen base, rank 4/8, bounded sequence length, and current adapter/optimizer API contracts apply. Arbitrary dimensions and unsupported shapes/modes must fall back or fail explicitly; they are not silently accepted as profile variants.
- The integrated graph base-training contract is limited to LM-head SGD+AdamW, output-projection SGD, and combined-QKV SGD for `1 <= rows <= 128`. Output/QKV AdamW, embeddings, positions, FFN, norm, dataset training, and retained training remain unsupported; standalone caller-supplied subsets are separate APIs. The host path is bounded orchestration and parity coverage, not a GUI-complete training service, distributed trainer, multi-device trainer, or unrestricted autonomous improvement loop.
- FP16 is not a gfx803 compute-throughput feature. Storage conversion may lose precision or erase any bandwidth benefit; no speedup is promised. FP16-storage evaluation exists, but FP16 arithmetic and FP16 training do not.
- The test suite proves correctness against CPU references and software Vulkan where specified. It does not prove RADV/RX580 performance, long-run stability, memory scaling, watchdog behavior, or production deployment packaging.
- The repository still does not claim full-parameter training, MoE training, frontier-scale RX580 pretraining, arbitrary Hugging Face architecture support, or unrestricted autonomous source modification.

For the narrower retained-command semantics, see [`../vulkan-runtime/docs/IMMUTABLE_COMMAND_REUSE.md`](../../vulkan-runtime/docs/IMMUTABLE_COMMAND_REUSE.md). For storage precision boundaries, see [`../vulkan-runtime/REDUCED_PRECISION.md`](../../vulkan-runtime/REDUCED_PRECISION.md).

## Acceptance interpretation

The current evidence supports the label **implemented experimental bounded GPU LoRA training and fixed forward-command prototype, with passing host and runtime correctness gates**. It does **not** support labels such as “fully GPU-trained Tiny,” “production trainer,” “retained full graph,” “FP16 acceleration,” or “useful model.”
