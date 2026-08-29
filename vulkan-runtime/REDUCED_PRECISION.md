# Reduced-precision storage boundary

The runtime exposes an optional FP16 **representation** and evaluation-input path
for frozen Tiny weights and activations. It is deliberately implemented as a
capability-gated conversion utility: the Tiny FP16-input constructor widens values
to FP32 before creating the existing graph. FP32 remains the default and all
shader arithmetic/accumulation remains FP32.

## API

`src/api/reduced_precision_storage.h` exposes `query_capabilities()` and portable
IEEE-754 binary16 conversion helpers. `packedFp16Storage` is always available:
two 16-bit values may be represented in one `uint32_t` word without requiring
`shaderFloat16` or `storageBuffer16BitAccess`. The C ABI reports the same contract
through `spaceslug_storage_capabilities_query()` and
`spaceslug_storage_capability()`. The current Tiny FP16-input constructor accepts
binary16 host arrays and widens them to FP32 before the existing graph is built; it
does not currently provide device-side FP16 storage for that graph.

The physical `shaderFloat16` bit is reported for diagnostics only. It is never
enabled by `create_context()` and must not be interpreted as a claim of useful
FP16 arithmetic on gfx803/Polaris. Consumers must convert FP16 storage to FP32
before arithmetic and accumulate in FP32.

## Precision and performance boundaries

* FP16 storage rounds values to binary16 (about 3 decimal digits of relative
  precision and a finite range up to 65504); out-of-range finite values become
  infinity. NaN/Infinity and signed zero are preserved.
* The verified Tiny FP16-input constructor is limited to frozen embeddings,
  positions, Q/K/V/O weights, and the LM head. Trainable parameters, optimizer
  state, logits, loss computation, and numerically sensitive reductions remain
  FP32 in the existing graph.
* A binary16 representation can reduce host-side input storage, but the current
  Tiny constructor widens inputs to FP32 before upload. It therefore makes no
  device-memory or bandwidth reduction claim. No speedup is promised, and no FP16
  throughput claim is made for gfx803.
* The Tiny forward graph has a safe evaluation constructor accepting binary16
  host input arrays for frozen embeddings, positions, Q/K/V/O weights, and the LM
  head. It widens every value to FP32 before uploading to the existing graph; no
  shader, descriptor, or arithmetic path is changed and no FP16 arithmetic is
  claimed.
* The end-to-end Tiny test compares the reduced-storage run with the original
  FP32-input baseline on the same CPU/Vulkan implementation. The acceptance
  bound is `abs(error) <= 0.02 + 0.02 * abs(baseline)` per finite logit. This is
  an evaluation guardrail, not a universal model-quality guarantee; callers
  should tighten it for their model and data.
* The existing FP32 constructor and graph remain the default. Training,
  optimizer state, logits, losses, and backward buffers remain FP32.

## Verification

`reduced_precision_storage` performs CPU conversion parity checks and queries the
actual Vulkan device capability contract. `tiny_forward_persistent` additionally
runs the complete Tiny forward graph through the FP16-input constructor and
compares CPU-generated FP32 baseline inputs against the widened-input logits.
Run the focused checks on RADV and lavapipe:

```sh
ctest --test-dir build/debug -R 'reduced_precision_storage|tiny_forward_persistent' --output-on-failure
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ctest --test-dir build/debug -R 'reduced_precision_storage|tiny_forward_persistent' --output-on-failure
```
