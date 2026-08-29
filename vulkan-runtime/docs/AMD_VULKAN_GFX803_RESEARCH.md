# AMD Vulkan and ISA Research for RX580/gfx803

**Status:** Reviewed 2026-08-27
**Scope:** Actionable findings from the requested AMD repositories, GPUOpen documentation, and the local ISA PDFs in `/home/oddsoul/Desktop/Training samples/AMD`.

## Executive summary

The RX580 is **GCN4 / gfx803 (Polaris10)**, not RDNA. The requested RDNA performance material is useful for general GPU-performance methodology, but RDNA-specific advice (wave32, WGP sizing, rapid packed math assumptions, and RDNA-only cache behavior) must not be copied into this runtime. For this project, the GCN3 ISA PDF is the closest supplied architectural reference and the existing gfx803 rules remain authoritative: wave64, FP32 arithmetic, at most 32 KiB LDS per Vulkan workgroup, no cooperative matrices, and bounded submissions.

The most valuable additions from this review are:

1. Treat ISA/resource inspection as a repeatable optimization loop, not a one-off disassembly check.
2. Keep synchronization validation enabled, and use synchronization validation specifically for cross-dispatch and cross-queue hazards.
3. Use GPU-assisted validation only as a targeted debug mode because shader instrumentation and readback are expensive.
4. Record the exact driver/compiler/device in benchmark artifacts. AMDVLK is discontinued and should not be adopted as a second production backend for this project; RADV/ACO is the target.
5. Preserve the supplied ISA PDFs as offline references, but use a matching machine-readable ISA XML when decoding binary code objects.

## Sources reviewed

### Local ISA references

The user-provided directory contains:

- `gcn3-instruction-set-architecture.pdf` (348 pages; two copies)
- Vega 7nm ISA
- RDNA, RDNA2, RDNA3, RDNA3.5, and RDNA4 ISA documents
- CDNA1 through CDNA5 ISA documents

For RX580 work, use **`gcn3-instruction-set-architecture.pdf`**, not the RDNA or CDNA PDFs. Relevant GCN3 sections include LDS, GPR allocation, work-groups, wait counters, and LDS/buffer instructions. The PDF states that GCN instructions operate over 64 threads per wavefront, describes 32-bank LDS, and documents `S_WAITCNT`/`LGKM_CNT`/`VM_CNT` ordering behavior. The ISA uses the term GCN3 even though the target device is the later GCN4/gfx803 family; verify any opcode or encoding assumption against a gfx803-compatible machine-readable spec before relying on it.

### Repositories

- [isa_spec_manager](https://github.com/GPUOpen-Tools/isa_spec_manager) — AMD's C++ [`IsaDecoder` API](https://github.com/GPUOpen-Tools/isa_spec_manager/blob/main/include/amdisa/isa_decoder.h) and experimental `explorer::Spec` APIs for machine-readable XML ISA specifications. The [basic decoder example](https://github.com/GPUOpen-Tools/isa_spec_manager/blob/main/source/examples/basic_decoder.cpp) shows initialization with a matching XML and decoding single instructions or complete shaders. This is suitable for a future offline shader-inspection helper, but it is not a replacement for SPIR-V validation.
- [Vulkan-ValidationLayers](https://github.com/KhronosGroup/Vulkan-ValidationLayers) — core validation, synchronization validation, and GPU-assisted validation. The [sync-validation usage guide](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/syncval_usage.md) documents the focused mode. The repository documents that synchronization hazards can only be fully diagnosed with submission context and that GPU-AV instruments shaders and reads GPU state back. The focused sync-validation pass uses `VK_VALIDATION_VALIDATE_SYNC=1` in this project’s workflow.
- [AMDVLK](https://github.com/GPUOpen-Drivers/AMDVLK) — explicitly [discontinued](https://github.com/GPUOpen-Drivers/AMDVLK/discussions/416). Its [README product-support section](https://github.com/GPUOpen-Drivers/AMDVLK/blob/master/README.md#product-support) says pre-GFX10 users should use an old release (v-2023.Q3.3 or earlier). It is useful as historical PAL/LLPC context only, not as a supported runtime dependency.

### GPUOpen documentation

- [RDNA Performance Guide](https://gpuopen.com/learn/rdna-performance-guide/) — useful general methodology: avoid bank conflicts, use LDS intentionally, minimize bandwidth, and profile rather than guess. RDNA-specific wave/WGP recommendations do not apply directly to gfx803.
- [Using Vulkan Validation Layers](https://gpuopen.com/learn/using-the-vulkan-validation-layers/) — supports making validation a normal development workflow.
- [Occupancy and resource usage](https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/) — reinforces balancing registers, LDS, and group size; translate its architectural examples to GCN rather than copying RDNA numbers.
- [Context rolls](https://gpuopen.com/learn/understanding-gpu-context-rolls/) — relevant to submission/state-change analysis, but the runtime is headless compute and should first measure dispatch, queue, barrier, and watchdog behavior directly.
- [Radeon Vulkan versions](https://gpuopen.com/learn/decoding-radeon-vulkan-versions/) — useful when recording driver/version identity in reports.
- [GPU Performance API](https://gpuopen.com/manuals/gpu_performance_api_manual/), [RGP](https://gpuopen.com/manuals/rgp_manual/), [RMV](https://gpuopen.com/manuals/rmv_manual/), [RGA](https://gpuopen.com/manuals/rga_manual/), [RDP](https://gpuopen.com/manuals/rdp_manual/), [RRA](https://gpuopen.com/manuals/rra_manual/), [OCAT](https://gpuopen.com/manuals/ocat/), [ADLX](https://gpuopen.com/manuals/adlx/), [FSR SDK](https://gpuopen.com/manuals/fsr_sdk/), [FidelityFX SDK](https://gpuopen.com/manuals/fidelityfx_sdk/), and [Schola](https://gpuopen.com/manuals/schola/) — tool/manual index reviewed for applicability. RGP/RGA/RMV are the most relevant to this headless runtime; RRA, FSR, FidelityFX, ADLX, OCAT, and Schola are not current dependencies.
- [Using the ISA decoder API](https://gpuopen.com/learn/using-isadecoder-api/) — confirms the XML-driven decode workflow.

## Concrete implications for vulkan-runtime

### Kernel design

- Keep every local size a multiple of 64. The current 256-thread/4-wave choices are compatible with the ISA model.
- Keep statically allocated/shared LDS below the 32 KiB Vulkan workgroup ceiling. Continue using padded row strides and validate the actual generated code when changing tile layouts.
- Treat VGPR and SGPR use as occupancy constraints. A larger tile or more accumulators is not automatically faster if register allocation reduces resident waves.
- For memory/LDS pipelines, reason about visibility and completion separately: shader-level ordering (`S_WAITCNT` in generated ISA) is not a substitute for Vulkan barriers/semaphore dependencies.
- Continue compiling GLSL with `glslc -O`; inspect SPIR-V and, where available, disassemble the final code object for resource counts and instruction mix.

### Validation workflow

1. Build with normal VVL enabled and do not suppress messages.
2. For a focused synchronization pass, run with `VK_VALIDATION_VALIDATE_SYNC=1` and inspect barriers, synchronization2 transitions, timeline semaphore waits, and cross-queue ownership/use patterns.
3. Enable GPU-AV only on a focused failing test or kernel. It consumes a descriptor slot, requires suitable features (including timeline semaphores for efficient operation), and can be dramatically slower; never use it for benchmarks.
4. Compare CPU reference output on lavapipe and RADV before interpreting a performance result.

### Profiling workflow

- Record GPU name, driver, Vulkan API version, shader compiler command, workgroup dimensions, LDS usage, and benchmark warmup/sample policy with every result.
- Prefer RADV/ACO and the existing timestamp-query benchmark path. RGP/RADV tracing may help where available, but tool support and counters vary by driver/device.
- Use median timings after clock warmup; do not infer occupancy from throughput alone.
- Use the ISA decoder as an offline inspection aid only after obtaining the correct machine-readable XML for the target architecture. Do not decode gfx803 binaries with an RDNA XML.

## Deliberate non-actions

- Do **not** add AMDVLK as a supported backend: the project is discontinued and its own documentation directs pre-GFX10 users to an old release.
- Do **not** change RX580 kernels to RDNA wave32/WGP recommendations.
- Do **not** add the entire Radeon Developer Tool suite as build dependencies.
- Do **not** vendor the large PDF collection into this repository; the source PDFs remain at the user's supplied path and this document records the relevant selection and applicability boundary.

## Follow-up candidates

1. Add a small optional `isa_inspect` utility outside the normal runtime build that accepts an ISA XML and a code-object/disassembly input, reports instruction counts, and records VGPR/SGPR/LDS metadata.
2. Add a documented VVL synchronization-validation test invocation to the verification workflow.
3. Add benchmark metadata output so RADV device/driver/compiler settings are captured alongside medians.
4. Revisit RGP/RADV trace capture only after the current kernel correctness and benchmark gates remain clean.

## Repository review checkout

For this review, the requested repositories were cloned shallowly under `/tmp/amd-vulkan-research/` and left outside the project tree so they do not affect project history or dependencies.

## Existing project constraints retained

This note supplements, and does not replace, `RESEARCH.md`, `AGENTS.md`, and `.opencode/rules/vulkan.md`. Those files remain authoritative for milestone acceptance gates and the non-negotiable gfx803 hardware rules.

## Citation links

- [isa_spec_manager GitHub](https://github.com/GPUOpen-Tools/isa_spec_manager)
- [Vulkan-ValidationLayers GitHub](https://github.com/KhronosGroup/Vulkan-ValidationLayers)
- [Vulkan-ValidationLayers sync-validation usage](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/syncval_usage.md)
- [Vulkan-ValidationLayers GPU-assisted validation](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/main/docs/gpu_validation.md)
- [AMDVLK GitHub](https://github.com/GPUOpen-Drivers/AMDVLK)
- [AMDVLK discontinuation discussion](https://github.com/GPUOpen-Drivers/AMDVLK/discussions/416)
- [GPUOpen machine-readable ISA](https://gpuopen.com/machine-readable-isa)
- [Vulkan Validation Layers guide](https://gpuopen.com/learn/using-the-vulkan-validation-layers/)
- [GPUOpen ISA decoder API guide](https://gpuopen.com/learn/using-isadecoder-api/)
- [GPUOpen context rolls](https://gpuopen.com/learn/understanding-gpu-context-rolls/)
- [GPUOpen occupancy/resource usage](https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/)
- [GPUOpen Radeon Vulkan versions](https://gpuopen.com/learn/decoding-radeon-vulkan-versions/)
- [GPUOpen RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/)
- [GPUOpen manuals](https://gpuopen.com/manuals/)

## Review date

2026-08-27
