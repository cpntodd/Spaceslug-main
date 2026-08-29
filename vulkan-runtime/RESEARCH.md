# Custom Vulkan Compute Runtime for the RX580 — Research Report

**Date:** 2026-08-21
**Status:** Research complete — ready for design decision
**Author:** Orchestrated research report (4 parallel research tracks + environment verification)

---

## 0. TL;DR

| Question | Answer |
|---|---|
| Can we build a custom GPU runtime from scratch? | **Yes.** RADV (mesa) fully supports your RX580 today — Vulkan 1.4, ACO compiler, no ROCm anywhere. |
| What language? | **C++20 + Vulkan-Hpp (RAII) + VMA (memory) + GLSL shaders + CMake.** Best tooling, biggest AI-assist corpus, proven LLM precedents. |
| What's the compute target? | **FP32, not fp16.** *(Correction: I told you earlier "11.6 TFLOPS fp16" — that was wrong. Polaris has no 2x fp16. The real ceiling is ~5.8–6.2 TFLOPS fp32.)* |
| Is there reference code for *our exact GPU*? | **Yes — `lakitu12/vkblas`: a Vulkan fp32 GEMM tuned for gfx803 hitting 2.54 TFLOPS (62% of peak), 5× faster than rocBLAS on the same card.** |
| Realistic performance envelope | 4B model ≈ **40 tok/s**; 7B Q4 ≈ **3–5 tok/s** (llama.cpp-on-Vulkan community data on this exact GPU). |
| Buildable solo, non-coder, with AI help? | **Yes** — if we take it in small verified steps. This is a *months* project, not weeks. |

---

## 1. Why this works at all (the good news)

AMD's ROCm *userspace* (HIP, rocBLAS, etc.) dropped gfx803 support after ROCm 4.5. But the important things still work on your card:

- **amdgpu kernel driver** — fully supports gfx803 (your `/dev/kfd` + `/dev/dri` are live)
- **amdkfd** — the compute kernel driver — fully supports it
- **RADV** — mesa's Vulkan driver — fully supports it, **no deprecation planned**, Vulkan **1.4**, and compiles shaders with the excellent ACO backend
- **LLVM** — still ships gfx803 codegen (`-mcpu=gfx803`) for any custom tooling

So: the *kernel stack* is alive and well; only AMD's *userspace libraries* were abandoned. We replace those userspace libraries with our own — exactly your vision.

Your system is verified ready: Vulkan 1.4.305, `RADV POLARIS10`, mesa 25.0.7, 8.6 GB VRAM, g++ 14.2, cmake 3.31.

---

## 2. Hardware reality — what the RX580 can actually do

**Corrected numbers (sourced):**

| Property | Value |
|---|---|
| Compute units | 36 CUs (2304 shader lanes) |
| FP32 peak | **~5.8–6.2 TFLOPS** |
| FP16 | **~1× FP32** — *no* 2x packed math (that's Vega/GCN5). RADV even hides fp16 by default (`radv_enable_float16_gfx8`) because it's "not beneficial". **We compute in fp32.** |
| Memory | 8 GB GDDR5 @ 256 GB/s |
| Wavefront | **64 lanes** (not 32 like NVIDIA) — workgroups must be multiples of 64 |
| Shared memory (LDS) | 64 KB per CU, but Vulkan exposes **32 KB max per workgroup** |
| Async compute engines | 4 ACEs + 1 graphics + 2 hardware schedulers |
| PCIe | 3.0 x16 (~12–13 GB/s practical) — keep weights on-device, avoid host round-trips |
| Subgroups | Fixed 64 — no wave32 control |
| Tensor cores / coop matrices | **None** — pure FMA. (llama.cpp's coop-matrix shaders literally fail to compile on your GPU.) |

**Implications for kernel design:**
- GEMM design: 64×64 tiles, 256 threads (4 waves), LDS-resident operands ≤32 KB, 12–16 waves/CU occupancy target, VGPR budget ≤64–128 per lane
- Compute in fp32; use fp16/bf16 *only as storage* to halve memory traffic
- Long dispatches must be chunked (amdgpu compute-ring timeout workaround)
- Performance ceiling for LLM decode is memory/ALU-bound, not PCIe-bound

**Real-world datapoints on this exact GPU (Vulkan):**
- llama.cpp Vulkan backend on RX 580 (RADV): **~40 tok/s on a 4B Q4_K_M** model
- Community aggregate: 7B Q4 ≈ 3–5 tok/s generation
- **vkblas fp32 GEMM (4096³): 2.54 TFLOPS = 62% of Polaris FMA peak** — vs rocBLAS's 0.51 TFLOPS on the same card. The hardware can be pushed hard; we have a target to beat.

---

## 3. Language & tooling recommendation

**Host runtime: C++20 + Vulkan-Hpp (RAII) + VulkanMemoryAllocator (VMA) + CMake**
**Shaders: GLSL `#version 450` → compiled at build time by `glslc` → SPIR-V → embedded → driver (ACO) compiles to GCN ISA at runtime**

Why (ranked):

1. **AI-assistant reliability is the deciding factor.** ~90% of all Vulkan code ever written is C/C++ + GLSL. An AI co-developer can generate correct `vkCmdDispatch`/`#version 450` code with far less hallucination risk than in Rust/Zig/WGSL, where the corpus is thin.
2. **The two hardest Vulkan subsystems come pre-solved and MIT-licensed:**
   - Memory management → **VMA** (used by Blender, Godot, Qt, Cyberpunk 2077)
   - Object lifetimes → **Vulkan-Hpp RAII wrappers**
   What's left for us is queue/command/dispatch logic — the well-trodden core.
3. **Direct precedents in the same niche** (LLM/GEMM on Vulkan): llama.cpp's Vulkan backend and Kompute are both C++ + GLSL. We can study and legally copy (MIT).
4. **Tooling safety net for a non-coder:** Khronos Validation Layers (catch barrier/OOB bugs with readable messages), RenderDoc (debug a compute dispatch, inspect buffers), lavapipe (software Vulkan — run tests without a GPU at all).

Alternatives considered and rejected:
- **C**: no safety, every handle manual — foot-gun city for a non-professional
- **Rust (ash)**: viable fallback, but ash is "C with extra ceremony" (unsafe everywhere) and the LLM-compute corpus is thin; `vulkano` is pre-1.0 with API churn
- **Zig**: bindings + language both pre-1.0, near-zero AI corpus
- **WGSL/WebGPU**: wgpu hides exactly the memory/submission control we're deliberately building
- **OpenCL C via clspv**: viable escape hatch but extra moving parts; GLSL is the proven route

Toolchain: CMake (every library in the stack ships CMake; best IDE/CI/AI support) + glslc + spirv-val (build-time SPIR-V validation) + VVL in debug + lavapipe for CI + timestamp queries / `MESA_VK_TRACE=rgp` for profiling.

---

## 4. Architecture — the design plan

```
┌─────────────────────────────────────────────────────┐
│  APPLICATION LAYER (later: Cactus engine backend)    │
├─────────────────────────────────────────────────────┤
│  KERNEL LIBRARY  (the "math": what makes it useful)  │
│  SGEMM · transpose · dequant-int4 · RMSNorm · RoPE   │
│  attention (flash-style) · (later) backward kernels  │
├─────────────────────────────────────────────────────┤
│  EXECUTION ENGINE (the "OS")                         │
│  queue mgmt · N-slot ring · timeline semaphores      │
│  pipeline barriers · async-copy overlap (ACE queues) │
├─────────────────────────────────────────────────────┤
│  MEMORY MANAGER (VMA wrapper)                        │
│  device-local pools · linear staging allocator       │
├─────────────────────────────────────────────────────┤
│  CORE (vk-bootstrap pattern)                         │
│  instance → device → compute queue family            │
├─────────────────────────────────────────────────────┤
│  DRIVER: RADV (ACO) → GCN ISA → RX580                │
└─────────────────────────────────────────────────────┘
```

Key design decisions (from research):

1. **Core:** headless compute only — no windowing, no present queue, no WSI. Smallest correct skeleton is vk-bootstrap's `simple_compute.cpp` (~200 lines).
2. **Memory:** VMA with two pools — a device-local pool (weights, activations, KV cache) and a *linear allocator* pool (per-frame staging/scratch). Keep weights resident; never stream per-token over PCIe.
3. **Execution:** modern pattern — **N-slot ring buffering** (e.g. 3 slots), each owning a command buffer + descriptor set; **timeline semaphores** for queue/host sync; **synchronization2 pipeline barriers** between dependent dispatches. Avoid `vkQueueWaitIdle` except in debug/benchmark.
4. **Shaders:** GLSL 450 at build time → SPIR-V embedded in the binary. `VkPipelineCache` + RADV's on-disk cache avoid per-run compile latency. Later, vkFFT-style *runtime codegen + autotuning* is the performance upgrade path.
5. **GEMM (the heart):** copy vkblas's proven gfx803 recipe — 64×64 tile, 256 threads (4 waves, 3 workgroups/CU = 12 waves), LDS row stride `≡ 2 mod 32` floats to avoid bank conflicts, 16 accumulators/thread, `ds_read_b64` + FMA inner loop. Verify with `spirv-dis | grep -c Fma` (the #1 silent killer: forgetting `-O` in glslang costs ~1000×).
6. **Async:** use the 4 ACE (compute) queues to overlap staging copies / dequant passes with GEMM — vkFFT-style "append to command buffer" design gives us that freedom.
7. **Testing:** correctness vs CPU reference → lavapipe CI runs → real perf on RX580 via timestamp queries. `spirv-val` on every shader at build time.

---

## 5. Open-source reference map (what to study, what we can copy)

| Project | License | Role in our project |
|---|---|---|
| **`lakitu12/vkblas`** | ⚠️ **no license** | **STUDY ONLY (don't copy code).** A Vulkan fp32 GEMM tuned specifically for **gfx803/RADV** — 2.54 TFLOPS (62% peak). Its README documents every Polaris trap: bank conflicts, workgroup sizing, write-bandwidth, dispatch chunking. This is our performance blueprint. |
| **`minsuk22/exynos-gemm-vulkan-bench`** | ⚠️ no license | **STUDY ONLY.** GEMM tuning methodology: LDS double-buffering, 126-kernel sweep, timestamp-query timing, the `-O` disaster warning. Our tuning harness template. |
| **llama.cpp Vulkan backend** (`ggml-vulkan.cpp` + `vulkan-shaders/*.comp`) | MIT ✅ | Architecture reference for LLM compute: dequant-into-registers kernels, flash attention, split-K, descriptor/pipeline management. Copy the *patterns* freely. |
| **vkFFT** (`DTolm/VkFFT`) | MIT ✅ | The gold standard for runtime shader codegen + autotuning and in-place memory discipline. Its design philosophy (append-to-command-buffer, no internal queue ownership) shapes our execution engine. |
| **VulkanMemoryAllocator (VMA)** | MIT ✅ | **Dependency.** Memory type selection, sub-allocation, linear allocator. |
| **Erkaman/vulkan_minimal_compute** | MIT ✅ | The ~400-line minimal compute skeleton to start from. |
| **lumina37/vulkan-compute-demo** | MIT ✅ | Multi-kernel C runtime with GEMM + Flash Attention 2 + tests — good middle-weight template. |
| **Kompute** (LF AI & Data) | Apache-2.0 ✅ | Purpose-built Vulkan compute framework (C++/Python), used by GPT4ALL. Good for API-shape inspiration. |
| **Sascha Willems vulkan-examples** | MIT ✅ | `computeheadless`, `computenbody`, `timelinesemaphore` — canonical compute examples. |
| **Khronos guide + Vulkan-Samples + vkguide.dev + vk-bootstrap** | MIT/Apache ✅ | Tutorial/reference docs; `simple_compute.cpp` is our skeleton. |
| **clspv** (Google) | Apache-2.0 ✅ | OpenCL C → SPIR-V. Escape hatch if GLSL ever feels limiting. |

**License rule of thumb:** everything useful in this space is MIT/Apache except the two gfx803-tuned repos (study their *techniques* — bank-conflict math and tuning methodology are ideas, not copyrightable code).

---

## 6. Milestone plan (solo, non-coder, AI-assisted)

Each milestone ends with a **verifiable acceptance test** before we move on. Nothing here is speculative — each step builds on a working previous step.

| # | Milestone | Deliverable | Acceptance test |
|---|---|---|---|
| **M0** | Project scaffold | `vulkan-runtime/` repo, CMake skeleton, VMA + Vulkan-Hpp vendored, VVL enabled, lavapipe fallback | `cmake --build .` clean; `vulkaninfo` confirms RADV |
| **M1** | "Hello compute" | vk-bootstrap-style skeleton: init → compute queue → allocate SSBO → dispatch trivial kernel (vector add) → read back | Correct result vs CPU; runs on both lavapipe and RADV |
| **M2** | First GEMM | 64×64-tile fp32 SGEMM with LDS, 256 threads | Correct vs CPU; ≥1.0 TFLOPS on 1024³; **stretch: approach vkblas 2.54 TFLOPS** |
| **M3** | Tuning harness | Timestamp-query benchmark, median-of-runs, per-size sweep (exynos methodology) | Reproducible numbers; ranking of variants |
| **M4** | Kernel library | transpose, int4-dequant-GEMM, RMSNorm, RoPE, flash-style attention | Each kernel correct vs CPU reference; GEMM perf measured |
| **M5** | Execution engine v2 | N-slot ring, timeline semaphores, async-copy overlap on ACE queues | Pipelined dispatches overlap; no stalls on long chains |
| **M6** | Cactus engine integration | `gfx803_backend` behind the same interface as `metal_backend` — cactus graph ops dispatch to our kernels | `cactus run Cactus-Compute/needle` executes with GPU backend active (the stated baseline) |
| **M7** | Training kernels (moonshot) | fp32 backward GEMM + AdamW on GPU | LoRA training step runs on GPU with correct gradients vs CPU |

**Suggested initial repository layout:**
```
vulkan-runtime/
  cmake/          # find Vulkan, VMA, glslc
  src/core/       # instance, device, queue
  src/mem/        # VMA wrapper, pools
  src/exec/       # command/ring/semaphores
  src/kernels/    # GLSL + launchers (gemm, transpose, ...)
  shaders/        # *.glsl + build script
  bench/          # tuning harness
  tests/          # CPU-reference correctness + lavapipe CI
```

---

## 7. Risks & honest expectations

1. **This is a months-long project**, not a weekend. M0–M2 (getting a real GEMM running) is achievable in weeks with AI help. M6 (cactus on GPU) is the milestone where it becomes genuinely hard.
2. **The performance ceiling is real:** ~6 TFLOPS fp32 and 256 GB/s. Expect a 7B Q4 at 3–5 tok/s and a 4B at ~40 tok/s if we do everything right (that's llama.cpp-Vulkan's numbers; beating them is our goal).
3. **"No CPU offload" is achievable** — 8 GB VRAM fits up to ~7B at 4-bit. But per-token host transfers must be avoided (PCIe is the bottleneck).
4. **The Cactus integration is the hard part**, not the runtime. The engine's `metal_backend` interface is Apple-specific; adding a GPU backend means understanding cactus's graph/engine call paths. We'll study that interface during M5–M6, not before.
5. **We must verify each kernel on hardware.** RADV + ACO will silently miscompile nothing, but our *logic* will be wrong often — hence CPU-reference tests at every step.
6. **Known gfx803 traps to design around from day 1:** wave64-only, 32 KB LDS cap, no coop-matrices, compute-ring timeouts on long submissions, no resizable BAR (keep transfers batchy).

---

## 8. Sources

- Mesa RADV docs (gfx803 support, Vulkan 1.4, ACO, fp16 gating): docs.mesa3d.org/drivers/radv.html · docs.mesa3d.org/envvars.html
- Wikipedia: Radeon RX 500 series · Graphics Core Next (wave64, LDS, ACE counts, GCN4=GCN3 ISA)
- PC Perspective RX 480 review (36 CU, 4 ACEs + 2 HWS, native fp16) · AnandTech RX 480 preview
- llama.cpp issues #23311 (Polaris device log), #25985 (coopmat fails on gfx8), #26163 (LDS gating), PR #26813 (RX 580 ~40 tok/s 4B, ring-timeout workaround)
- Ollama issue #2453 + rocm_sdk_builder #173 (ROCm userspace dropped gfx803; 5.4.x last workable)
- vkblas (lakitu12) — gfx803 Vulkan GEMM 2.54 TFLOPS; exynos-gemm-vulkan-bench (minsuk22) — tuning methodology
- vkFFT (DTolm) — runtime codegen/autotuning; Kompute; Erkaman/vulkan_minimal_compute; SaschaWillems/Vulkan; Khronos Vulkan guide + Vulkan-Samples; vkguide.dev; vk-bootstrap
- LLVM AMDGPUUsage (gfx803 = amdgpu8.03, still supported in current LLVM)
