# AGENTS.md — vulkan-runtime

C++20 **headless Vulkan compute runtime** for an AMD RX580 (RADV/gfx803): fp32 kernels for LLM inference (GEMM → dequant/RMSNorm/RoPE/attention). No WSI, no present, no windows.

**This repo is the `vulkan-runtime/` subdirectory**, not the workspace root. The parent contains `cactus-main/` (OUR fork of upstream Cactus with the gfx803 backend — remote `origin` → https://github.com/cpntodd/cactus) and `needle-main/` (upstream model package, **never touch** — it's unrelated to the integration).

Authoritative sources, read before changing milestone behavior:
- `.opencode/rules/vulkan.md` — per-milestone acceptance gates (M0…M7) + verify loop
- `RESEARCH.md` §6 — milestone plan, §2 — hardware analysis
- `.opencode/commands/verify.md`, `bench.md` — the acceptance workflows

## Commands

```sh
# Configure + build (debug default; release preset mirrors it)
cmake --preset debug && cmake --build build/debug

# Fresh build (do this for a trustworthy gate): 0 errors, 0 warnings expected
rm -rf build/debug && cmake --preset debug && cmake --build build/debug

# Correctness on real GPU (RADV default ICD)
ctest --test-dir build/debug --output-on-failure

# Same tests on software Vulkan (lavapipe) — the CI path
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json ctest --test-dir build/debug --output-on-failure

# Perf gate (RADV only): bench_sgemm is labeled "bench"
./build/debug/bench_sgemm          # or: ctest --test-dir build/debug -L bench
```

After any fresh configure, restore the clangd symlink (rules requirement; `rm -rf build` leaves it dangling):
```sh
ln -sf build/debug/compile_commands.json compile_commands.json
```

## Verification gotchas (all hard-earned)

- **RX580 clock ramp**: idle clock is 300–600 MHz; boost is 1366 MHz. Bench medians are garbage unless you warm up ~100+ dispatches first (M2's median read 0.97 TFLOPS at 3 warmups, 3.19 TFLOPS at 128). Always report median (never mean) of ≥7 runs.
- **Benchmarks skip on non-discrete GPUs** — all `bench_*` targets guard on `is_discrete_gpu()` (from `bench/bench_common`), so on lavapipe they exit in ~0.05 s instead of running a sweep at software speed. Perf numbers are RADV-only.
- **glslc `-O` is mandatory** (~1000× slowdown on GCN without it). After building, prove it fired: `spirv-dis build/debug/shaders/<k>.spv | grep -c "OpFAdd\|OpFMul"` must be substantial.
- VVL (`VK_LAYER_KHRONOS_validation`) is installed and enabled when enumerated — never mask validation errors.

## Dependencies / toolchain

- **Vendored headers are git-ignored.** Run `./scripts/fetch_third_party.sh` before configuring (fresh clones/builds). Versions: VMA 3.4.0, Vulkan-Hpp **pinned to tag `v1.4.309`**.
- **Do not bump Vulkan-Hpp to `main`** — it asserts `VK_HEADER_VERSION ==` the installed SDK (309). A bump means bumping `VULKAN_HPP_TAG` in `scripts/fetch_third_party.sh` to match the SDK, then checking the static_assert.
- VMA macros are configured once in `src/core/vk_setup.h` (`VMA_STATIC_VULKAN_FUNCTIONS 0`, `VMA_DYNAMIC_VULKAN_FUNCTIONS 1`) and must stay identical in every TU that includes it. `VMA_IMPLEMENTATION` lives only in `src/core/vk_setup.cpp`.
- Shaders compile at build time via `cmake/ShaderToolchain.cmake` — new kernels must use `add_shader(vulkan_runtime_shaders shaders/<k>.comp)` + `embed_shaders(...)`, then load at runtime via `vulkan_runtime::shaders::get("<k>.spv")`. Never load shaders from disk.
- `-Wall -Wextra -Wpedantic` always on; `-Werror` is opt-in (`VR_WARNINGS_AS_ERRORS`). Milestone gate expects 0 warnings.

## gfx803 hardware rules (non-negotiable)

- Workgroup sizes must be multiples of **64** (wave64); 256 threads = 4 waves.
- **LDS ≤ 32 KB/workgroup** hard cap (a 64×64 fp32 tile = 16 KB).
- LDS row stride ≡ **2 (mod 32)** floats (e.g. 66 for 64-wide rows) to avoid bank conflicts.
- **fp32 math only** — no fp16/bf16 compute, no coop matrices (hardware lacks them); fp16/bf16 are storage-only.

## C++ / Vulkan traps (verified at compile/runtime)

- `vk::Device::createComputePipeline` has **no single-arg throwing overload** — it returns `vk::ResultValue<vk::Pipeline>`; check `.result` then read `.value`.
- VMA 3.4 aborts at `vmaMapMemory` on AUTO-usage allocations without a `VMA_ALLOCATION_CREATE_HOST_ACCESS_*` flag — staging buffers need `HOST_ACCESS_RANDOM_BIT` (or similar).
- Aggregate-initializing raw `Vk*` structs with one member (`VkBufferCreateInfo x{VK_STRUCTURE_TYPE_...}`) fires 8× `-Wmissing-field-initializers` — use `{}` + explicit `sType`.
- Prefer `vk::` C++ wrappers over raw `Vk*` structs where they exist; keep destruction in reverse-creation order.

## Conventions

- **Tests are self-contained per kernel**: each `tests/test_*.cpp` duplicates its small buffer/descriptor/pipeline helpers and reuses only `create_context()` from `src/core/vk_setup.h` (instance + device + VMA). Don't extract shared plumbing without a milestone reason.
- Every test compares against a **CPU reference** (double precision) and must pass on **both** RADV and lavapipe.
- Directory roles: `src/core` setup · `src/mem` VMA pools · `src/exec` ring engine (M5: N-slot ring, one timeline semaphore per queue, `ExecEngine::submit(record, queueIndex, waitValues)`) · `src/cactus` cactus-compat CPU lib + clang bridge · `shaders/` GLSL · `bench/` perf · `tests/` correctness.
- **M5 engine semantics worth knowing**: timeline value namespace is a single global monotonic counter, but there is ONE timeline semaphore per queue (cross-queue signals from a shared semaphore violate ordering — the M5b deadlock). Slot reuse host-waits on the slot's own last value; `waitValues` are device-side waits for other queues' work. `vkQueueWaitIdle` is banned inside engine internals.
- Bench executables are built with `-O2` (their host-side CPU references are unusable at -O0); test executables stay at the preset build type.
- Formatting: `.clang-format` (LLVM base, 4-space indent, 120 col, pointers left).
- git (this repo): `main`, remote `origin` → https://github.com/cpntodd/Spaceslug.git. Commit per milestone.
- Routing: shader/kernel/CMake work → `vulkan-pro` agent (per rules); load the `vulkan-gfx803` skill for hardware gotchas.

## M6 — cactus gfx803 integration (hard-earned; read before touching)

- **Kernel inventory**: fp32 set (vector_add, sgemm + 3 variants, transpose, rmsnorm, rope, q4gemm Q4_0, attention flash) + **fp16-storage set** for cactus (cast, unary 13-ops, rmsnorm_f16, sgemm_f16, **sgemm_f16_am** arbitrary-M, kvquantize int8, attention_f16, attention_i8kv, transform_act + dequant_w + embed_cq4 for Cactus-Quants CQ4, fp16pack bridging). Each fp16 kernel is tested against the matching `cactus_x86` CPU function (via the clang bridge); see `docs/TESTING.md` for the current generated CTest inventory.
- **M7 packing (current storage convention)**: the CQ4 hot path uses **packed fp16 = 2 values per uint32** (element i → word i>>1, even idx = low 16 bits) — byte-identical to a little-endian host `__fp16` array, so no expand/shrink for those buffers. Legacy word-per-value (1 per uint32) remains on rmsnorm_f16/attention_*/kvquantize/embed_cq4. `fp16pack.comp` modes: 0 u16→u32, 1 u32→u16, 2 i8→i32 sign-extend, 3 i32→i8, 4 u8→u32 **zero-extend** (CQ4 packed_indices!), 5 u32→u8.
- **Cactus C-API is C++ linkage, NOT `extern "C"`** — `cactus_x86` symbols must be mangled to link (the M6b-2 "every symbol undefined" bug). `__fp16` only compiles under **clang++-21** (gcc rejects it); the cactus build (build.sh, compile.py, runtime.py) is switched to clang++-21.
- **Cactus quant weights** (Cactus-Quants, ≠ our Q4_0): LSB-first bit packing, INTERLEAVED_4ROW panel addressing, FWHT-128 activation transform, signs/permutation/rotation paths. **Interleaved weights store norms PANEL-MAJOR** `[(n/4*ng+g)*4 + (n&3)]` — proven from the writer (python/cactus/convert/quantization/cq.py:366) and the ARM reference (matmul.cpp:1493); the GPU dequant_w mode 1 AND cactus_x86 both read panel-major now (M7 fix — the x86 ref was row-major for all modes, silently wrong for ng>1 interleaved).
- **The M7 chain is**: transform_act (packed A′ out, no cast) → dequant_w (cached by `packed_indices`, packed W′) → sgemm_f16_am (packed). bench_cq4gemm: **1.19 TFLOPS @ 1024³**.
- **Backend traps** (all real bugs, all fixed): a `Step` must OWN its push constants (static struct aliasing corrupted every op after the first); `acquire_reg` clamps `n≥1` (empty KV cache → VK_NULL_HANDLE → nullDescriptor VUID); cross-submit GPU→GPU visibility needs timeline `waitValues`; host buffers need `HOST_ACCESS_RANDOM_BIT` + flush/invalidate; the M7b flush-cadence batching (48-op cadence, drain at cadence/sync/end) is PROVEN safe (CACTUS_EAGER_SYNC=1 gave identical results).
- **`embed_cq4` has NO interleaved path** — needle's `token_embeddings` is orthogonal+interleaved, so embedding stays on CPU fallback by design (also the tied lm_head via the MATMUL guard: `ORTHOGONAL && INTERLEAVED_4ROW → false`).
- **Ops on CPU fallback** (documented, by design): binary fp16 elementwise, rope, softmax, layernorm, kv-cache-append, gather, GELU, non-pretransposed fp16 GEMM, attention prefill/sliding/sink paths.
- **Default backend is now gfx803 when available** — a pure-CPU cactus run needs `--backend cpu`.
- **Acceptance (M7)**: `cactus-main/.venv/bin/cactus run /tmp/needle_bundle --prompt "hi" --max-new-tokens 8 --backend gfx803` → GPU-active (127 ops/decode-step), deterministic, VVL zero, ~2.9–3.3 tok/s vs CPU ~9.9 s/slow-path. **Documented caveat (accepted, Option 1)**: CPU output `________` vs gfx803 `<tool_call>_______` — one borderline argmax token at position 1, fp16-rounding-scale, stable and non-growing (verified to 16 tokens), both degenerate tool-call artifacts on this toy prompt. Not a logic error; training is immune to argmax boundaries (uses probabilities, not picks). (Bundle: `hf_hub_download(repo_id="Cactus-Compute/needle", filename="needle-cq4.zip")` → unzip → pass dir.)
