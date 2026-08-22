# Architectural decisions

## D001 — Separate products

**Decision:** Keep `vulkan-runtime` as Spaceslug, the RX580/gfx803 backend product, and create `spaceslug-main` as the host engine and model laboratory.

**Reason:** Backend stability, model orchestration, training, and GUI concerns have different release and validation cycles.

## D002 — Canonical IR

**Decision:** Cactus graphs and Colibri containers are import sources; neither is the permanent Spaceslug-main contract.

**Reason:** Their formats and execution assumptions are coupled to their current implementations and are not sufficient for training, adapters, or strict provenance.

## D003 — Tiny-first training

**Decision:** Validate full training on Spaceslug-Tiny before targeting Spaceslug-0.5B.

**Reason:** Tiny models allow rapid CPU/Vulkan gradient, optimizer, checkpoint, and GUI validation under RX580 limits.

## D004 — Dense before MoE

**Decision:** Make the first trainable model dense. Add MoE as an inference subsystem before attempting MoE training.

**Reason:** Dense training is substantially easier to validate; MoE adds routing, capacity, expert placement, and gradient complexity.

## D005 — Explicit fallback

**Decision:** CPU fallback is allowed but must be visible in the execution plan and metrics.

**Reason:** Silent fallback caused misleading performance conclusions in prior integration work and makes semantic debugging difficult.

## D006 — Bounded self-improvement

**Decision:** Automated improvement operates in isolated worktrees with fixed budgets, fixed evaluation, and human approval.

**Reason:** The system must improve models and code without being able to redefine its own success criteria or bypass safety gates.

## D007 — Inspectable dataset bundles

**Decision:** Use a proposed `.dts` Spaceslug Dataset Bundle with lossless canonical UTF-8 records, optional lossless Zstandard compression, separately identified derived training data, per-file checksums, and explicit lineage.

**Reason:** Training and evaluation claims must be tied to exact, reproducible inputs. An inspectable container supports independent verification without coupling users to the engine implementation.

## D008 — Evidence-based development log

**Decision:** Maintain chronological devlog entries alongside machine-readable manifests and experiment records. Entries must distinguish proposed, experimental, and validated work and must include reproducible commands and measured evidence when making performance or loss claims.

**Reason:** Other users need to audit progress and reproduce comparisons rather than rely on informal status descriptions.
