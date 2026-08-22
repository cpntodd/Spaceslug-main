# Implementation roadmap

## Phase 0 — contracts

Freeze the product boundary, canonical IR, tensor/layout rules, artifact schemas, capability model, correctness policy, and versioning policy.

**Gate:** design review complete.

## Phase 1 — backend API

Expose the existing Spaceslug runtime through a stable asynchronous backend interface with explicit capabilities, ownership, events, errors, and metrics. Preserve current runtime tests.

**Gate:** standalone host executes validated primitives through the API.

## Phase 1A — dataset and measurement foundation

Implement the `.dts` manifest and bundle contracts, deterministic record normalization, checksum verification, dataset lineage, protected benchmark references, experiment metadata, and the first devlog/reporting workflow.

**Gate:** a small fixture round-trips byte-for-byte, reproduces pinned token IDs, verifies checksums, and produces a reproducible statistics report.

## Phase 2 — tiny CPU engine

Implement a tiny dense transformer, CPU reference backend, tokenizer contract, teacher-forced loss, checkpointing, and CLI chat/training.

**Gate:** train, save, reload, and chat with Spaceslug-Tiny.

## Phase 3 — GUI shell

Implement the local service, dashboard, model registry, training controls, loss plots, chat, logs, and job lifecycle.

**Gate:** the complete tiny-model workflow is usable without the CLI.

## Phase 4 — Vulkan inference

Lower validated operations to Spaceslug: GEMM, CQ4, RMSNorm, attention, packing, and fused MLP. Keep fallback explicit.

**Gate:** CPU/Vulkan layer parity and correct cached generation.

## Phase 5 — useful model

Train or import Spaceslug-125M, add instruction tuning, basic tools, and coding-oriented evaluation.

**Gate:** narrow-domain chat and simple coding assistance.

## Phase 6 — Spaceslug-0.5B

Freeze the architecture and artifact, establish pretraining provenance, quantize for RX580, and support adapter import and local LoRA adaptation.

**Gate:** reproducible quantized chat model with documented limitations.

## Phase 7 — MoE subsystem

Add router, expert union, residency, hot-set/LRU, CPU/GPU placement, prefetch, and metrics. Start with inference only.

**Gate:** routing and output parity under mixed placement.

## Phase 8 — CPU LoRA training

Implement CPU gradients, AdamW, accumulation, validation, checkpoint resume, and adapter export.

**Gate:** tiny reference gradient and optimizer parity.

## Phase 9 — Vulkan LoRA training

Add Vulkan LoRA forward/backward GEMM, gradient accumulation, and optimizer updates.

**Gate:** CPU/Vulkan gradients, optimizer state, and loss trajectory agree.

## Phase 10 — bounded self-improvement

Add isolated experiments, fixed budgets, evaluation, keep/reject decisions, GUI history, and supervised coding-agent patches.

**Gate:** a model or configuration improvement is reproduced and reviewed.

## Deferred research

Full 0.5B pretraining on RX580, full-parameter training, MoE training, multi-device training, distributed expert execution, and unrestricted autonomous engine modification are later research tracks.
