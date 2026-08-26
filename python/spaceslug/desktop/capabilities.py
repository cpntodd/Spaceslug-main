"""Accurate capability statements for the Phase 1 desktop shell.

These strings are assembled from the live capability metadata already exported by
the model/training modules rather than re-stating a stale snapshot.  They reflect
the corrected Tiny boundaries (STAGED_ROADMAP.md, revision ``8474903``): graph-owned
AdamW is available for the LM-head, output projection, and combined Q/K/V, and the
two fixed Tiny profiles are not described as blanket "frozen base" (only the LoRA
adapter path freezes base weights).
"""

from __future__ import annotations

from typing import Any

from ..gpu_lora_training import (
    gpu_lora_capability,
    persistent_graph_boundary,
    persistent_tiny_capability,
)
from ..native_training import native_fp32_lm_head_capability


def structured_capabilities() -> dict[str, Any]:
    """Assemble the current capability facts from the live metadata functions."""
    lora = gpu_lora_capability()
    persistent = persistent_tiny_capability()
    standalone = native_fp32_lm_head_capability()
    return {
        "gpu_lora": {
            "status": lora["status"],
            "production_status": lora["production_status"],
            "base_weights": lora["base_weights"],  # frozen for the LoRA adapter path
            "optimizers": list(lora["optimizers"]),
            "supported_tiny_profiles": list(lora["supported_tiny_profiles"]),
            "dataset_training": lora["dataset_training"],
        },
        "persistent_tiny": {
            "status": persistent["status"],
            "production_status": persistent["production_status"],
            "optimizers": list(persistent["optimizers"]),
            "native_adamw_state_checkpoint": persistent["native_adamw_state_checkpoint"],
            "fixed_forward_tokens": persistent["fixed_forward_tokens"],
            "dataset_device_resident": persistent["dataset_device_resident"],
        },
        "native_fp32_base": {
            "status": standalone["status"],
            "trainable_parameter_groups": list(standalone["trainable_parameter_groups"]),
            "frozen_parameter_groups": list(standalone["frozen_parameter_groups"]),
            "dataset_training": standalone["dataset_training"],
        },
    }


def native_training_groups() -> dict[str, Any]:
    """Return the native parameter-group and optimizer facts for the display.

    The trainable/frozen groups come from the standalone native FP32 subset
    contract; the optimizers and base-weights come from the fixed native Tiny
    contract.  These are facts about the native boundary, not desktop toggles.
    """
    standalone = native_fp32_lm_head_capability()
    contract = persistent_graph_boundary()["contract"]
    return {
        "trainable_parameter_groups": list(standalone["trainable_parameter_groups"]),
        "frozen_parameter_groups": list(standalone["frozen_parameter_groups"]),
        "optimizers": list(contract["optimizers"]),
        "base_weights": contract["base_weights"],
    }


def native_training_groups_text() -> str:
    groups = native_training_groups()
    return (
        "trainable: " + ", ".join(groups["trainable_parameter_groups"]) + "\n"
        "frozen: " + ", ".join(groups["frozen_parameter_groups"]) + "\n"
        "optimizers: " + ", ".join(groups["optimizers"]) + "\n"
        "base_weights: " + groups["base_weights"]
    )


def training_capability_text() -> str:
    return (
        "CPU projected Tiny training is wired end-to-end in the desktop: it runs in a "
        "background worker thread with a live per-step loss callback and writes a "
        "checkpoint, a checksummed artifact, and an experiment record. CPU reference "
        "training is authoritative: deterministic dataset-backed projected-attention "
        "training (AdamW), masked causal loss, checkpoint/resume, held-out metrics, and "
        "deterministic inference.\n\n"
        "GPU training is bounded to the two fixed Tiny profiles "
        "tiny_h64_v259_vp320_t128_rank4 and tiny_h64_v259_vp320_t128_rank8 "
        "(fp32, H=64, V=259, Vp=320, T=128); only the LoRA rank differs (4 or 8). "
        "Unlisted shapes and ranks are rejected, never coerced. The LoRA adapter path "
        "trains with frozen base weights using SGD and, when the native ABI exposes it, "
        "AdamW.\n\n"
        "Graph-owned SGD is available for token embeddings, the LM-head, output "
        "projection, and combined Q/K/V for 1..128 rows. Graph-owned AdamW is available "
        "for the LM-head, output projection, and combined Q/K/V (QKV applies AdamW to "
        "existing gradients) with persistent state and unified checkpoint/resume.\n\n"
        "Positions, FFN, and normalization are unsupported; dataset-resident training "
        "remains bounded by its documented group/path contract, and the retained-command "
        "path is fixed forward/loss only. FP16 is storage-only. CPU is the transparent "
        "fallback outside the bounded GPU path; the GPU is primary only for a supported "
        "native operation selected by a job."
    )


def chat_capability_text() -> str:
    return (
        "Interact starts an embedded responder at desktop launch. It uses the configured "
        "Vulkan Tiny forward path (vulkan-radv-tiny) when the fixed V=259,H=64 native "
        "contract is available, and transparently reports a CPU Tiny reference fallback "
        "(TinyCpuEchoResponder) when GPU initialization is unavailable. Both paths are deterministic, "
        "non-generative, echo-safe single argmax next-token references; streaming is not supported."
    )


def api_capability_text() -> str:
    return (
        "The Local API is the committed loopback-only OpenAI-compatible HTTP service "
        "(spaceslug.openai_api.OpenAICompatibleServer) with start/stop controls in the "
        "desktop. It binds to 127.0.0.1/::1/localhost and serves GET /health, "
        "GET /v1/models, and POST /v1/chat/completions in OpenAI shape. Only "
        "non-streaming completions are supported; streaming requests are rejected "
        "explicitly."
    )


def datasets_capability_text() -> str:
    return (
        "Datasets uses the committed headless workspace service "
        "(spaceslug.workspace.WorkspaceService): local .txt/.md/.json/.jsonl import (including Discord chat exports), "
        "local .pdf import through the locally installed pdftotext tool, HTTP(S) URL "
        "import only after explicit approval, SearXNG-compatible search (results are "
        "never auto-fetched), optional license metadata, content-addressed "
        "SHA-256 staging, and deterministic .dts bundle creation. PDF import fails "
        "explicitly if pdftotext is unavailable, errors, or yields no text."
    )
