"""Explicit boundary metadata for the native FP32 base-training plan.

This module is deliberately metadata-only.  The current ctypes ABI does not
expose a native base-training step, so these functions must not be used as an
implementation or as evidence that full-model training is available.
"""

from __future__ import annotations

from typing import Any


_NATIVE_FP32_LM_HEAD_GROUP = "lm_head"
_UNSUPPORTED_FULL_BASE_GROUPS = (
    "embeddings",
    "attention_qkv",
    "attention_output",
    "feed_forward",
    "normalization",
    "positional_embeddings",
)


def native_fp32_lm_head_capability() -> dict[str, Any]:
    """Return the truthful native base-training capability contract."""
    return {
        "operation": "native_fp32_lm_head_only_base_training",
        "status": "planned-not-implemented",
        "production_status": "unsupported",
        "native_binding": False,
        "dtype": "fp32",
        "trainable_parameter_groups": [_NATIVE_FP32_LM_HEAD_GROUP],
        "frozen_parameter_groups": list(_UNSUPPORTED_FULL_BASE_GROUPS),
        "full_base_training": False,
        "full_base_training_status": "unsupported",
        "unsupported_full_base_groups": list(_UNSUPPORTED_FULL_BASE_GROUPS),
        "optimizer": "not-bound",
        "dataset_training": False,
        "cpu_reference": True,
        "boundary": (
            "Plan boundary only: native ctypes FP32 LM-head forward/backward/update "
            "ABI is not implemented; CPU remains authoritative. Full-base groups "
            "are unsupported and must not be inferred from this metadata."
        ),
    }


def native_fp32_lm_head_training_plan(*, hidden_size: int, vocab_size: int) -> dict[str, Any]:
    """Describe the future LM-head-only plan without claiming execution."""
    if hidden_size <= 0 or vocab_size <= 0:
        raise ValueError("hidden_size and vocab_size must be positive")
    capability = native_fp32_lm_head_capability()
    return {
        **capability,
        "dimensions": {"hidden_size": hidden_size, "vocab_size": vocab_size},
        "steps": [
            "native_forward_frozen_backbone",
            "native_lm_head_fp32_logits",
            "native_lm_head_loss",
            "native_lm_head_backward",
            "native_lm_head_optimizer_update",
        ],
        "unsupported_steps": [
            "native_full_base_backward",
            "native_full_base_optimizer_update",
        ],
    }


__all__ = ["native_fp32_lm_head_capability", "native_fp32_lm_head_training_plan"]
