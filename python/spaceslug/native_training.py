"""Capability metadata for the native FP32 LM-head training boundary.

The runtime exposes a standalone LM-head gradient/SGD ABI.  It consumes
caller-supplied projected Tiny activations and dlogits; it is not wired into
the Tiny forward graph or dataset training, and it does not train the
backbone.
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
        "status": "implemented-standalone",
        "production_status": "bounded",
        "native_binding": True,
        "forward_integration": False,
        "activation_source": "caller-supplied_projected_tiny_activations",
        "dataset_integration": False,
        "dtype": "fp32",
        "trainable_parameter_groups": [_NATIVE_FP32_LM_HEAD_GROUP],
        "frozen_parameter_groups": list(_UNSUPPORTED_FULL_BASE_GROUPS),
        "full_base_training": False,
        "full_base_training_status": "unsupported",
        "unsupported_full_base_groups": list(_UNSUPPORTED_FULL_BASE_GROUPS),
        "optimizer": "sgd",
        "dataset_training": False,
        "cpu_reference": True,
        "boundary": (
            "Standalone native FP32 LM-head gradient/SGD ABI is implemented, but "
            "it consumes caller-supplied projected activations and dlogits; it is "
            "not integrated into Tiny forward activations or dataset training. "
            "Full-base groups remain unsupported."
        ),
    }


def native_fp32_lm_head_training_plan(*, hidden_size: int, vocab_size: int) -> dict[str, Any]:
    """Describe the standalone LM-head-only execution contract."""
    if hidden_size <= 0 or vocab_size <= 0:
        raise ValueError("hidden_size and vocab_size must be positive")
    capability = native_fp32_lm_head_capability()
    return {
        **capability,
        "dimensions": {"hidden_size": hidden_size, "vocab_size": vocab_size},
        "steps": [
            "caller_supplied_projected_activations",
            "caller_supplied_dlogits",
            "native_lm_head_backward",
            "native_lm_head_optimizer_update",
        ],
        "unsupported_steps": [
            "native_full_base_backward",
            "native_full_base_optimizer_update",
        ],
    }


__all__ = ["native_fp32_lm_head_capability", "native_fp32_lm_head_training_plan"]
