"""Capability metadata for the native standalone FP32 base-training subsets.

The runtime exposes three deliberately narrow FP32 gradient/SGD ABIs: LM-head,
output projection, and combined QKV projection training.  Each consumes
caller-supplied activations and upstream gradients; none is wired into the
Tiny forward graph or dataset training, and together they are not full base
training.
"""

from __future__ import annotations

from typing import Any


_NATIVE_FP32_BASE_SUBSETS = ("lm_head", "output_projection", "combined_qkv")
_UNSUPPORTED_FULL_BASE_GROUPS = (
    "embeddings",
    "attention_qkv",
    "attention_output",
    "feed_forward",
    "normalization",
    "positional_embeddings",
)


def native_fp32_lm_head_capability() -> dict[str, Any]:
    """Return the truthful native FP32 base-training subset contract."""
    return {
        "operation": "native_fp32_standalone_base_training_subsets",
        "status": "implemented-standalone",
        "production_status": "bounded",
        "native_binding": True,
        "standalone_api": True,
        "graph_owned_lm_head": False,
        "forward_integration": False,
        "tiny_graph_integration": False,
        "activation_source": "caller-supplied",
        "dataset_integration": False,
        "dtype": "fp32",
        "optimizer": "sgd",
        "trainable_parameter_groups": list(_NATIVE_FP32_BASE_SUBSETS),
        "implemented_subsets": list(_NATIVE_FP32_BASE_SUBSETS),
        "subset_contract": {
            "lm_head": "caller-supplied projected activations and dlogits",
            "output_projection": "caller-supplied activations and upstream gradients",
            "combined_qkv": "caller-supplied states and dquery/dkey/dvalue gradients",
        },
        "frozen_parameter_groups": list(_UNSUPPORTED_FULL_BASE_GROUPS),
        "full_base_training": False,
        "full_base_training_status": "unsupported",
        "unsupported_full_base_groups": list(_UNSUPPORTED_FULL_BASE_GROUPS),
        "dataset_training": False,
        "cpu_reference": True,
        "boundary": (
            "Standalone native FP32 LM-head, output-projection, and combined-QKV "
            "gradient/SGD subsets are implemented. Each consumes caller-supplied "
            "tensors, is not integrated into the Tiny graph or dataset training, "
            "and these subsets do not constitute full-base training."
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
