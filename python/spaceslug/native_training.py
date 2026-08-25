"""Native training contracts and the optional graph-owned Tiny LM-head trainer.

The standalone FP32 APIs consume caller-supplied tensors.  The integrated Tiny
API instead owns activations and the LM-head inside a persistent forward graph;
it is exposed only when its runtime symbols are present.  The two contracts are
intentionally reported separately.
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


def integrated_tiny_lm_head_capability(*, group: str = "lm_head", available: bool = False, runtime_capability: str | None = None) -> dict[str, Any]:
    """Return a graph-owned Tiny SGD contract for one parameter group."""
    if group not in {"lm_head", "output", "qkv"}:
        raise ValueError("group must be lm_head, output, or qkv")
    return {
        "operation": f"tiny_graph_owned_{group}_sgd",
        "parameter_group": group,
        "status": "available" if available else "unsupported",
        "native_binding": available,
        "integrated_graph_sgd": True,
        "graph_owned_lm_head": group == "lm_head",
        "graph_owned_parameter_group": True,
        "standalone_api": False,
        "activation_source": "graph-owned-forward-activations",
        "optimizer": "sgd",
        "adamw": False,
        "integrated_graph_adamw": False,
        "trainable_parameter_groups": [group],
        "dataset_integration": False,
        "adamw_unsupported": True,
        "adamw_return_code": -4,
        "contract": f"persistent Tiny forward graph owns activations and {group}; fixed-window token/target/mask SGD",
        "runtime_capability": runtime_capability,
    }


def integrated_tiny_lm_head_adamw_capability(*, available: bool = False, runtime_capability: str | None = None) -> dict[str, Any]:
    """Return the conditional graph-owned Tiny LM-head AdamW contract."""
    return integrated_tiny_group_adamw_capability(group="lm_head", available=available, runtime_capability=runtime_capability)


def integrated_tiny_group_adamw_capability(*, group: str, available: bool = False, runtime_capability: str | None = None) -> dict[str, Any]:
    """Return a conditional graph-owned Tiny AdamW contract for one supported group."""
    if group not in {"lm_head", "output", "qkv"}:
        raise ValueError("group must be lm_head, output, or qkv")
    return {
        "operation": f"tiny_graph_owned_{group}_adamw",
        "parameter_group": group,
        "status": "available" if available else "unsupported",
        "native_binding": available,
        "integrated_graph_adamw": available,
        "from_existing_gradients": group == "qkv",
        "graph_owned_lm_head": group == "lm_head",
        "graph_owned_parameter_group": True,
        "standalone_api": False,
        "optimizer": "adamw",
        "optimizer_state": "graph-owned-m-v-step",
        "trainable_parameter_groups": [group],
        "dataset_integration": False,
        "retained_training": False,
        "return_code_when_unavailable": -4,
        "return_code": 0 if available else -4,
        "gradient_source": "existing-qkv-gradients" if group == "qkv" else "graph-owned-forward-activations",
        "contract": f"persistent Tiny forward graph owns activations, {group}, AdamW moments, and step; host owns token windows and checkpoints" if group != "qkv" else "persistent Tiny forward graph applies AdamW to existing QKV gradients; no recompute or double update; host owns token windows and checkpoints",
        "runtime_capability": runtime_capability,
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


__all__ = ["integrated_tiny_group_adamw_capability", "integrated_tiny_lm_head_adamw_capability", "integrated_tiny_lm_head_capability", "native_fp32_lm_head_capability", "native_fp32_lm_head_training_plan"]
