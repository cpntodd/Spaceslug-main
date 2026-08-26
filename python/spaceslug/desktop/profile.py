"""Fixed native Tiny profile display for the Phase 1 desktop shell.

The desktop shell is deliberately restricted to the fixed Spaceslug-Tiny
profile.  The numbers shown are the *actual* native contract — ``H=64``,
``V=259``, ``Vp=320``, ``T=128``, ranks ``4``/``8``, fp32, sgd/adamw — taken
live from :func:`spaceslug.gpu_lora_training.persistent_graph_boundary` rather
than the planning-only ``model_profiles`` estimate (which reported a fake
``1M``-parameter / multi-layer / ``512``-context shape).  The desktop never
exposes arbitrary architecture configuration.
"""

from __future__ import annotations

from typing import Any

from ..gpu_lora_training import persistent_graph_boundary

TINY_PROFILE_ID = "Spaceslug-Tiny"
TINY_RANKS = (4, 8)


def tiny_profile_id(rank: int) -> str:
    """Return the fixed profile id for a valid Tiny rank (``4`` or ``8``)."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank not in TINY_RANKS:
        raise ValueError(f"rank must be 4 or 8, got {rank!r}")
    return f"tiny_h64_v259_vp320_t128_rank{rank}"


def native_tiny_contract() -> dict[str, Any]:
    """Return the live native Tiny contract (the single source of truth)."""
    return persistent_graph_boundary()["contract"]


def fixed_tiny_profile() -> dict[str, Any]:
    """Return the fixed native Tiny profile as a render-ready mapping."""
    contract = native_tiny_contract()
    return {
        "model_id": TINY_PROFILE_ID,
        "hidden_size": contract["hidden"],
        "vocab_size": contract["vocab"],
        "projected_vocab_size": contract["logits_stride"],
        "sequence_capacity": contract["sequence_capacity"],
        "ranks": list(TINY_RANKS),
        "dtype": contract["dtype"],
        "optimizers": list(contract["optimizers"]),
        "base_weights": contract["base_weights"],
        "profiles": [tiny_profile_id(rank) for rank in TINY_RANKS],
    }


def fixed_tiny_profile_lines() -> list[str]:
    profile = fixed_tiny_profile()
    return [
        "Spaceslug-Tiny (fixed native profile)",
        f"hidden_size (H): {profile['hidden_size']}",
        f"vocab_size (V): {profile['vocab_size']}",
        f"projected_vocab_size (Vp): {profile['projected_vocab_size']}",
        f"sequence_capacity (T): {profile['sequence_capacity']}",
        f"ranks: {', '.join(str(rank) for rank in profile['ranks'])}",
        f"dtype: {profile['dtype']}",
        f"optimizers: {', '.join(profile['optimizers'])}",
        f"base_weights: {profile['base_weights']}",
    ]
