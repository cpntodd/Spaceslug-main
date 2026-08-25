"""Fixed Tiny profile display for the Phase 1 desktop shell.

The desktop shell is deliberately restricted to the fixed Spaceslug-Tiny
profile.  It reuses the existing :func:`spaceslug.model_profiles.resolve_profile`
so the displayed numbers are the same numbers the CLI/TUI report.
"""

from __future__ import annotations

from typing import Any

from ..model_profiles import resolve_profile

TINY_PROFILE_ID = "Spaceslug-Tiny"


def fixed_tiny_profile() -> dict[str, Any]:
    return resolve_profile(TINY_PROFILE_ID)


def fixed_tiny_profile_lines() -> list[str]:
    profile = fixed_tiny_profile()
    return [
        "Spaceslug-Tiny (fixed MVP profile)",
        f"target_parameters: {profile['target_parameters']}",
        f"hidden_size: {profile['hidden_size']}",
        f"layers: {profile['layers']}",
        f"attention_heads: {profile['attention_heads']}",
        f"context_length: {profile['context_length']}",
        f"training_mode: {profile['training_mode']}",
        f"dtype: {profile['dtype']}",
        f"lora_rank: {profile['lora_rank']}",
        f"estimated_parameters: {profile['estimated_parameters']}",
    ]
