"""Testable desktop shell controller.

The controller owns all desktop state and logic and never imports or constructs
a Tk root, so it can be exercised headlessly.  Rendering and Tk wiring live in
:mod:`spaceslug.desktop.app`, which consumes a controller instance.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .loss_graph import LossWormGraph
from .profile import fixed_tiny_profile
from .runtime import RuntimePlacement, default_runtime_probe, resolve_placement

TAB_NAMES = ("Home", "Datasets", "Build & Train", "Interact", "Local API")

# Explicit capability boundaries shown on each placeholder tab.  Phase 1 is the
# GUI shell only: training/chat/API are not wired and their boundaries are
# surfaced rather than hidden.
CAPABILITY_BOUNDARIES: dict[str, str] = {
    "training": (
        "Training is not wired in this Phase 1 GUI shell. Authorized boundaries: "
        "fixed Tiny profiles only (tiny_h64_v259_vp320_t128_rank4 / "
        "tiny_h64_v259_vp320_t128_rank8); H=64, V=259, Vp=320, T=128; LoRA rank 4 "
        "or 8; frozen base weights; fp32 arithmetic. QKV AdamW, positions/FFN/"
        "normalization, and dataset-integrated all-parameter training remain "
        "unsupported. CPU remains authoritative outside the bounded GPU path."
    ),
    "chat": (
        "Interact is a placeholder. Streaming chat requires the local service, "
        "which is not implemented in Phase 1. CPU inference and the validated "
        "attention gate exist in the CLI/TUI but are not exposed here as a "
        "streaming chat."
    ),
    "api": (
        "Local API is a placeholder. No server or API implementation is provided "
        "in Phase 1. A local-only service is planned but is out of scope for "
        "this shell."
    ),
}


class DesktopController:
    """State and actions for the desktop shell; no Tk imports."""

    def __init__(
        self,
        runtime_probe: Callable[[], dict[str, Any]] | None = None,
        *,
        history: Sequence[float] | None = None,
    ) -> None:
        self.runtime_probe = runtime_probe or default_runtime_probe
        self.active_tab = "Home"
        self.profile = fixed_tiny_profile()
        self.loss = LossWormGraph(history)
        # Static until refresh_runtime() runs; keeps construction side-effect free.
        self.placement: RuntimePlacement = RuntimePlacement(
            mode="cpu-fallback",
            device=None,
            software_vulkan=False,
            gpu_primary=False,
            cpu_fallback=True,
            reason="runtime not probed yet",
        )
        self.dataset_file = ""
        self.dataset_url = ""
        self.dataset_search = ""
        self.training_steps = 10
        self.chat_prompt = ""
        self.api_address = "127.0.0.1:8123"

    # -- tabs ----------------------------------------------------------------
    def set_active_tab(self, name: str) -> str:
        if name not in TAB_NAMES:
            raise ValueError(f"unknown tab: {name}")
        self.active_tab = name
        return name

    # -- runtime / placement -------------------------------------------------
    def refresh_runtime(self) -> RuntimePlacement:
        self.placement = resolve_placement(self.runtime_probe())
        return self.placement

    def runtime_status(self) -> RuntimePlacement:
        return self.placement

    # -- loss worm -----------------------------------------------------------
    def record_loss(self, value: float) -> int:
        return self.loss.record(value)

    def clear_loss(self) -> None:
        self.loss.clear()

    # -- dataset settings placeholders --------------------------------------
    def set_dataset_file(self, value: str) -> None:
        self.dataset_file = value

    def set_dataset_url(self, value: str) -> None:
        self.dataset_url = value

    def set_dataset_search(self, value: str) -> None:
        self.dataset_search = value

    # -- training / chat / api placeholders ----------------------------------
    def set_training_steps(self, steps: int) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.training_steps = steps

    def set_chat_prompt(self, value: str) -> None:
        self.chat_prompt = value

    def set_api_address(self, value: str) -> None:
        self.api_address = value

    # -- capability boundaries ----------------------------------------------
    def capability_boundary(self, area: str) -> str:
        return CAPABILITY_BOUNDARIES[area]

    # -- pure read model -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return a render-ready view without probing, training, or model work."""
        return {
            "active_tab": self.active_tab,
            "profile": dict(self.profile),
            "placement": self.placement.to_dict(),
            "loss_history": self.loss.history,
            "dataset_file": self.dataset_file,
            "dataset_url": self.dataset_url,
            "dataset_search": self.dataset_search,
            "training_steps": self.training_steps,
            "chat_prompt": self.chat_prompt,
            "api_address": self.api_address,
            "capabilities": {
                "training": self.capability_boundary("training"),
                "chat": self.capability_boundary("chat"),
                "api": self.capability_boundary("api"),
            },
        }
