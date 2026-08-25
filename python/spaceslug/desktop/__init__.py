"""Native Tkinter/ttk Phase 1 desktop application shell.

The headless-testable pieces (controller, profile, runtime placement, and loss
graph) are imported eagerly; the Tk application is imported lazily so importing
this package never requires a display.
"""

from __future__ import annotations

from typing import Any

from .controller import CAPABILITY_BOUNDARIES, TAB_NAMES, DesktopController
from .loss_graph import LossWormGraph
from .profile import TINY_PROFILE_ID, fixed_tiny_profile, fixed_tiny_profile_lines
from .runtime import (
    RUNTIME_ROOT,
    RuntimePlacement,
    backend_runtime_probe,
    default_runtime_probe,
    resolve_placement,
)

__all__ = [
    "CAPABILITY_BOUNDARIES",
    "DesktopApp",
    "DesktopController",
    "LossWormGraph",
    "RUNTIME_ROOT",
    "RuntimePlacement",
    "TAB_NAMES",
    "TINY_PROFILE_ID",
    "backend_runtime_probe",
    "default_runtime_probe",
    "fixed_tiny_profile",
    "fixed_tiny_profile_lines",
    "resolve_placement",
    "run_desktop",
]


def __getattr__(name: str) -> Any:
    if name == "DesktopApp":
        from .app import DesktopApp

        return DesktopApp
    if name == "run_desktop":
        from .app import run_desktop

        return run_desktop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
