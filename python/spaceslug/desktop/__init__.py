"""Native Tkinter/ttk Phase 1 desktop application shell.

The headless-testable pieces (controller, profile, runtime placement, and loss
graph) are imported eagerly; the Tk application is imported lazily so importing
this package never requires a display.
"""

from __future__ import annotations

from typing import Any

from .capabilities import native_training_groups, native_training_groups_text
from .controller import (
    CAPABILITY_BOUNDARIES,
    TAB_NAMES,
    DesktopController,
    detect_code_revision,
    parse_api_address,
)
from .loss_graph import LossWormGraph
from .profile import (
    TINY_PROFILE_ID,
    TINY_RANKS,
    fixed_tiny_profile,
    fixed_tiny_profile_lines,
    tiny_profile_id,
)
from .runtime import (
    RUNTIME_ROOT,
    RuntimePlacement,
    TrainingPlacement,
    backend_runtime_probe,
    default_runtime_probe,
    resolve_placement,
    resolve_training_placement,
)
from .settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_WORKSPACE_ROOT,
    WorkspacePaths,
    default_workspace_paths,
    load_workspace_paths,
    save_workspace_paths,
)

__all__ = [
    "CAPABILITY_BOUNDARIES",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_WORKSPACE_ROOT",
    "RUNTIME_ROOT",
    "TAB_NAMES",
    "TINY_PROFILE_ID",
    "TINY_RANKS",
    "DesktopApp",
    "DesktopController",
    "LossWormGraph",
    "RuntimePlacement",
    "TrainingPlacement",
    "WorkspacePaths",
    "backend_runtime_probe",
    "default_runtime_probe",
    "default_workspace_paths",
    "detect_code_revision",
    "fixed_tiny_profile",
    "fixed_tiny_profile_lines",
    "load_workspace_paths",
    "native_training_groups",
    "native_training_groups_text",
    "parse_api_address",
    "resolve_placement",
    "resolve_training_placement",
    "run_desktop",
    "save_workspace_paths",
    "tiny_profile_id",
]


def __getattr__(name: str) -> Any:
    if name == "DesktopApp":
        from .app import DesktopApp

        return DesktopApp
    if name == "run_desktop":
        from .app import run_desktop

        return run_desktop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
