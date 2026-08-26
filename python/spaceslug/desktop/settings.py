"""Persisted desktop workspace-path settings (no Tk, no third-party deps).

The desktop shell keeps one small JSON file that records the user's workspace
layout — the workspace root and the dataset / checkpoint / artifact / experiment
/ temp directories the Phase 1 workflows write into.  This module is plain
Python so the controller can load and persist it headlessly, and tests can point
it at an isolated config file.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_WORKSPACE_ROOT = Path.home() / ".spaceslug" / "workspace"
DEFAULT_CONFIG_PATH = Path.home() / ".spaceslug" / "desktop.json"

# The five directories derived from the workspace root by default, in the order
# the UI and JSON expose them.
_PATH_FIELDS = (
    "dataset_dir",
    "checkpoint_dir",
    "artifact_dir",
    "experiment_dir",
    "temp_dir",
)

# Subdirectory name under the workspace root for each derived path.
_SUBDIR_NAMES: dict[str, str] = {
    "dataset_dir": "datasets",
    "checkpoint_dir": "checkpoints",
    "artifact_dir": "artifacts",
    "experiment_dir": "experiments",
    "temp_dir": "tmp",
}


def coerce_path(value: Any) -> Path:
    """Resolve *value* to an absolute path, rejecting empty/None input."""
    if value is None:
        raise ValueError("path must not be empty")
    text = str(value).strip()
    if not text:
        raise ValueError("path must not be empty")
    return Path(text).expanduser().resolve()


def _coerce_optional(value: Any) -> Path | None:
    """Tolerantly coerce a persisted value, returning None for blank/bad input."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Path(text).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


@dataclass(frozen=True)
class WorkspacePaths:
    """The six persisted directories the desktop writes into."""

    workspace_root: Path
    dataset_dir: Path
    checkpoint_dir: Path
    artifact_dir: Path
    experiment_dir: Path
    temp_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "workspace_root": str(self.workspace_root),
            "dataset_dir": str(self.dataset_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "artifact_dir": str(self.artifact_dir),
            "experiment_dir": str(self.experiment_dir),
            "temp_dir": str(self.temp_dir),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkspacePaths:
        """Build paths from a persisted mapping, deriving any missing subdir."""
        root = _coerce_optional(data.get("workspace_root")) or DEFAULT_WORKSPACE_ROOT
        derived = default_workspace_paths(root)
        values = dict(derived.to_dict())
        for field in _PATH_FIELDS:
            resolved = _coerce_optional(data.get(field))
            if resolved is not None:
                values[field] = str(resolved)
        values["workspace_root"] = str(root)
        return cls(
            workspace_root=root,
            dataset_dir=Path(values["dataset_dir"]),
            checkpoint_dir=Path(values["checkpoint_dir"]),
            artifact_dir=Path(values["artifact_dir"]),
            experiment_dir=Path(values["experiment_dir"]),
            temp_dir=Path(values["temp_dir"]),
        )


def default_workspace_paths(workspace_root: str | Path | None = None) -> WorkspacePaths:
    """Return the standard layout for a workspace root (or the user default)."""
    root = coerce_path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    return WorkspacePaths(
        workspace_root=root,
        dataset_dir=root / _SUBDIR_NAMES["dataset_dir"],
        checkpoint_dir=root / _SUBDIR_NAMES["checkpoint_dir"],
        artifact_dir=root / _SUBDIR_NAMES["artifact_dir"],
        experiment_dir=root / _SUBDIR_NAMES["experiment_dir"],
        temp_dir=root / _SUBDIR_NAMES["temp_dir"],
    )


def load_workspace_paths(config_path: str | Path | None = None) -> WorkspacePaths:
    """Load persisted paths, falling back to defaults on any missing/bad file."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            return WorkspacePaths.from_dict(data)
    return default_workspace_paths()


def save_workspace_paths(paths: WorkspacePaths, config_path: str | Path | None = None) -> Path:
    """Write *paths* to the JSON config file and return the written path."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(paths.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
