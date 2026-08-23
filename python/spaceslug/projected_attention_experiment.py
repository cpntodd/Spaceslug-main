"""Reproducible projected-attention Tiny run records."""

from __future__ import annotations

import json
from pathlib import Path


def write_projected_experiment(output: str | Path, experiment_id: str, metrics: dict, *, code_revision: str = "unrecorded") -> Path:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    record = {"experiment_id": experiment_id, "parent_model": "Spaceslug-Tiny-projected-attention",
              "code_revision": code_revision, "dataset_revision": metrics["dataset_revision"],
              "config": metrics["config"], "hardware": {"backend": "cpu-reference"},
              "budget": {"seconds": 1, "steps": metrics["optimizer_step"]}, "metrics": metrics,
              "resource": {"precision": "float64", "fallback": False}, "status": "kept"}
    path = root / "experiment.json"
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path
