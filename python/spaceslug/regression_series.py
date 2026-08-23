"""Persisted bounded regression-series records for Tiny experiments."""

from __future__ import annotations

import json
from pathlib import Path

from .regression import compare_reports


def write_regression_series(output: str | Path, series_id: str, reports: list[dict], *, max_loss_increase: float = 0.0, max_accuracy_drop: float = 0.0) -> Path:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    comparison = compare_reports(reports, max_loss_increase=max_loss_increase, max_accuracy_drop=max_accuracy_drop)
    record = {"format": "spaceslug-regression-series", "schema_version": 1, "series_id": series_id,
              "runs": [{"experiment_id": report.get("experiment_id"), "code_revision": report.get("code_revision"), "dataset_revision": report.get("dataset_revision"), "config": report.get("config"), "metrics": report.get("metrics")} for report in reports],
              "comparison": comparison, "thresholds": {"max_loss_increase": max_loss_increase, "max_accuracy_drop": max_accuracy_drop}}
    path = root / "regression.json"
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def load_regression_series(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("format") != "spaceslug-regression-series" or record.get("schema_version") != 1:
        raise ValueError("unsupported regression series")
    return record
