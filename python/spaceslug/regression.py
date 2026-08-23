"""Bounded multi-run regression comparison for Tiny reports."""

from __future__ import annotations


def compare_reports(reports: list[dict], *, max_loss_increase: float = 0.0) -> dict:
    if not reports:
        raise ValueError("at least one report is required")
    losses = [report["metrics"]["test_loss"] for report in reports]
    baseline = losses[0]
    maximum = max(losses)
    return {"runs": len(reports), "baseline_test_loss": baseline, "max_test_loss": maximum,
            "max_allowed_test_loss": baseline + max_loss_increase,
            "pass": maximum <= baseline + max_loss_increase}
