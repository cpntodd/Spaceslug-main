"""Bounded multi-run regression comparison for Tiny reports."""

from __future__ import annotations


def compare_reports(reports: list[dict], *, max_loss_increase: float = 0.0, max_accuracy_drop: float = 0.0) -> dict:
    if not reports:
        raise ValueError("at least one report is required")
    losses = [report["metrics"]["test_loss"] for report in reports]
    accuracies = [report["metrics"].get("test_token_accuracy") for report in reports]
    baseline = losses[0]
    maximum = max(losses)
    accuracy_baseline = accuracies[0]
    minimum_accuracy = min(accuracies) if all(value is not None for value in accuracies) else None
    accuracy_pass = minimum_accuracy is None or minimum_accuracy >= accuracy_baseline - max_accuracy_drop
    return {"runs": len(reports), "baseline_test_loss": baseline, "max_test_loss": maximum,
            "max_allowed_test_loss": baseline + max_loss_increase, "baseline_test_token_accuracy": accuracy_baseline,
            "min_test_token_accuracy": minimum_accuracy, "max_allowed_accuracy_drop": max_accuracy_drop,
            "pass": maximum <= baseline + max_loss_increase and accuracy_pass}
