"""Separate reproducibility and quality status for Tiny experiment reports."""

from __future__ import annotations


def classify_report(metrics: dict) -> dict[str, str | bool]:
    """Do not conflate a reproducible run with a useful-model quality result."""
    reproducible = bool(metrics.get("artifact_revision")) and metrics.get("inference", {}).get("next_token") is not None and metrics.get("test_loss") is not None
    accuracy = metrics.get("test_token_accuracy")
    quality = accuracy is not None and accuracy > 0.0
    return {"reproducibility": "pass" if reproducible else "fail", "quality": "pass" if quality else "not-established", "ready_for_bounded_testing": reproducible}
