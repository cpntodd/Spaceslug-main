"""Acceptance gate proving the Tiny CPU training path before GPU enablement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from .dataset import verify_bundle
from .projected_attention_training import ProjectedAttentionConfig, run_projected_training
from .tokenizer import default_tokenizer


@dataclass(frozen=True)
class CpuVerification:
    passed: bool
    backend: str
    dataset_revision: str
    initial_loss: float
    final_loss: float
    artifact_revision: str
    inference_token: int
    reason: str


def verify_cpu_training(bundle_path: str | Path, *, steps: int = 2, learning_rate: float = 0.1) -> CpuVerification:
    bundle = bundle_path if hasattr(bundle_path, "manifest") else verify_bundle(bundle_path)
    with tempfile.TemporaryDirectory(prefix="spaceslug-cpu-gate-") as directory:
        root = Path(directory)
        result = run_projected_training(bundle, ProjectedAttentionConfig(steps=steps, learning_rate=learning_rate, batch_size=2), tokenizer=default_tokenizer(), checkpoint=root / "checkpoint.json", artifact=root / "artifact.spaceslug", experiment=root / "experiment", code_revision="cpu-verification")
    metrics = result["metrics"]
    passed = metrics["final_train_loss"] <= metrics["initial_train_loss"] and metrics["inference"]["next_token"] is not None and result["artifact_revision"]
    return CpuVerification(bool(passed), "cpu-reference", bundle.manifest["revision"], metrics["initial_train_loss"], metrics["final_train_loss"], result["artifact_revision"], metrics["inference"]["next_token"], "loss-reduced-and-artifact-loaded" if passed else "cpu-acceptance-failed")
