"""Dataset-backed deterministic training and checkpointing for dense Spaceslug-Tiny."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .dataset import DatasetBundle
from .tiny_dense_model import TinyDenseCausalModel
from .tiny_training import sequences_from_records
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class DenseTinyTrainingConfig:
    steps: int
    learning_rate: float
    hidden_size: int = 16
    seed: int = 0

    def validate(self) -> None:
        if self.steps <= 0 or self.learning_rate <= 0.0 or self.hidden_size <= 0:
            raise ValueError("steps, learning_rate, and hidden_size must be positive")


def train_dense_tiny(bundle: DatasetBundle, config: DenseTinyTrainingConfig, *, tokenizer: ByteTokenizer,
                     model: TinyDenseCausalModel | None = None, prior_steps: int = 0) -> tuple[TinyDenseCausalModel, dict]:
    config.validate()
    sequences = sequences_from_records(bundle.records("train"), tokenizer)
    model = model or TinyDenseCausalModel.create(tokenizer.vocab_size, config.hidden_size)
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabularies differ")
    before, _ = model.loss_and_gradients(sequences)
    for _ in range(config.steps):
        model.train_step(sequences, config.learning_rate)
    after, _ = model.loss_and_gradients(sequences)
    validation = bundle.records("validation")
    validation_loss = None
    if validation:
        validation_loss, _ = model.loss_and_gradients(sequences_from_records(validation, tokenizer))
    return model, {"initial_train_loss": before, "final_train_loss": after, "validation_loss": validation_loss,
                   "optimizer_step": prior_steps + config.steps, "dataset_revision": bundle.manifest["revision"],
                   "tokenizer_fingerprint": tokenizer.fingerprint(), "config": asdict(config)}


def save_dense_checkpoint(path: str | Path, model: TinyDenseCausalModel, metrics: dict) -> None:
    payload = {"format": "spaceslug-tiny-dense-training-checkpoint", "schema_version": 1,
               "model": {"vocab_size": model.vocab_size, "hidden_size": model.hidden_size,
                         "embedding": model.embedding, "output": model.output, "output_bias": model.output_bias},
               "metrics": metrics}
    Path(path).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def load_dense_checkpoint(path: str | Path) -> tuple[TinyDenseCausalModel, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "spaceslug-tiny-dense-training-checkpoint" or payload.get("schema_version") != 1:
        raise ValueError("unsupported dense Tiny checkpoint")
    model = payload["model"]
    return TinyDenseCausalModel(model["vocab_size"], model["hidden_size"], model["embedding"], model["output"], model["output_bias"]), payload["metrics"]


def write_experiment_record(output: str | Path, experiment_id: str, metrics: dict, *, code_revision: str = "unrecorded") -> Path:
    """Write the schema-compatible immutable metadata record for a CPU Tiny run."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    record = {"experiment_id": experiment_id, "parent_model": "Spaceslug-Tiny-dense-cpu-reference",
              "code_revision": code_revision, "dataset_revision": metrics["dataset_revision"],
              "config": metrics["config"], "hardware": {"backend": "cpu-reference"},
              "budget": {"seconds": 1, "steps": metrics["optimizer_step"]}, "metrics": metrics,
              "resource": {"precision": "float64"}, "status": "kept"}
    path = root / "experiment.json"
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path
