"""Reproducible dataset-backed training workflow for Spaceslug-Tiny."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from .dataset import DatasetBundle
from .tiny_model import TinyBigramModel
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class TinyTrainingConfig:
    steps: int
    learning_rate: float
    weight_decay: float = 0.0
    seed: int = 0

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must not be negative")


def sequences_from_records(records: Iterable[dict], tokenizer: ByteTokenizer) -> list[list[int]]:
    sequences: list[list[int]] = []
    for record in records:
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("every training record requires a string text field")
        sequence = tokenizer.encode(text)
        if len(sequence) >= 2:
            sequences.append(sequence)
    if not sequences:
        raise ValueError("training records contain no next-token pairs")
    return sequences


def train_tiny(
    bundle: DatasetBundle,
    config: TinyTrainingConfig,
    *,
    tokenizer: ByteTokenizer,
    model: TinyBigramModel | None = None,
    optimizer_state: dict | None = None,
) -> tuple[TinyBigramModel, dict, dict]:
    """Train on the canonical train split and return model, optimizer, metrics."""
    config.validate()
    sequences = sequences_from_records(bundle.records("train"), tokenizer)
    model = model or TinyBigramModel.create(tokenizer.vocab_size)
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabularies differ")
    state = optimizer_state if optimizer_state is not None else {}
    initial_loss, _ = model.loss_and_gradients(sequences)
    for _ in range(config.steps):
        model.train_step(sequences, config.learning_rate, weight_decay=config.weight_decay, optimizer_state=state)
    final_loss, _ = model.loss_and_gradients(sequences)
    validation_records = bundle.records("validation")
    validation_loss = None
    if validation_records:
        validation_loss, _ = model.loss_and_gradients(sequences_from_records(validation_records, tokenizer))
    metrics = {
        "initial_train_loss": initial_loss, "final_train_loss": final_loss,
        "validation_loss": validation_loss, "optimizer_step": state["step"],
        "train_sequences": len(sequences), "dataset_revision": bundle.manifest["revision"],
        "tokenizer_fingerprint": tokenizer.fingerprint(), "config": asdict(config),
    }
    return model, state, metrics


def save_training_checkpoint(path: str | Path, model: TinyBigramModel, optimizer_state: dict, metrics: dict) -> None:
    """Write a deterministic JSON checkpoint containing model and optimizer state."""
    payload = {"format": "spaceslug-tiny-training-checkpoint", "schema_version": 1,
               "model": {"vocab_size": model.vocab_size, "weights": model.weights},
               "optimizer": optimizer_state, "metrics": metrics}
    Path(path).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def load_training_checkpoint(path: str | Path) -> tuple[TinyBigramModel, dict, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "spaceslug-tiny-training-checkpoint" or payload.get("schema_version") != 1:
        raise ValueError("unsupported Tiny training checkpoint")
    model_data = payload["model"]
    return TinyBigramModel(int(model_data["vocab_size"]), model_data["weights"]), payload["optimizer"], payload["metrics"]
