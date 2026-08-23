"""Deterministic dataset training for the projected Tiny attention reference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .batching import target_only_batches
from .dataset import DatasetBundle
from .projected_attention_reference import ProjectedTinyAttentionModel
from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class ProjectedAttentionConfig:
    steps: int
    learning_rate: float
    batch_size: int = 1

    def validate(self) -> None:
        if self.steps <= 0 or self.learning_rate <= 0.0 or self.batch_size <= 0:
            raise ValueError("steps, learning_rate, and batch_size must be positive")


def train_projected_attention(bundle: DatasetBundle, config: ProjectedAttentionConfig, *, tokenizer: ByteTokenizer,
                              model: ProjectedTinyAttentionModel | None = None) -> tuple[ProjectedTinyAttentionModel, dict]:
    config.validate()
    records = bundle.records("train")
    batches = target_only_batches(records, tokenizer, config.batch_size)
    model = model or ProjectedTinyAttentionModel(tokenizer.vocab_size)
    before = sum(model.loss(batch) for batch in batches) / len(batches)
    for _ in range(config.steps):
        for batch in batches:
            model.train_step(batch, config.learning_rate)
    after = sum(model.loss(batch) for batch in batches) / len(batches)
    validation_records = bundle.records("validation")
    validation_loss = None
    if validation_records:
        validation_batches = target_only_batches(validation_records, tokenizer, config.batch_size)
        validation_loss = sum(model.loss(batch) for batch in validation_batches) / len(validation_batches)
    return model, {"initial_train_loss": before, "final_train_loss": after, "validation_loss": validation_loss, "optimizer_step": config.steps,
                   "dataset_revision": bundle.manifest["revision"], "tokenizer_fingerprint": tokenizer.fingerprint(),
                   "config": asdict(config)}


def save_projected_checkpoint(path: str | Path, model: ProjectedTinyAttentionModel, metrics: dict) -> None:
    payload = {"format": "spaceslug-tiny-projected-attention-checkpoint", "schema_version": 1,
               "model": {name: getattr(model, name) for name in ("vocab_size", "hidden_size", "use_positions", "embedding", "query", "key", "value", "output", "lm_head")},
               "metrics": metrics}
    Path(path).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def load_projected_checkpoint(path: str | Path) -> tuple[ProjectedTinyAttentionModel, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "spaceslug-tiny-projected-attention-checkpoint" or payload.get("schema_version") != 1:
        raise ValueError("unsupported projected attention checkpoint")
    data = payload["model"]
    model = ProjectedTinyAttentionModel(data["vocab_size"], data["hidden_size"], data.get("use_positions", True))
    for name in ("embedding", "query", "key", "value", "output", "lm_head"):
        setattr(model, name, data[name])
    return model, payload["metrics"]
