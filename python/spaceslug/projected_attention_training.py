"""Deterministic dataset training for the projected Tiny attention reference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .batching import target_only_batches
from .dataset import DatasetBundle
from .projected_attention_reference import ProjectedTinyAttentionModel
from .tokenizer import ByteTokenizer
from .projected_attention_artifact import write_projected_artifact
from .projected_attention_experiment import write_projected_experiment
from .projected_attention_inference import next_token
from .quality_metrics import token_accuracy


@dataclass(frozen=True)
class ProjectedAttentionConfig:
    steps: int
    learning_rate: float
    batch_size: int = 1
    optimizer: str = "adamw"
    weight_decay: float = 0.0

    def validate(self) -> None:
        if self.steps <= 0 or self.learning_rate <= 0.0 or self.batch_size <= 0:
            raise ValueError("steps, learning_rate, and batch_size must be positive")
        if self.optimizer != "adamw" or self.weight_decay < 0.0:
            raise ValueError("only adamw is supported and weight_decay must not be negative")


def train_projected_attention(bundle: DatasetBundle, config: ProjectedAttentionConfig, *, tokenizer: ByteTokenizer,
                              model: ProjectedTinyAttentionModel | None = None, prior_steps: int = 0, optimizer_state: dict | None = None) -> tuple[ProjectedTinyAttentionModel, dict, dict]:
    config.validate()
    records = bundle.records("train")
    batches = target_only_batches(records, tokenizer, config.batch_size)
    model = model or ProjectedTinyAttentionModel(tokenizer.vocab_size)
    before = sum(model.loss(batch) for batch in batches) / len(batches)
    state = optimizer_state if optimizer_state is not None else {}
    for _ in range(config.steps):
        for batch in batches:
            model.train_step(batch, config.learning_rate, optimizer_state=state, weight_decay=config.weight_decay)
    after = sum(model.loss(batch) for batch in batches) / len(batches)
    validation_records = bundle.records("validation")
    validation_loss = None
    if validation_records:
        validation_batches = target_only_batches(validation_records, tokenizer, config.batch_size)
        validation_loss = sum(model.loss(batch) for batch in validation_batches) / len(validation_batches)
    test_records = bundle.records("test")
    test_loss = None
    test_accuracy = None
    if test_records:
        test_batches = target_only_batches(test_records, tokenizer, config.batch_size)
        test_loss = sum(model.loss(batch) for batch in test_batches) / len(test_batches)
        test_accuracy = token_accuracy(model, test_batches)
    metrics = {"initial_train_loss": before, "final_train_loss": after, "validation_loss": validation_loss, "test_loss": test_loss, "test_token_accuracy": test_accuracy, "optimizer_step": state.get("step", prior_steps + config.steps),
               "dataset_revision": bundle.manifest["revision"], "tokenizer_fingerprint": tokenizer.fingerprint(), "config": asdict(config)}
    return model, state, metrics


def save_projected_checkpoint(path: str | Path, model: ProjectedTinyAttentionModel, optimizer_state: dict, metrics: dict) -> None:
    payload = {"format": "spaceslug-tiny-projected-attention-checkpoint", "schema_version": 2,
               "optimizer": optimizer_state,
               "model": {name: getattr(model, name) for name in ("vocab_size", "hidden_size", "use_positions", "embedding", "query", "key", "value", "output", "lm_head")},
               "metrics": metrics}
    Path(path).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _assert_resume_compatible(metrics: dict, bundle: DatasetBundle, config: ProjectedAttentionConfig, tokenizer: ByteTokenizer) -> None:
    if metrics.get("dataset_revision") != bundle.manifest["revision"]:
        raise ValueError("checkpoint dataset revision does not match training bundle")
    if metrics.get("tokenizer_fingerprint") != tokenizer.fingerprint():
        raise ValueError("checkpoint tokenizer does not match training tokenizer")
    previous_config = metrics.get("config", {})
    if previous_config.get("batch_size") != config.batch_size or previous_config.get("optimizer") != config.optimizer or previous_config.get("weight_decay") != config.weight_decay:
        raise ValueError("checkpoint training configuration is incompatible")


def run_projected_training(
    bundle: DatasetBundle,
    config: ProjectedAttentionConfig,
    *,
    tokenizer: ByteTokenizer,
    checkpoint: str | Path,
    artifact: str | Path,
    experiment: str | Path,
    resume: str | Path | None = None,
    code_revision: str = "unrecorded",
) -> dict:
    model = None
    prior_steps = 0
    optimizer_state = None
    if resume is not None:
        model, previous_metrics, optimizer_state = load_projected_checkpoint(resume)
        _assert_resume_compatible(previous_metrics, bundle, config, tokenizer)
        prior_steps = int(previous_metrics["optimizer_step"])
    model, optimizer_state, metrics = train_projected_attention(bundle, config, tokenizer=tokenizer, model=model, prior_steps=prior_steps, optimizer_state=optimizer_state)
    save_projected_checkpoint(checkpoint, model, optimizer_state, metrics)
    manifest = write_projected_artifact(artifact, model, tokenizer)
    metrics = dict(metrics)
    metrics["artifact_revision"] = manifest["revision"]
    metrics["checkpoint_identity"] = {"path": str(checkpoint), "schema_version": 2}
    metrics["inference"] = {"prompt": "Q: ", "next_token": next_token(model, tokenizer, "Q: ")}
    experiment_path = write_projected_experiment(experiment, Path(experiment).name, metrics, code_revision=code_revision, command="spaceslug tiny-attention-train")
    return {"metrics": metrics, "artifact_revision": manifest["revision"], "experiment": str(experiment_path)}


def load_projected_checkpoint(path: str | Path) -> tuple[ProjectedTinyAttentionModel, dict, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "spaceslug-tiny-projected-attention-checkpoint" or payload.get("schema_version") != 2:
        raise ValueError("unsupported projected attention checkpoint")
    data = payload["model"]
    model = ProjectedTinyAttentionModel(data["vocab_size"], data["hidden_size"], data.get("use_positions", True))
    for name in ("embedding", "query", "key", "value", "output", "lm_head"):
        setattr(model, name, data[name])
    return model, payload["metrics"], payload["optimizer"]
