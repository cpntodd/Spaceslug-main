"""Strict native Vulkan training job for the fixed rank-4 Tiny desktop profile."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .backend import BackendSession
from .dataset import DatasetBundle, verify_bundle
from .gpu_lora_training import PersistentTinyTrainer
from .lora import TinyLoRAAdapter
from .projected_attention_reference import ProjectedTinyAttentionModel
from .tokenizer import ByteTokenizer, default_tokenizer

REQUIRED_OPERATIONS = frozenset({
    "tiny_forward_abi", "lora_train_step_abi", "attention_causal_backward_abi",
    "lora_gradients_multi_abi", "lora_sgd_multi_abi",
})


def readiness(probe: dict[str, Any] | None, rank: int) -> dict[str, Any]:
    """Return a fail-closed gate; the current native graph constructor is rank-4."""
    if not probe:
        return {"ready": False, "reason": "refresh GPU readiness first", "missing": sorted(REQUIRED_OPERATIONS)}
    operations = set(probe.get("operations") or ())
    missing = sorted(REQUIRED_OPERATIONS - operations)
    device = str(probe.get("device") or "")
    reasons = []
    if probe.get("software_vulkan"):
        reasons.append("software Vulkan/lavapipe is not native GPU training")
    if "RADV" not in device.upper():
        reasons.append("a RADV hardware device is required")
    if rank != 4:
        reasons.append("the current native graph constructor supports rank 4 only")
    if missing:
        reasons.append("missing native operations: " + ", ".join(missing))
    return {"ready": not reasons, "reason": "; ".join(reasons) if reasons else "RADV fixed-Tiny rank-4 training ready", "missing": missing}


def _windows(bundle: DatasetBundle, tokenizer: ByteTokenizer) -> list[tuple[list[int], list[int], list[int]]]:
    result = []
    for record in bundle.records("train"):
        prompt = str(record.get("prompt", ""))
        target = str(record.get("target", record.get("text", "")))
        prompt_ids = tokenizer.encode(prompt)
        target_ids = tokenizer.encode(target)
        ids = prompt_ids + target_ids
        # Train only on response tokens when a prompt/target record is present;
        # prompt context remains visible to the forward pass but contributes no
        # loss. Plain text records retain all-token masking.
        response_start = max(0, len(prompt_ids) - 1) if target else 0
        # Keep each RX580 submission below the compute-ring watchdog budget.
        # The ABI permits 128 positions, but a full backward window can trigger
        # gfx803 soft recovery; bounded 8-token chunks are the verified desktop gate.
        for start in range(0, max(0, len(ids) - 1), 8):
            source = ids[start:start + 9]
            if len(source) >= 2:
                mask = [1 if (start + index + 1) >= response_start else 0 for index in range(len(source) - 1)]
                result.append((source[:-1], source[1:], mask))
    if not result:
        raise ValueError("dataset has no token windows for native training")
    return result


def run_native_training(
    bundle_path: str | Path, *, runtime_root: str | Path, runtime_revision: str,
    rank: int, steps: int, learning_rate: float, checkpoint: str | Path,
    artifact: str | Path, experiment: str | Path,
    on_step: Callable[[int, float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    backend_factory: Callable[..., Any] = BackendSession,
    trainer_factory: Callable[..., Any] = PersistentTinyTrainer,
) -> dict[str, Any]:
    """Run host-staged dataset windows through the native persistent Vulkan graph."""
    bundle = verify_bundle(bundle_path)
    tokenizer = default_tokenizer()
    windows = _windows(bundle, tokenizer)
    split_counts = {split: len(bundle.records(split)) for split in ("train", "validation", "test")}
    backend = backend_factory(runtime_root, runtime_revision)
    caps = backend.capabilities()
    gate = readiness({"device": caps.device, "software_vulkan": caps.software_vulkan,
                      "operations": list(caps.operations)}, rank)
    if not gate["ready"]:
        raise RuntimeError("native GPU training unavailable: " + gate["reason"])
    model = ProjectedTinyAttentionModel(259, 64)
    adapter = TinyLoRAAdapter(rank=rank)
    trainer = trainer_factory(backend, model, adapter, learning_rate, optimizer="sgd")
    losses: list[float] = []
    stopped = "steps"
    device_batch_attempted = False
    device_batch_status = "not-attempted"
    validation_metrics = {"records": split_counts["validation"], "loss": None, "status": "not-run"}
    test_metrics = {"records": split_counts["test"], "loss": None, "status": "not-run"}
    try:
        for step in range(1, steps + 1):
            if should_stop and should_stop():
                stopped = "cancelled"
                break
            tokens, targets, mask = windows[(step - 1) % len(windows)]
            # Full device-batch training is attempted only through its explicit
            # ABI. Until the native graph implements it, retain the verified
            # bounded token path rather than silently mislabeling LM-head SGD.
            outcome = trainer.train_tokens(tokens, targets, mask)
            values = [float(value) for value in outcome["loss"]]
            loss = sum(values) / max(1, len(values))
            losses.append(loss)
            if on_step:
                on_step(step, loss)
        checkpoint = Path(checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.checkpoint(checkpoint)
    finally:
        trainer.close()
    artifact = Path(artifact); artifact.mkdir(parents=True, exist_ok=False)
    (artifact / "native-training.json").write_text(json.dumps({
        "format": "spaceslug-native-tiny-adapter", "profile": "tiny_h64_v259_vp320_t128_rank4",
        "checkpoint": str(checkpoint), "backend": "vulkan-radv", "gpu_execution": True,
        "dataset_device_resident": device_batch_status == "ok", "device_batch_attempted": device_batch_attempted,
        "device_batch_status": device_batch_status, "split_record_counts": split_counts,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
    }, sort_keys=True, indent=2) + "\n")
    revision = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Keep the adapter checkpoint discoverable by the desktop responder.
    artifact_manifest = json.loads((artifact / "native-training.json").read_text())
    checkpoint_data = json.loads(checkpoint.read_text())
    artifact_manifest["adapter"] = checkpoint_data.get("adapter")
    (artifact / "native-training.json").write_text(json.dumps(artifact_manifest, sort_keys=True, indent=2) + "\n")
    experiment = Path(experiment); experiment.mkdir(parents=True, exist_ok=True)
    experiment_file = experiment / "experiment.json"
    metrics = {"initial_train_loss": losses[0] if losses else None, "final_train_loss": losses[-1] if losses else None,
               "stopped_reason": stopped, "steps": len(losses), "gpu_execution": True,
               "dataset_device_resident": device_batch_status == "ok", "device_batch_status": device_batch_status,
               "split_record_counts": split_counts, "validation_metrics": validation_metrics,
               "test_metrics": test_metrics}
    experiment_file.write_text(json.dumps({"created_at": datetime.now(UTC).isoformat(), "backend": "vulkan-radv",
                                            "metrics": metrics}, sort_keys=True, indent=2) + "\n")
    return {"artifact_revision": revision, "experiment": str(experiment_file), "metrics": metrics,
            "backend": "vulkan-radv", "gpu_execution": True}
