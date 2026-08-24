"""Bounded GPU LoRA training orchestration and checkpoint metadata."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

@dataclass
class GpuLoRATrainingState:
    step: int = 0
    learning_rate: float = 0.01
    backend: str = "vulkan-radv"
    device_resident: bool = False
    optimizer: str = "sgd"

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "step": self.step, "learning_rate": self.learning_rate, "backend": self.backend, "device_resident": self.device_resident, "optimizer": self.optimizer}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "GpuLoRATrainingState":
        if int(state.get("schema_version", 0)) != 1 or state.get("optimizer") != "sgd":
            raise ValueError("unsupported GPU LoRA training state")
        return cls(int(state["step"]), float(state["learning_rate"]), str(state["backend"]), bool(state["device_resident"]), str(state["optimizer"]))


def save_gpu_lora_checkpoint(path: str | Path, state: GpuLoRATrainingState, adapter_state: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps({"training": state.state_dict(), "adapter": adapter_state}, sort_keys=True, indent=2) + "\n")


def load_gpu_lora_checkpoint(path: str | Path) -> tuple[GpuLoRATrainingState, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    return GpuLoRATrainingState.from_state_dict(data["training"]), data["adapter"]


def gpu_lora_capability() -> dict[str, Any]:
    return {"status": "experimental", "base_weights": "frozen", "optimizer": "sgd", "device_resident": True, "gradient_accumulation": False, "adamw": False, "dataset_training": False, "persistent_command_buffer": True, "boundary": "persistent tensor LoRA session is available; full token-derived LM graph remains transient and host-orchestrated"}


def gpu_lora_training_plan() -> dict[str, Any]:
    return {"operation": "tiny_lora_gpu_training", "status": "persistent-tensor-session-plus-bounded-lm-graph", "steps": ["gpu_lora_forward", "gpu_causal_loss", "gpu_lm_head_backward", "gpu_projection_backward", "gpu_attention_backward", "gpu_multi_adapter_gradients", "gpu_multi_adapter_sgd", "persistent_lora_session"], "buffers": "persistent-for-tensor-session", "optimizer": "sgd", "base_weights": "frozen", "unsupported": ["persistent-token-derived-LM-graph", "gradient-accumulation", "adamw", "dataset-training"]}


class PersistentGpuLoRATrainer:
    """Persistent native LoRA tensor session; keeps A/B device-resident across steps."""
    def __init__(self, backend: Any, a: list[float], b: list[float], rank: int, learning_rate: float, rows: int, step: int = 0) -> None:
        self.backend, self.a, self.b, self.rank, self.learning_rate, self.rows, self.step_index = backend, list(a), list(b), rank, learning_rate, rows, step
        self.handle = backend.open_lora_session(self.a, self.b, rank, learning_rate, rows)

    def step(self, x: list[float], dy: list[float]) -> dict[str, Any]:
        y = self.backend.step_lora_session(self.handle, x, dy, self.rows)
        self.step_index += 1
        return {"status": "ok", "gpu_execution": True, "device_resident": True, "parity": "persistent-session", "step": self.step_index, "y": y}

    def checkpoint(self, path: str | Path) -> None:
        self.a, self.b = self.backend.readback_lora_session(self.handle, self.rank)
        save_gpu_lora_checkpoint(path, GpuLoRATrainingState(step=self.step_index, learning_rate=self.learning_rate, device_resident=True), {"rank": self.rank, "a": self.a, "b": self.b})

    def close(self) -> None:
        if self.handle is not None:
            self.backend.close_lora_session(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try: self.close()
        except Exception: pass


class GpuLoRATrainer:
    """Host-coordinated repeated GPU steps; deliberately not device-resident."""
    def __init__(self, backend: Any, model: Any, adapter: Any, state: GpuLoRATrainingState | None = None) -> None:
        self.backend, self.model, self.adapter = backend, model, adapter
        self.state = state or GpuLoRATrainingState()
        if self.state.device_resident:
            raise ValueError("device-resident GPU LoRA trainer is not implemented")

    def step(self, tokens: list[int], targets: list[int], mask: list[int]) -> dict[str, Any]:
        result = self.backend.execute_tiny_lora_train_graph(tokens, targets, mask, self.model, self.adapter, self.state.learning_rate)
        if result.status != "ok":
            raise RuntimeError(result.metrics.get("reason", "GPU LoRA step failed"))
        packed_a, packed_b = result.output["updated_a"], result.output["updated_b"]
        names = ("query", "key", "value", "output")
        for target, name in enumerate(names):
            matrix = self.adapter.matrices[name]
            start = target * 64 * matrix.rank
            matrix.A = [packed_a[start + row * matrix.rank:start + (row + 1) * matrix.rank] for row in range(64)]
            start = target * matrix.rank * 64
            matrix.B = [packed_b[start + row * 64:start + (row + 1) * 64] for row in range(matrix.rank)]
        self.state.step += 1
        return {"status": result.status, "loss": result.output["row_loss"], "step": self.state.step, "gpu_execution": result.metrics["gpu_execution"], "parity": result.metrics["parity"]}

    def run_steps(self, batches: list[tuple[list[int], list[int], list[int]]]) -> list[dict[str, Any]]:
        if not batches:
            raise ValueError("at least one GPU LoRA batch is required")
        reports = []
        for tokens, targets, mask in batches:
            reports.append(self.step(tokens, targets, mask))
        return reports

    def checkpoint(self, path: str | Path) -> None:
        save_gpu_lora_checkpoint(path, self.state, self.adapter.state_dict())
