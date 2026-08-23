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
    return {"status": "experimental", "base_weights": "frozen", "optimizer": "sgd", "device_resident": False, "gradient_accumulation": False, "adamw": False, "dataset_training": False, "boundary": "per-step GPU tensor orchestration with transient ABI buffers; no persistent device-resident loop"}
