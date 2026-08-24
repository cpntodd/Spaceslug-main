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


def gpu_lora_capability(native_adamw: bool = False) -> dict[str, Any]:
    return {"status": "experimental", "base_weights": "frozen", "optimizer": "sgd", "optimizers": ["sgd"] + (["adamw"] if native_adamw else []), "device_resident": True, "gradient_accumulation": True, "adamw": native_adamw, "native_adamw": native_adamw, "dataset_training": False, "persistent_command_buffer": True, "fixed_shape_retained_command_buffer_resubmit": True, "reusable_exec_submission": True, "fp16_storage": True, "fp16_arithmetic": False, "boundary": "native persistent Tiny token graph and device-resident gradient accumulation are available for the fixed H=64/V=259/Vp=320/T<=128/rank=4 or rank=8 profiles; Python orchestration remains bounded"}


def gpu_lora_training_plan() -> dict[str, Any]:
    return {"operation": "tiny_lora_gpu_training", "status": "persistent-tiny-token-graph", "steps": ["gpu_tiny_forward", "gpu_causal_loss", "gpu_lm_head_backward", "gpu_projection_backward", "gpu_attention_backward", "gpu_multi_adapter_gradients", "gpu_multi_adapter_sgd", "adapter_checkpoint_restore"], "buffers": "persistent-token-graph", "optimizer": "sgd", "base_weights": "frozen", "unsupported": ["other-model-shapes", "dataset-training", "immutable-command-buffer-reuse-for-mutable-inputs", "fp16-arithmetic", "fp16-tiny-forward-integration"]}


def persistent_graph_boundary() -> dict[str, Any]:
    return {"status": "implemented-bounded", "persistent": ["embeddings", "positions", "tokens", "targets", "mask", "states", "Q", "K", "V", "attention", "projected", "logits", "dLogits", "loss", "backward_intermediates", "adapter_A", "adapter_B", "adapter_dA", "adapter_dB"], "transient": ["host_input_staging", "host_readback", "CPU_reference"], "contract": {"hidden": 64, "vocab": 259, "logits_stride": 320, "sequence_capacity": 128, "ranks": [4, 8], "dtype": "fp32", "optimizer": "sgd", "base_weights": "frozen"}, "unsupported": ["other-model-shapes", "dataset-training", "immutable-command-buffer-reuse-for-mutable-inputs", "fp16-arithmetic", "fp16-tiny-forward-integration"]}


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

    @classmethod
    def resume(cls, backend: Any, path: str | Path, rows: int) -> "PersistentGpuLoRATrainer":
        state, adapter = load_gpu_lora_checkpoint(path)
        if not state.device_resident or "a" not in adapter or "b" not in adapter:
            raise ValueError("checkpoint is not a persistent GPU LoRA checkpoint")
        return cls(backend, adapter["a"], adapter["b"], int(adapter["rank"]), state.learning_rate, rows, state.step)

    def close(self) -> None:
        if self.handle is not None:
            self.backend.close_lora_session(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try: self.close()
        except Exception: pass


class PersistentTinyTrainer:
    """Fixed native Tiny graph trainer with bounded host-side window streaming.

    Dataset batches are sliced and staged by the host for each call.  The native
    persistent graph retains model/adapter state only; it never owns the dataset.
    """
    def __init__(self, backend: Any, model: Any, adapter: Any, learning_rate: float = 0.01, optimizer: str = "sgd", weight_decay: float = 0.0) -> None:
        if (model.hidden_size, model.vocab_size, adapter.hidden_size) != (64, 259, 64) or adapter.rank not in {4, 8}:
            raise ValueError("PersistentTiny supports only H=64, V=259, rank=4 or rank=8")
        if optimizer not in {"sgd", "adamw"} or weight_decay < 0.0:
            raise ValueError("optimizer must be sgd or adamw and weight_decay must not be negative")
        self.backend, self.model, self.adapter = backend, model, adapter
        self.learning_rate, self.weight_decay, self.optimizer, self.step_index, self.handle = learning_rate, weight_decay, optimizer, 0, backend.create_tiny_persistent_full(model, adapter)
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.sample_position = 0
        self.window_position = 0

    @staticmethod
    def iter_window_batches(tokens: list[int], targets: list[int], window_length: int,
                            mask: list[int] | None = None, *, start_window: int = 0,
                            max_windows: int | None = None, batch_windows: int | None = None):
        """Yield deterministic host-staged batches and their source positions.

        Each yielded item is ``(window_position, sample_position, tokens, targets,
        mask)``.  No dataset storage is transferred to, or retained on, the device.
        """
        if not 0 < window_length <= 128 or not tokens or len(tokens) % window_length:
            raise ValueError("PersistentTiny requires non-empty fixed windows of length 1..128")
        if len(targets) != len(tokens): raise ValueError("targets length must match tokens")
        mask = [1] * len(tokens) if mask is None else mask
        if len(mask) != len(tokens): raise ValueError("mask length must match tokens")
        total = len(tokens) // window_length
        if not 0 <= start_window <= total: raise ValueError("start_window is outside the dataset")
        if max_windows is not None and max_windows < 0: raise ValueError("max_windows must not be negative")
        if batch_windows is not None and batch_windows <= 0: raise ValueError("batch_windows must be positive")
        end = total if max_windows is None else min(total, start_window + max_windows)
        width = batch_windows or (end - start_window or 1)
        for first in range(start_window, end, width):
            last = min(end, first + width)
            lo, hi = first * window_length, last * window_length
            yield first, lo, tokens[lo:hi], targets[lo:hi], mask[lo:hi]

    def train_windows(self, tokens: list[int], targets: list[int], window_length: int,
                      mask: list[int] | None = None, *, start_window: int | None = None,
                      max_windows: int | None = None, batch_windows: int | None = None) -> dict[str, Any]:
        if self.optimizer != "sgd":
            raise NotImplementedError("train_windows currently supports the native SGD window path only")
        start = self.window_position if start_window is None else start_window
        batches = self.iter_window_batches(tokens, targets, window_length, mask,
                                           start_window=start, max_windows=max_windows,
                                           batch_windows=batch_windows)
        losses: list[Any] = []
        processed = 0
        self.backend.begin_tiny_accumulation(self.handle)
        for position, sample, bt, by, bm in batches:
            result = self.backend.accumulate_tiny_windows(self.handle, bt, by, bm, window_length)
            losses.extend(result)
            processed += len(bt) // window_length
        if not processed:
            raise ValueError("max_windows must select at least one window")
        self.backend.finalize_tiny_sgd(self.handle, self.learning_rate,
                                       float(sum(mask or [1] * len(tokens)) or 1))
        self.step_index += 1
        self.window_position = start + processed
        self.sample_position = self.window_position * window_length
        return {"status": "ok", "step": self.step_index, "loss": losses, "windows": processed,
                "window_length": window_length, "sample_position": self.sample_position,
                "window_position": self.window_position, "gpu_execution": True,
                "device_resident": True, "dataset_device_resident": False,
                "host_staging": True, "optimizer": "sgd",
                "contract": {"hidden": 64, "vocab": 259, "rank": self.adapter.rank}}

    def checkpoint(self, path: str | Path) -> None:
        """Save adapter and resumable stream metadata; dataset remains host-owned."""
        data: dict[str, Any] = {"schema_version": 1, "training": {
            "step": self.step_index, "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay, "optimizer": self.optimizer,
            "sample_position": self.sample_position, "window_position": self.window_position,
            "dataset_device_resident": False}, "adapter": self.readback_adapter()}
        if self.optimizer == "adamw":
            try: data["optimizer_state"] = self.readback_optimizer_state()
            except (AttributeError, NotImplementedError): pass
        Path(path).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")

    @classmethod
    def resume(cls, backend: Any, model: Any, adapter: Any, path: str | Path) -> "PersistentTinyTrainer":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != 1: raise ValueError("unsupported PersistentTiny checkpoint")
        state = data["training"]
        trainer = cls(backend, model, adapter, float(state["learning_rate"]),
                      str(state["optimizer"]), float(state.get("weight_decay", 0.0)))
        trainer.restore_adapter(data["adapter"])
        trainer.step_index = int(state["step"]); trainer.sample_position = int(state.get("sample_position", 0))
        trainer.window_position = int(state.get("window_position", 0))
        if "optimizer_state" in data:
            try: trainer.restore_optimizer_state(data["optimizer_state"])
            except (AttributeError, NotImplementedError): raise ValueError("checkpoint requires native AdamW state support")
        return trainer

    def train_tokens_adamw(self, tokens: list[int], targets: list[int], mask: list[int] | None = None) -> dict[str, Any]:
        return self._train_tokens(tokens, targets, mask, "adamw")

    def _train_tokens(self, tokens: list[int], targets: list[int], mask: list[int] | None, optimizer: str) -> dict[str, Any]:
        if not 0 < len(tokens) <= 128 or len(targets) != len(tokens):
            raise ValueError("PersistentTiny requires 1..128 tokens and equal targets")
        mask = mask or [1] * len(tokens)
        if len(mask) != len(tokens): raise ValueError("mask length must match tokens")
        begin = self.backend.begin_tiny_adamw if optimizer == "adamw" else self.backend.begin_tiny_accumulation
        accumulate = self.backend.accumulate_tiny_adamw if optimizer == "adamw" else self.backend.accumulate_tiny_backward
        finalize = self.backend.finalize_tiny_adamw if optimizer == "adamw" else self.backend.finalize_tiny_sgd
        begin(self.handle)
        losses = []
        for position, (token, target, included) in enumerate(zip(tokens, targets, mask)):
            result = accumulate(self.handle, token, position, target, included)
            losses.append(result["loss"][0])
        if optimizer == "adamw":
            finalize(self.handle, self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay, float(sum(mask) or 1))
        else:
            finalize(self.handle, self.learning_rate, float(sum(mask) or 1))
        self.step_index += 1
        return {"status": "ok", "step": self.step_index, "loss": losses, "gpu_execution": True, "device_resident": True, "optimizer": optimizer, "contract": {"hidden": 64, "vocab": 259, "rank": self.adapter.rank}}

    def train_tokens(self, tokens: list[int], targets: list[int], mask: list[int] | None = None) -> dict[str, Any]:
        return self._train_tokens(tokens, targets, mask, self.optimizer)

    def readback_adapter(self) -> list[list[float]]:
        return self.backend.readback_tiny_adapters(self.handle)

    def restore_adapter(self, values: list[list[float]]) -> None:
        self.backend.update_tiny_adapters(self.handle, values)

    def readback_optimizer_state(self) -> dict[str, Any]:
        return self.backend.readback_tiny_adamw_state(self.handle)

    def restore_optimizer_state(self, state: dict[str, Any]) -> None:
        self.backend.restore_tiny_adamw_state(self.handle, state)

    def close(self) -> None:
        if self.handle is not None:
            self.backend.close_tiny_persistent(self.handle)
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
