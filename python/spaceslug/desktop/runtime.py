"""Runtime placement and GPU-primary / CPU-fallback status for the desktop shell.

The desktop shell is a GUI client of the Spaceslug service/backend.  This module
decides the visible placement status from a plain capability probe dict so the
controller stays testable and never touches the runtime binary on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUNTIME_ROOT = "/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime"

# Operations that prove a GPU-accelerated path is actually present.  The
# ``sgemm`` executable/link is the coarse gate used elsewhere in the repo; the
# ABI names make the claim stronger when the native library is available.
_GPU_OPERATIONS = (
    "sgemm",
    "attention_causal_abi",
    "tiny_forward_abi",
    "tiny_forward_fixed_retained_abi",
    "lora_train_step_abi",
)


@dataclass(frozen=True)
class RuntimePlacement:
    mode: str
    device: str | None
    software_vulkan: bool
    gpu_primary: bool
    cpu_fallback: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "device": self.device,
            "software_vulkan": self.software_vulkan,
            "gpu_primary": self.gpu_primary,
            "cpu_fallback": self.cpu_fallback,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrainingPlacement:
    """Requested vs. actual placement for a training job.

    ``requested`` records what the user asked for (``gpu-primary`` or ``cpu``).
    ``actual`` records what the job really runs on. In Phase 1 the projected
    Tiny training job always resolves to ``cpu-fallback`` even when a hardware
    GPU primary is present, because dataset-integrated GPU training is not yet
    available and CPU remains authoritative. ``hardware`` keeps the resolved
    hardware capability separate so the two facts are never conflated.
    """

    requested: str
    actual: str
    hardware: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "actual": self.actual,
            "hardware": self.hardware,
            "reason": self.reason,
        }


def default_runtime_probe() -> dict[str, Any]:
    """Static probe that never spawns a process or loads the native library."""
    return {
        "backend": "spaceslug",
        "device": None,
        "runtime_revision": "runtime",
        "operations": [],
        "software_vulkan": False,
    }


def backend_runtime_probe(
    runtime_root: str | Path = RUNTIME_ROOT, runtime_revision: str = "runtime"
) -> dict[str, Any]:
    """Probe the real backend, falling back to a static CPU probe on failure.

    This is a capability read only: it never runs training or model forward
    work beyond the runtime's existing smoke check.
    """
    try:
        from ..backend import BackendSession

        capabilities = BackendSession(runtime_root, runtime_revision).capabilities()
        return {
            "backend": capabilities.backend,
            "device": capabilities.device,
            "runtime_revision": capabilities.runtime_revision,
            "operations": list(capabilities.operations),
            "software_vulkan": capabilities.software_vulkan,
        }
    except Exception:
        return default_runtime_probe()


def resolve_placement(probe: dict[str, Any]) -> RuntimePlacement:
    """Turn a capability probe into a single GPU-primary / CPU-fallback status."""
    device = probe.get("device")
    software_vulkan = bool(probe.get("software_vulkan"))
    operations = tuple(probe.get("operations") or ())
    gpu_op_present = any(operation in operations for operation in _GPU_OPERATIONS)
    gpu_primary = bool(device) and not software_vulkan and gpu_op_present

    if gpu_primary:
        reason = "GPU primary available; CPU reference remains the authoritative fallback"
    elif device and software_vulkan:
        reason = "software Vulkan (CPU) fallback active; no hardware GPU primary"
    else:
        reason = "no GPU primary detected; CPU reference fallback active"

    return RuntimePlacement(
        mode="gpu-primary" if gpu_primary else "cpu-fallback",
        device=device,
        software_vulkan=software_vulkan,
        gpu_primary=gpu_primary,
        cpu_fallback=True,
        reason=reason,
    )


def resolve_training_placement(requested_gpu_primary: bool, runtime: RuntimePlacement) -> TrainingPlacement:
    """Resolve the requested vs. actual placement for a projected Tiny training job.

    The request is honored as a recorded intent only. Phase 1 has no
    dataset-integrated GPU training path, so the actual placement is always
    ``cpu-fallback`` (the CPU reference is authoritative) regardless of the
    hardware capability reported in *runtime*. This keeps the GPU-primary
    claim honest: it is never shown as actual until that path truly exists.
    """
    requested = "gpu-primary" if requested_gpu_primary else "cpu"
    actual = "cpu-fallback"
    reason = (
        "dataset-integrated GPU training is not yet available; CPU projected "
        f"Tiny training is authoritative (hardware={runtime.mode})"
    )
    return TrainingPlacement(requested=requested, actual=actual, hardware=runtime.mode, reason=reason)
