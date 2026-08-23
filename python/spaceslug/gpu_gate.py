"""CPU-first guard for invoking the current Vulkan parity boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .backend import BackendSession, ExecutionResult
from .cpu_verification import CpuVerification


@dataclass(frozen=True)
class TinyGpuGateResult:
    cpu_verified: bool
    gpu_operation: str
    gpu_passed: bool
    runtime_revision: str
    device: str | None
    parity: str
    reason: str


def run_tiny_gemm_gate(cpu: CpuVerification, runtime_root: str | Path, runtime_revision: str, *, software_vulkan: bool = False) -> TinyGpuGateResult:
    if not cpu.passed:
        return TinyGpuGateResult(False, "sgemm", False, runtime_revision, None, "not-run", "CPU verification is required")
    session = BackendSession(runtime_root, runtime_revision, software_vulkan=software_vulkan)
    result: ExecutionResult = session.execute_sgemm_parity()
    return TinyGpuGateResult(True, result.operation, result.status == "ok", result.runtime_revision, result.device, result.metrics.get("parity", "unknown"), "CPU/RADV GEMM parity passed")
