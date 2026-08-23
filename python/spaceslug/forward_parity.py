"""Forward-result parity record for the CPU baseline and future Vulkan path."""

from __future__ import annotations

from .backend import BackendSession, ExecutionResult
from .parity import compare_float_arrays


def record_cpu_forward(session: BackendSession, tokens: list[int], model) -> dict:
    result: ExecutionResult = session.execute_projected_attention_forward(tokens, model)
    return {"operation": result.operation, "backend": result.backend, "tokens": list(tokens), "logits": result.output["logits"], "parity": {"status": "baseline", "tolerance": None}, "gpu_execution": False}


def build_gpu_forward_report(cpu_record: dict, gpu_result, *, tolerance: float = 1e-3) -> dict:
    """Compare a structured GPU execution result against a CPU baseline."""
    gpu_logits = gpu_result.output.get("logits")
    if gpu_logits is None:
        return {"operation": cpu_record["operation"], "cpu_backend": cpu_record["backend"], "gpu_backend": gpu_result.backend, "gpu_execution": False, "parity": {"status": "not-run", "reason": gpu_result.metrics.get("reason", "GPU result does not contain logits")}, "runtime_revision": gpu_result.runtime_revision, "device": gpu_result.device}
    return compare_gpu_logits(cpu_record, gpu_logits, tolerance=tolerance) | {"runtime_revision": gpu_result.runtime_revision, "device": gpu_result.device}


def compare_gpu_logits(cpu_record: dict, gpu_logits: list[float], *, tolerance: float = 1e-3) -> dict:
    report = compare_float_arrays(gpu_logits, cpu_record["logits"], tolerance=tolerance)
    return {"operation": cpu_record["operation"], "cpu_backend": cpu_record["backend"], "gpu_backend": "vulkan-radv", "gpu_execution": True, "parity": report}
