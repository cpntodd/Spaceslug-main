"""CPU-first inference session with explicit backend and parity metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import BackendSession
from .forward_parity import build_gpu_forward_report, record_cpu_forward


@dataclass
class InferenceSession:
    backend: BackendSession
    model: object
    last_record: dict | None = None

    def run(self, tokens: list[int]) -> dict:
        record = record_cpu_forward(self.backend, tokens, self.model)
        self.last_record = record
        return record

    def run_attention_gate(self, q: list[float], k: list[float], v: list[float], tokens: int, hidden_size: int) -> dict:
        result = self.backend.execute_attention_kernel_parity(q, k, v, tokens, hidden_size)
        return {"operation": result.operation, "backend": result.backend, "status": result.status, "parity": result.metrics.get("parity"), "runtime_revision": result.runtime_revision, "device": result.device, "report": result.output}

    def run_gpu_chain_plan(self, tokens: list[int]) -> dict:
        result = self.backend.execute_tiny_attention_kernel_chain(tokens, self.model)
        return {"operation": result.operation, "backend": result.backend, "status": result.status, "parity": result.metrics.get("parity"), "reason": result.metrics.get("reason"), "steps": result.metrics.get("steps", []), "runtime_revision": result.runtime_revision, "device": result.device, "gpu_execution": result.status == "ok"}

    def run_gpu_plan(self, tokens: list[int]) -> dict:
        gpu_result = self.backend.execute_projected_attention_gpu_plan(tokens, self.model)
        return self.compare_gpu_result(tokens, gpu_result)

    def compare_gpu_result(self, tokens: list[int], gpu_result) -> dict:
        cpu_record = self.run(tokens)
        return build_gpu_forward_report(cpu_record, gpu_result)

    def next_token(self, tokens: list[int]) -> int:
        record = self.run(tokens)
        logits = record["logits"]
        return max(range(len(logits)), key=logits.__getitem__)
