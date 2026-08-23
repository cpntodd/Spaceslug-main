"""CPU-first inference session with explicit backend and parity metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import BackendSession
from .forward_parity import record_cpu_forward


@dataclass
class InferenceSession:
    backend: BackendSession
    model: object
    last_record: dict | None = None

    def run(self, tokens: list[int]) -> dict:
        record = record_cpu_forward(self.backend, tokens, self.model)
        self.last_record = record
        return record

    def next_token(self, tokens: list[int]) -> int:
        record = self.run(tokens)
        logits = record["logits"]
        return max(range(len(logits)), key=logits.__getitem__)
