"""CPU-authoritative Tiny LoRA adapter and deterministic update reference."""
from __future__ import annotations
from dataclasses import dataclass, field
from .projected_attention_reference import ProjectedTinyAttentionModel

@dataclass
class LoRAMatrix:
    target: str
    input_size: int = 64
    output_size: int = 64
    rank: int = 4
    alpha: float = 4.0
    A: list[list[float]] = field(default_factory=list)
    B: list[list[float]] = field(default_factory=list)
    def __post_init__(self):
        if not self.A: self.A = [[0.01 * ((i + r) % 3 + 1) for r in range(self.rank)] for i in range(self.input_size)]
        if not self.B: self.B = [[0.0] * self.output_size for _ in range(self.rank)]
    def validate(self, hidden_size=64):
        if self.input_size != hidden_size or self.output_size != hidden_size or not 0 < self.rank <= hidden_size: raise ValueError("invalid LoRA dimensions")
        if len(self.A) != hidden_size or any(len(row) != self.rank for row in self.A) or len(self.B) != self.rank or any(len(row) != hidden_size for row in self.B): raise ValueError("invalid LoRA matrices")
    def delta(self):
        scale = self.alpha / self.rank
        return [[scale * sum(self.A[i][r] * self.B[r][j] for r in range(self.rank)) for j in range(self.output_size)] for i in range(self.input_size)]

@dataclass
class TinyLoRAAdapter:
    hidden_size: int = 64
    rank: int = 4
    alpha: float = 4.0
    matrices: dict[str, LoRAMatrix] = field(default_factory=dict)
    def __post_init__(self):
        if not self.matrices: self.matrices = {name: LoRAMatrix(name, self.hidden_size, self.hidden_size, self.rank, self.alpha) for name in ("query", "key", "value", "output")}
        self.validate()
    def validate(self, model=None):
        if model is not None and (model.hidden_size != self.hidden_size): raise ValueError("adapter/model hidden size mismatch")
        for name, matrix in self.matrices.items():
            if name not in ("query", "key", "value", "output"): raise ValueError("unsupported LoRA target")
            matrix.validate(self.hidden_size)
    def effective_matrix(self, target, base):
        delta = self.matrices[target].delta()
        return [[base[i][j] + delta[i][j] for j in range(self.hidden_size)] for i in range(self.hidden_size)]
    def state_dict(self):
        return {"hidden_size": self.hidden_size, "rank": self.rank, "alpha": self.alpha, "matrices": {k: {"A": v.A, "B": v.B} for k,v in self.matrices.items()}}

class LoRAProjectedTinyAttention:
    def __init__(self, base: ProjectedTinyAttentionModel, adapter: TinyLoRAAdapter):
        adapter.validate(base); self.base, self.adapter = base, adapter
    def effective_projection(self, name): return self.adapter.effective_matrix(name, getattr(self.base, name))
    def logits_for_tokens(self, tokens):
        model = ProjectedTinyAttentionModel(self.base.vocab_size, self.base.hidden_size, self.base.use_positions)
        model.embedding, model.lm_head = self.base.embedding, self.base.lm_head
        model.query, model.key, model.value, model.output = (self.effective_projection(n) for n in ("query", "key", "value", "output"))
        return model.logits_for_tokens(tokens)
