"""Dependency-free tiny CPU reference language model for acceptance tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass
class TinyBigramModel:
    vocab_size: int
    weights: list[list[float]]

    @classmethod
    def create(cls, vocab_size: int) -> "TinyBigramModel":
        return cls(vocab_size, [[0.0 for _ in range(vocab_size)] for _ in range(vocab_size)])

    def loss_and_gradients(self, sequences: list[list[int]]) -> tuple[float, list[list[float]]]:
        gradients = [[0.0 for _ in range(self.vocab_size)] for _ in range(self.vocab_size)]
        total = 0.0
        count = 0
        for sequence in sequences:
            for previous, target in zip(sequence, sequence[1:]):
                logits = self.weights[previous]
                maximum = max(logits)
                exp_values = [math.exp(value - maximum) for value in logits]
                normalizer = sum(exp_values)
                probabilities = [value / normalizer for value in exp_values]
                total -= math.log(max(probabilities[target], 1e-30))
                count += 1
                for token in range(self.vocab_size):
                    gradients[previous][token] += probabilities[token] - (token == target)
        if count == 0:
            raise ValueError("training sequences contain no next-token pairs")
        scale = 1.0 / count
        return total * scale, [[value * scale for value in row] for row in gradients]

    def train_step(self, sequences: list[list[int]], learning_rate: float) -> float:
        loss, gradients = self.loss_and_gradients(sequences)
        for row, gradient_row in zip(self.weights, gradients):
            for index, gradient in enumerate(gradient_row):
                row[index] -= learning_rate * gradient
        return loss

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocab_size": self.vocab_size, "weights": self.weights}, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TinyBigramModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(int(payload["vocab_size"]), payload["weights"])
