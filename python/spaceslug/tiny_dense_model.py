"""Small dense causal CPU reference model with explicit analytic gradients.

This deliberately tiny neural language model is the next acceptance step after the
bigram baseline.  Each next-token prediction uses the previous token only, making
causality and its full backward pass easy to inspect before adding attention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class TinyDenseCausalModel:
    vocab_size: int
    hidden_size: int
    embedding: list[list[float]]
    output: list[list[float]]
    output_bias: list[float]

    @classmethod
    def create(cls, vocab_size: int, hidden_size: int = 16) -> "TinyDenseCausalModel":
        if vocab_size <= 0 or hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        # Deterministic non-zero initialization without a hidden RNG dependency.
        embedding = [[((token + 1) * (unit + 3) % 17 - 8) / 100.0 for unit in range(hidden_size)] for token in range(vocab_size)]
        output = [[((unit + 5) * (token + 7) % 19 - 9) / 100.0 for token in range(vocab_size)] for unit in range(hidden_size)]
        return cls(vocab_size, hidden_size, embedding, output, [0.0] * vocab_size)

    def loss_and_gradients(self, sequences: list[list[int]]) -> tuple[float, dict[str, list]]:
        embedding_grad = [[0.0] * self.hidden_size for _ in range(self.vocab_size)]
        output_grad = [[0.0] * self.vocab_size for _ in range(self.hidden_size)]
        bias_grad = [0.0] * self.vocab_size
        total = 0.0
        count = 0
        for sequence in sequences:
            for previous, target in zip(sequence, sequence[1:]):
                if not 0 <= previous < self.vocab_size or not 0 <= target < self.vocab_size:
                    raise ValueError("token outside vocabulary")
                hidden = self.embedding[previous]
                logits = [self.output_bias[token] + sum(hidden[unit] * self.output[unit][token] for unit in range(self.hidden_size)) for token in range(self.vocab_size)]
                maximum = max(logits)
                exp_values = [math.exp(logit - maximum) for logit in logits]
                normalizer = sum(exp_values)
                probabilities = [value / normalizer for value in exp_values]
                total -= math.log(max(probabilities[target], 1e-30))
                count += 1
                for token, probability in enumerate(probabilities):
                    delta = probability - (token == target)
                    bias_grad[token] += delta
                    for unit in range(self.hidden_size):
                        output_grad[unit][token] += hidden[unit] * delta
                        embedding_grad[previous][unit] += self.output[unit][token] * delta
        if count == 0:
            raise ValueError("training sequences contain no next-token pairs")
        scale = 1.0 / count
        gradients = {
            "embedding": [[value * scale for value in row] for row in embedding_grad],
            "output": [[value * scale for value in row] for row in output_grad],
            "output_bias": [value * scale for value in bias_grad],
        }
        return total * scale, gradients

    def train_step(self, sequences: list[list[int]], learning_rate: float, *, gradient_clip: float | None = None) -> float:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if gradient_clip is not None and gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        loss, gradients = self.loss_and_gradients(sequences)
        if gradient_clip is not None:
            squared_norm = sum(value * value for rows in (gradients["embedding"], gradients["output"]) for row in rows for value in row) + sum(value * value for value in gradients["output_bias"])
            norm = math.sqrt(squared_norm)
            if norm > gradient_clip:
                scale = gradient_clip / norm
                gradients = {name: ([value * scale for value in values] if name == "output_bias" else [[value * scale for value in row] for row in values]) for name, values in gradients.items()}
        for parameter, gradient in ((self.embedding, gradients["embedding"]), (self.output, gradients["output"])):
            for row, gradient_row in zip(parameter, gradient):
                for index, value in enumerate(gradient_row):
                    row[index] -= learning_rate * value
        for index, value in enumerate(gradients["output_bias"]):
            self.output_bias[index] -= learning_rate * value
        return loss

    def parameter(self, name: str, first: int, second: int | None = None) -> float:
        parameter = getattr(self, name)
        return parameter[first] if second is None else parameter[first][second]

    def set_parameter(self, name: str, first: int, value: float, second: int | None = None) -> None:
        parameter = getattr(self, name)
        if second is None:
            parameter[first] = value
        else:
            parameter[first][second] = value
