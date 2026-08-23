"""Single-head causal attention with explicit Q/K/V/output projection parameters."""

from __future__ import annotations

import math

from .batching import CausalBatch
from .positional_encoding import sinusoidal_positions


def _matvec(vector: list[float], matrix: list[list[float]]) -> list[float]:
    return [sum(value * matrix[row][column] for row, value in enumerate(vector)) for column in range(len(matrix[0]))]


class ProjectedTinyAttentionModel:
    """Inspectable forward reference for a projected causal-attention block.

    Parameters are stored separately to make each tensor path and finite-difference
    acceptance target explicit before a complete analytic backward implementation.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 2, use_positions: bool = True) -> None:
        if vocab_size <= 0 or hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        self.vocab_size, self.hidden_size, self.use_positions = vocab_size, hidden_size, use_positions
        self.embedding = [[((token + 1) * (channel + 3) % 17 - 8) / 10.0 for channel in range(hidden_size)] for token in range(vocab_size)]
        def matrix(offset: int) -> list[list[float]]:
            return [[((row + offset) * (column + 5) % 11 - 5) / 10.0 for column in range(hidden_size)] for row in range(hidden_size)]
        self.query, self.key, self.value, self.output = matrix(2), matrix(3), matrix(5), matrix(7)
        self.lm_head = [[((channel + 11) * (token + 3) % 13 - 6) / 10.0 for token in range(vocab_size)] for channel in range(hidden_size)]

    def logits_for_tokens(self, tokens: list[int]) -> list[float]:
        if not tokens:
            raise ValueError("inference requires at least one token")
        states = [self.embedding[token][:] for token in tokens]
        if self.use_positions:
            positions = sinusoidal_positions(len(states), self.hidden_size)
            states = [[value + positions[index][channel] for channel, value in enumerate(state)] for index, state in enumerate(states)]
        keys = [_matvec(state, self.key) for state in states]
        values = [_matvec(state, self.value) for state in states]
        query = _matvec(states[-1], self.query)
        scale = 1.0 / math.sqrt(self.hidden_size)
        scores = [sum(query[channel] * keys[index][channel] for channel in range(self.hidden_size)) * scale for index in range(len(states))]
        maximum = max(scores)
        raw = [math.exp(score - maximum) for score in scores]
        weights = [value / sum(raw) for value in raw]
        context = [sum(weights[index] * values[index][channel] for index in range(len(states))) for channel in range(self.hidden_size)]
        return _matvec(_matvec(context, self.output), self.lm_head)

    def loss_and_gradients(self, batch: CausalBatch) -> tuple[float, dict[str, list[list[float]]]]:
        total, count, scale = 0.0, 0, 1.0 / math.sqrt(self.hidden_size)
        gradients = {name: [[0.0] * self.hidden_size for _ in range(self.hidden_size)] for name in ("query", "key", "value", "output")}
        for inputs, targets, mask in zip(batch.input_tokens, batch.target_tokens, batch.loss_mask):
            states = [self.embedding[token][:] for token in inputs]
            if self.use_positions:
                positions = sinusoidal_positions(len(states), self.hidden_size)
                states = [[value + positions[index][channel] for channel, value in enumerate(state)] for index, state in enumerate(states)]
            keys, values = [_matvec(state, self.key) for state in states], [_matvec(state, self.value) for state in states]
            for position, (target, include) in enumerate(zip(targets, mask)):
                if not include:
                    continue
                query = _matvec(states[position], self.query)
                scores = [sum(query[channel] * keys[index][channel] for channel in range(self.hidden_size)) * scale for index in range(position + 1)]
                maximum = max(scores)
                raw = [math.exp(score - maximum) for score in scores]
                weights = [value / sum(raw) for value in raw]
                context = [sum(weights[index] * values[index][channel] for index in range(position + 1)) for channel in range(self.hidden_size)]
                projected = _matvec(context, self.output)
                logits = _matvec(projected, self.lm_head)
                maximum = max(logits)
                raw = [math.exp(logit - maximum) for logit in logits]
                probabilities = [value / sum(raw) for value in raw]
                total -= math.log(max(probabilities[target], 1e-30))
                count += 1
                d_projected = [sum((probabilities[token] - (token == target)) * self.lm_head[channel][token] for token in range(self.vocab_size)) for channel in range(self.hidden_size)]
                d_context = [sum(d_projected[column] * self.output[channel][column] for column in range(self.hidden_size)) for channel in range(self.hidden_size)]
                for row in range(self.hidden_size):
                    for column in range(self.hidden_size):
                        gradients["output"][row][column] += context[row] * d_projected[column]
                d_weight = [sum(d_context[channel] * values[index][channel] for channel in range(self.hidden_size)) for index in range(position + 1)]
                d_value = [[weights[index] * d_context[channel] for channel in range(self.hidden_size)] for index in range(position + 1)]
                expected = sum(weights[index] * d_weight[index] for index in range(position + 1))
                d_score = [weights[index] * (d_weight[index] - expected) for index in range(position + 1)]
                d_query = [sum(d_score[index] * scale * keys[index][channel] for index in range(position + 1)) for channel in range(self.hidden_size)]
                d_key = [[d_score[index] * scale * query[channel] for channel in range(self.hidden_size)] for index in range(position + 1)]
                for row in range(self.hidden_size):
                    for column in range(self.hidden_size):
                        gradients["query"][row][column] += states[position][row] * d_query[column]
                for index in range(position + 1):
                    for row in range(self.hidden_size):
                        for column in range(self.hidden_size):
                            gradients["key"][row][column] += states[index][row] * d_key[index][column]
                            gradients["value"][row][column] += states[index][row] * d_value[index][column]
        if not count:
            raise ValueError("loss mask contains no target tokens")
        return total / count, {name: [[value / count for value in row] for row in matrix] for name, matrix in gradients.items()}

    def loss(self, batch: CausalBatch) -> float:
        return self.loss_and_gradients(batch)[0]

    def train_step(self, batch: CausalBatch, learning_rate: float) -> float:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        loss, gradients = self.loss_and_gradients(batch)
        for name, matrix in gradients.items():
            parameter = getattr(self, name)
            for row in range(self.hidden_size):
                for column in range(self.hidden_size):
                    parameter[row][column] -= learning_rate * matrix[row][column]
        return loss

    def parameter(self, tensor: str, row: int, column: int) -> float:
        return getattr(self, tensor)[row][column]

    def set_parameter(self, tensor: str, row: int, column: int, value: float) -> None:
        getattr(self, tensor)[row][column] = value
