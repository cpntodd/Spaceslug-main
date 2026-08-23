"""Tiny trainable causal-attention CPU reference with explicit masked loss gradient."""

from __future__ import annotations

import math

from .batching import CausalBatch


class TinyAttentionModel:
    """One-head causal attention with a trainable logit scale.

    Token embeddings and output projection are deterministic reference constants;
    the trainable attention scale isolates and verifies causal-attention backward
    semantics before adding the larger projection-parameter surface.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 4, attention_scale: float = 1.0) -> None:
        if vocab_size <= 0 or hidden_size <= 0:
            raise ValueError("vocab_size and hidden_size must be positive")
        self.vocab_size, self.hidden_size, self.attention_scale = vocab_size, hidden_size, attention_scale
        self.embedding = [[((token + 1) * (unit + 3) % 17 - 8) / 10.0 for unit in range(hidden_size)] for token in range(vocab_size)]
        self.output = [[((unit + 5) * (token + 7) % 19 - 9) / 10.0 for token in range(vocab_size)] for unit in range(hidden_size)]

    def loss_and_gradient(self, batch: CausalBatch) -> tuple[float, float]:
        total_loss, total_gradient, count = 0.0, 0.0, 0
        inv_root = 1.0 / math.sqrt(self.hidden_size)
        for inputs, targets, mask in zip(batch.input_tokens, batch.target_tokens, batch.loss_mask):
            hidden = [self.embedding[token] for token in inputs]
            for position, (target, include) in enumerate(zip(targets, mask)):
                if not include:
                    continue
                scores = [sum(hidden[position][channel] * hidden[key][channel] for channel in range(self.hidden_size)) * inv_root for key in range(position + 1)]
                logits = [self.attention_scale * score for score in scores]
                maximum = max(logits)
                raw_weights = [math.exp(value - maximum) for value in logits]
                normalizer = sum(raw_weights)
                weights = [value / normalizer for value in raw_weights]
                attended = [sum(weights[key] * hidden[key][channel] for key in range(position + 1)) for channel in range(self.hidden_size)]
                output_logits = [sum(attended[channel] * self.output[channel][token] for channel in range(self.hidden_size)) for token in range(self.vocab_size)]
                output_maximum = max(output_logits)
                raw_probabilities = [math.exp(value - output_maximum) for value in output_logits]
                probability_normalizer = sum(raw_probabilities)
                probabilities = [value / probability_normalizer for value in raw_probabilities]
                total_loss -= math.log(max(probabilities[target], 1e-30))
                # dL/d attended state.
                state_gradient = [sum((probabilities[token] - (token == target)) * self.output[channel][token] for token in range(self.vocab_size)) for channel in range(self.hidden_size)]
                expected_score = sum(weights[key] * scores[key] for key in range(position + 1))
                # d attended/d scale = sum softmax derivative * value.
                attended_gradient = [sum(weights[key] * (scores[key] - expected_score) * hidden[key][channel] for key in range(position + 1)) for channel in range(self.hidden_size)]
                total_gradient += sum(state_gradient[channel] * attended_gradient[channel] for channel in range(self.hidden_size))
                count += 1
        if count == 0:
            raise ValueError("loss mask contains no target tokens")
        return total_loss / count, total_gradient / count

    def train_step(self, batch: CausalBatch, learning_rate: float) -> float:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        loss, gradient = self.loss_and_gradient(batch)
        self.attention_scale -= learning_rate * gradient
        return loss
