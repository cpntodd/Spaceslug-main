"""Deterministic quality metrics for Tiny acceptance reports."""

from __future__ import annotations


def token_accuracy(model, batches) -> float:
    """Compute masked greedy token accuracy over causal batches."""
    correct = total = 0
    for batch in batches:
        for inputs, targets, mask in zip(batch.input_tokens, batch.target_tokens, batch.loss_mask):
            for position, (target, include) in enumerate(zip(targets, mask)):
                if not include:
                    continue
                prediction = max(range(model.vocab_size), key=model.logits_for_tokens(inputs[:position + 1]).__getitem__)
                correct += prediction == target
                total += 1
    if total == 0:
        raise ValueError("accuracy requires at least one masked target")
    return correct / total
