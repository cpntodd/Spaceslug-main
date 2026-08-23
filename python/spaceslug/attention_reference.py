"""Dependency-free causal self-attention reference for Spaceslug-Tiny."""

from __future__ import annotations

import math


def causal_self_attention(inputs: list[list[float]]) -> list[list[float]]:
    """Single-head scaled dot-product attention with a strict causal mask.

    Query, key, and value are intentionally the supplied hidden states. This is
    a forward semantic reference used before introducing trainable projections.
    """
    if not inputs or not inputs[0]:
        raise ValueError("attention requires non-empty inputs")
    width = len(inputs[0])
    if any(len(row) != width for row in inputs):
        raise ValueError("attention inputs must be rectangular")
    scale = 1.0 / math.sqrt(width)
    outputs: list[list[float]] = []
    for index, query in enumerate(inputs):
        logits = [sum(query[channel] * inputs[key][channel] for channel in range(width)) * scale for key in range(index + 1)]
        maximum = max(logits)
        weights = [math.exp(logit - maximum) for logit in logits]
        normalizer = sum(weights)
        outputs.append([sum(weights[key] / normalizer * inputs[key][channel] for key in range(index + 1)) for channel in range(width)])
    return outputs
