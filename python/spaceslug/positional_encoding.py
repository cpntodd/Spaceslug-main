"""Deterministic sinusoidal positional encoding reference."""

from __future__ import annotations

import math


def sinusoidal_positions(length: int, width: int) -> list[list[float]]:
    if length <= 0 or width <= 0:
        raise ValueError("length and width must be positive")
    positions = []
    for position in range(length):
        row = []
        for channel in range(width):
            exponent = 2 * (channel // 2) / width
            angle = position / (10000.0 ** exponent)
            row.append(math.sin(angle) if channel % 2 == 0 else math.cos(angle))
        positions.append(row)
    return positions
