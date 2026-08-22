"""Independent host-side reference for the initial vector-add contract."""

from __future__ import annotations


def vector_add(left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("vector lengths differ")
    return [a + b for a, b in zip(left, right)]
