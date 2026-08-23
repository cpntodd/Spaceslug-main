"""Numerical parity reports shared by CPU and Vulkan integration gates."""

from __future__ import annotations


def compare_float_arrays(actual: list[float], expected: list[float], *, tolerance: float = 1e-3) -> dict:
    if len(actual) != len(expected):
        raise ValueError("parity arrays must have equal length")
    maximum = 0.0
    first_bad = None
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        error = abs(float(observed) - float(reference)) / max(1.0, abs(float(reference)))
        maximum = max(maximum, error)
        if first_bad is None and error > tolerance:
            first_bad = index
    return {"status": "pass" if first_bad is None else "fail", "tolerance": tolerance, "max_relative_error": maximum, "first_bad_index": first_bad, "count": len(actual)}
