"""Canvas loss "worm" graph: headless series math plus Canvas rendering.

The math lives here, in plain Python, so tests can assert the computed points
without a Tk root.  ``draw`` accepts any canvas-like object exposing
``delete``/``create_line``/``create_oval``, which keeps the renderer decoupled
from a live ``tkinter.Canvas``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class LossWormGraph:
    """A bounded ring of loss values that renders as a connected line ("worm")."""

    def __init__(self, history: Sequence[float] | None = None, *, max_points: int = 100000) -> None:
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        self.max_points = max_points
        self._history = [float(value) for value in (history or [])]
        if len(self._history) > self.max_points:
            self._history = self._history[-self.max_points :]

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def record(self, value: float) -> int:
        """Append one loss value and return the retained point count."""
        self._history.append(float(value))
        if len(self._history) > self.max_points:
            self._history = self._history[-self.max_points :]
        return len(self._history)

    def clear(self) -> None:
        self._history.clear()

    def latest(self) -> float | None:
        return self._history[-1] if self._history else None

    def bounds(self) -> tuple[float, float] | None:
        if not self._history:
            return None
        return min(self._history), max(self._history)

    def series(self) -> list[tuple[int, float]]:
        """Return ``(step_index, loss)`` pairs for the retained window."""
        return list(enumerate(self._history))

    def viewport(self, start: int = 0, count: int | None = None) -> list[tuple[int, float]]:
        """Return absolute-step/value pairs for a scrollable viewport."""
        begin = max(0, min(int(start), len(self._history)))
        end = len(self._history) if count is None else begin + max(0, int(count))
        return [(begin + index, value) for index, value in enumerate(self._history[begin:end])]

    def scaled_points(self, width: int, height: int, *, padding: int = 8) -> list[tuple[float, float]]:
        """Compute canvas coordinates for the retained window.

        The y axis is inverted so lower losses appear higher on the canvas.  A
        flat series spans vertically so a single value does not divide by zero.
        """
        series = self.series()
        if not series or width <= 0 or height <= 0 or padding < 0:
            return []
        low, high = self.bounds()  # type: ignore[misc]
        span = (high - low) or 1.0
        inner_width = max(1, width - 2 * padding)
        inner_height = max(1, height - 2 * padding)
        count = len(series)
        x_step = inner_width / (count - 1) if count > 1 else 0.0
        points: list[tuple[float, float]] = []
        for index, value in series:
            x = padding + index * x_step
            y = padding + (1.0 - (value - low) / span) * inner_height
            points.append((x, y))
        return points

    def draw(self, canvas: Any, width: int, height: int, *, padding: int = 8, start: int = 0, count: int | None = None) -> None:
        """Render a viewport with a connected line, markers, and value labels."""
        canvas.delete("all")
        start = max(0, min(int(start), len(self._history)))
        values = self._history[start:] if count is None else self._history[start:start + max(0, int(count))]
        if not values:
            canvas.create_text(padding, padding, anchor="nw", text="No loss history", fill="#8b949e")
            return
        view = LossWormGraph(values, max_points=max(1, len(values)))
        points = view.scaled_points(width, height, padding=padding + 18)
        if len(points) >= 2:
            flat = [coordinate for point in points for coordinate in point]
            # Do not use smoothing: every segment must join the measured points.
            canvas.create_line(*flat, fill="#58a6ff", width=2, joinstyle="round")
        radius = 3
        label_step = max(1, len(points) // 24)
        for index, ((x, y), value) in enumerate(zip(points, values)):
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="#58a6ff", outline="")
            if (index % label_step == 0 or index == len(points) - 1) and hasattr(canvas, "create_text"):
                canvas.create_text(x, max(2, y - 10), anchor="s", text=f"{value:.4f}", fill="#f0f6fc", font=("TkDefaultFont", 8))
