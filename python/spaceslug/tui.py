"""Barebones mouse-capable terminal UI for the Spaceslug-main workflow.

The UI owns interaction and rendering only; training remains in the service/session
layer. It can be rendered headlessly for tests and run with curses when attached to a
terminal supporting mouse reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import curses
from pathlib import Path

from .filesystem_picker import FileSelection, pick_files
from .model_profiles import profile_names, resolve_profile


@dataclass
class TuiState:
    screen: str = "dashboard"
    selected_path: Path | None = None
    selection: FileSelection | None = None
    model_id: str = "Spaceslug-Tiny"
    model_config: dict = field(default_factory=lambda: resolve_profile("Spaceslug-Tiny"))
    steps: int = 10
    epochs: int = 1
    loss_history: list[float] = field(default_factory=list)
    backend: str = "cpu-reference"
    status: str = "ready"


class SpaceslugTui:
    """Stateful TUI controller with deterministic rendering and key actions."""

    def __init__(self) -> None:
        self.state = TuiState()

    def select_directory(self, path: str | Path) -> FileSelection:
        selection = pick_files(path)
        self.state.selected_path, self.state.selection = Path(path), selection
        self.state.screen = "dataset"
        self.state.status = f"selected {len(selection.files)} files"
        return selection

    def set_model(self, model_id: str, **overrides: object) -> dict:
        self.state.model_id = model_id
        self.state.model_config = resolve_profile(model_id, **overrides)
        self.state.screen = "model"
        return self.state.model_config

    def set_training(self, *, steps: int | None = None, epochs: int | None = None) -> None:
        if steps is not None and steps <= 0 or epochs is not None and epochs <= 0:
            raise ValueError("steps and epochs must be positive")
        if steps is not None:
            self.state.steps = steps
        if epochs is not None:
            self.state.epochs = epochs

    def add_loss(self, value: float) -> None:
        self.state.loss_history.append(float(value))
        self.state.screen = "training"
        self.state.status = f"step {len(self.state.loss_history)}/{self.state.steps}"

    def worm_graph(self, width: int = 48, height: int = 8) -> list[str]:
        values = self.state.loss_history[-width:]
        if not values:
            return [" " * width for _ in range(height)]
        low, high = min(values), max(values)
        span = high - low or 1.0
        canvas = [[" " for _ in range(len(values))] for _ in range(height)]
        for column, value in enumerate(values):
            row = height - 1 - int((value - low) / span * (height - 1))
            canvas[row][column] = "●"
        return ["".join(row).ljust(width) for row in canvas]

    def render(self, width: int = 80) -> str:
        state = self.state
        lines = ["Spaceslug-main :: Tiny workstation", "=" * min(width, 80),
                 f"[{state.screen}] backend={state.backend} status={state.status}",
                 "[d] dataset  [m] model  [t] training  [q] quit", ""]
        if state.screen == "dashboard":
            lines += [f"model: {state.model_id}", f"steps/epochs: {state.steps}/{state.epochs}", "GPU gate: CPU verification required before Vulkan"]
        elif state.screen == "dataset":
            lines += [f"root: {state.selected_path}", f"files: {len(state.selection.files) if state.selection else 0}", "filesystem picker: recursive .txt/.md/.jsonl"]
        elif state.screen == "model":
            lines += [f"profile: {state.model_id}", f"target parameters: {state.model_config['target_parameters']}", f"resolved: {state.model_config['estimated_parameters']}"]
        elif state.screen == "training":
            lines += [f"configured steps={state.steps} epochs={state.epochs}", "worm graph (loss)", *self.worm_graph(width - 2)]
        return "\n".join(lines)

    def run(self) -> None:
        curses.wrapper(self._run_curses)

    def _run_curses(self, window) -> None:
        curses.curs_set(0)
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        window.keypad(True)
        while True:
            window.erase()
            for row, line in enumerate(self.render(window.getmaxyx()[1])):
                if row >= window.getmaxyx()[0] - 1:
                    break
                window.addstr(row, 0, line[:window.getmaxyx()[1] - 1])
            window.refresh()
            key = window.getch()
            if key in (ord("q"), 27):
                return
            if key == ord("d"):
                self.state.screen = "dataset"
            elif key == ord("m"):
                self.state.screen = "model"
            elif key == ord("t"):
                self.state.screen = "training"
            elif key == curses.KEY_MOUSE:
                _, x, y, _, _ = curses.getmouse()
                if y == 3:
                    self.state.screen = ("dataset", "model", "training")[min(x // 12, 2)]
