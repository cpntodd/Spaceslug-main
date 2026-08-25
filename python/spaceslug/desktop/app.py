"""Native Tkinter/ttk Phase 1 desktop application shell.

The shell renders a ``DesktopController`` and never runs training, chat, or a
server itself.  Training is represented only by the loss worm graph fed through
the controller; the "start" controls are explicit placeholders.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from .controller import TAB_NAMES, DesktopController
from .profile import fixed_tiny_profile_lines

_LOSS_CANVAS_WIDTH = 480
_LOSS_CANVAS_HEIGHT = 200


class DesktopApp:
    """Tk/ttk shell wired to a controller; owns presentation only."""

    def __init__(self, controller: DesktopController | None = None, root: tk.Tk | None = None) -> None:
        self.controller = controller or DesktopController()
        self.root = root or tk.Tk()
        self.root.title("Spaceslug-main — desktop shell (Phase 1)")
        self.root.geometry("760x640")

        self._vars: dict[str, tk.StringVar] = {}
        self._build()
        self.refresh()

    # -- construction --------------------------------------------------------
    def _variable(self, key: str, value: str, setter: Any | None = None) -> tk.StringVar:
        variable = tk.StringVar(value=value)
        if setter is not None:
            variable.trace_add("write", lambda *_: setter(variable.get()))
        self._vars[key] = variable
        return variable

    def _readonly_text(self, parent: tk.Widget, text: str, height: int = 6) -> tk.Text:
        widget = tk.Text(parent, height=height, wrap="word", relief="flat")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        return widget

    def _build(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self.notebook = notebook

        self._home_frame = self._build_home(notebook)
        self._datasets_frame = self._build_datasets(notebook)
        self._training_frame = self._build_training(notebook)
        self._interact_frame = self._build_interact(notebook)
        self._api_frame = self._build_api(notebook)

        notebook.add(self._home_frame, text="Home")
        notebook.add(self._datasets_frame, text="Datasets")
        notebook.add(self._training_frame, text="Build & Train")
        notebook.add(self._interact_frame, text="Interact")
        notebook.add(self._api_frame, text="Local API")

        self._status = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self._status, anchor="w", relief="sunken", padding=(6, 3))
        status_bar.pack(fill="x", side="bottom")

    def _build_home(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Spaceslug-main desktop shell", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")

        profile_box = ttk.LabelFrame(frame, text="Fixed Tiny profile", padding=8)
        profile_box.pack(fill="x", pady=(8, 4))
        self._profile_labels: list[ttk.Label] = []
        for line in fixed_tiny_profile_lines():
            label = ttk.Label(profile_box, text=line, anchor="w")
            label.pack(fill="x")
            self._profile_labels.append(label)

        placement_box = ttk.LabelFrame(frame, text="GPU-primary / CPU fallback status", padding=8)
        placement_box.pack(fill="x", pady=4)
        self._placement_mode = ttk.Label(placement_box, text="", anchor="w")
        self._placement_mode.pack(fill="x")
        self._placement_device = ttk.Label(placement_box, text="", anchor="w")
        self._placement_device.pack(fill="x")
        self._placement_reason = ttk.Label(placement_box, text="", anchor="w", wraplength=680)
        self._placement_reason.pack(fill="x")

        ttk.Label(frame, text="Tabs: Home · Datasets · Build & Train · Interact · Local API").pack(anchor="w", pady=(8, 0))
        return frame

    def _build_datasets(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Dataset settings (placeholder)", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 2))
        ttk.Label(row, text="File path:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_file", self.controller.dataset_file, self.controller.set_dataset_file)
        ttk.Entry(row, textvariable=self._vars["dataset_file"]).pack(side="left", fill="x", expand=True)

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="URL:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_url", self.controller.dataset_url, self.controller.set_dataset_url)
        ttk.Entry(row, textvariable=self._vars["dataset_url"]).pack(side="left", fill="x", expand=True)

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Search:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_search", self.controller.dataset_search, self.controller.set_dataset_search)
        ttk.Entry(row, textvariable=self._vars["dataset_search"]).pack(side="left", fill="x", expand=True)

        ttk.Label(frame, text="Placeholder: these settings are recorded but not wired to a dataset service in Phase 1.",
                  wraplength=680, foreground="#555").pack(anchor="w", pady=(8, 0))
        return frame

    def _build_training(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Build & Train (placeholder)", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(8, 4))
        ttk.Label(controls, text="Steps:").pack(side="left")
        self._variable("training_steps", str(self.controller.training_steps), self._on_steps)
        ttk.Spinbox(controls, from_=1, to=100000, textvariable=self._vars["training_steps"], width=10).pack(side="left", padx=4)

        self._loss_canvas = tk.Canvas(frame, width=_LOSS_CANVAS_WIDTH, height=_LOSS_CANVAS_HEIGHT, background="#0d1117", highlightthickness=0)
        self._loss_canvas.pack(fill="x", pady=4)

        demo = ttk.Frame(frame)
        demo.pack(fill="x", pady=4)
        ttk.Label(demo, text="Loss value:").pack(side="left")
        self._variable("loss_value", "1.0")
        ttk.Entry(demo, textvariable=self._vars["loss_value"], width=12).pack(side="left", padx=4)
        ttk.Button(demo, text="Record loss", command=self._record_loss).pack(side="left", padx=2)
        ttk.Button(demo, text="Clear", command=self._clear_loss).pack(side="left", padx=2)
        ttk.Button(demo, text="Start training", state="disabled").pack(side="left", padx=2)

        ttk.Label(frame, text="Loss worm graph (controller-driven; no training runs on render)").pack(anchor="w", pady=(4, 2))
        self._readonly_text(frame, self.controller.capability_boundary("training")).pack(fill="x", pady=(2, 0))
        return frame

    def _build_interact(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Interact (placeholder)", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 2))
        ttk.Label(row, text="Prompt:", width=10, anchor="w").pack(side="left")
        self._variable("chat_prompt", self.controller.chat_prompt, self.controller.set_chat_prompt)
        ttk.Entry(row, textvariable=self._vars["chat_prompt"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Send", state="disabled").pack(side="left", padx=(6, 0))

        self._readonly_text(frame, self.controller.capability_boundary("chat")).pack(fill="x", pady=(8, 0))
        return frame

    def _build_api(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Local API (placeholder)", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 2))
        ttk.Label(row, text="Address:", width=10, anchor="w").pack(side="left")
        self._variable("api_address", self.controller.api_address, self.controller.set_api_address)
        ttk.Entry(row, textvariable=self._vars["api_address"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Start API", state="disabled").pack(side="left", padx=(6, 0))

        self._readonly_text(frame, self.controller.capability_boundary("api")).pack(fill="x", pady=(8, 0))
        return frame

    # -- actions -------------------------------------------------------------
    def _on_steps(self, value: str) -> None:
        try:
            self.controller.set_training_steps(int(value))
        except ValueError:
            pass

    def _record_loss(self) -> None:
        try:
            value = float(self._vars["loss_value"].get())
        except ValueError:
            return
        self.controller.record_loss(value)
        self._redraw_loss()

    def _clear_loss(self) -> None:
        self.controller.clear_loss()
        self._redraw_loss()

    # -- rendering -----------------------------------------------------------
    def _redraw_loss(self) -> None:
        self.controller.loss.draw(self._loss_canvas, _LOSS_CANVAS_WIDTH, _LOSS_CANVAS_HEIGHT)

    def refresh(self) -> None:
        snapshot = self.controller.snapshot()
        placement = snapshot["placement"]
        self._placement_mode.configure(text=f"mode: {placement['mode']} (gpu_primary={placement['gpu_primary']}, cpu_fallback={placement['cpu_fallback']})")
        self._placement_device.configure(text=f"device: {placement['device']}")
        self._placement_reason.configure(text=f"reason: {placement['reason']}")
        profile_id = snapshot["profile"]["model_id"]
        self._status.set(
            f"{profile_id} · tab={snapshot['active_tab']} · placement={placement['mode']}"
            f" · device={placement['device']} · steps={snapshot['training_steps']}"
        )
        self._redraw_loss()

    def run(self) -> None:
        try:
            self.controller.refresh_runtime()
        except Exception:
            pass
        self.refresh()
        self.root.mainloop()


def run_desktop(runtime_root: str | None = None, runtime_revision: str = "runtime") -> int:
    """Build and run the desktop shell against a real (but safe) backend probe."""
    from .runtime import RUNTIME_ROOT, backend_runtime_probe

    probe = backend_runtime_probe(runtime_root or RUNTIME_ROOT, runtime_revision)
    app = DesktopApp(controller=DesktopController(runtime_probe=probe))
    app.run()
    return 0
