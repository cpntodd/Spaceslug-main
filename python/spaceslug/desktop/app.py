"""Native Tkinter/ttk Phase 1 desktop application.

The shell renders a :class:`spaceslug.desktop.controller.DesktopController` and
keeps every Tk operation on the main thread. The only long-running work —
projected Tiny CPU training — runs in a controller-owned background worker
thread; the shell polls its thread-safe loss queue with ``root.after`` so live
loss updates and completion are drawn on the main thread. Dataset import, URL
import, and SearXNG search are user-initiated, time-bounded calls surfaced
through the same controller.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

from .controller import DesktopController, detect_code_revision
from .profile import fixed_tiny_profile_lines

_LOSS_CANVAS_WIDTH = 480
_LOSS_CANVAS_HEIGHT = 200
_POLL_INTERVAL_MS = 200


class DesktopApp:
    """Tk/ttk shell wired to a controller; owns presentation only."""

    def __init__(self, controller: DesktopController | None = None, root: tk.Tk | None = None) -> None:
        self.controller = controller or DesktopController(code_revision=detect_code_revision())
        self.root = root or tk.Tk()
        self.root.title("Spaceslug-main — desktop (Phase 1)")
        self.root.geometry("820x720")

        self._vars: dict[str, tk.StringVar] = {}
        self._polling = False
        self._build()
        self.refresh()

    # -- construction --------------------------------------------------------
    def _variable(self, key: str, value: str, setter: Any | None = None) -> tk.StringVar:
        variable = tk.StringVar(value=value)
        if setter is not None:
            variable.trace_add("write", lambda *_: setter(variable.get()))
        self._vars[key] = variable
        return variable

    def _readonly_text(self, parent: tk.Widget, text: str = "", height: int = 6) -> tk.Text:
        widget = tk.Text(parent, height=height, wrap="word", relief="flat")
        if text:
            widget.insert("1.0", text)
        widget.configure(state="disabled")
        return widget

    def _append_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", text)
        widget.see("end")
        widget.configure(state="disabled")

    def _status_label(self, parent: tk.Widget, text: str = "") -> ttk.Label:
        label = ttk.Label(parent, text=text, wraplength=760, foreground="#444", anchor="w")
        label.pack(fill="x", pady=(4, 0))
        return label

    def _set_label(self, label: ttk.Label, text: str) -> None:
        label.configure(text=text)

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
        ttk.Label(frame, text="Spaceslug-main desktop", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")

        profile_box = ttk.LabelFrame(frame, text="Fixed Tiny profile", padding=8)
        profile_box.pack(fill="x", pady=(8, 4))
        self._profile_labels: list[ttk.Label] = []
        for line in fixed_tiny_profile_lines():
            label = ttk.Label(profile_box, text=line, anchor="w")
            label.pack(fill="x")
            self._profile_labels.append(label)

        placement_box = ttk.LabelFrame(frame, text="GPU-primary / CPU fallback status", padding=8)
        placement_box.pack(fill="x", pady=4)
        self._placement_requested = ttk.Label(placement_box, text="", anchor="w")
        self._placement_requested.pack(fill="x")
        self._placement_actual = ttk.Label(placement_box, text="", anchor="w")
        self._placement_actual.pack(fill="x")
        self._placement_device = ttk.Label(placement_box, text="", anchor="w")
        self._placement_device.pack(fill="x")
        self._placement_reason = ttk.Label(placement_box, text="", anchor="w", wraplength=740)
        self._placement_reason.pack(fill="x")

        ttk.Label(frame, text="Tabs: Home · Datasets · Build & Train · Interact · Local API").pack(anchor="w", pady=(8, 0))
        return frame

    def _build_datasets(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Datasets", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        license_row = ttk.Frame(frame)
        license_row.pack(fill="x", pady=(8, 2))
        ttk.Label(license_row, text="License:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_license", self.controller.dataset_license, self.controller.set_dataset_license)
        ttk.Entry(license_row, textvariable=self._vars["dataset_license"]).pack(side="left", fill="x", expand=True)
        ttk.Label(license_row, text="(required before any import)").pack(side="left", padx=(6, 0))

        local_box = ttk.LabelFrame(frame, text="Local source", padding=6)
        local_box.pack(fill="x", pady=6)
        row = ttk.Frame(local_box)
        row.pack(fill="x")
        ttk.Label(row, text="File path:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_file", self.controller.dataset_file, self.controller.set_dataset_file)
        ttk.Entry(row, textvariable=self._vars["dataset_file"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Choose…", command=self._choose_local_file).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Import local", command=self._import_local).pack(side="left", padx=(6, 0))

        url_box = ttk.LabelFrame(frame, text="Approved URL", padding=6)
        url_box.pack(fill="x", pady=6)
        row = ttk.Frame(url_box)
        row.pack(fill="x")
        ttk.Label(row, text="URL:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_url", self.controller.dataset_url, self.controller.set_dataset_url)
        ttk.Entry(row, textvariable=self._vars["dataset_url"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Import URL", command=self._import_url).pack(side="left", padx=(6, 0))

        search_box = ttk.LabelFrame(frame, text="SearXNG search", padding=6)
        search_box.pack(fill="x", pady=6)
        row = ttk.Frame(search_box)
        row.pack(fill="x")
        ttk.Label(row, text="Query:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_search", self.controller.dataset_search, self.controller.set_dataset_search)
        ttk.Entry(row, textvariable=self._vars["dataset_search"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Search", command=self._run_search).pack(side="left", padx=(6, 0))
        row = ttk.Frame(search_box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="SearXNG URL:", width=14, anchor="w").pack(side="left")
        self._variable("searxng_base_url", self.controller.searxng_base_url, self.controller.set_searxng_base_url)
        ttk.Entry(row, textvariable=self._vars["searxng_base_url"]).pack(side="left", fill="x", expand=True)

        self._search_results = tk.Listbox(search_box, height=5)
        self._search_results.pack(fill="x", pady=(4, 0))
        row = ttk.Frame(search_box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Select result", command=self._select_search_result).pack(side="left")
        ttk.Button(row, text="Import selected", command=self._import_selected).pack(side="left", padx=(6, 0))

        bundle_box = ttk.LabelFrame(frame, text="Dataset bundle (.dts)", padding=6)
        bundle_box.pack(fill="x", pady=6)
        row = ttk.Frame(bundle_box)
        row.pack(fill="x")
        ttk.Label(row, text="Dataset id:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_id", self.controller.dataset_id, self.controller.set_dataset_id)
        ttk.Entry(row, textvariable=self._vars["dataset_id"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Create .dts", command=self._create_dataset).pack(side="left", padx=(6, 0))

        self._datasets_status = self._status_label(frame)
        return frame

    def _build_training(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Build & Train", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(8, 4))
        ttk.Label(controls, text="Steps:").pack(side="left")
        self._variable("training_steps", str(self.controller.training_steps), self._on_steps)
        ttk.Spinbox(controls, from_=1, to=100000, textvariable=self._vars["training_steps"], width=8).pack(side="left", padx=4)
        ttk.Label(controls, text="LR:").pack(side="left", padx=(8, 0))
        self._variable("training_learning_rate", str(self.controller.training_learning_rate), self._on_learning_rate)
        ttk.Entry(controls, textvariable=self._vars["training_learning_rate"], width=8).pack(side="left", padx=4)
        ttk.Label(controls, text="Batch:").pack(side="left", padx=(8, 0))
        self._variable("training_batch_size", str(self.controller.training_batch_size), self._on_batch_size)
        ttk.Spinbox(controls, from_=1, to=64, textvariable=self._vars["training_batch_size"], width=6).pack(side="left", padx=4)
        self._train_button = ttk.Button(controls, text="Start training", command=self._start_training)
        self._train_button.pack(side="left", padx=(8, 0))
        self._stop_button = ttk.Button(controls, text="Stop", command=self._stop_training, state="disabled")
        self._stop_button.pack(side="left", padx=(4, 0))

        self._loss_canvas = tk.Canvas(frame, width=_LOSS_CANVAS_WIDTH, height=_LOSS_CANVAS_HEIGHT, background="#0d1117", highlightthickness=0)
        self._loss_canvas.pack(fill="x", pady=4)

        ttk.Label(frame, text="Live loss worm (fed by the training worker thread)").pack(anchor="w", pady=(4, 2))
        self._training_status_text = self._readonly_text(frame, self.controller.capability_boundary("training"), height=10)
        self._training_status_text.pack(fill="x", pady=(2, 0))
        return frame

    def _build_interact(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Interact", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        self._responder_label = ttk.Label(frame, text="", anchor="w", foreground="#555")
        self._responder_label.pack(anchor="w", pady=(2, 4))

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(4, 2))
        ttk.Label(row, text="Prompt:", width=10, anchor="w").pack(side="left")
        self._variable("chat_prompt", self.controller.chat_prompt, self.controller.set_chat_prompt)
        ttk.Entry(row, textvariable=self._vars["chat_prompt"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Send", command=self._send_chat).pack(side="left", padx=(6, 0))

        ttk.Label(frame, text="Transcript").pack(anchor="w", pady=(6, 2))
        self._chat_text = self._readonly_text(frame, height=14)
        self._chat_text.pack(fill="both", expand=True)
        return frame

    def _build_api(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Local API", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(8, 2))
        ttk.Label(row, text="Address:", width=10, anchor="w").pack(side="left")
        self._variable("api_address", self.controller.api_address, self.controller.set_api_address)
        ttk.Entry(row, textvariable=self._vars["api_address"]).pack(side="left", fill="x", expand=True)
        self._api_start_button = ttk.Button(row, text="Start API", command=self._start_api)
        self._api_start_button.pack(side="left", padx=(6, 0))
        self._api_stop_button = ttk.Button(row, text="Stop API", command=self._stop_api, state="disabled")
        self._api_stop_button.pack(side="left", padx=(4, 0))

        self._api_status_label = self._status_label(frame)
        self._readonly_text(frame, self.controller.capability_boundary("api")).pack(fill="x", pady=(8, 0))
        return frame

    # -- Datasets actions ----------------------------------------------------
    def _choose_local_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a local dataset source",
            filetypes=[("Dataset sources", "*.txt *.md *.jsonl *.pdf"), ("All files", "*.*")],
        )
        if path:
            self.controller.set_dataset_file(path)
            self._vars["dataset_file"].set(path)

    def _import_local(self) -> None:
        try:
            imported = self.controller.import_local_source(self._vars["dataset_file"].get())
        except Exception as exc:  # surfaced to the user, never silent
            self._set_label(self._datasets_status, f"import failed: {type(exc).__name__}: {exc}")
            return
        self._set_label(
            self._datasets_status,
            f"imported local source sha256={imported.sha256} ({imported.bytes} bytes, {len(imported.records)} records)",
        )
        self.refresh()

    def _import_url(self) -> None:
        try:
            imported = self.controller.import_approved_url(self._vars["dataset_url"].get())
        except Exception as exc:
            self._set_label(self._datasets_status, f"import failed: {type(exc).__name__}: {exc}")
            return
        self._set_label(
            self._datasets_status,
            f"imported URL {imported.source} sha256={imported.sha256} ({len(imported.records)} records)",
        )
        self.refresh()

    def _run_search(self) -> None:
        try:
            results = self.controller.run_search()
        except Exception as exc:
            self._set_label(self._datasets_status, f"search failed: {type(exc).__name__}: {exc}")
            return
        self._search_results.delete(0, "end")
        for result in results:
            self._search_results.insert("end", f"{result.title} — {result.url}")
        self._set_label(self._datasets_status, f"{len(results)} search results (select one, then import it)")
        self.refresh()

    def _select_search_result(self) -> None:
        selection = self._search_results.curselection()
        if not selection:
            self._set_label(self._datasets_status, "select a search result first")
            return
        try:
            result = self.controller.select_search_result(selection[0])
        except ValueError as exc:
            self._set_label(self._datasets_status, str(exc))
            return
        self._set_label(self._datasets_status, f"selected: {result.title} — {result.url}")
        self.refresh()

    def _import_selected(self) -> None:
        try:
            imported = self.controller.import_selected_search_result()
        except Exception as exc:
            self._set_label(self._datasets_status, f"import failed: {type(exc).__name__}: {exc}")
            return
        self._set_label(self._datasets_status, f"imported selected URL sha256={imported.sha256}")
        self.refresh()

    def _create_dataset(self) -> None:
        try:
            bundle = self.controller.create_dataset()
        except Exception as exc:
            self._set_label(self._datasets_status, f"create .dts failed: {type(exc).__name__}: {exc}")
            return
        self._set_label(
            self._datasets_status,
            f"created .dts {bundle.root} (revision={bundle.manifest['revision']}, "
            f"records={bundle.manifest['record_count']})",
        )
        self.refresh()

    # -- Training actions ----------------------------------------------------
    def _on_steps(self, value: str) -> None:
        try:
            self.controller.set_training_steps(int(value))
        except ValueError:
            pass

    def _on_learning_rate(self, value: str) -> None:
        try:
            self.controller.set_training_learning_rate(float(value))
        except ValueError:
            pass

    def _on_batch_size(self, value: str) -> None:
        try:
            self.controller.set_training_batch_size(int(value))
        except ValueError:
            pass

    def _start_training(self) -> None:
        try:
            self.controller.start_training()
        except Exception as exc:
            self._set_training_status(f"start failed: {type(exc).__name__}: {exc}")
            return
        self._train_button.configure(state="disabled")
        self._stop_button.configure(state="normal")
        self._schedule_training_poll()

    def _stop_training(self) -> None:
        self.controller.stop_training()

    def _schedule_training_poll(self) -> None:
        if self._polling:
            return
        self._polling = True
        self.root.after(_POLL_INTERVAL_MS, self._poll_training)

    def _poll_training(self) -> None:
        status = self.controller.poll_training()
        self._redraw_loss()
        self._refresh_training_status(status)
        if status["state"] == "running":
            self.root.after(_POLL_INTERVAL_MS, self._poll_training)
        else:
            self._polling = False
            self._train_button.configure(state="normal")
            self._stop_button.configure(state="disabled")

    def _refresh_training_status(self, status: dict | None = None) -> None:
        status = status or self.controller.training_status()
        placement = status["placement"]
        lines = [
            f"state: {status['state']} · completed_steps: {status['completed_steps']}",
            f"requested: {placement['requested']} · actual: {placement['actual']} · hardware: {placement['hardware']}",
            placement["reason"],
        ]
        if status["latest_loss"] is not None:
            lines.append(f"latest_loss: {status['latest_loss']:.6f}")
        if status["paths"]:
            paths = status["paths"]
            lines.append(f"run_id: {paths['run_id']}")
            lines.append(f"checkpoint: {paths['checkpoint']}")
            lines.append(f"artifact: {paths['artifact']}")
            lines.append(f"experiment: {paths['experiment']}")
        if status["result"]:
            result = status["result"]
            lines.append(
                f"artifact_revision: {result['artifact_revision']} · stopped: {result['stopped_reason']} · "
                f"loss {result['initial_train_loss']:.6f} → {result['final_train_loss']:.6f}"
            )
        if status["error"]:
            lines.append(f"error: {status['error']}")
        lines.append("")
        lines.append(self.controller.capability_boundary("training"))
        self._set_training_status("\n".join(lines))

    def _set_training_status(self, text: str) -> None:
        self._training_status_text.configure(state="normal")
        self._training_status_text.delete("1.0", "end")
        self._training_status_text.insert("1.0", text)
        self._training_status_text.configure(state="disabled")

    # -- Interact actions ----------------------------------------------------
    def _send_chat(self) -> None:
        prompt = self._vars["chat_prompt"].get()
        if not prompt.strip():
            return
        try:
            reply = self.controller.send_chat(prompt)
        except Exception as exc:
            self._append_text(self._chat_text, f"\n[error] {type(exc).__name__}: {exc}\n")
            return
        self._append_text(self._chat_text, f"\n> {prompt}\n{reply}\n")
        self._vars["chat_prompt"].set("")
        self.refresh()

    # -- API actions ---------------------------------------------------------
    def _start_api(self) -> None:
        try:
            status = self.controller.start_api()
        except Exception as exc:
            self._set_label(self._api_status_label, f"start failed: {type(exc).__name__}: {exc}")
            return
        self._api_start_button.configure(state="disabled")
        self._api_stop_button.configure(state="normal")
        self._set_label(self._api_status_label, f"running at {status['base_url']} ({status['backend']}/{status['model']})")
        self.refresh()

    def _stop_api(self) -> None:
        self.controller.stop_api()
        self._api_start_button.configure(state="normal")
        self._api_stop_button.configure(state="disabled")
        self._set_label(self._api_status_label, "stopped")
        self.refresh()

    # -- rendering -----------------------------------------------------------
    def _redraw_loss(self) -> None:
        self.controller.loss.draw(self._loss_canvas, _LOSS_CANVAS_WIDTH, _LOSS_CANVAS_HEIGHT)

    def refresh(self) -> None:
        snapshot = self.controller.snapshot()
        placement = snapshot["placement"]
        training_placement = snapshot["training_placement"]
        self._placement_requested.configure(text=f"requested: {training_placement['requested']} (gpu_primary_requested={snapshot['gpu_primary_requested']})")
        self._placement_actual.configure(text=f"actual: {training_placement['actual']} · hardware: {training_placement['hardware']}")
        self._placement_device.configure(text=f"device: {placement['device']}")
        self._placement_reason.configure(text=f"reason: {training_placement['reason']}")
        responder = snapshot["responder"]
        self._responder_label.configure(text=f"responder: {responder['backend']} / {responder['model']}")
        profile_id = snapshot["profile"]["model_id"]
        api = snapshot["api"]
        self._status.set(
            f"{profile_id} · tab={snapshot['active_tab']} · placement={training_placement['actual']}"
            f" · api={'on' if api['running'] else 'off'} · training={snapshot['training']['state']}"
        )
        self._redraw_loss()
        self._refresh_training_status(snapshot["training"])
        if not snapshot["api"]["running"]:
            self._api_start_button.configure(state="normal")
            self._api_stop_button.configure(state="disabled")

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
    app = DesktopApp(controller=DesktopController(runtime_probe=probe, code_revision=detect_code_revision()))
    app.run()
    return 0
