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
from collections.abc import Callable
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

    def _path_row(self, parent: tk.Widget, key: str, label: str, *, browse: bool = False) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18, anchor="w").pack(side="left")
        self._variable(key, str(getattr(self.controller.paths, key)))
        ttk.Entry(row, textvariable=self._vars[key]).pack(side="left", fill="x", expand=True)
        if browse:
            ttk.Button(row, text="Browse…", command=lambda k=key: self._browse_path(k)).pack(side="left", padx=(6, 0))

    def _browse_path(self, key: str) -> None:
        path = filedialog.askdirectory(title=f"Choose {key.replace('_', ' ')}")
        if path:
            self._vars[key].set(path)

    def _sync_path_entries(self) -> None:
        for key in ("workspace_root", "dataset_dir", "checkpoint_dir", "artifact_dir", "experiment_dir", "temp_dir"):
            self._vars[key].set(str(getattr(self.controller.paths, key)))

    def _apply_workspace_paths(self) -> None:
        try:
            self.controller.set_workspace_paths(
                workspace_root=self._vars["workspace_root"].get(),
                dataset_dir=self._vars["dataset_dir"].get(),
                checkpoint_dir=self._vars["checkpoint_dir"].get(),
                artifact_dir=self._vars["artifact_dir"].get(),
                experiment_dir=self._vars["experiment_dir"].get(),
                temp_dir=self._vars["temp_dir"].get(),
            )
        except Exception as exc:  # surfaced to the user, never silent
            self._set_label(self._workspace_status, f"failed to save paths: {type(exc).__name__}: {exc}")
            return
        self._sync_path_entries()
        self._set_label(self._workspace_status, f"saved workspace paths to {self.controller.config_path}")
        self.refresh()

    def _refresh_gpu_readiness(self) -> None:
        """Refresh the safe native-runtime diagnostic from the Home tab."""
        try:
            self.controller.set_runtime_target(
                self._vars["runtime_root"].get(), self._vars["runtime_revision"].get()
            )
            report = self.controller.refresh_gpu_readiness()
        except Exception as exc:
            text = f"GPU readiness check failed: {type(exc).__name__}: {exc}"
        else:
            text = report.summary_text()
        self._gpu_readiness_text.configure(state="normal")
        self._gpu_readiness_text.delete("1.0", "end")
        self._gpu_readiness_text.insert("1.0", text)
        self._gpu_readiness_text.configure(state="disabled")
        self.refresh()

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

        workspace_box = ttk.LabelFrame(frame, text="Workspace paths (persisted)", padding=8)
        workspace_box.pack(fill="x", pady=4)
        for key, label in (
            ("workspace_root", "Workspace root:"),
            ("dataset_dir", "Dataset dir:"),
            ("checkpoint_dir", "Checkpoint dir:"),
            ("artifact_dir", "Artifact dir:"),
            ("experiment_dir", "Experiment dir:"),
            ("temp_dir", "Temp dir:"),
        ):
            self._path_row(workspace_box, key, label, browse=(key == "workspace_root"))
        apply_row = ttk.Frame(workspace_box)
        apply_row.pack(fill="x", pady=(6, 0))
        ttk.Button(apply_row, text="Apply & save paths", command=self._apply_workspace_paths).pack(side="left")
        self._workspace_status = self._status_label(workspace_box)
        self._set_label(self._workspace_status, f"config: {self.controller.config_path}")

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

        readiness_box = ttk.LabelFrame(frame, text="Native GPU readiness diagnostic", padding=8)
        readiness_box.pack(fill="x", pady=4)
        target = ttk.Frame(readiness_box)
        target.pack(fill="x")
        ttk.Label(target, text="Runtime root:", width=14, anchor="w").pack(side="left")
        self._variable("runtime_root", self.controller.runtime_root)
        ttk.Entry(target, textvariable=self._vars["runtime_root"]).pack(side="left", fill="x", expand=True)
        ttk.Label(target, text="Revision:", width=9, anchor="w").pack(side="left", padx=(6, 0))
        self._variable("runtime_revision", self.controller.runtime_revision)
        ttk.Entry(target, textvariable=self._vars["runtime_revision"], width=12).pack(side="left")
        ttk.Button(target, text="Refresh GPU readiness", command=self._refresh_gpu_readiness).pack(side="left", padx=(6, 0))
        self._gpu_readiness_text = self._readonly_text(readiness_box, "Refresh to inspect the native runtime.", height=10)
        self._gpu_readiness_text.pack(fill="x", pady=(6, 0))

        ttk.Label(frame, text="Tabs: Home · Datasets · Build & Train · Interact · Local API").pack(anchor="w", pady=(8, 0))
        return frame

    def _build_datasets(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Datasets", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        local_box = ttk.LabelFrame(frame, text="Local source", padding=6)
        local_box.pack(fill="x", pady=6)
        row = ttk.Frame(local_box)
        row.pack(fill="x")
        ttk.Label(row, text="File path:", width=14, anchor="w").pack(side="left")
        self._variable("dataset_file", self.controller.dataset_file, self.controller.set_dataset_file)
        ttk.Entry(row, textvariable=self._vars["dataset_file"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Choose file…", command=self._choose_local_file).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Import file", command=self._import_local).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Import folder…", command=self._import_folder).pack(side="left", padx=(6, 0))

        crawl_box = ttk.LabelFrame(frame, text="Controlled documentation crawl (static HTML; explicit approval)", padding=6)
        crawl_box.pack(fill="x", pady=6)
        row = ttk.Frame(crawl_box); row.pack(fill="x")
        ttk.Label(row, text="Start URL:", width=14, anchor="w").pack(side="left")
        self._variable("documentation_url", self.controller.documentation_url, self.controller.set_documentation_url)
        ttk.Entry(row, textvariable=self._vars["documentation_url"]).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Crawl approved", command=self._crawl_documentation).pack(side="left", padx=(6,0))
        self._documentation_status = self._status_label(crawl_box, "Same-origin, bounded, approval-controlled; JavaScript rendering is not enabled.")

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

        sources_box = ttk.LabelFrame(frame, text="Imported documents and information sources", padding=6)
        sources_box.pack(fill="both", expand=True, pady=6)
        columns = ("filename", "extension", "date_imported", "file_size")
        self._source_table = ttk.Treeview(sources_box, columns=columns, show="headings", height=7)
        for key, heading, width in (
            ("filename", "Filename / source", 310),
            ("extension", "Extension / type", 120),
            ("date_imported", "Date imported", 210),
            ("file_size", "File size", 100),
        ):
            self._source_table.heading(key, text=heading)
            self._source_table.column(key, width=width, anchor="w")
        self._source_table.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(sources_box, orient="vertical", command=self._source_table.yview)
        scrollbar.pack(side="right", fill="y")
        self._source_table.configure(yscrollcommand=scrollbar.set)

        self._datasets_status = self._status_label(frame)
        return frame

    def _build_training(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook, padding=10)
        ttk.Label(frame, text="Build & Train", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        tiny_box = ttk.LabelFrame(frame, text="Fixed Tiny settings (rank 4/8, native groups)", padding=6)
        tiny_box.pack(fill="x", pady=(8, 4))
        rank_row = ttk.Frame(tiny_box)
        rank_row.pack(fill="x")
        ttk.Label(rank_row, text="LoRA rank:").pack(side="left")
        self._rank_var = tk.IntVar(value=self.controller.training_rank)
        for rank in (4, 8):
            ttk.Radiobutton(
                rank_row, text=str(rank), value=rank, variable=self._rank_var, command=self._on_rank
            ).pack(side="left", padx=(6, 0))
        self._native_groups_label = ttk.Label(
            tiny_box,
            text=self.controller.native_training_groups_text(),
            anchor="w",
            justify="left",
            foreground="#555",
        )
        self._native_groups_label.pack(fill="x", pady=(4, 0))

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
        self._gpu_train_button = ttk.Button(controls, text="Start native GPU training", command=self._start_gpu_training)
        self._gpu_train_button.pack(side="left", padx=(8, 0))
        self._cpu_train_button = ttk.Button(controls, text="Start CPU reference training", command=self._start_cpu_training)
        self._cpu_train_button.pack(side="left", padx=(4, 0))
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

    def _import_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose a folder of dataset sources")
        if not folder:
            return
        try:
            result = self.controller.import_local_folder(folder, recursive=True)
        except Exception as exc:
            self._set_label(self._datasets_status, f"folder import failed: {type(exc).__name__}: {exc}")
            return
        self._set_label(
            self._datasets_status,
            f"folder import: {len(result['imported'])} imported, {len(result['skipped'])} skipped, "
            f"{len(result['errors'])} failed",
        )
        self.refresh()

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

    def _crawl_documentation(self) -> None:
        try:
            session = self.controller.crawl_documentation(approve_all_same_origin=True)
        except Exception as exc:
            self._set_label(self._documentation_status, f"crawl failed: {type(exc).__name__}: {exc}")
            return
        self._set_label(self._documentation_status, f"crawl complete: {len(session.fetched)} fetched, {len(session.rejected)} rejected, {len(session.errors)} errors; review session provenance before creating .dts")
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

    def _on_rank(self) -> None:
        try:
            self.controller.set_training_rank(self._rank_var.get())
        except ValueError:
            pass
        self.refresh()

    def _begin_training(self, starter: Callable[[], dict]) -> None:
        try:
            starter()
        except Exception as exc:
            self._set_training_status(f"start failed: {type(exc).__name__}: {exc}")
            return
        self._gpu_train_button.configure(state="disabled")
        self._cpu_train_button.configure(state="disabled")
        self._stop_button.configure(state="normal")
        self._schedule_training_poll()

    def _start_gpu_training(self) -> None:
        self._begin_training(self.controller.start_native_gpu_training)

    def _start_cpu_training(self) -> None:
        self._begin_training(self.controller.start_training)

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
            gate = status["native_gpu_readiness"]
            self._gpu_train_button.configure(state="normal" if gate["ready"] else "disabled")
            self._cpu_train_button.configure(state="normal")
            self._stop_button.configure(state="disabled")

    def _refresh_training_status(self, status: dict | None = None) -> None:
        status = status or self.controller.training_status()
        placement = status["placement"]
        lines = [
            f"state: {status['state']} · mode: {status['mode']} · completed_steps: {status['completed_steps']}",
            f"tiny_profile: {status['tiny_profile']} · rank: {status['rank']}",
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
            lines.append(f"backend: {result['backend']} · gpu_execution: {result['gpu_execution']}")
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
        for item in self._source_table.get_children():
            self._source_table.delete(item)
        for row in self.controller.imported_source_rows():
            size = f"{row['file_size']:,} B"
            self._source_table.insert(
                "", "end", values=(row["filename"], row["extension"], row["date_imported"], size)
            )
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
        if snapshot["training"]["state"] != "running":
            gate = snapshot["training"]["native_gpu_readiness"]
            self._gpu_train_button.configure(state="normal" if gate["ready"] else "disabled")
            self._cpu_train_button.configure(state="normal")
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


def runtime_probe_factory(
    runtime_root: str | None = None, runtime_revision: str = "runtime"
) -> Callable[[], dict[str, Any]]:
    """Return a deferred probe callable for :class:`DesktopController`.

    ``backend_runtime_probe`` returns a snapshot dict, but the controller
    expects a zero-argument callable so placement is resolved lazily on
    ``refresh_runtime`` rather than eagerly at construction time.
    """
    from . import runtime

    root = runtime_root or runtime.RUNTIME_ROOT

    def probe() -> dict[str, Any]:
        return runtime.backend_runtime_probe(root, runtime_revision)

    return probe


def run_desktop(runtime_root: str | None = None, runtime_revision: str = "runtime") -> int:
    """Build and run the desktop shell against a real (but safe) backend probe."""
    app = DesktopApp(
        controller=DesktopController(
            runtime_probe=runtime_probe_factory(runtime_root, runtime_revision),
            code_revision=detect_code_revision(),
        )
    )
    app.run()
    return 0
