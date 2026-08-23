"""Barebones mouse-capable terminal UI for the Spaceslug-main workflow.

The UI owns interaction and rendering only; training remains in the service/session
layer. It can be rendered headlessly for tests and run with curses when attached to a
terminal supporting mouse reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import curses
from pathlib import Path
import tempfile

from .cpu_verification import verify_cpu_training
from .dataset import verify_bundle
from .filesystem_picker import FileSelection, pick_files
from .gpu_gate import run_tiny_gemm_gate
from .inference_session import InferenceSession
from .projected_attention_reference import ProjectedTinyAttentionModel
from .model_profiles import resolve_profile
from .projected_attention_training import ProjectedAttentionConfig, train_projected_attention
from .tokenizer import default_tokenizer


@dataclass
class TuiState:
    screen: str = "dashboard"
    selected_path: Path | None = None
    selection: FileSelection | None = None
    dataset_bundle: Path | None = None
    model_id: str = "Spaceslug-Tiny"
    model_config: dict = field(default_factory=lambda: resolve_profile("Spaceslug-Tiny"))
    steps: int = 10
    epochs: int = 1
    loss_history: list[float] = field(default_factory=list)
    backend: str = "cpu-reference"
    status: str = "ready"
    cpu_verified: bool = False
    last_result: dict | None = None


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

    def select_bundle(self, path: str | Path) -> Path:
        bundle = path if hasattr(path, "manifest") else verify_bundle(path)
        self.state.dataset_bundle = bundle.root
        self.state.screen = "dataset"
        self.state.status = f"dataset verified: {bundle.manifest['revision']}"
        return bundle.root

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
        self.state.status = f"step {len(self.state.loss_history)}/{self.state.steps * self.state.epochs}"

    def verify_cpu(self, bundle_path: str | Path | None = None) -> dict:
        path = bundle_path or self.state.dataset_bundle
        if path is None:
            raise ValueError("select a verified dataset bundle first")
        result = verify_cpu_training(path, steps=min(self.state.steps, 2))
        self.state.cpu_verified = result.passed
        self.state.backend = result.backend
        self.state.status = "CPU gate passed" if result.passed else "CPU gate failed"
        return result.__dict__

    def run_gpu_chain_plan(self, tokens: list[int], *, vocab_size: int = 259) -> dict:
        from .backend import BackendSession
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(vocab_size))
        report = session.run_gpu_chain_plan(tokens)
        self.state.status = f"GPU chain: {report['status']}"
        self.state.backend = report["backend"]
        return report

    def run_gpu_forward_plan(self, tokens: list[int], *, vocab_size: int = 259) -> dict:
        from .backend import BackendSession
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(vocab_size))
        report = session.run_gpu_plan(tokens)
        self.state.status = f"GPU forward: {report['parity']['status']}"
        self.state.backend = report.get("gpu_backend", "vulkan-radv")
        return report

    def compare_gpu_forward_result(self, tokens: list[int], gpu_result, *, vocab_size: int = 259) -> dict:
        from .backend import BackendSession
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(vocab_size))
        report = session.compare_gpu_result(tokens, gpu_result)
        self.state.status = f"GPU forward parity: {report['parity']['status']}"
        self.state.backend = report.get("gpu_backend", "vulkan")
        return report

    def run_cpu_inference(self, tokens: list[int], *, vocab_size: int = 259) -> dict:
        from .backend import BackendSession
        session = InferenceSession(BackendSession("/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", "runtime"), ProjectedTinyAttentionModel(vocab_size))
        record = session.run(tokens)
        self.state.screen = "dashboard"
        self.state.backend = record["backend"]
        self.state.status = f"CPU inference complete: next token {session.next_token(tokens)}"
        return record

    def plan_gpu_forward(self, runtime_root: str | Path, runtime_revision: str) -> dict:
        from .backend import BackendSession
        config = self.state.model_config
        plan = BackendSession(runtime_root, runtime_revision).projected_attention_forward_plan(hidden_size=2, sequence_length=min(self.state.model_config["context_length"], 512), vocab_size=259)
        self.state.status = "Tiny Vulkan forward plan ready"
        return plan

    def verify_gpu_gemm(self, runtime_root: str | Path, runtime_revision: str, *, software_vulkan: bool = False) -> dict:
        if not self.state.cpu_verified:
            raise ValueError("CPU verification is required before Vulkan GEMM")
        cpu = type("CpuResult", (), {"passed": True})()
        result = run_tiny_gemm_gate(cpu, runtime_root, runtime_revision, software_vulkan=software_vulkan)
        self.state.backend = "vulkan-lavapipe" if software_vulkan else "vulkan-radv"
        self.state.status = "Vulkan GEMM parity passed" if result.gpu_passed else "Vulkan GEMM parity failed"
        return result.__dict__

    def train_cpu(self, bundle_path: str | Path | None = None) -> dict:
        path = bundle_path or self.state.dataset_bundle
        if path is None:
            raise ValueError("select a verified dataset bundle first")
        if not self.state.cpu_verified:
            self.verify_cpu(path)
        bundle = path if hasattr(path, "manifest") else verify_bundle(path)
        self.state.loss_history.clear()
        self.state.screen, self.state.status = "training", "training CPU reference"
        with tempfile.TemporaryDirectory(prefix="spaceslug-tui-") as directory:
            model, _, metrics = train_projected_attention(
                bundle,
                ProjectedAttentionConfig(steps=self.state.steps * self.state.epochs, learning_rate=0.1),
                tokenizer=default_tokenizer(),
                on_step=lambda step, loss: self._record_training_loss(step, loss),
            )
        self.state.last_result = metrics
        self.state.status = f"CPU training complete: {metrics['stopped_reason']}"
        return metrics

    def _record_training_loss(self, step: int, loss: float) -> None:
        self.state.loss_history.append(float(loss))
        self.state.status = f"step {step}/{self.state.steps * self.state.epochs}"

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

    def runtime_capabilities(self, runtime_root: str | Path = "/mnt/Data/Projects/Cpntodd_Cactus/vulkan-runtime", runtime_revision: str = "runtime") -> dict:
        from .backend import BackendSession
        capabilities = BackendSession(runtime_root, runtime_revision).capabilities()
        return {"backend": capabilities.backend, "device": capabilities.device, "runtime_revision": capabilities.runtime_revision, "operations": list(capabilities.operations), "software_vulkan": capabilities.software_vulkan}

    def render(self, width: int = 80) -> str:
        state = self.state
        lines = ["Spaceslug-main :: Tiny workstation", "=" * min(width, 80),
                 f"[{state.screen}] backend={state.backend} status={state.status}",
                 "[d] dataset  [m] model  [t] training  [v] verify CPU  [r] run CPU  [g] Vulkan GEMM  [q] quit", ""]
        if state.screen == "dashboard":
            capabilities = self.runtime_capabilities()
            lines += [f"model: {state.model_id}", f"steps/epochs: {state.steps}/{state.epochs}", f"CPU verified: {state.cpu_verified}", f"runtime: {capabilities['device']} ops={','.join(capabilities['operations'])}", "GPU gate: CPU verification required before Vulkan"]
        elif state.screen == "dataset":
            lines += [f"root: {state.selected_path}", f"bundle: {state.dataset_bundle}", f"files: {len(state.selection.files) if state.selection else 0}", "filesystem picker: recursive .txt/.md/.jsonl"]
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
            elif key == ord("v"):
                self.state.screen = "training"
                self.state.status = "use controller API to select bundle and verify"
            elif key == ord("r"):
                self.state.screen = "training"
                self.state.status = "use controller API to start CPU training"
            elif key == ord("g"):
                self.state.screen = "training"
                self.state.status = "use controller API to run Vulkan GEMM parity"
            elif key == curses.KEY_MOUSE:
                _, x, y, _, _ = curses.getmouse()
                if y == 3:
                    self.state.screen = ("dataset", "model", "training")[min(x // 12, 2)]
