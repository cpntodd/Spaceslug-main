"""Testable desktop shell controller.

The controller owns all desktop state and workflow logic and never imports or
constructs a Tk root, so it can be exercised headlessly.  Rendering and Tk
wiring live in :mod:`spaceslug.desktop.app`, which consumes a controller
instance and runs only presentation work on the main thread.

Every Phase 1 workflow is driven through this object: local source
choosing/import, approved URL import, SearXNG search/selection, deterministic
``.dts`` creation, CPU projected Tiny training (run in a background worker
thread with a live per-step loss callback feeding a thread-safe queue),
explicit GPU-primary requested/actual placement, loopback OpenAI API start/stop,
and chat through an injected responder (or the server's default responder).
"""

from __future__ import annotations

import queue
import subprocess
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from ..dataset import verify_bundle
from ..filesystem_picker import FileSelection, pick_files
from ..openai_api import (
    LOOPBACK_HOSTS,
    ModelResponder,
    OpenAICompatibleServer,
    TinyCpuEchoResponder,
)
from ..projected_attention_training import (
    ProjectedAttentionConfig,
    build_training_bundle,
    run_projected_training,
)
from ..tokenizer import default_tokenizer
from ..workspace import IngestionError, SearchResult, WorkspaceService, searxng_search
from .capabilities import (
    api_capability_text,
    chat_capability_text,
    datasets_capability_text,
    structured_capabilities,
    training_capability_text,
)
from .loss_graph import LossWormGraph
from .profile import fixed_tiny_profile
from .runtime import (
    RuntimePlacement,
    TrainingPlacement,
    default_runtime_probe,
    resolve_placement,
    resolve_training_placement,
)

TAB_NAMES = ("Home", "Datasets", "Build & Train", "Interact", "Local API")

DEFAULT_WORKSPACE_ROOT = Path.home() / ".spaceslug" / "workspace"
DEFAULT_SEARXNG_BASE_URL = "http://127.0.0.1:8888"
DEFAULT_API_ADDRESS = "127.0.0.1:8123"
LOCAL_SOURCE_EXTENSIONS = (".txt", ".md", ".jsonl")

# Explicit capability boundaries shown on each tab. Phase 1 workflows are wired
# and these texts are assembled from the live module capability metadata rather
# than a stale snapshot (see .capabilities).
CAPABILITY_BOUNDARIES: dict[str, str] = {
    "training": training_capability_text(),
    "chat": chat_capability_text(),
    "api": api_capability_text(),
    "datasets": datasets_capability_text(),
}


def detect_code_revision() -> str:
    """Return the current git revision, or ``unrecorded`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
        )
    except (OSError, subprocess.SubprocessError):
        return "unrecorded"
    revision = result.stdout.strip()
    return revision or "unrecorded"


def parse_api_address(address: str, *, default_port: int = 8123) -> tuple[str, int]:
    """Parse a ``host:port`` address into a loopback host and an integer port."""
    if not isinstance(address, str) or not address.strip():
        raise ValueError("API address must be a non-empty host:port string")
    value = address.strip()
    if value.startswith("["):  # bracketed IPv6 literal
        host, _, rest = value[1:].partition("]")
        port_text = rest[1:] if rest.startswith(":") else ""
    elif ":" in value:
        host, _, port_text = value.rpartition(":")
    else:
        host, port_text = value, ""
    host = host.strip() or "127.0.0.1"
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"API host must be loopback, got {host!r}")
    if not port_text:
        port = default_port
    else:
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError(f"invalid API port: {port_text!r}") from exc
    if not (0 <= port <= 65535):
        raise ValueError(f"API port out of range: {port}")
    return host, port


@dataclass(frozen=True)
class TrainingJobPaths:
    """Resolved output paths for one projected Tiny training run."""

    run_id: str
    checkpoint: Path
    artifact: Path
    experiment: Path


class DesktopController:
    """State and actions for the desktop shell; no Tk imports."""

    def __init__(
        self,
        runtime_probe: Callable[[], dict[str, Any]] | None = None,
        *,
        history: Sequence[float] | None = None,
        workspace: WorkspaceService | None = None,
        workspace_root: str | Path | None = None,
        responder: ModelResponder | None = None,
        code_revision: str | None = None,
    ) -> None:
        self.runtime_probe = runtime_probe or default_runtime_probe
        self.active_tab = "Home"
        self.profile = fixed_tiny_profile()
        self.loss = LossWormGraph(history)
        # Static until refresh_runtime() runs; keeps construction side-effect free.
        self.placement: RuntimePlacement = RuntimePlacement(
            mode="cpu-fallback",
            device=None,
            software_vulkan=False,
            gpu_primary=False,
            cpu_fallback=True,
            reason="runtime not probed yet",
        )
        self.gpu_primary_requested = True

        # Workspace service (lazily constructed; injectable for tests).
        self.workspace_root = Path(workspace_root) if workspace_root else DEFAULT_WORKSPACE_ROOT
        self._workspace = workspace

        # Responder shared by chat and the local API server.
        self.responder = responder
        self.code_revision = code_revision if code_revision is not None else "unrecorded"

        # Dataset workflow state.
        self.dataset_file = ""
        self.dataset_url = ""
        self.dataset_search = ""
        self.dataset_license = ""
        self.dataset_id = "spaceslug-dataset"
        self.searxng_base_url = DEFAULT_SEARXNG_BASE_URL
        self.imports: list[Any] = []
        self.last_import: Any = None
        self.search_results: list[SearchResult] = []
        self.selected_search_result: int | None = None
        self.created_bundle = ""
        self.dataset_revision: str | None = None
        self.training_bundle_path = ""

        # Training workflow state.
        self.training_steps = 10
        self.training_learning_rate = 0.2
        self.training_batch_size = 1
        self.training_prompt = ""
        self.latest_loss: float | None = None
        self.completed_steps = 0
        self._training_thread: threading.Thread | None = None
        self._training_state = "idle"  # idle | running | finished | error
        self._training_error: BaseException | None = None
        self._training_result: dict | None = None
        self._training_paths: TrainingJobPaths | None = None
        self._loss_queue: "queue.Queue[tuple[int, float]]" = queue.Queue()
        self._cancel_event = threading.Event()

        # Chat and API workflow state.
        self.chat_prompt = ""
        self.chat_history: list[tuple[str, str]] = []
        self.api_address = DEFAULT_API_ADDRESS
        self._server: OpenAICompatibleServer | None = None

    # -- tabs ----------------------------------------------------------------
    def set_active_tab(self, name: str) -> str:
        if name not in TAB_NAMES:
            raise ValueError(f"unknown tab: {name}")
        self.active_tab = name
        return name

    # -- runtime / placement -------------------------------------------------
    def refresh_runtime(self) -> RuntimePlacement:
        self.placement = resolve_placement(self.runtime_probe())
        return self.placement

    def runtime_status(self) -> RuntimePlacement:
        return self.placement

    def set_gpu_primary_requested(self, value: bool) -> None:
        self.gpu_primary_requested = bool(value)

    def training_placement(self) -> TrainingPlacement:
        return resolve_training_placement(self.gpu_primary_requested, self.placement)

    # -- loss worm -----------------------------------------------------------
    def record_loss(self, value: float) -> int:
        return self.loss.record(value)

    def clear_loss(self) -> None:
        self.loss.clear()

    # -- workspace -----------------------------------------------------------
    def _workspace_service(self) -> WorkspaceService:
        if self._workspace is None:
            self._workspace = WorkspaceService(self.workspace_root)
        return self._workspace

    def _resolve_license(self, license: str | None) -> str:
        return license if license is not None else self.dataset_license

    def _record_import(self, imported: Any) -> Any:
        self.imports.append(imported)
        self.last_import = imported
        return imported

    def list_imports(self) -> list[Any]:
        return list(self._workspace_service().list_imports())

    # -- dataset settings ----------------------------------------------------
    def set_dataset_file(self, value: str) -> None:
        self.dataset_file = value

    def set_dataset_url(self, value: str) -> None:
        self.dataset_url = value

    def set_dataset_search(self, value: str) -> None:
        self.dataset_search = value

    def set_dataset_license(self, value: str) -> None:
        self.dataset_license = value

    def set_dataset_id(self, value: str) -> None:
        self.dataset_id = value

    def set_searxng_base_url(self, value: str) -> None:
        self.searxng_base_url = value

    # -- local source choosing / import --------------------------------------
    def choose_local_sources(
        self,
        root: str | Path,
        *,
        recursive: bool = True,
    ) -> FileSelection:
        """Discover importable local text sources (no PDF; separate work item)."""
        return pick_files(
            root,
            recursive=recursive,
            extensions=LOCAL_SOURCE_EXTENSIONS,
        )

    def import_local_source(self, path: str | Path, *, license: str | None = None) -> Any:
        imported = self._workspace_service().import_local(path, license=self._resolve_license(license))
        self.dataset_file = str(path)
        return self._record_import(imported)

    # -- approved URL import -------------------------------------------------
    def import_approved_url(self, url: str, *, license: str | None = None) -> Any:
        imported = self._workspace_service().import_url(
            url, license=self._resolve_license(license), approved=True
        )
        self.dataset_url = url
        return self._record_import(imported)

    # -- SearXNG search / selection ------------------------------------------
    def run_search(self, query: str | None = None, *, base_url: str | None = None) -> list[SearchResult]:
        text = query if query is not None else self.dataset_search
        base = base_url if base_url is not None else self.searxng_base_url
        self.search_results = searxng_search(text, base_url=base)
        self.selected_search_result = None
        return list(self.search_results)

    def select_search_result(self, index: int) -> SearchResult:
        if not 0 <= index < len(self.search_results):
            raise ValueError(f"search result index out of range: {index}")
        self.selected_search_result = index
        return self.search_results[index]

    def import_selected_search_result(self, *, license: str | None = None) -> Any:
        if self.selected_search_result is None:
            raise IngestionError("no search result selected")
        return self.import_approved_url(
            self.search_results[self.selected_search_result].url,
            license=license,
        )

    # -- .dts creation -------------------------------------------------------
    def create_dataset(self, dataset_id: str | None = None, *, split: str = "train") -> Any:
        bundle = self._workspace_service().create_dataset(
            dataset_id if dataset_id is not None else self.dataset_id,
            self.imports,
            split=split,
        )
        self.created_bundle = str(bundle.root)
        self.dataset_revision = bundle.manifest["revision"]
        return bundle

    def build_training_bundle(self, *, prompt: str | None = None) -> Any:
        """Derive a prompt/target training ``.dts`` from the canonical bundle."""
        if not self.created_bundle:
            raise IngestionError("create a .dts from imported sources before training")
        output = self._workspace_service().bundles_dir / f"{Path(self.created_bundle).name}-train.dts"
        bundle = build_training_bundle(
            verify_bundle(self.created_bundle),
            output,
            prompt=self.training_prompt if prompt is None else prompt,
        )
        self.training_bundle_path = str(bundle.root)
        return bundle

    # -- training ------------------------------------------------------------
    def set_training_steps(self, steps: int) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.training_steps = steps

    def set_training_learning_rate(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.training_learning_rate = value

    def set_training_batch_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("batch_size must be positive")
        self.training_batch_size = value

    def set_training_prompt(self, value: str) -> None:
        self.training_prompt = value

    def _resolve_training_bundle(self, bundle_path: str | Path | None) -> Path:
        if bundle_path is not None:
            return Path(bundle_path)
        if self.training_bundle_path:
            return Path(self.training_bundle_path)
        if self.created_bundle:
            self.build_training_bundle()
            return Path(self.training_bundle_path)
        raise IngestionError("no dataset bundle available for training")

    def _resolve_training_paths(
        self,
        checkpoint: str | Path | None,
        artifact: str | Path | None,
        experiment: str | Path | None,
    ) -> TrainingJobPaths:
        run_id = f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        root = self._workspace_service().root
        checkpoint_path = Path(checkpoint) if checkpoint is not None else root / "checkpoints" / f"{run_id}.json"
        artifact_path = Path(artifact) if artifact is not None else root / "artifacts" / f"{run_id}.spaceslug"
        experiment_path = Path(experiment) if experiment is not None else root / "experiments" / run_id
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
        return TrainingJobPaths(run_id, checkpoint_path, artifact_path, experiment_path)

    def start_training(
        self,
        *,
        bundle_path: str | Path | None = None,
        steps: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        checkpoint: str | Path | None = None,
        artifact: str | Path | None = None,
        experiment: str | Path | None = None,
    ) -> dict:
        """Start CPU projected Tiny training in a background worker thread."""
        if self._training_thread is not None and self._training_thread.is_alive():
            raise RuntimeError("training is already running")
        bundle = self._resolve_training_bundle(bundle_path)
        config = ProjectedAttentionConfig(
            steps=self.training_steps if steps is None else steps,
            learning_rate=self.training_learning_rate if learning_rate is None else learning_rate,
            batch_size=self.training_batch_size if batch_size is None else batch_size,
        )
        paths = self._resolve_training_paths(checkpoint, artifact, experiment)
        self._training_paths = paths
        self._training_result = None
        self._training_error = None
        self._training_state = "running"
        self.latest_loss = None
        self.completed_steps = 0
        self._cancel_event.clear()
        self._training_thread = threading.Thread(
            target=self._run_training,
            args=(bundle, config, paths),
            name="spaceslug-desktop-training",
            daemon=True,
        )
        self._training_thread.start()
        return self.training_status()

    def _run_training(self, bundle: Path, config: ProjectedAttentionConfig, paths: TrainingJobPaths) -> None:
        try:
            result = run_projected_training(
                verify_bundle(bundle),
                config,
                tokenizer=default_tokenizer(),
                checkpoint=paths.checkpoint,
                artifact=paths.artifact,
                experiment=paths.experiment,
                code_revision=self.code_revision,
                on_step=self._on_training_step,
                should_stop=self._cancel_event.is_set,
            )
            self._training_result = result
            self._training_state = "finished"
        except BaseException as exc:  # surfaced to the UI and tests, never silent
            self._training_error = exc
            self._training_state = "error"

    def _on_training_step(self, step: int, loss: float) -> None:
        """Thread-safe per-step callback; the main thread drains the queue."""
        self._loss_queue.put((step, loss))

    def poll_training(self) -> dict:
        """Drain pending loss callbacks on the calling (main) thread."""
        while True:
            try:
                step, loss = self._loss_queue.get_nowait()
            except queue.Empty:
                break
            self.record_loss(loss)
            self.latest_loss = loss
            self.completed_steps = step
        return self.training_status()

    def wait_training(self, timeout: float | None = None) -> dict:
        """Join the worker thread (best-effort) and drain remaining callbacks."""
        thread = self._training_thread
        if thread is not None:
            thread.join(timeout)
        return self.poll_training()

    def stop_training(self) -> bool:
        """Request cooperative cancellation; returns True when a run is active."""
        if self._training_thread is not None and self._training_thread.is_alive():
            self._cancel_event.set()
            return True
        return False

    def training_status(self) -> dict:
        status: dict[str, Any] = {
            "state": self._training_state,
            "completed_steps": self.completed_steps,
            "latest_loss": self.latest_loss,
            "loss_history": self.loss.history,
            "placement": self.training_placement().to_dict(),
            "paths": None,
            "result": None,
            "error": None,
        }
        if self._training_paths is not None:
            status["paths"] = {
                "run_id": self._training_paths.run_id,
                "checkpoint": str(self._training_paths.checkpoint),
                "artifact": str(self._training_paths.artifact),
                "experiment": str(self._training_paths.experiment),
            }
        if self._training_result is not None:
            status["result"] = {
                "artifact_revision": self._training_result.get("artifact_revision"),
                "experiment": self._training_result.get("experiment"),
                "stopped_reason": self._training_result.get("metrics", {}).get("stopped_reason"),
                "final_train_loss": self._training_result.get("metrics", {}).get("final_train_loss"),
                "initial_train_loss": self._training_result.get("metrics", {}).get("initial_train_loss"),
            }
        if self._training_error is not None:
            status["error"] = f"{type(self._training_error).__name__}: {self._training_error}"
        return status

    # -- chat ----------------------------------------------------------------
    def _responder(self) -> ModelResponder:
        if self.responder is None:
            self.responder = TinyCpuEchoResponder()
        return self.responder

    def set_chat_prompt(self, value: str) -> None:
        self.chat_prompt = value

    def send_chat(self, prompt: str | None = None) -> str:
        text = prompt if prompt is not None else self.chat_prompt
        result = self._responder().respond([{"role": "user", "content": text}])
        self.chat_history.append(("user", text))
        self.chat_history.append(("assistant", result.content))
        self.chat_prompt = ""
        return result.content

    def responder_identity(self) -> dict[str, str]:
        responder = self._responder()
        return {"backend": responder.backend_id, "model": responder.model_id}

    # -- local API -----------------------------------------------------------
    def set_api_address(self, value: str) -> None:
        parse_api_address(value)  # validate eagerly; raises on a bad address
        self.api_address = value

    def start_api(self, address: str | None = None) -> dict:
        if self._server is not None and self._server.is_running:
            raise RuntimeError("API server is already running")
        target = self.api_address if address is None else address
        host, port = parse_api_address(target)
        self._server = OpenAICompatibleServer(host=host, port=port, responder=self._responder())
        self._server.start()
        self.api_address = self._server.base_url
        return self.api_status()

    def stop_api(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None

    def api_status(self) -> dict[str, Any]:
        if self._server is not None and self._server.is_running:
            responder = self._server.responder
            return {
                "running": True,
                "base_url": self._server.base_url,
                "backend": responder.backend_id,
                "model": responder.model_id,
            }
        responder = self._responder()
        return {
            "running": False,
            "base_url": self.api_address,
            "backend": responder.backend_id,
            "model": responder.model_id,
        }

    # -- capability boundaries ----------------------------------------------
    def capability_boundary(self, area: str) -> str:
        return CAPABILITY_BOUNDARIES[area]

    def capabilities(self) -> dict[str, Any]:
        return structured_capabilities()

    # -- pure read model -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return a render-ready view without probing, training, or model work."""
        return {
            "active_tab": self.active_tab,
            "profile": dict(self.profile),
            "placement": self.placement.to_dict(),
            "training_placement": self.training_placement().to_dict(),
            "gpu_primary_requested": self.gpu_primary_requested,
            "loss_history": self.loss.history,
            "dataset_file": self.dataset_file,
            "dataset_url": self.dataset_url,
            "dataset_search": self.dataset_search,
            "dataset_license": self.dataset_license,
            "dataset_id": self.dataset_id,
            "searxng_base_url": self.searxng_base_url,
            "import_count": len(self.imports),
            "search_results": [{"url": r.url, "title": r.title} for r in self.search_results],
            "selected_search_result": self.selected_search_result,
            "created_bundle": self.created_bundle,
            "dataset_revision": self.dataset_revision,
            "training_bundle_path": self.training_bundle_path,
            "training_steps": self.training_steps,
            "training_learning_rate": self.training_learning_rate,
            "training_batch_size": self.training_batch_size,
            "training": self.training_status(),
            "chat_prompt": self.chat_prompt,
            "chat_history": list(self.chat_history),
            "api_address": self.api_address,
            "api": self.api_status(),
            "responder": self.responder_identity(),
            "code_revision": self.code_revision,
            "capabilities": {
                "training": self.capability_boundary("training"),
                "chat": self.capability_boundary("chat"),
                "api": self.capability_boundary("api"),
                "datasets": self.capability_boundary("datasets"),
                "structured": self.capabilities(),
            },
        }
