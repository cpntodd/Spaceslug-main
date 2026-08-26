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
import urllib.parse
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..dataset import verify_bundle
from ..documentation_crawler import DocumentationCrawler, CrawlSession, same_origin
from ..filesystem_picker import FileSelection, pick_files
from ..native_desktop_training import readiness as native_gpu_readiness, run_native_training
from ..backend import BackendSession
from ..openai_api import (
    LOOPBACK_HOSTS,
    ModelResponder,
    OpenAICompatibleServer,
    TinyCpuEchoResponder,
    TinyGpuResponder,
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
    native_training_groups,
    native_training_groups_text,
    structured_capabilities,
    training_capability_text,
)
from .loss_graph import LossWormGraph
from .gpu_readiness import GpuReadiness, refresh_gpu_readiness_probe
from .profile import fixed_tiny_profile, tiny_profile_id
from .runtime import (
    RUNTIME_ROOT,
    RuntimePlacement,
    TrainingPlacement,
    default_runtime_probe,
    resolve_placement,
    resolve_training_placement,
)
from .settings import (
    DEFAULT_CONFIG_PATH,
    WorkspacePaths,
    coerce_path,
    default_workspace_paths,
    load_workspace_paths,
    save_workspace_paths,
)

TAB_NAMES = ("Home", "Datasets", "Build & Train", "Interact", "Local API")

DEFAULT_SEARXNG_BASE_URL = "http://127.0.0.1:8888"
DEFAULT_API_ADDRESS = "127.0.0.1:8123"
LOCAL_SOURCE_EXTENSIONS = (".txt", ".md", ".json", ".jsonl", ".pdf")

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
        config_path: str | Path | None = None,
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
        self.runtime_root = RUNTIME_ROOT
        self.runtime_revision = "runtime"
        self.gpu_readiness: GpuReadiness | None = None

        # Persisted workspace paths. An explicit workspace_root wins over any
        # persisted config so callers (and tests) stay deterministic.
        self.config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        if workspace_root is not None:
            self.paths = default_workspace_paths(workspace_root)
        else:
            self.paths = load_workspace_paths(self.config_path)
        self.workspace_root = self.paths.workspace_root
        # Workspace service (lazily constructed; injectable for tests).
        self._workspace = workspace

        # Responder shared by chat and the local API server.
        self.responder = responder
        self._inference_explicit = responder is not None
        self.inference_status = {"backend": "uninitialized", "ready": False, "reason": "inference engine has not been initialized"}
        self.code_revision = code_revision if code_revision is not None else "unrecorded"

        # Dataset workflow state.
        self.dataset_file = ""
        self.dataset_url = ""
        self.documentation_url = ""
        self.documentation_max_pages = 10
        self.documentation_max_depth = 1
        self.documentation_same_origin = True
        self.documentation_session: CrawlSession | None = None
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
        self.training_rank = 4
        self.training_prompt = ""
        self.latest_loss: float | None = None
        self.completed_steps = 0
        self._training_thread: threading.Thread | None = None
        self._training_state = "idle"  # idle | running | finished | error
        self._training_error: BaseException | None = None
        self._training_result: dict | None = None
        self._training_paths: TrainingJobPaths | None = None
        self._training_mode = "idle"
        self._gpu_readiness_error = "GPU readiness has not been refreshed"
        self._loss_queue: queue.Queue[tuple[int, float]] = queue.Queue()
        self._cancel_event = threading.Event()

        # Chat and API workflow state. The native desktop app initializes the
        # embedded engine at launch; headless controllers remain side-effect free.
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

    def set_runtime_target(self, runtime_root: str | Path, runtime_revision: str = "runtime") -> None:
        """Set the runtime location used by the in-app readiness check."""
        root = str(runtime_root).strip()
        if not root:
            raise ValueError("runtime root must not be empty")
        self.runtime_root = root
        self.runtime_revision = str(runtime_revision).strip() or "runtime"

    def refresh_gpu_readiness(self) -> GpuReadiness:
        """Safely inspect the configured native runtime and update placement."""
        try:
            readiness = refresh_gpu_readiness_probe(self.runtime_root, self.runtime_revision)
        except Exception as exc:
            self._gpu_readiness_error = f"{type(exc).__name__}: {exc}"
            self.gpu_readiness = None
            raise
        self.gpu_readiness = readiness
        self._gpu_readiness_error = ""
        self.placement = resolve_placement(readiness.placement_probe())
        return readiness

    def refresh_gpu_readiness_if_needed(self) -> None:
        if self.gpu_readiness is None:
            try:
                self.refresh_gpu_readiness()
            except Exception:
                pass

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
            self._workspace = WorkspaceService(
                self.paths.workspace_root,
                bundles_dir=self.paths.dataset_dir,
                temp_dir=self.paths.temp_dir,
            )
        return self._workspace

    def _persist_paths(self) -> Path:
        return save_workspace_paths(self.paths, self.config_path)

    def _apply_paths(self, paths: WorkspacePaths, *, reset_imports: bool, reset_bundles: bool) -> None:
        self.paths = paths
        self.workspace_root = paths.workspace_root
        self._workspace = None
        if reset_imports:
            self.imports = []
            self.last_import = None
            self.search_results = []
            self.selected_search_result = None
        if reset_bundles:
            self.created_bundle = ""
            self.dataset_revision = None
            self.training_bundle_path = ""
        self._persist_paths()

    def set_workspace_paths(
        self,
        *,
        workspace_root: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        artifact_dir: str | Path | None = None,
        experiment_dir: str | Path | None = None,
        temp_dir: str | Path | None = None,
    ) -> WorkspacePaths:
        """Update persisted paths; a new workspace root re-derives its subdirs."""
        if workspace_root is not None:
            root = coerce_path(workspace_root)
            paths = default_workspace_paths(root)
            overrides = {
                "dataset_dir": dataset_dir,
                "checkpoint_dir": checkpoint_dir,
                "artifact_dir": artifact_dir,
                "experiment_dir": experiment_dir,
                "temp_dir": temp_dir,
            }
            for field, value in overrides.items():
                if value is not None:
                    paths = replace(paths, **{field: coerce_path(value)})
            self._apply_paths(paths, reset_imports=True, reset_bundles=True)
            return self.paths
        overrides = {
            "dataset_dir": dataset_dir,
            "checkpoint_dir": checkpoint_dir,
            "artifact_dir": artifact_dir,
            "experiment_dir": experiment_dir,
            "temp_dir": temp_dir,
        }
        if not any(value is not None for value in overrides.values()):
            return self.paths
        paths = replace(
            self.paths,
            **{field: coerce_path(value) for field, value in overrides.items() if value is not None},
        )
        self._apply_paths(
            paths,
            reset_imports=False,
            reset_bundles=dataset_dir is not None,
        )
        return self.paths

    def set_workspace_root(self, value: str | Path) -> WorkspacePaths:
        return self.set_workspace_paths(workspace_root=value)

    def set_dataset_dir(self, value: str | Path) -> WorkspacePaths:
        return self.set_workspace_paths(dataset_dir=value)

    def set_checkpoint_dir(self, value: str | Path) -> WorkspacePaths:
        return self.set_workspace_paths(checkpoint_dir=value)

    def set_artifact_dir(self, value: str | Path) -> WorkspacePaths:
        return self.set_workspace_paths(artifact_dir=value)

    def set_experiment_dir(self, value: str | Path) -> WorkspacePaths:
        return self.set_workspace_paths(experiment_dir=value)

    def set_temp_dir(self, value: str | Path) -> WorkspacePaths:
        return self.set_workspace_paths(temp_dir=value)

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

    def set_documentation_url(self, value: str) -> None:
        self.documentation_url = value

    def crawl_documentation(self, url: str | None = None, *, approve_all_same_origin: bool = False) -> CrawlSession:
        seed = url if url is not None else self.documentation_url
        if not seed.strip():
            raise IngestionError("documentation URL is required")
        crawler = DocumentationCrawler(
            self._workspace_service(), max_pages=self.documentation_max_pages,
            max_depth=self.documentation_max_depth, same_origin_only=self.documentation_same_origin,
        )
        session_path = self.paths.temp_dir / "documentation-crawl.json"
        approved_origin = seed
        session = crawler.start(seed, approve=(lambda candidate: approve_all_same_origin and same_origin(approved_origin, candidate)), session_path=session_path)
        self.documentation_session = session
        self.imports.extend(item for item in self._workspace_service().list_imports() if item not in self.imports)
        return session

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

    def import_local_folder(
        self, root: str | Path, *, recursive: bool = True, license: str | None = None
    ) -> dict[str, Any]:
        """Scan *root* and import every supported file, retaining partial success."""
        selection = self.choose_local_sources(root, recursive=recursive)
        imported: list[Any] = []
        errors: list[dict[str, str]] = []
        for path in selection.files:
            try:
                imported.append(self.import_local_source(path, license=license))
            except Exception as exc:
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        return {
            "root": str(selection.root),
            "imported": imported,
            "skipped": [str(path) for path in selection.skipped],
            "errors": errors,
        }

    def imported_source_rows(self) -> list[dict[str, Any]]:
        """Return display metadata for local and approved online imports."""
        rows: list[dict[str, Any]] = []
        for imported in self.imports:
            source = str(imported.source)
            if imported.retrieval == "local":
                path = Path(source)
                name = path.name
                extension = path.suffix.lower() or imported.kind
                try:
                    timestamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
                except OSError:
                    timestamp = imported.fetched_at or ""
            else:
                parsed_path = Path(urllib.parse.urlsplit(source).path)
                name = parsed_path.name or source
                extension = parsed_path.suffix.lower() or imported.kind
                timestamp = imported.fetched_at or ""
            rows.append({
                "filename": name,
                "extension": extension,
                "date_imported": timestamp,
                "file_size": imported.bytes,
                "source": source,
                "retrieval": imported.retrieval,
            })
        return rows

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
    def list_dataset_bundles(self) -> list[Path]:
        """Return verified canonical .dts bundles in the configured dataset dir."""
        result: list[Path] = []
        for path in sorted(self.paths.dataset_dir.glob("*.dts")):
            try:
                verify_bundle(path)
            except Exception:
                continue
            result.append(path)
        return result

    def import_dataset_bundle(self, path: str | Path) -> Any:
        """Verify and load an existing canonical or training .dts bundle."""
        bundle = verify_bundle(path)
        self.created_bundle = str(bundle.root)
        self.dataset_id = str(bundle.manifest.get("dataset_id", Path(path).stem))
        self.dataset_revision = bundle.manifest.get("revision")
        self.training_bundle_path = str(bundle.root) if bundle.manifest.get("preprocessing", {}).get("pipeline") == "spaceslug-prompt-target" else ""
        return bundle

    # -- .dts creation -------------------------------------------------------
    def create_dataset(self, dataset_id: str | None = None, *, split: str = "train") -> Any:
        """Create or reuse the named canonical bundle and update training state.

        Dataset IDs map to immutable bundle paths. Repeated clicks therefore reuse
        a verified existing bundle instead of leaving the controller with no
        ``created_bundle`` after a ``FileExistsError``.
        """
        name = dataset_id if dataset_id is not None else self.dataset_id
        output = self._workspace_service().bundles_dir / f"{name}.dts"
        if output.exists():
            bundle = verify_bundle(output)
        else:
            bundle = self._workspace_service().create_dataset(name, self.imports, split=split)
        self.created_bundle = str(bundle.root)
        self.dataset_revision = bundle.manifest["revision"]
        return bundle

    def build_training_bundle(self, *, prompt: str | None = None) -> Any:
        """Derive a prompt/target training ``.dts`` from the canonical bundle."""
        if not self.created_bundle:
            raise IngestionError("create a .dts from imported sources before training")
        output = self._workspace_service().bundles_dir / f"{Path(self.created_bundle).name}-train.dts"
        if output.exists():
            bundle = verify_bundle(output)
        else:
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

    def set_training_rank(self, rank: int) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in (4, 8):
            raise ValueError(f"rank must be 4 or 8, got {rank!r}")
        self.training_rank = rank

    def set_training_prompt(self, value: str) -> None:
        self.training_prompt = value

    def _resolve_training_bundle(self, bundle_path: str | Path | None) -> Path:
        if bundle_path is not None:
            return Path(bundle_path)
        if self.training_bundle_path:
            return Path(self.training_bundle_path)
        if self.created_bundle:
            try:
                self.build_training_bundle()
            except FileExistsError:
                candidate = self._workspace_service().bundles_dir / f"{Path(self.created_bundle).name}-train.dts"
                verify_bundle(candidate)
                self.training_bundle_path = str(candidate)
            return Path(self.training_bundle_path)
        raise IngestionError("no dataset bundle available for training")

    def _resolve_training_paths(
        self,
        checkpoint: str | Path | None,
        artifact: str | Path | None,
        experiment: str | Path | None,
    ) -> TrainingJobPaths:
        run_id = f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        checkpoint_path = Path(checkpoint) if checkpoint is not None else self.paths.checkpoint_dir / f"{run_id}.json"
        artifact_path = Path(artifact) if artifact is not None else self.paths.artifact_dir / f"{run_id}.spaceslug"
        experiment_path = Path(experiment) if experiment is not None else self.paths.experiment_dir / run_id
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
            rank=self.training_rank,
        )
        paths = self._resolve_training_paths(checkpoint, artifact, experiment)
        self._training_paths = paths
        self._training_result = None
        self._training_error = None
        self._training_mode = "cpu-reference"
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

    def native_gpu_training_readiness(self) -> dict[str, Any]:
        probe = self.gpu_readiness.placement_probe() if self.gpu_readiness is not None else None
        if probe is None:
            return {"ready": False, "reason": self._gpu_readiness_error or "GPU readiness has not been refreshed", "missing": []}
        return native_gpu_readiness(probe, self.training_rank)

    def start_native_gpu_training(
        self, *, bundle_path: str | Path | None = None, steps: int | None = None,
        learning_rate: float | None = None, checkpoint: str | Path | None = None,
        artifact: str | Path | None = None, experiment: str | Path | None = None,
    ) -> dict:
        if self._training_thread is not None and self._training_thread.is_alive():
            raise RuntimeError("training is already running")
        gate = self.native_gpu_training_readiness()
        if not gate["ready"]:
            raise RuntimeError("native GPU training unavailable: " + gate["reason"])
        bundle = self._resolve_training_bundle(bundle_path)
        paths = self._resolve_training_paths(checkpoint, artifact, experiment)
        self._training_paths = paths
        self._training_result = None
        self._training_error = None
        self._training_mode = "native-gpu"
        self._training_state = "running"
        self.latest_loss = None
        self.completed_steps = 0
        self._cancel_event.clear()
        self._training_thread = threading.Thread(
            target=self._run_native_gpu_training,
            args=(bundle, self.training_steps if steps is None else steps,
                  self.training_learning_rate if learning_rate is None else learning_rate, paths),
            name="spaceslug-desktop-native-gpu-training", daemon=True,
        )
        self._training_thread.start()
        return self.training_status()

    def _run_native_gpu_training(self, bundle: Path, steps: int, learning_rate: float, paths: TrainingJobPaths) -> None:
        try:
            self._training_result = run_native_training(
                bundle, runtime_root=self.runtime_root, runtime_revision=self.runtime_revision,
                rank=self.training_rank, steps=steps, learning_rate=learning_rate,
                checkpoint=paths.checkpoint, artifact=paths.artifact, experiment=paths.experiment,
                on_step=self._on_training_step, should_stop=self._cancel_event.is_set,
            )
            self._training_state = "finished"
        except BaseException as exc:
            self._training_error = exc
            self._training_state = "error"

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
            "mode": self._training_mode,
            "native_gpu_readiness": self.native_gpu_training_readiness(),
            "completed_steps": self.completed_steps,
            "latest_loss": self.latest_loss,
            "loss_history": self.loss.history,
            "rank": self.training_rank,
            "tiny_profile": tiny_profile_id(self.training_rank),
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
                "backend": self._training_result.get("backend", "cpu-reference"),
                "gpu_execution": bool(self._training_result.get("gpu_execution", False)),
            }
        if self._training_error is not None:
            status["error"] = f"{type(self._training_error).__name__}: {self._training_error}"
        return status

    # -- chat ----------------------------------------------------------------
    def initialize_inference(self) -> dict[str, Any]:
        if self.responder is not None and self._inference_explicit:
            self.inference_status = {"backend": self.responder.backend_id, "ready": True, "reason": "injected responder"}
            return self.inference_status
        self.responder = None
        try:
            backend = BackendSession(self.runtime_root, self.runtime_revision)
            backend.capabilities()
            self.responder = TinyGpuResponder(backend)
            self.inference_status = {"backend": self.responder.backend_id, "ready": True, "reason": "native Vulkan inference initialized"}
        except Exception as exc:
            self.responder = TinyCpuEchoResponder()
            self.inference_status = {"backend": self.responder.backend_id, "ready": False, "reason": f"GPU inference unavailable: {type(exc).__name__}: {exc}"}
        return self.inference_status

    def _responder(self) -> ModelResponder:
        if self.responder is None:
            self.responder = TinyCpuEchoResponder()
            self.inference_status = {"backend": self.responder.backend_id, "ready": False, "reason": "headless controller CPU reference; desktop launch initializes GPU"}
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

    def native_training_groups(self) -> dict[str, Any]:
        return native_training_groups()

    def native_training_groups_text(self) -> str:
        return native_training_groups_text()

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
            "workspace_root": str(self.paths.workspace_root),
            "config_path": str(self.config_path),
            "workspace_paths": self.paths.to_dict(),
            "dataset_file": self.dataset_file,
            "dataset_url": self.dataset_url,
            "documentation_url": self.documentation_url,
            "documentation_max_pages": self.documentation_max_pages,
            "documentation_max_depth": self.documentation_max_depth,
            "documentation_same_origin": self.documentation_same_origin,
            "documentation_session": None if self.documentation_session is None else asdict(self.documentation_session),
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
            "training_rank": self.training_rank,
            "training": self.training_status(),
            "chat_prompt": self.chat_prompt,
            "chat_history": list(self.chat_history),
            "inference": dict(self.inference_status),
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
