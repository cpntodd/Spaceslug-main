"""Coarse-grained initial boundary for the Spaceslug runtime.

The first implementation intentionally wraps an existing validated executable.
It keeps runtime ownership and host orchestration separate while a native ABI is
being designed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import time
from typing import Any


@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    runtime_revision: str
    operations: tuple[str, ...]
    device: str | None = None
    software_vulkan: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    operation: str
    backend: str
    runtime_revision: str
    device: str | None
    fallback_used: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)


class BackendError(RuntimeError):
    """A structured backend invocation failure."""


class BackendSession:
    """Initial subprocess-backed session for validated runtime operations."""

    def __init__(
        self,
        runtime_root: str | Path,
        runtime_revision: str,
        build_dir: str = "build/debug",
        *,
        software_vulkan: bool = False,
    ) -> None:
        self.runtime_root = Path(runtime_root).resolve()
        self.runtime_revision = runtime_revision
        self.build_dir = self.runtime_root / build_dir
        self.software_vulkan = software_vulkan
        self._device: str | None = None

    def capabilities(self) -> BackendCapabilities:
        if self._device is None:
            smoke = self._run("smoke")
            self._device = next((line[8:] for line in smoke.stdout.splitlines() if line.startswith("Device: ")), None)
        return BackendCapabilities(
            backend="spaceslug",
            runtime_revision=self.runtime_revision,
            operations=("vector_add",),
            device=self._device,
            software_vulkan=self.software_vulkan,
        )

    def execute_vector_add(self) -> ExecutionResult:
        started = time.perf_counter()
        completed = self._run("vector_add")
        elapsed = time.perf_counter() - started
        output = completed.stdout.strip()
        if not output.endswith("PASS"):
            raise BackendError(f"vector_add did not pass: {output}")
        return ExecutionResult(
            status="ok",
            operation="vector_add",
            backend="spaceslug",
            runtime_revision=self.runtime_revision,
            device=self.capabilities().device,
            fallback_used=False,
            metrics={
                "operation_count": 1,
                "host_elapsed_seconds": elapsed,
                "software_vulkan": self.software_vulkan,
            },
            output={"runtime_report": output},
        )

    def _run(self, executable: str) -> subprocess.CompletedProcess[str]:
        path = self.build_dir / executable
        if not path.is_file():
            raise BackendError(f"runtime executable is missing: {path}")
        environment = None
        if self.software_vulkan:
            environment = {"VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json"}
        try:
            completed = subprocess.run(
                [str(path)],
                cwd=self.runtime_root,
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
        except OSError as exc:
            raise BackendError(f"failed to launch {path}: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            raise BackendError(json.dumps({"returncode": exc.returncode, "stderr": exc.stderr})) from exc
        return completed
