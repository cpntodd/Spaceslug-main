"""Coarse-grained host boundary for the Spaceslug runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
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
    native_library: str | None = None


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
    """Runtime session using the native ABI when built, executable otherwise."""

    def __init__(self, runtime_root: str | Path, runtime_revision: str, build_dir: str = "build/debug", *, software_vulkan: bool = False) -> None:
        self.runtime_root = Path(runtime_root).resolve()
        self.runtime_revision = runtime_revision
        self.build_dir = self.runtime_root / build_dir
        self.software_vulkan = software_vulkan
        self._device: str | None = None
        self._library_path = self.build_dir / "libvulkan_runtime_api.so"
        self._library: ctypes.CDLL | None = None

    def _native(self) -> ctypes.CDLL:
        if self._library is None:
            if not self._library_path.is_file():
                raise BackendError(f"runtime shared library is missing: {self._library_path}")
            try:
                self._library = ctypes.CDLL(str(self._library_path))
                function = self._library.spaceslug_vector_add
                function.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
                function.restype = ctypes.c_int
            except OSError as exc:
                raise BackendError(f"failed to load {self._library_path}: {exc}") from exc
        return self._library

    def capabilities(self) -> BackendCapabilities:
        if self._device is None:
            smoke = self._run("smoke")
            self._device = next((line[8:] for line in smoke.stdout.splitlines() if line.startswith("Device: ")), None)
        return BackendCapabilities("spaceslug", self.runtime_revision, ("vector_add",), self._device, self.software_vulkan, str(self._library_path) if self._library_path.is_file() else None)

    def execute_vector_add(self, left: list[float] | None = None, right: list[float] | None = None) -> ExecutionResult:
        left = left or [1.0] * (1 << 20)
        right = right or [2.0] * len(left)
        if len(left) != len(right) or len(left) == 0 or len(left) % 256:
            raise BackendError("vector_add requires equal non-empty lengths divisible by 256")
        started = time.perf_counter()
        if self._library_path.is_file():
            output = (ctypes.c_float * len(left))()
            a = (ctypes.c_float * len(left))(*left)
            b = (ctypes.c_float * len(right))(*right)
            environment_note = "native-shared-library"
            # Vulkan reads the process environment during instance creation.
            import os
            old_icd = os.environ.get("VK_ICD_FILENAMES")
            if self.software_vulkan:
                os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
            try:
                code = self._native().spaceslug_vector_add(a, b, output, len(left))
            finally:
                if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
                else: os.environ["VK_ICD_FILENAMES"] = old_icd
            if code != 0:
                raise BackendError(f"native vector_add returned {code}")
            expected = [left_value + right_value for left_value, right_value in zip(left, right)]
            max_abs_error = max(abs(float(actual) - expected_value) for actual, expected_value in zip(output, expected))
            if max_abs_error != 0.0:
                raise BackendError(f"native vector_add parity failure: max_abs_error={max_abs_error}")
            return ExecutionResult("ok", "vector_add", "spaceslug", self.runtime_revision, self.capabilities().device, False, {"operation_count": 1, "host_elapsed_seconds": time.perf_counter() - started, "software_vulkan": self.software_vulkan, "execution": environment_note, "parity": "cpu-reference", "max_abs_error": max_abs_error}, {"first": float(output[0]), "last": float(output[-1]), "count": len(left)})
        completed = self._run("vector_add")
        if not completed.stdout.strip().endswith("PASS"):
            raise BackendError(f"vector_add did not pass: {completed.stdout.strip()}")
        return ExecutionResult("ok", "vector_add", "spaceslug", self.runtime_revision, self.capabilities().device, False, {"operation_count": 1, "host_elapsed_seconds": time.perf_counter() - started, "software_vulkan": self.software_vulkan, "execution": "validated-executable"}, {"runtime_report": completed.stdout.strip()})

    def verify_tiny_gpu_prerequisites(self) -> dict[str, Any]:
        """Report the currently available Vulkan gate without claiming Tiny GPU training."""
        capabilities = self.capabilities()
        return {
            "backend": capabilities.backend,
            "device": capabilities.device,
            "runtime_revision": capabilities.runtime_revision,
            "software_vulkan": capabilities.software_vulkan,
            "tiny_gpu_inference": False,
            "tiny_gpu_training": False,
            "validated_operations": list(capabilities.operations),
            "next_operation": "tensor_gemm_parity",
            "cpu_gate_required": True,
        }

    def _run(self, executable: str) -> subprocess.CompletedProcess[str]:
        path = self.build_dir / executable
        if not path.is_file(): raise BackendError(f"runtime executable is missing: {path}")
        environment = None
        if self.software_vulkan: environment = {"VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.json"}
        try:
            return subprocess.run([str(path)], cwd=self.runtime_root, check=True, capture_output=True, text=True, env=environment)
        except OSError as exc: raise BackendError(f"failed to launch {path}: {exc}") from exc
        except subprocess.CalledProcessError as exc: raise BackendError(json.dumps({"returncode": exc.returncode, "stderr": exc.stderr})) from exc
