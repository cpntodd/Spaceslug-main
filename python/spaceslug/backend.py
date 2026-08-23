"""Coarse-grained host boundary for the Spaceslug runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from .parity import compare_float_arrays


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
                sgemm = self._library.spaceslug_sgemm
                sgemm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
                sgemm.restype = ctypes.c_int
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

    def execute_sgemm_native(self, a: list[float], b: list[float], m: int, n: int, k: int) -> ExecutionResult:
        if len(a) != m * k or len(b) != k * n or m % 64 or n % 64 or k % 32:
            raise BackendError("native sgemm dimensions or input lengths are invalid")
        self._native()
        import array
        aa, bb, cc = (array.array("f", values) for values in (a, b, [0.0] * (m * n)))
        error_value = ctypes.c_float()
        code = self._library.spaceslug_sgemm((ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb), (ctypes.c_float * len(cc)).from_buffer(cc), m, n, k, ctypes.byref(error_value))
        if code != 0:
            raise BackendError(f"native sgemm returned {code}")
        expected_values = [sum(a[row * k + inner] * b[inner * n + column] for inner in range(k)) for row in range(m) for column in range(n)]
        parity = compare_float_arrays(cc.tolist(), expected_values)
        return ExecutionResult("ok", "sgemm", "spaceslug", self.runtime_revision, self.capabilities().device, False, {"parity": "cpu-reference", **parity}, {"m": m, "n": n, "k": k, "first": float(cc[0]), "values": cc.tolist()})

    def execute_projected_qkv(self, states: list[float], projection: list[float], sequence_length: int, hidden_size: int, *, cpu_reference: list[float] | None = None) -> ExecutionResult:
        if len(states) != sequence_length * hidden_size or len(projection) != hidden_size * hidden_size:
            raise BackendError("projected QKV dimensions are invalid")
        result = self.execute_sgemm_native(states, projection, sequence_length // 64 * 64, hidden_size, hidden_size) if sequence_length >= 64 and hidden_size >= 64 and hidden_size % 64 == 0 else None
        if result is None:
            return ExecutionResult("not-run", "qkv_projection_sgemm", "spaceslug", self.runtime_revision, self.capabilities().device, False, {"reason": "runtime SGEMM requires padded production dimensions", "parity": "not-run"}, {"sequence_length": sequence_length, "hidden_size": hidden_size})
        result = ExecutionResult(result.status, "qkv_projection_sgemm", result.backend, result.runtime_revision, result.device, result.fallback_used, dict(result.metrics), dict(result.output))
        result.metrics["parity"] = "CPU projection contract"
        result.output["sequence_length"] = sequence_length
        result.output["hidden_size"] = hidden_size
        if cpu_reference is not None:
            if len(cpu_reference) != len(result.output["values"]):
                raise BackendError("CPU projection reference length does not match GPU output")
            result.metrics["cpu_projection_parity"] = compare_float_arrays(result.output["values"], cpu_reference)
            result.metrics["parity"] = "cpu-projection-output"
        return result

    def projected_attention_forward_plan(self, *, hidden_size: int, sequence_length: int, vocab_size: int) -> dict[str, Any]:
        if hidden_size <= 0 or sequence_length <= 0 or vocab_size <= 0:
            raise ValueError("projected attention dimensions must be positive")
        return {
            "operation": "tiny_projected_attention_forward",
            "status": "planned-not-implemented",
            "backend": "spaceslug",
            "runtime_revision": self.runtime_revision,
            "device": self.capabilities().device,
            "cpu_reference": True,
            "gpu_execution": False,
            "steps": ["embedding_upload", "qkv_projection_sgemm", "causal_softmax", "output_projection_sgemm", "lm_head_sgemm"],
            "dimensions": {"hidden_size": hidden_size, "sequence_length": sequence_length, "vocab_size": vocab_size},
            "parity_gate": "CPU logits vs RADV logits",
            "next_kernel": "qkv_projection_sgemm",
        }

    def execute_sgemm_parity(self) -> ExecutionResult:
        """Run the validated fp32 GEMM parity executable as the first GPU gate."""
        if not self._library_path.is_file():
            raise BackendError("native runtime library is missing; build vulkan-runtime first")
        started = time.perf_counter()
        completed = self._run("sgemm")
        output = completed.stdout.strip()
        if "PASS" not in output:
            raise BackendError(f"sgemm parity did not pass: {output}")
        return ExecutionResult("ok", "sgemm", "spaceslug", self.runtime_revision, self.capabilities().device, False,
                               {"execution": "validated-executable", "parity": "cpu-reference", "host_elapsed_seconds": time.perf_counter() - started},
                               {"runtime_report": output})

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
            "next_operation": "tiny_projected_attention_forward",
            "gemm_parity_available": self._library_path.is_file(),
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
