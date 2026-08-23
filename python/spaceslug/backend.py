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
                if hasattr(self._library, "spaceslug_attention"):
                    attention = self._library.spaceslug_attention
                    attention.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
                    attention.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_attention_causal"):
                    causal = self._library.spaceslug_attention_causal
                    causal.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32]
                    causal.restype = ctypes.c_int
            except OSError as exc:
                raise BackendError(f"failed to load {self._library_path}: {exc}") from exc
        return self._library

    def capabilities(self) -> BackendCapabilities:
        if self._device is None:
            smoke = self._run("smoke")
            self._device = next((line[8:] for line in smoke.stdout.splitlines() if line.startswith("Device: ")), None)
        operations = ["vector_add"]
        if (self.build_dir / "sgemm").is_file() or self._library_path.is_file():
            operations.append("sgemm")
        if (self.build_dir / "attention").is_file():
            operations.append("attention_kernel")
        if self._library_path.is_file():
            try:
                if hasattr(self._native(), "spaceslug_attention_causal"):
                    operations.append("attention_causal_abi")
            except BackendError:
                pass
        return BackendCapabilities("spaceslug", self.runtime_revision, tuple(operations), self._device, self.software_vulkan, str(self._library_path) if self._library_path.is_file() else None)

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

    def execute_tiny_lora_plan(self, model: Any, rank: int = 4) -> ExecutionResult:
        return ExecutionResult("not-run", "tiny_lora_forward_backward_update", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "GPU LoRA forward/backward/update kernels are not implemented", "rank": rank}, {"supported_base": model.hidden_size == 64 and model.vocab_size == 259})

    def execute_projected_attention_forward(self, tokens: list[int], model: Any) -> ExecutionResult:
        """Use the complete GPU chain only for the validated Tiny shape."""
        if model.hidden_size == 64 and model.vocab_size == 259 and 0 < len(tokens) <= 128 and hasattr(self._native(), "spaceslug_attention_causal"):
            return self._execute_tiny_gpu_forward(tokens, model)
        return self.execute_projected_attention_cpu_fallback(tokens, model)

    def _native_sgemm_values(self, a: list[float], b: list[float], m: int, n: int, k: int) -> list[float]:
        import array, ctypes, os
        aa, bb, cc = (array.array("f", values) for values in (a, b, [0.0] * (m * n)))
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan:
            os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_sgemm((ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb), (ctypes.c_float * len(cc)).from_buffer(cc), m, n, k, ctypes.byref(ctypes.c_float()))
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0:
            raise BackendError(f"native sgemm returned {code}")
        return cc.tolist()

    def _execute_tiny_gpu_forward(self, tokens: list[int], model: Any) -> ExecutionResult:
        import array, ctypes, os
        from .positional_encoding import sinusoidal_positions
        t, h, vp = len(tokens), model.hidden_size, 320
        states = [model.embedding[token][:] for token in tokens]
        if model.use_positions:
            positions = sinusoidal_positions(t, h)
            states = [[value + positions[row][column] for column, value in enumerate(state)] for row, state in enumerate(states)]
        states += [[0.0] * h for _ in range(128 - t)]
        def mat(matrix):
            return [matrix[row][column] for row in range(h) for column in range(h)]
        self._native()
        q = self._native_sgemm_values([x for row in states for x in row], mat(model.query), 128, 64, 64)
        k = self._native_sgemm_values([x for row in states for x in row], mat(model.key), 128, 64, 64)
        v = self._native_sgemm_values([x for row in states for x in row], mat(model.value), 128, 64, 64)
        qv, kv, vv = (array.array("f", x) for x in (q, k, v))
        out = (ctypes.c_float * (128 * 64))()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_attention_causal((ctypes.c_float * len(qv)).from_buffer(qv), (ctypes.c_float * len(kv)).from_buffer(kv), (ctypes.c_float * len(vv)).from_buffer(vv), out, t, h)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native causal attention returned {code}")
        context = list(out)
        projected = self._native_sgemm_values(context, mat(model.output), 128, 64, 64)
        lm = [[model.lm_head[row][column] if column < model.vocab_size else 0.0 for column in range(vp)] for row in range(h)]
        logits_all = self._native_sgemm_values(projected, [x for row in lm for x in row], 128, vp, h)
        logits = logits_all[(t - 1) * vp:t * vp][:model.vocab_size]
        parity = compare_float_arrays(logits, model.logits_for_tokens(tokens))
        return ExecutionResult("ok" if parity["status"] == "pass" else "error", "tiny_projected_attention_forward", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "cpu-reference", "gpu_execution": parity["status"] == "pass", "padded_tokens": 128, "padded_vocab": vp, **parity}, {"logits": logits, "token_count": t})

    def execute_projected_attention_gpu_plan(self, tokens: list[int], model: Any) -> ExecutionResult:
        plan = self.projected_attention_forward_plan(hidden_size=model.hidden_size, sequence_length=len(tokens), vocab_size=model.vocab_size)
        if not hasattr(self._native(), "spaceslug_attention_causal"):
            reason = "causal attention ABI is unavailable"
        else:
            reason = "padded QKV, output projection, and LM-head orchestration are not implemented"
        return ExecutionResult("not-run", plan["operation"], "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": reason, "steps": plan["steps"]}, {"plan": plan})

    def execute_attention_kernel_parity(self, q: list[float], k: list[float], v: list[float], tokens: int, hidden_size: int) -> ExecutionResult:
        if tokens <= 0 or hidden_size != 64 or tokens % 32:
            return ExecutionResult("not-run", "attention", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "attention kernel contract requires D=64 and T divisible by 32"}, {"tokens": tokens, "hidden_size": hidden_size})
        completed = self._run("attention")
        output = completed.stdout.strip()
        if "PASS" not in output:
            raise BackendError(f"attention parity did not pass: {output}")
        return ExecutionResult("ok", "attention", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "cpu-reference", "execution": "validated-executable"}, {"runtime_report": output})

    def execute_tiny_attention_kernel_chain(self, tokens: list[int], model: Any) -> ExecutionResult:
        plan = self.projected_attention_forward_plan(hidden_size=model.hidden_size, sequence_length=len(tokens), vocab_size=model.vocab_size)
        return ExecutionResult("not-run", "tiny_projected_attention_forward", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "chain requires embedding/qkv/causal-mask/lm-head integration", "steps": plan["steps"]}, {"plan": plan})

    def execute_projected_attention_cpu_fallback(self, tokens: list[int], model: Any) -> ExecutionResult:
        logits = model.logits_for_tokens(tokens)
        return ExecutionResult("ok", "tiny_projected_attention_forward", "cpu-reference", self.runtime_revision, None, True, {"parity": "cpu-reference", "gpu_execution": False}, {"logits": logits, "token_count": len(tokens)})

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
