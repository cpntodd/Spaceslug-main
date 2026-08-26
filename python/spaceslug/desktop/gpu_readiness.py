"""In-app GPU readiness diagnostic for the Phase 1 desktop shell.

This module turns the host's *capability probe* into an honest, actionable
readiness report without running any training or model forward work.  It is the
single place the desktop asks: "is the Vulkan runtime API present, loadable, and
backed by a hardware GPU, and which of the expected native Tiny ops are
actually exported?"

The probe is deliberately safe: it checks the runtime library on disk, loads it
through :mod:`ctypes` to enumerate the native ABI symbols we care about, and —
when the library exists — asks :class:`spaceslug.backend.BackendSession` for its
capability summary (device, backend, software-Vulkan flag, and operations).
That summary runs the runtime's existing ``smoke`` executable, never a training
step or a model forward.

The result is a frozen :class:`GpuReadiness` value with:

* the **runtime root** and library path actually inspected,
* **libvulkan_runtime_api.so existence / loadability**,
* **device / backend / software-Vulkan status** (RADV vs lavapipe),
* the **expected native Tiny ops** and which are present, and
* **actionable diagnostics** — build the runtime API, fix the runtime root, or
  resolve RADV-vs-lavapipe.

Dataset-integrated GPU training is *not* claimed anywhere here: it remains
unavailable in Phase 1 and the report says so explicitly.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import RUNTIME_ROOT, resolve_placement

# Expected native Tiny ops, in display order.  Each operation maps to the native
# ABI symbol(s) that must be exported by libvulkan_runtime_api.so for that
# operation to be considered present.  The list mirrors the operation names
# already reported by BackendSession.capabilities() so the diagnostic never
# invents a stronger claim than the backend itself makes.
EXPECTED_TINY_OPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sgemm", ("spaceslug_sgemm",)),
    ("attention_causal_abi", ("spaceslug_attention_causal",)),
    ("causal_loss_abi", ("spaceslug_causal_loss",)),
    ("attention_causal_backward_abi", ("spaceslug_attention_causal_backward",)),
    ("lora_train_step_abi", ("spaceslug_lora_train_step",)),
    ("lora_gradients_multi_abi", ("spaceslug_lora_gradients_multi",)),
    ("lora_sgd_multi_abi", ("spaceslug_lora_sgd_multi",)),
    ("tiny_forward_abi", ("spaceslug_tiny_forward",)),
    ("tiny_forward_fixed_retained_abi", ("spaceslug_tiny_forward_fixed_retained",)),
    (
        "tiny_lora_adamw_abi",
        (
            "spaceslug_tiny_forward_begin_lora_adamw",
            "spaceslug_tiny_forward_finalize_lora_adamw",
            "spaceslug_tiny_forward_readback_lora_adamw_state",
            "spaceslug_tiny_forward_update_lora_adamw_state",
            "spaceslug_tiny_forward_token_step_training_backward_accumulate",
        ),
    ),
)


def expected_tiny_op_names() -> tuple[str, ...]:
    """Return the ordered names of the expected native Tiny ops."""
    return tuple(operation for operation, _ in EXPECTED_TINY_OPS)


def classify_device(device: str | None, software_vulkan: bool = False) -> str:
    """Classify a reported device name as ``radv``, ``lavapipe``, or ``unknown``.

    RADV is the Mesa driver for the RX580/gfx803 (device name contains "RADV");
    lavapipe/llvmpipe is software Vulkan running on the CPU.  A forced
    ``software_vulkan`` flag is always treated as lavapipe regardless of name.
    """
    if software_vulkan:
        return "lavapipe"
    if not device:
        return "none"
    lowered = device.lower()
    if "llvmpipe" in lowered or "lavapipe" in lowered:
        return "lavapipe"
    if "radv" in lowered:
        return "radv"
    return "unknown"


def _library_paths(runtime_root: Path) -> tuple[Path, Path]:
    """Return the debug and release library paths the backend could load."""
    return (
        runtime_root / "build" / "debug" / "libvulkan_runtime_api.so",
        runtime_root / "build" / "release" / "libvulkan_runtime_api.so",
    )


@dataclass(frozen=True)
class GpuReadiness:
    """A single immutable snapshot of the GPU/backend readiness state."""

    runtime_root: str
    runtime_revision: str
    backend: str
    library_path: str | None
    library_exists: bool
    library_loadable: bool
    device: str | None
    device_class: str
    software_vulkan: bool
    operations: tuple[str, ...]
    expected_tiny_ops: tuple[dict[str, Any], ...]
    status: str
    gpu_primary: bool
    cpu_fallback: bool
    dataset_training: bool
    probe_error: str | None
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_root": self.runtime_root,
            "runtime_revision": self.runtime_revision,
            "backend": self.backend,
            "library_path": self.library_path,
            "library_exists": self.library_exists,
            "library_loadable": self.library_loadable,
            "device": self.device,
            "device_class": self.device_class,
            "software_vulkan": self.software_vulkan,
            "operations": list(self.operations),
            "expected_tiny_ops": [dict(operation) for operation in self.expected_tiny_ops],
            "status": self.status,
            "gpu_primary": self.gpu_primary,
            "cpu_fallback": self.cpu_fallback,
            "dataset_training": self.dataset_training,
            "probe_error": self.probe_error,
            "diagnostics": list(self.diagnostics),
        }

    def placement_probe(self) -> dict[str, Any]:
        """Return the subset :func:`resolve_placement` consumes."""
        return {
            "device": self.device,
            "operations": list(self.operations),
            "software_vulkan": self.software_vulkan,
        }

    def summary_text(self) -> str:
        lines = [
            f"runtime root: {self.runtime_root}",
            f"runtime library: {self.library_path or '(missing)'}",
            f"library exists: {self.library_exists} · loadable: {self.library_loadable}",
            f"backend: {self.backend} · device: {self.device or '(none)'}",
            f"device class: {self.device_class} · software Vulkan: {self.software_vulkan}",
            f"status: {self.status} (gpu_primary={self.gpu_primary}, cpu_fallback={self.cpu_fallback})",
            f"dataset-integrated GPU training: {'available' if self.dataset_training else 'not available (CPU authoritative)'}",
            "expected native Tiny ops:",
        ]
        for operation in self.expected_tiny_ops:
            mark = "present" if operation["present"] else "missing"
            missing = operation.get("missing_symbols") or []
            detail = f" (missing: {', '.join(missing)})" if missing else ""
            lines.append(f"  - {operation['operation']}: {mark}{detail}")
        if self.diagnostics:
            lines.append("diagnostics:")
            lines.extend(f"  - {line}" for line in self.diagnostics)
        return "\n".join(lines)


def _static_readiness(runtime_root: str | Path, runtime_revision: str) -> GpuReadiness:
    """The side-effect-free placeholder used before any probe has run."""
    root = str(Path(runtime_root).resolve())
    return GpuReadiness(
        runtime_root=root,
        runtime_revision=runtime_revision,
        backend="spaceslug",
        library_path=None,
        library_exists=False,
        library_loadable=False,
        device=None,
        device_class="none",
        software_vulkan=False,
        operations=(),
        expected_tiny_ops=(),
        status="not-probed",
        gpu_primary=False,
        cpu_fallback=True,
        dataset_training=False,
        probe_error=None,
        diagnostics=(),
    )


def gpu_readiness_probe(
    runtime_root: str | Path = RUNTIME_ROOT,
    runtime_revision: str = "runtime",
    *,
    safe: bool = True,
) -> GpuReadiness:
    """Probe the backend for GPU readiness; never runs training or a forward.

    When *safe* is true (the default) every backend touch is wrapped so a
    missing runtime, a failed load, or a broken ``smoke`` executable degrades to
    a diagnostic instead of an exception.  A ``ctypes`` load of the runtime
    library is itself a probe of loadability, not an execution of kernels.
    """
    root = Path(runtime_root).resolve()
    debug_lib, release_lib = _library_paths(root)
    library_exists = debug_lib.is_file()
    library_loadable = False
    device: str | None = None
    software_vulkan = False
    operations: tuple[str, ...] = ()
    probe_error: str | None = None
    symbol_presence: dict[str, bool] = {}
    diagnostics: list[str] = []

    if not root.is_dir():
        diagnostics.append(
            f"runtime root {str(root)!r} does not exist; pass --runtime-root "
            "(spaceslug desktop --runtime-root /path/to/vulkan-runtime) or correct the configured root."
        )

    if not library_exists:
        diagnostics.append(
            "libvulkan_runtime_api.so is not built at build/debug/libvulkan_runtime_api.so; "
            "build the runtime API with: cd ../vulkan-runtime && cmake --preset debug && "
            "cmake --build build/debug, then refresh."
        )
        if release_lib.is_file():
            diagnostics.append(
                "libvulkan_runtime_api.so exists only under build/release; the host backend "
                "loads build/debug, so build the debug preset (or a matching build) to load it."
            )
    else:
        try:
            from ..backend import BackendSession

            capabilities = BackendSession(root, runtime_revision).capabilities()
            device = capabilities.device
            software_vulkan = capabilities.software_vulkan
            operations = capabilities.operations
        except Exception as exc:  # degrade to a diagnostic, never crash the shell
            probe_error = f"{type(exc).__name__}: {exc}"
            diagnostics.append(f"backend capability probe failed: {probe_error}")

        try:
            library = ctypes.CDLL(str(debug_lib))
            library_loadable = True
            for _, symbols in EXPECTED_TINY_OPS:
                for symbol in symbols:
                    symbol_presence[symbol] = hasattr(library, symbol)
        except OSError as exc:
            probe_error = f"{type(exc).__name__}: {exc}"
            diagnostics.append(f"failed to load {debug_lib}: {exc}")

    expected: list[dict[str, Any]] = []
    for operation, symbols in EXPECTED_TINY_OPS:
        if operation == "sgemm":
            present = bool(symbol_presence.get("spaceslug_sgemm", False)) or "sgemm" in operations
        else:
            present = library_loadable and all(symbol_presence.get(symbol, False) for symbol in symbols)
        missing = [] if present else [symbol for symbol in symbols if not symbol_presence.get(symbol, False)]
        expected.append(
            {
                "operation": operation,
                "expected_symbols": list(symbols),
                "present": present,
                "missing_symbols": missing,
            }
        )

    device_class = classify_device(device, software_vulkan)
    placement = resolve_placement(
        {"device": device, "operations": list(operations), "software_vulkan": software_vulkan}
    )

    if not library_exists:
        status = "runtime-missing"
    elif device_class == "lavapipe":
        status = "software-vulkan"
    elif placement.gpu_primary:
        status = "gpu-primary"
    else:
        status = "cpu-fallback"

    if device_class == "lavapipe":
        diagnostics.append(
            "software Vulkan (lavapipe/llvmpipe) is active — no hardware GPU primary. "
            "RADV reports 'AMD Radeon RX 580 Series (RADV POLARIS10)'. Ensure the RADV "
            "ICD is installed and that VK_ICD_FILENAMES is not forcing the lvp_icd.json driver."
        )
    elif device_class == "radv" and placement.gpu_primary:
        diagnostics.append(
            "hardware GPU (RADV) detected and the expected native Tiny ops gate is satisfied; "
            "the GPU is primary only for supported native operations, and the CPU reference "
            "remains the authoritative fallback."
        )
    elif device is None and library_exists:
        diagnostics.append(
            "the runtime library is present but no Vulkan device was reported; run "
            "./build/debug/smoke inside the runtime root to see the device/ICD and confirm "
            "a Vulkan ICD is installed."
        )

    diagnostics.append(
        "dataset-integrated GPU training is not available in Phase 1; the CPU projected Tiny "
        "training path is authoritative regardless of GPU readiness."
    )

    return GpuReadiness(
        runtime_root=str(root),
        runtime_revision=runtime_revision,
        backend="spaceslug",
        library_path=str(debug_lib) if library_exists else (str(release_lib) if release_lib.is_file() else None),
        library_exists=library_exists,
        library_loadable=library_loadable,
        device=device,
        device_class=device_class,
        software_vulkan=software_vulkan,
        operations=operations,
        expected_tiny_ops=tuple(expected),
        status=status,
        gpu_primary=placement.gpu_primary,
        cpu_fallback=placement.cpu_fallback,
        dataset_training=False,
        probe_error=probe_error,
        diagnostics=tuple(diagnostics),
    )


def refresh_gpu_readiness_probe(
    runtime_root: str | Path = RUNTIME_ROOT, runtime_revision: str = "runtime"
) -> GpuReadiness:
    """Alias kept for a clear "refresh" call site in the controller/UI."""
    return gpu_readiness_probe(runtime_root, runtime_revision)
