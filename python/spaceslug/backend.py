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
    metadata: dict[str, Any] = field(default_factory=dict)


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
                if hasattr(self._library, "spaceslug_causal_loss"):
                    causal_loss = self._library.spaceslug_causal_loss
                    causal_loss.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32]
                    causal_loss.restype = ctypes.c_int
                if hasattr(self._library, "vulkan_runtime_dataset_batch_capability"):
                    capability = self._library.vulkan_runtime_dataset_batch_capability
                    capability.argtypes = []
                    capability.restype = ctypes.c_char_p
                if hasattr(self._library, "vulkan_runtime_dataset_batch_create"):
                    create_batch = self._library.vulkan_runtime_dataset_batch_create
                    create_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
                    create_batch.restype = ctypes.c_void_p
                if hasattr(self._library, "vulkan_runtime_dataset_batch_destroy"):
                    destroy_batch = self._library.vulkan_runtime_dataset_batch_destroy
                    destroy_batch.argtypes = [ctypes.c_void_p]
                    destroy_batch.restype = None
                if hasattr(self._library, "vulkan_runtime_dataset_batch_process"):
                    process_batch = self._library.vulkan_runtime_dataset_batch_process
                    process_batch.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint32)] * 4 + [ctypes.POINTER(ctypes.c_float)]
                    process_batch.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_lora_delta"):
                    lora_delta = self._library.spaceslug_lora_delta
                    lora_delta.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32]
                    lora_delta.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_projection_backward"):
                    projection_backward = self._library.spaceslug_projection_backward
                    projection_backward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
                    projection_backward.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_lora_gradients_multi"):
                    multi_grad = self._library.spaceslug_lora_gradients_multi
                    multi_grad.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
                    multi_grad.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_attention_causal_backward"):
                    attention_backward = self._library.spaceslug_attention_causal_backward
                    attention_backward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
                    attention_backward.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_profile_count"):
                    profile_count = self._library.spaceslug_tiny_profile_count
                    profile_count.argtypes = []
                    profile_count.restype = ctypes.c_uint32
                    profile_query = self._library.spaceslug_tiny_profile_query
                    profile_query.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
                    profile_query.restype = ctypes.c_int
                    profile_validate = self._library.spaceslug_tiny_profile_validate
                    profile_validate.argtypes = [ctypes.c_uint32] * 5
                    profile_validate.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_base_train_capability"):
                    base_train_capability = self._library.spaceslug_tiny_forward_base_train_capability
                    base_train_capability.argtypes = []
                    base_train_capability.restype = ctypes.c_char_p
                if hasattr(self._library, "spaceslug_tiny_forward_base_train_group_supported"):
                    base_train_group_supported = self._library.spaceslug_tiny_forward_base_train_group_supported
                    base_train_group_supported.argtypes = [ctypes.c_uint32]
                    base_train_group_supported.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_import_base_train_lm_head"):
                    import_base_train_lm_head = self._library.spaceslug_tiny_forward_import_base_train_lm_head
                    import_base_train_lm_head.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
                    import_base_train_lm_head.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_readback_base_train_lm_head"):
                    readback_base_train_lm_head = self._library.spaceslug_tiny_forward_readback_base_train_lm_head
                    readback_base_train_lm_head.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
                    readback_base_train_lm_head.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_import_base_train_output"):
                    self._library.spaceslug_tiny_forward_import_base_train_output.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
                    self._library.spaceslug_tiny_forward_import_base_train_output.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_readback_base_train_output"):
                    self._library.spaceslug_tiny_forward_readback_base_train_output.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
                    self._library.spaceslug_tiny_forward_readback_base_train_output.restype = ctypes.c_int
                for symbol in ("import_base_train_qkv", "readback_base_train_qkv", "readback_base_train_qkv_gradients"):
                    full_symbol = f"spaceslug_tiny_forward_{symbol}"
                    if hasattr(self._library, full_symbol):
                        function = getattr(self._library, full_symbol)
                        function.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)] * (3 if symbol != "import_base_train_qkv" else 3)
                        function.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_train_lm_head_sgd"):
                    train_lm_head_sgd = self._library.spaceslug_tiny_forward_train_lm_head_sgd
                    train_lm_head_sgd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_float]
                    train_lm_head_sgd.restype = ctypes.c_int
                for group in ("output", "qkv"):
                    symbol = f"spaceslug_tiny_forward_train_{group}_sgd"
                    if hasattr(self._library, symbol):
                        function = getattr(self._library, symbol)
                        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_float]
                        function.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_train_lm_head_adamw"):
                    train_lm_head_adamw = self._library.spaceslug_tiny_forward_train_lm_head_adamw
                    train_lm_head_adamw.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
                    train_lm_head_adamw.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_train_qkv_adamw_from_gradients"):
                     function = self._library.spaceslug_tiny_forward_train_qkv_adamw_from_gradients
                     function.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
                     function.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_base_checkpoint_create"):
                    checkpoint_create = self._library.spaceslug_tiny_base_checkpoint_create
                    checkpoint_create.argtypes = []
                    checkpoint_create.restype = ctypes.c_void_p
                    checkpoint_destroy = self._library.spaceslug_tiny_base_checkpoint_destroy
                    checkpoint_destroy.argtypes = [ctypes.c_void_p]
                    checkpoint_destroy.restype = None
                    readback_checkpoint = self._library.spaceslug_tiny_forward_readback_base_checkpoint
                    readback_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    readback_checkpoint.restype = ctypes.c_int
                    update_checkpoint = self._library.spaceslug_tiny_forward_update_base_checkpoint
                    update_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    update_checkpoint.restype = ctypes.c_int
                    for name in ("group_mask", "adamw_step", "profile_rank"):
                        function = getattr(self._library, f"spaceslug_tiny_base_checkpoint_{name}")
                        function.argtypes = [ctypes.c_void_p]
                        function.restype = ctypes.c_uint64 if name == "adamw_step" else ctypes.c_uint32
                    for name in ("float_count", "weights", "qkv_weights", "adamw_m", "adamw_v"):
                        function = getattr(self._library, f"spaceslug_tiny_base_checkpoint_{name}")
                        function.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                        function.restype = ctypes.c_uint32 if name == "float_count" else ctypes.POINTER(ctypes.c_float)
                if hasattr(self._library, "spaceslug_tiny_forward_create_dataset_batch"):
                    function = self._library.spaceslug_tiny_forward_create_dataset_batch
                    function.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
                    function.restype = ctypes.c_void_p
                if hasattr(self._library, "spaceslug_tiny_forward_destroy_dataset_batch"):
                    function = self._library.spaceslug_tiny_forward_destroy_dataset_batch
                    function.argtypes = [ctypes.c_void_p]
                    function.restype = None
                if hasattr(self._library, "spaceslug_tiny_forward_upload_dataset_batch"):
                    function = self._library.spaceslug_tiny_forward_upload_dataset_batch
                    function.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint32)] * 4
                    function.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_train_dataset_batch"):
                    function = self._library.spaceslug_tiny_forward_train_dataset_batch
                    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
                    function.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_readback_base_train_lm_head_adamw_state"):
                    readback_lm_head_adamw = self._library.spaceslug_tiny_forward_readback_base_train_lm_head_adamw_state
                    readback_lm_head_adamw.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64)]
                    readback_lm_head_adamw.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_update_base_train_lm_head_adamw_state"):
                    update_lm_head_adamw = self._library.spaceslug_tiny_forward_update_base_train_lm_head_adamw_state
                    update_lm_head_adamw.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64]
                    update_lm_head_adamw.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_tiny_forward_capability"):
                    tiny_capability = self._library.spaceslug_tiny_forward_capability
                    tiny_capability.restype = ctypes.c_char_p
                    tiny_forward = self._library.spaceslug_tiny_forward
                    tiny_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
                    tiny_forward.restype = ctypes.c_int
                    if hasattr(self._library, "spaceslug_tiny_forward_fixed_retained"):
                        fixed_retained = self._library.spaceslug_tiny_forward_fixed_retained
                        fixed_retained.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_float)]
                        fixed_retained.restype = ctypes.c_int
                    if hasattr(self._library, "spaceslug_tiny_forward_loss_fixed_retained"):
                        fixed_retained_loss = self._library.spaceslug_tiny_forward_loss_fixed_retained
                        fixed_retained_loss.argtypes = [ctypes.c_void_p,
                                                         ctypes.POINTER(ctypes.c_uint32),
                                                         ctypes.POINTER(ctypes.c_uint32),
                                                         ctypes.POINTER(ctypes.c_uint32),
                                                         ctypes.POINTER(ctypes.c_float),
                                                         ctypes.POINTER(ctypes.c_float)]
                        fixed_retained_loss.restype = ctypes.c_int
                    if hasattr(self._library, "spaceslug_tiny_forward_loss_fixed_metrics"):
                        fixed_metrics = self._library.spaceslug_tiny_forward_loss_fixed_metrics
                        fixed_metrics.argtypes = [ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_uint32),
                                                  ctypes.POINTER(ctypes.c_uint32),
                                                  ctypes.POINTER(ctypes.c_uint32),
                                                  ctypes.POINTER(ctypes.c_float),
                                                  ctypes.POINTER(ctypes.c_uint32)]
                        fixed_metrics.restype = ctypes.c_int
                    tiny_destroy = self._library.spaceslug_tiny_forward_destroy
                    tiny_destroy.argtypes = [ctypes.c_void_p]
                    if hasattr(self._library, "spaceslug_tiny_forward_create_full"):
                        create_full = self._library.spaceslug_tiny_forward_create_full
                        create_full.argtypes = [ctypes.POINTER(ctypes.c_float)] * 7
                        create_full.restype = ctypes.c_void_p
                        if hasattr(self._library, "spaceslug_tiny_forward_begin_lora_adamw"):
                            begin_adamw = self._library.spaceslug_tiny_forward_begin_lora_adamw
                            begin_adamw.argtypes = [ctypes.c_void_p]
                            begin_adamw.restype = ctypes.c_int
                        if hasattr(self._library, "spaceslug_tiny_forward_accumulate_lora_adamw"):
                            accumulate_adamw = self._library.spaceslug_tiny_forward_accumulate_lora_adamw
                            accumulate_adamw.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32] + [ctypes.POINTER(ctypes.c_float)] * 9
                            accumulate_adamw.restype = ctypes.c_int
                        if hasattr(self._library, "spaceslug_tiny_forward_finalize_lora_adamw"):
                            finalize_adamw = self._library.spaceslug_tiny_forward_finalize_lora_adamw
                            finalize_adamw.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
                            finalize_adamw.restype = ctypes.c_int
                        if hasattr(self._library, "spaceslug_tiny_forward_readback_lora_adamw_state"):
                            readback_adamw = self._library.spaceslug_tiny_forward_readback_lora_adamw_state
                            readback_adamw.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64)]
                            readback_adamw.restype = ctypes.c_int
                        if hasattr(self._library, "spaceslug_tiny_forward_update_lora_adamw_state"):
                            update_adamw = self._library.spaceslug_tiny_forward_update_lora_adamw_state
                            update_adamw.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint64]
                            update_adamw.restype = ctypes.c_int
                        begin_accumulation = self._library.spaceslug_tiny_forward_begin_lora_accumulation
                        begin_accumulation.argtypes = [ctypes.c_void_p]
                        begin_accumulation.restype = ctypes.c_int
                        token_backward_accumulate = self._library.spaceslug_tiny_forward_token_step_training_backward_accumulate
                        token_backward_accumulate.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32] + [ctypes.POINTER(ctypes.c_float)] * 9
                        token_backward_accumulate.restype = ctypes.c_int
                        finalize_sgd = self._library.spaceslug_tiny_forward_finalize_lora_sgd
                        finalize_sgd.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
                        finalize_sgd.restype = ctypes.c_int
                        readback_adapters = self._library.spaceslug_tiny_forward_readback_lora_adapters
                        readback_adapters.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)] * 8
                        readback_adapters.restype = ctypes.c_int
                        update_adapters = self._library.spaceslug_tiny_forward_update_lora_adapters
                        update_adapters.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)] * 8
                        update_adapters.restype = ctypes.c_int
                    tiny_destroy.restype = None
                        
                if hasattr(self._library, "spaceslug_lora_session_token_step"):
                    token_step = self._library.spaceslug_lora_session_token_step
                    token_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
                    token_step.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_lora_session_create"):
                    session_create = self._library.spaceslug_lora_session_create
                    session_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_void_p)]
                    session_create.restype = ctypes.c_int
                    session_step = self._library.spaceslug_lora_session_step
                    session_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
                    session_step.restype = ctypes.c_int
                    session_readback = self._library.spaceslug_lora_session_readback
                    session_readback.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
                    session_readback.restype = ctypes.c_int
                    session_destroy = self._library.spaceslug_lora_session_destroy
                    session_destroy.argtypes = [ctypes.c_void_p]
                    session_destroy.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_lora_sgd_multi"):
                    sgd_multi = self._library.spaceslug_lora_sgd_multi
                    sgd_multi.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_uint32]
                    sgd_multi.restype = ctypes.c_int
                if hasattr(self._library, "spaceslug_lora_train_step"):
                    lora_train = self._library.spaceslug_lora_train_step
                    lora_train.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_uint32, ctypes.c_uint32]
                    lora_train.restype = ctypes.c_int
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
                native = self._native()
                if hasattr(native, "spaceslug_attention_causal"):
                    operations.append("attention_causal_abi")
                if hasattr(native, "spaceslug_tiny_forward"):
                    operations.append("tiny_forward_abi")
                if hasattr(native, "spaceslug_tiny_forward_fixed_retained"):
                    operations.append("tiny_forward_fixed_retained_abi")
                if hasattr(native, "spaceslug_tiny_forward_loss_fixed_retained"):
                    operations.append("tiny_forward_loss_fixed_retained_abi")
                if hasattr(native, "spaceslug_lora_train_step"):
                    operations.append("lora_train_step_abi")
                if hasattr(native, "spaceslug_attention_causal_backward"):
                    operations.append("attention_causal_backward_abi")
                if hasattr(native, "spaceslug_lora_gradients_multi"):
                    operations.append("lora_gradients_multi_abi")
                if hasattr(native, "spaceslug_lora_sgd_multi"):
                    operations.append("lora_sgd_multi_abi")
                if hasattr(native, "spaceslug_lora_session_create"):
                    operations.append("lora_session_persistent_abi")
                if all(hasattr(native, name) for name in ("spaceslug_tiny_forward_begin_lora_adamw", "spaceslug_tiny_forward_finalize_lora_adamw", "spaceslug_tiny_forward_readback_lora_adamw_state", "spaceslug_tiny_forward_update_lora_adamw_state")) and hasattr(native, "spaceslug_tiny_forward_token_step_training_backward_accumulate"):
                    operations.append("tiny_lora_adamw_abi")
                if hasattr(native, "spaceslug_causal_loss"):
                    operations.append("causal_loss_abi")
                if hasattr(native, "vulkan_runtime_dataset_batch_process"):
                    operations.append("dataset_batch_buffer_abi")
            except BackendError:
                pass
        native_retained = False
        native_retained_loss = False
        native_training = False
        if self._library_path.is_file():
            try:
                native = self._native()
                native_retained = hasattr(native, "spaceslug_tiny_forward_fixed_retained")
                native_retained_loss = hasattr(native, "spaceslug_tiny_forward_loss_fixed_retained")
                native_training = all(hasattr(native, name) for name in (
                    "spaceslug_tiny_forward_begin_lora_accumulation",
                    "spaceslug_tiny_forward_token_step_training_backward_accumulate",
                    "spaceslug_tiny_forward_finalize_lora_sgd",
                ))
            except BackendError:
                pass
        dataset_batch = False
        dataset_batch_capability = None
        if self._library_path.is_file():
            try:
                native = self._native()
                dataset_batch = all(hasattr(native, name) for name in (
                    "vulkan_runtime_dataset_batch_capability",
                    "vulkan_runtime_dataset_batch_create",
                    "vulkan_runtime_dataset_batch_destroy",
                    "vulkan_runtime_dataset_batch_process",
                ))
                if dataset_batch:
                    dataset_batch_capability = native.vulkan_runtime_dataset_batch_capability().decode("utf-8")
            except (BackendError, AttributeError):
                pass
        from .native_training import integrated_tiny_group_adamw_capability, integrated_tiny_lm_head_adamw_capability, integrated_tiny_lm_head_capability, native_fp32_lm_head_capability
        native_fp32_base_training = native_fp32_lm_head_capability()
        graph_lm_head = False
        graph_lm_head_capability = None
        graph_lm_head_group_supported = False
        graph_lm_head_training_methods = False
        graph_output_training_methods = False
        graph_qkv_training_methods = False
        graph_lm_head_adamw = False
        graph_output_adamw = False
        graph_qkv_adamw = False
        if self._library_path.is_file():
            try:
                native = self._native()
                graph_lm_head = all(hasattr(native, name) for name in (
                    "spaceslug_tiny_forward_base_train_capability",
                    "spaceslug_tiny_forward_base_train_group_supported",
                    "spaceslug_tiny_forward_import_base_train_lm_head",
                    "spaceslug_tiny_forward_readback_base_train_lm_head",
                ))
                graph_lm_head_training_methods = hasattr(native, "spaceslug_tiny_forward_train_lm_head_sgd")
                graph_output_training_methods = hasattr(native, "spaceslug_tiny_forward_train_output_sgd")
                graph_qkv_training_methods = hasattr(native, "spaceslug_tiny_forward_train_qkv_sgd")
                graph_output_adamw = all(hasattr(native, name) for name in (
                "spaceslug_tiny_forward_train_output_adamw",
                "spaceslug_tiny_forward_readback_base_train_output_adamw_state",
                "spaceslug_tiny_forward_update_base_train_output_adamw_state",
                ))
                graph_qkv_adamw = all(hasattr(native, name) for name in (
                "spaceslug_tiny_forward_train_qkv_adamw_from_gradients",
                "spaceslug_tiny_forward_readback_base_train_qkv_adamw_state",
                "spaceslug_tiny_forward_update_base_train_qkv_adamw_state",
                ))
                graph_lm_head_adamw = all(hasattr(native, name) for name in (
                "spaceslug_tiny_forward_train_lm_head_adamw",
                "spaceslug_tiny_forward_readback_base_train_lm_head_adamw_state",
                "spaceslug_tiny_forward_update_base_train_lm_head_adamw_state",
                ))
                if graph_lm_head:
                    graph_lm_head_capability = native.spaceslug_tiny_forward_base_train_capability().decode("utf-8")
                    graph_lm_head_group_supported = bool(native.spaceslug_tiny_forward_base_train_group_supported(1))
                    if graph_lm_head_group_supported and "base_train_group_lm_head_owned_fp32_fixed_window_sgd" not in graph_lm_head_capability:
                        graph_lm_head_capability += ";base_train_group_lm_head_owned_fp32_fixed_window_sgd"
            except (BackendError, AttributeError):
                pass
        metadata = {"tiny_forward_fixed_retained": native_retained,
                    "tiny_forward_loss_fixed_retained": native_retained_loss,
                    "tiny_training_production": native_training,
                    "native_fp32_base_training_subsets": native_fp32_base_training,
                    "native_fp32_lm_head_only_base_training": native_fp32_base_training,
                     "tiny_graph_base_train_lm_head": graph_lm_head,
                     "tiny_graph_base_train_lm_head_capability": graph_lm_head_capability,
                     "tiny_graph_base_train_lm_head_group_supported": graph_lm_head_group_supported,
                     "tiny_graph_base_train_lm_head_training_methods": graph_lm_head_training_methods,
                     "tiny_graph_base_train_lm_head_training": False,
                     "tiny_graph_base_train_lm_head_training_return_code": -4 if graph_lm_head_training_methods else None,
                     "tiny_graph_integrated_lm_head_sgd": integrated_tiny_lm_head_capability(available=graph_lm_head_training_methods, runtime_capability=graph_lm_head_capability),
                     "tiny_graph_integrated_output_sgd": integrated_tiny_lm_head_capability(group="output", available=graph_output_training_methods, runtime_capability=graph_lm_head_capability),
                     "tiny_graph_integrated_qkv_sgd": integrated_tiny_lm_head_capability(group="qkv", available=graph_qkv_training_methods, runtime_capability=graph_lm_head_capability),
                     "tiny_graph_integrated_lm_head_adamw": integrated_tiny_lm_head_adamw_capability(available=graph_lm_head_adamw, runtime_capability=graph_lm_head_capability),
                      "tiny_graph_integrated_output_adamw": integrated_tiny_group_adamw_capability(group="output", available=graph_output_adamw, runtime_capability=graph_lm_head_capability),
                     "tiny_graph_integrated_qkv_adamw": integrated_tiny_group_adamw_capability(group="qkv", available=graph_qkv_adamw, runtime_capability=graph_lm_head_capability),
                    "tiny_forward_token_count": 128 if native_retained else None,
                    "tiny_forward_loss_token_count": 128 if native_retained_loss else None,
                    "tiny_forward_loss_target_count": 128 if native_retained_loss else None,
                    "tiny_forward_loss_mask_count": 128 if native_retained_loss else None,
                    "dataset_batch_buffer": dataset_batch,
                    "dataset_batch_buffer_capability": dataset_batch_capability,
                    "dataset_batch_buffer_training": False}
        return BackendCapabilities("spaceslug", self.runtime_revision, tuple(operations), self._device, self.software_vulkan, str(self._library_path) if self._library_path.is_file() else None, metadata)

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

    def execute_tiny_lora(self, x: list[float], a: list[float], b: list[float], dy: list[float] | None = None, rank: int = 4, learning_rate: float = 0.01) -> ExecutionResult:
        rows = len(x) // 64 if x else 0
        if rank != 4 or rows == 0 or rows > 128 or len(x) % 64 or len(a) != 64 * rank or len(b) != rank * 64:
            return ExecutionResult("not-run", "tiny_lora_forward_backward_update", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "LoRA MVP requires 1<=M<=128, rank=4, and H=64", "rank": rank}, {"rows": rows})
        if dy is None or len(dy) != len(x):
            return ExecutionResult("not-run", "tiny_lora_forward_backward_update", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "train-step requires dY[M,64]", "rank": rank}, {"rows": rows})
        self._native()
        if not hasattr(self._library, "spaceslug_lora_train_step"):
            return ExecutionResult("not-run", "tiny_lora_forward_backward_update", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "GPU LoRA train-step ABI is unavailable", "rank": rank}, {"rows": rows})
        import array, ctypes, os
        xx, dd, aa, bb = (array.array("f", values) for values in (x, dy, a, b))
        yy = (ctypes.c_float * len(xx))()
        a_before, b_before = aa.tolist(), bb.tolist()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_lora_train_step((ctypes.c_float * len(xx)).from_buffer(xx), (ctypes.c_float * len(dd)).from_buffer(dd), (ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb), yy, learning_rate, rows, rank)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native LoRA train-step returned {code}")
        y_ref = [sum(x[row * 64 + k] * a_before[k * rank + r] * b_before[r * 64 + column] for r in range(rank) for k in range(64)) for row in range(rows) for column in range(64)]
        da = [sum(x[row * 64 + k] * dy[row * 64 + column] * b_before[r * 64 + column] for row in range(rows) for column in range(64)) for k in range(64) for r in range(rank)]
        db = [sum(sum(x[row * 64 + k] * a_before[k * rank + r] for k in range(64)) * dy[row * 64 + column] for row in range(rows)) for r in range(rank) for column in range(64)]
        a_ref = [value - learning_rate * gradient for value, gradient in zip(a_before, da)]
        b_ref = [value - learning_rate * gradient for value, gradient in zip(b_before, db)]
        y_parity, a_parity, b_parity = compare_float_arrays(list(yy), y_ref), compare_float_arrays(aa.tolist(), a_ref), compare_float_arrays(bb.tolist(), b_ref)
        passed = all(report["status"] == "pass" for report in (y_parity, a_parity, b_parity))
        return ExecutionResult("ok" if passed else "error", "tiny_lora_forward_backward_update", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "cpu-reference", "gpu_execution": passed, "rank": rank, "forward": y_parity, "dA_update": a_parity, "dB_update": b_parity}, {"y": list(yy), "a": aa.tolist(), "b": bb.tolist(), "rows": rows})

    def execute_tiny_lora_plan(self, model: Any, rank: int = 4) -> ExecutionResult:
        supported = model.hidden_size == 64 and model.vocab_size == 259 and rank == 4 and hasattr(self._native(), "spaceslug_lora_train_step")
        return ExecutionResult("ready" if supported else "not-run", "tiny_lora_forward_backward_update", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "cpu-reference-required", "reason": "supply X and dY tensors to execute the native LoRA train-step" if supported else "LoRA MVP requires H=64, V=259, rank=4, and native ABI", "rank": rank, "gpu_execution": False}, {"supported_base": supported})

    def execute_projected_attention_forward(self, tokens: list[int], model: Any) -> ExecutionResult:
        """CPU-authoritative forward reference."""
        return self.execute_projected_attention_cpu_fallback(tokens, model)

    def execute_projected_attention_gpu(self, tokens: list[int], model: Any) -> ExecutionResult:
        """Run the composed Vulkan Tiny chain when its fixed contract is available."""
        if model.hidden_size == 64 and model.vocab_size == 259 and 0 < len(tokens) <= 128 and hasattr(self._native(), "spaceslug_attention_causal"):
            return self._execute_tiny_gpu_forward(tokens, model)
        return self.execute_projected_attention_gpu_plan(tokens, model)

    def execute_tiny_lora_train_graph(self, tokens: list[int], targets: list[int], mask: list[int], model: Any, adapter: Any, learning_rate: float = 0.01) -> ExecutionResult:
        if model.hidden_size != 64 or model.vocab_size != 259 or adapter.rank != 4 or not tokens or len(tokens) > 128 or len(targets) != len(tokens) or len(mask) != len(tokens):
            return ExecutionResult("not-run", "tiny_lora_train_graph", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "graph requires Tiny V=259,H=64,rank=4 and equal token/target/mask lengths"}, {})
        if not all(hasattr(self._native(), name) for name in ("spaceslug_lora_delta", "spaceslug_causal_loss", "spaceslug_lm_head_backward", "spaceslug_projection_backward", "spaceslug_attention_causal_backward", "spaceslug_lora_gradients_multi")):
            return ExecutionResult("not-run", "tiny_lora_train_graph", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "one or more graph ABIs unavailable"}, {})
        from .positional_encoding import sinusoidal_positions
        from .lora import LoRAProjectedTinyAttention
        import math
        h, vp, actual_rows, rows = 64, 320, len(tokens), 128
        states = [model.embedding[token][:] for token in tokens]
        if model.use_positions:
            positions = sinusoidal_positions(actual_rows, h)
            states = [[value + positions[i][c] for c, value in enumerate(state)] for i, state in enumerate(states)]
        states += [[0.0] * h for _ in range(rows - actual_rows)]
        flat = [value for state in states for value in state]
        def mat(values): return [value for row in values for value in row]
        projections = {}
        for name in ("query", "key", "value"):
            base = self._native_sgemm_values(flat, mat(getattr(model, name)), rows, h, h)
            lm = adapter.matrices[name]
            scaled_b = [value * (adapter.alpha / adapter.rank) for row in lm.B for value in row]
            delta = self._native_lora_delta_values(flat, mat(lm.A), scaled_b, rows, adapter.rank)
            projections[name] = [x + y for x, y in zip(base, delta)]
        import array, ctypes, os
        q, k, v = (array.array("f", projections[name]) for name in ("query", "key", "value"))
        context = (ctypes.c_float * (rows * h))()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_attention_causal((ctypes.c_float * len(q)).from_buffer(q), (ctypes.c_float * len(k)).from_buffer(k), (ctypes.c_float * len(v)).from_buffer(v), context, rows, h)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native causal attention returned {code}")
        context_values = list(context)
        output_base = self._native_sgemm_values(context_values, mat(model.output), rows, h, h)
        out_lora = adapter.matrices["output"]
        output_delta = self._native_lora_delta_values(context_values, mat(out_lora.A), [x * (adapter.alpha / adapter.rank) for row in out_lora.B for x in row], rows, adapter.rank)
        projected = [x + y for x, y in zip(output_base, output_delta)]
        lm = [model.lm_head[r][c] if c < model.vocab_size else 0.0 for r in range(h) for c in range(vp)]
        logits = self._native_sgemm_values(projected, lm, rows, vp, h)
        padded_targets = list(targets) + [0] * (rows - actual_rows)
        padded_mask = list(mask) + [0] * (rows - actual_rows)
        loss_result = self.execute_causal_loss(logits, padded_targets, padded_mask, vp)
        if loss_result.status != "ok": return loss_result
        dprojected = (ctypes.c_float * (rows * h))()
        ll, gg = array.array("f", loss_result.output["dlogits"]), None
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_lm_head_backward((ctypes.c_float * len(ll)).from_buffer(ll), (ctypes.c_float * len(lm)).from_buffer(array.array("f", lm)), dprojected, rows, vp, h)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native lm head backward returned {code}")
        dprojected_values = list(dprojected)
        output_grad = self._native_projection_backward(dprojected_values, mat(model.output), rows, h, h)
        output_input = context_values
        output_gradients = self._native_multi_gradients(output_input, dprojected_values, adapter_a=[value for target in range(4) for row in adapter.matrices["output"].A for value in row], adapter_b=[value for target in range(4) for row in adapter.matrices["output"].B for value in row], rows=rows, rank=adapter.rank, target=3) if False else None
        attention_gradients = {mode: self._native_attention_backward(projections["query"], projections["key"], projections["value"], output_grad, actual_rows, mode) for mode in range(3)}
        packed_a = [value for name in ("query", "key", "value", "output") for row in adapter.matrices[name].A for value in row]
        packed_b = [value for name in ("query", "key", "value", "output") for row in adapter.matrices[name].B for value in row]
        projection_inputs = {name: flat for name, flat in (("query", flat), ("key", flat), ("value", flat), ("output", context_values))}
        lora_gradients = {}
        for target, name in enumerate(("query", "key", "value", "output")):
            source = attention_gradients[target] if target < 3 else output_grad
            lora_gradients[name] = self._native_multi_gradients(projection_inputs[name], source, packed_a, packed_b, actual_rows, adapter.rank, target)
        learning_rate = float(learning_rate)
        if learning_rate <= 0.0: raise ValueError("learning_rate must be positive")
        if not hasattr(self._native(), "spaceslug_lora_sgd_multi"):
            return ExecutionResult("ready", "tiny_lora_train_graph", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "gradient-partial", "gpu_execution": True, "reason": "GPU gradients complete but multi-adapter SGD ABI unavailable"}, {"dprojected": dprojected_values, "lora_gradients": lora_gradients})
        packed_da = [value for target in range(4) for value in lora_gradients[("query", "key", "value", "output")[target]][0][target * 64 * adapter.rank:(target + 1) * 64 * adapter.rank]]
        packed_db = [value for target in range(4) for value in lora_gradients[("query", "key", "value", "output")[target]][1][target * adapter.rank * 64:(target + 1) * adapter.rank * 64]]
        updated_a, updated_b = self._native_multi_sgd(packed_a, packed_b, packed_da, packed_db, learning_rate, adapter.rank)
        return ExecutionResult("ok", "tiny_lora_train_graph", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "gradient-update-partial", "gpu_execution": True, "loss": loss_result.metrics.get("loss"), "dlogits": loss_result.metrics.get("dlogits"), "reason": "GPU forward, loss, backward, four gradients, and host-coordinated SGD are complete; persistent device-resident update remains"}, {"dprojected": dprojected_values, "d_output_projection": output_grad, "d_attention": attention_gradients, "lora_gradients": lora_gradients, "updated_a": updated_a, "updated_b": updated_b, "logits": logits[:actual_rows * vp], "row_loss": loss_result.output["row_loss"][:actual_rows]})

    def execute_tiny_lora_backward_chain(self, q: list[float], k: list[float], v: list[float], d_context: list[float], projection_inputs: dict[str, list[float]], adapter_a: list[float], adapter_b: list[float], rank: int) -> ExecutionResult:
        tokens = len(q) // 64
        if tokens == 0 or tokens > 128 or any(len(values) != tokens * 64 for values in (k, v, d_context)) or rank < 1 or rank > 8 or len(adapter_a) != 4 * 64 * rank or len(adapter_b) != 4 * rank * 64:
            return ExecutionResult("not-run", "tiny_lora_backward_chain", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "backward chain requires equal Q/K/V/dContext [T,64] and packed four-target factors"}, {})
        required = ("query", "key", "value", "output")
        if any(name not in projection_inputs or len(projection_inputs[name]) != tokens * 64 for name in required):
            return ExecutionResult("not-run", "tiny_lora_backward_chain", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "backward chain requires projection inputs for query/key/value/output"}, {})
        if not all(hasattr(self._native(), name) for name in ("spaceslug_attention_causal_backward", "spaceslug_lora_gradients_multi")):
            return ExecutionResult("not-run", "tiny_lora_backward_chain", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "attention backward or multi-gradient ABI unavailable"}, {})
        attention_outputs = {}
        for mode, name in enumerate(("query", "key", "value")):
            attention_outputs[name] = self._native_attention_backward(q, k, v, d_context, tokens, mode)
        gradients = {}
        for target, name in enumerate(required):
            gradients[name] = self._native_multi_gradients(projection_inputs[name], attention_outputs[name] if name != "output" else d_context, adapter_a, adapter_b, tokens, rank, target)
        return ExecutionResult("ok", "tiny_lora_backward_chain", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "native-composed", "gpu_execution": True, "targets": list(required)}, {"attention_gradients": attention_outputs, "lora_gradients": gradients})

    def execute_lora_gradients_multi(self, x: list[float], dy: list[float], a: list[float], b: list[float], rank: int, target: int) -> ExecutionResult:
        rows = len(x) // 64
        if rows == 0 or rows > 128 or len(x) % 64 or len(dy) != len(x) or len(a) != 4 * 64 * rank or len(b) != 4 * rank * 64 or rank < 1 or rank > 8 or target < 0 or target > 3:
            return ExecutionResult("not-run", "lora_gradients_multi", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "four-adapter gradient contract requires [M,64], packed 4-target factors, rank<=8, target 0..3"}, {})
        if not hasattr(self._native(), "spaceslug_lora_gradients_multi"):
            return ExecutionResult("not-run", "lora_gradients_multi", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "multi-adapter gradient ABI unavailable"}, {})
        da, db = self._native_multi_gradients(x, dy, a, b, rows, rank, target)
        return ExecutionResult("ok", "lora_gradients_multi", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "native-cpu-gate", "gpu_execution": True, "target": target}, {"dA": da, "dB": db})

    def execute_attention_backward(self, q: list[float], k: list[float], v: list[float], d_output: list[float], mode: int) -> ExecutionResult:
        tokens = len(q) // 64
        if tokens == 0 or tokens > 128 or len(q) % 64 or any(len(values) != len(q) for values in (k, v, d_output)) or mode not in (0, 1, 2):
            return ExecutionResult("not-run", "attention_causal_backward", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "attention backward requires equal [T,64] tensors and mode 0/1/2"}, {})
        if not hasattr(self._native(), "spaceslug_attention_causal_backward"):
            return ExecutionResult("not-run", "attention_causal_backward", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "attention backward ABI unavailable"}, {})
        result = self._native_attention_backward(q, k, v, d_output, tokens, mode)
        return ExecutionResult("ok", "attention_causal_backward", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "native-cpu-gate", "gpu_execution": True, "mode": mode}, {"output": result, "tokens": tokens})

    def execute_causal_loss(self, logits: list[float], targets: list[int], mask: list[int], vocab: int) -> ExecutionResult:
        if not logits or len(targets) != len(mask) or len(logits) != len(targets) * vocab or vocab <= 0 or vocab > 320:
            return ExecutionResult("not-run", "causal_loss", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "loss requires logits[M,V], targets[M], mask[M], and V<=320"}, {})
        if not hasattr(self._native(), "spaceslug_causal_loss"):
            return ExecutionResult("not-run", "causal_loss", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "causal loss ABI is unavailable"}, {})
        losses, gradients = self._native_causal_loss(logits, targets, mask, vocab)
        import math
        ref_losses, ref_gradients = [], []
        for row, target in enumerate(targets):
            values = logits[row * vocab:(row + 1) * vocab]
            maximum = max(values)
            normalizer = sum(math.exp(value - maximum) for value in values)
            included = bool(mask[row]) and target < vocab
            ref_losses.append(math.log(normalizer) + maximum - values[target] if included else 0.0)
            ref_gradients.extend((math.exp(value - maximum) / normalizer - (1.0 if col == target else 0.0)) if included else 0.0 for col, value in enumerate(values))
        loss_parity = compare_float_arrays(losses, ref_losses)
        gradient_parity = compare_float_arrays(gradients, ref_gradients)
        passed = loss_parity["status"] == "pass" and gradient_parity["status"] == "pass"
        return ExecutionResult("ok" if passed else "error", "causal_loss", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "cpu-reference", "gpu_execution": passed, "loss": loss_parity, "dlogits": gradient_parity}, {"row_loss": losses, "dlogits": gradients})

    def execute_tiny_lora_forward(self, tokens: list[int], model: Any, adapter: Any) -> ExecutionResult:
        """Compose frozen base projections with GPU LoRA deltas for Tiny inference."""
        if model.hidden_size != 64 or model.vocab_size != 259 or not 0 < len(tokens) <= 128 or adapter.hidden_size != 64 or adapter.rank != 4:
            return ExecutionResult("not-run", "tiny_lora_forward", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "GPU LoRA forward requires V=259, H=64, rank=4, and 1<=T<=128"}, {"token_count": len(tokens)})
        if not hasattr(self._native(), "spaceslug_lora_delta") or not hasattr(self._native(), "spaceslug_attention_causal"):
            return ExecutionResult("not-run", "tiny_lora_forward", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "LoRA delta or causal attention ABI is unavailable"}, {"token_count": len(tokens)})
        import array, ctypes, os
        from .lora import LoRAProjectedTinyAttention
        from .positional_encoding import sinusoidal_positions
        t, h, vp = len(tokens), 64, 320
        states = [model.embedding[token][:] for token in tokens]
        if model.use_positions:
            positions = sinusoidal_positions(t, h)
            states = [[value + positions[row][column] for column, value in enumerate(state)] for row, state in enumerate(states)]
        states += [[0.0] * h for _ in range(128 - t)]
        flat_states = [value for row in states for value in row]
        def flat(matrix): return [value for row in matrix for value in row]
        def projected(name):
            base = self._native_sgemm_values(flat_states, flat(getattr(model, name)), 128, h, h)
            matrix = adapter.matrices[name]
            scaled_b = [value * (adapter.alpha / adapter.rank) for row in matrix.B for value in row]
            delta = self._native_lora_delta_values(flat_states, flat(matrix.A), scaled_b, 128, adapter.rank)
            return [left + right for left, right in zip(base, delta)]
        q, k, v = projected("query"), projected("key"), projected("value")
        qv, kv, vv = (array.array("f", values) for values in (q, k, v))
        context = (ctypes.c_float * (128 * h))()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_attention_causal((ctypes.c_float * len(qv)).from_buffer(qv), (ctypes.c_float * len(kv)).from_buffer(kv), (ctypes.c_float * len(vv)).from_buffer(vv), context, t, h)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native causal attention returned {code}")
        context_values = list(context)
        base_projected = self._native_sgemm_values(context_values, flat(model.output), 128, h, h)
        output_matrix = adapter.matrices["output"]
        scaled_output_b = [value * (adapter.alpha / adapter.rank) for row in output_matrix.B for value in row]
        delta_projected = self._native_lora_delta_values(context_values, flat(output_matrix.A), scaled_output_b, 128, adapter.rank)
        projected_values = [left + right for left, right in zip(base_projected, delta_projected)]
        lm = [model.lm_head[row][column] if column < model.vocab_size else 0.0 for row in range(h) for column in range(vp)]
        logits_all = self._native_sgemm_values(projected_values, lm, 128, vp, h)
        logits = logits_all[(t - 1) * vp:t * vp][:model.vocab_size]
        reference = LoRAProjectedTinyAttention(model, adapter).logits_for_tokens(tokens)
        parity = compare_float_arrays(logits, reference)
        return ExecutionResult("ok" if parity["status"] == "pass" else "error", "tiny_lora_forward", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "cpu-lora-reference", "gpu_execution": parity["status"] == "pass", "adapter_targets": ["query", "key", "value", "output"], **parity}, {"logits": logits, "token_count": t})

    def _native_projection_backward(self, dy: list[float], weight: list[float], rows: int, input_size: int, output_size: int) -> list[float]:
        import array, ctypes, os
        dd, ww = array.array("f", dy), array.array("f", weight)
        result = (ctypes.c_float * (rows * input_size))()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_projection_backward((ctypes.c_float * len(dd)).from_buffer(dd), (ctypes.c_float * len(ww)).from_buffer(ww), result, rows, input_size, output_size)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native projection backward returned {code}")
        return list(result)

    def dataset_batch_buffer_capability(self) -> str:
        """Return the optional standalone dataset prototype capability string."""
        native = self._native()
        fn = getattr(native, "vulkan_runtime_dataset_batch_capability", None)
        if fn is None:
            return "unsupported"
        return fn().decode("utf-8")

    def create_dataset_batch_buffer(self, window_count: int, window_tokens: int) -> ctypes.c_void_p:
        """Create the optional standalone fixed-window dataset buffer.

        This prototype owns dataset staging/device buffers only; it is not a
        PersistentTiny training graph and does not perform model training.
        """
        if window_count <= 0 or window_tokens <= 0:
            raise ValueError("dataset batch dimensions must be positive")
        fn = getattr(self._native(), "vulkan_runtime_dataset_batch_create", None)
        if fn is None:
            raise BackendError("native dataset batch buffer ABI unavailable")
        handle = fn(None, window_count, window_tokens)
        if not handle:
            raise BackendError("dataset batch buffer creation failed")
        return ctypes.c_void_p(handle)

    def process_dataset_batch_buffer(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], controls: list[int], window_count: int, window_tokens: int) -> list[float]:
        """Process one rectangular batch through the standalone prototype."""
        elements = window_count * window_tokens
        if window_count <= 0 or window_tokens <= 0 or any(len(values) != elements for values in (tokens, targets, masks)) or len(controls) != window_count:
            raise ValueError("dataset batch inputs do not match fixed dimensions")
        import array
        arrays = [array.array("I", values) for values in (tokens, targets, masks, controls)]
        output = array.array("f", [0.0] * (window_count * 2))
        pointers = [(ctypes.c_uint32 * len(values)).from_buffer(values) for values in arrays]
        code = self._native().vulkan_runtime_dataset_batch_process(handle, *pointers, (ctypes.c_float * len(output)).from_buffer(output))
        if code != 0:
            raise BackendError(f"dataset batch buffer processing returned {code}")
        return output.tolist()

    def close_dataset_batch_buffer(self, handle: ctypes.c_void_p) -> None:
        fn = getattr(self._native(), "vulkan_runtime_dataset_batch_destroy", None)
        if fn is None:
            raise BackendError("native dataset batch buffer ABI unavailable")
        fn(handle)

    def create_tiny_dataset_batch(self, graph: ctypes.c_void_p, window_count: int, window_tokens: int) -> ctypes.c_void_p:
        """Create optional device-resident dataset storage (max 32 x 128)."""
        if not 0 < window_count <= 32 or not 0 < window_tokens <= 128:
            raise ValueError("device dataset batches require 1..32 windows and 1..128 tokens")
        fn = getattr(self._native(), "spaceslug_tiny_forward_create_dataset_batch", None)
        if fn is None:
            raise BackendError("device-resident dataset training ABI unavailable")
        handle = fn(graph, window_count, window_tokens)
        if not handle:
            raise BackendError("device dataset batch creation failed")
        return ctypes.c_void_p(handle)

    def upload_tiny_dataset_batch(self, batch: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], controls: list[int], window_count: int, window_tokens: int) -> None:
        elements = window_count * window_tokens
        if not 0 < window_count <= 32 or not 0 < window_tokens <= 128 or any(len(values) != elements for values in (tokens, targets, masks)) or len(controls) != window_count:
            raise ValueError("device dataset batch inputs exceed 32 windows or 128 tokens")
        fn = getattr(self._native(), "spaceslug_tiny_forward_upload_dataset_batch", None)
        if fn is None:
            raise BackendError("device-resident dataset training ABI unavailable")
        import array
        arrays = [array.array("I", values) for values in (tokens, targets, masks, controls)]
        pointers = [(ctypes.c_uint32 * len(values)).from_buffer(values) for values in arrays]
        code = fn(batch, *pointers)
        if code != 0:
            raise BackendError(f"device dataset batch upload returned {code}")

    def train_tiny_dataset_batch_lm_head_sgd(self, graph: ctypes.c_void_p, batch: ctypes.c_void_p, learning_rate: float, normalizer: float = 1.0) -> None:
        """Train only the graph-owned LM head from uploaded device windows.

        This is bounded dataset metadata/training, not full-dataset or all-parameter training.
        """
        if learning_rate <= 0.0 or normalizer <= 0.0:
            raise ValueError("learning_rate and normalizer must be positive")
        fn = getattr(self._native(), "spaceslug_tiny_forward_train_dataset_batch", None)
        if fn is None:
            raise BackendError("device-resident dataset LM-head SGD ABI unavailable")
        code = fn(graph, batch, learning_rate, normalizer)
        if code != 0:
            raise BackendError(f"device dataset LM-head SGD returned {code}")

    def close_tiny_dataset_batch(self, batch: ctypes.c_void_p) -> None:
        fn = getattr(self._native(), "spaceslug_tiny_forward_destroy_dataset_batch", None)
        if fn is None:
            raise BackendError("device-resident dataset training ABI unavailable")
        fn(batch)

    def create_tiny_persistent_full(self, model: Any, adapter: Any) -> ctypes.c_void_p:
        """Create the fixed Tiny graph; only H=64, V=259, rank=4 is supported."""
        if (model.hidden_size, model.vocab_size, adapter.hidden_size, adapter.rank) != (64, 259, 64, 4):
            raise BackendError("persistent Tiny contract requires H=64, V=259, rank=4")
        from .positional_encoding import sinusoidal_positions
        import array
        flat = lambda matrix: [value for row in matrix for value in row]
        arrays = [array.array("f", flat(model.embedding)), array.array("f", flat(sinusoidal_positions(128, 64))),
                  *[array.array("f", flat(getattr(model, name))) for name in ("query", "key", "value", "output")],
                  array.array("f", [model.lm_head[r][c] if c < model.vocab_size else 0.0 for r in range(64) for c in range(320)])]
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in arrays]
        handle = self._native().spaceslug_tiny_forward_create_full(*pointers)
        if not handle:
            raise BackendError("persistent Tiny graph creation failed")
        self._tiny_arrays = getattr(self, "_tiny_arrays", {})
        self._tiny_arrays[int(handle)] = arrays
        return ctypes.c_void_p(handle)

    def readback_tiny_graph_base_checkpoint(self, handle: ctypes.c_void_p) -> dict[str, Any]:
        """Read the graph-owned base checkpoint ABI into JSON-ready metadata."""
        native = self._native()
        required = ("spaceslug_tiny_base_checkpoint_create", "spaceslug_tiny_forward_readback_base_checkpoint")
        if not all(hasattr(native, name) for name in required):
            raise BackendError("graph-owned base checkpoint ABI unavailable")
        checkpoint = native.spaceslug_tiny_base_checkpoint_create()
        if not checkpoint:
            raise BackendError("graph-owned base checkpoint allocation failed")
        try:
            code = native.spaceslug_tiny_forward_readback_base_checkpoint(handle, checkpoint)
            if code != 0:
                raise BackendError(f"graph-owned base checkpoint readback returned {code}")
            mask = int(native.spaceslug_tiny_base_checkpoint_group_mask(checkpoint))
            result: dict[str, Any] = {"version": 1, "group_mask": mask,
                "adamw_step": int(native.spaceslug_tiny_base_checkpoint_adamw_step(checkpoint)),
                "profile_rank": int(native.spaceslug_tiny_base_checkpoint_profile_rank(checkpoint))}
            counts = {"lm_head": (1, 64 * 320), "output": (2, 64 * 64), "qkv": (4, 64 * 64)}
            for group, bit in (("lm_head", 1), ("output", 2), ("qkv", 4)):
                if not mask & bit:
                    continue
                if group == "qkv":
                    result[group] = []
                    for projection in range(3):
                        pointer = native.spaceslug_tiny_base_checkpoint_qkv_weights(checkpoint, projection)
                        result[group].append([pointer[i] for i in range(64 * 64)])
                else:
                    count = int(native.spaceslug_tiny_base_checkpoint_float_count(checkpoint, bit)) or counts[group][1]
                    pointer = native.spaceslug_tiny_base_checkpoint_weights(checkpoint, bit)
                    result[group] = [pointer[i] for i in range(count)]
                    if group == "lm_head" or group == "output":
                        for state_name, symbol in (("m", "adamw_m"), ("v", "adamw_v")):
                            state_pointer = getattr(native, f"spaceslug_tiny_base_checkpoint_{symbol}")(checkpoint, bit)
                            if state_pointer:
                                result[f"{group}_{state_name}"] = [state_pointer[i] for i in range(count)]
            return result
        finally:
            native.spaceslug_tiny_base_checkpoint_destroy(checkpoint)

    def update_tiny_graph_base_checkpoint(self, handle: ctypes.c_void_p, checkpoint: dict[str, Any]) -> None:
        """Restore graph-owned weights and supported AdamW state from metadata."""
        native = self._native()
        if not all(hasattr(native, name) for name in ("spaceslug_tiny_forward_import_base_train_lm_head", "spaceslug_tiny_forward_import_base_train_output")):
            raise BackendError("graph-owned base checkpoint restore ABI unavailable")
        mask = int(checkpoint.get("group_mask", 0))
        import array
        def floats(values: list[float]):
            return (ctypes.c_float * len(values))(*values)
        if mask & 1 and checkpoint.get("lm_head"):
            values = floats(checkpoint["lm_head"]); code = native.spaceslug_tiny_forward_import_base_train_lm_head(handle, values)
            if code != 0: raise BackendError(f"LM-head checkpoint restore returned {code}")
        if mask & 2 and checkpoint.get("output"):
            values = floats(checkpoint["output"]); code = native.spaceslug_tiny_forward_import_base_train_output(handle, values)
            if code != 0: raise BackendError(f"output checkpoint restore returned {code}")
        if mask & 4:
            values = [floats(values) for values in checkpoint.get("qkv", [])]
            fn = getattr(native, "spaceslug_tiny_forward_import_base_train_qkv", None)
            if fn is None or len(values) != 3: raise BackendError("QKV checkpoint restore ABI unavailable")
            code = fn(handle, *values)
            if code != 0: raise BackendError(f"QKV checkpoint restore returned {code}")
        # AdamW is intentionally limited to LM-head/output; QKV remains SGD-only.
        for group, count in (("lm_head", 64 * 320), ("output", 64 * 64)):
            if mask & (1 if group == "lm_head" else 2) and all(f"{group}_{key}" in checkpoint for key in ("m", "v")):
                fn = getattr(native, f"spaceslug_tiny_forward_update_base_train_{group}_adamw_state", None)
                if fn is not None:
                    values = [floats(checkpoint[group]), floats(checkpoint[f"{group}_m"]), floats(checkpoint[f"{group}_v"])]
                    code = fn(handle, *values, int(checkpoint.get("adamw_step", 0)))
                    if code != 0: raise BackendError(f"{group} AdamW checkpoint restore returned {code}")

    def train_tiny_graph_lm_head_sgd(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], learning_rate: float) -> None:
        """Run one graph-owned Tiny LM-head SGD step when the runtime exports it."""
        if not tokens or len(tokens) != len(targets) or len(tokens) != len(masks) or len(tokens) > 128:
            raise ValueError("integrated Tiny LM-head SGD requires equal 1..128 token/target/mask values")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        fn = getattr(self._native(), "spaceslug_tiny_forward_train_lm_head_sgd", None)
        if fn is None:
            raise BackendError("integrated graph LM-head SGD ABI unavailable")
        import array
        arrays = [array.array("I", values) for values in (tokens, targets, masks)]
        pointers = [(ctypes.c_uint32 * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, len(tokens), learning_rate)
        if code != 0:
            raise BackendError(f"integrated graph LM-head SGD returned {code}")

    def _train_tiny_graph_group_sgd(self, group: str, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], learning_rate: float) -> None:
        if not tokens or len(tokens) != len(targets) or len(tokens) != len(masks) or len(tokens) > 128:
            raise ValueError(f"integrated Tiny {group} SGD requires equal 1..128 token/target/mask values")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        fn = getattr(self._native(), f"spaceslug_tiny_forward_train_{group}_sgd", None)
        if fn is None:
            raise BackendError(f"integrated graph {group} SGD ABI unavailable")
        import array
        arrays = [array.array("I", values) for values in (tokens, targets, masks)]
        pointers = [(ctypes.c_uint32 * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, len(tokens), learning_rate)
        if code != 0:
            raise BackendError(f"integrated graph {group} SGD returned {code}")

    def train_tiny_graph_output_sgd(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], learning_rate: float) -> None:
        """Run one graph-owned Tiny output-projection SGD step."""
        self._train_tiny_graph_group_sgd("output", handle, tokens, targets, masks, learning_rate)

    def train_tiny_graph_qkv_sgd(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], learning_rate: float) -> None:
        """Run one graph-owned Tiny combined-QKV SGD step."""
        self._train_tiny_graph_group_sgd("qkv", handle, tokens, targets, masks, learning_rate)

    def _train_tiny_graph_group_adamw(self, group: str, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0) -> None:
        if group not in {"lm_head", "output"}:
            raise ValueError("graph AdamW group must be lm_head or output")
        if not tokens or len(tokens) != len(targets) or len(tokens) != len(masks) or len(tokens) > 128:
            raise ValueError(f"integrated Tiny {group} AdamW requires equal 1..128 token/target/mask values")
        if learning_rate <= 0.0 or not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0 or epsilon <= 0.0 or weight_decay < 0.0:
            raise ValueError("invalid Tiny graph AdamW hyperparameters")
        fn = getattr(self._native(), f"spaceslug_tiny_forward_train_{group}_adamw", None)
        if fn is None:
            raise BackendError(f"integrated graph {group} AdamW ABI unavailable (return code -4)")
        import array
        arrays = [array.array("I", values) for values in (tokens, targets, masks)]
        pointers = [(ctypes.c_uint32 * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, len(tokens), learning_rate, beta1, beta2, epsilon, weight_decay)
        if code != 0:
            raise BackendError(f"integrated graph {group} AdamW returned {code}")

    def train_tiny_graph_qkv_adamw(self, handle: ctypes.c_void_p, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0) -> None:
        """Apply QKV AdamW to gradients already produced by graph backward."""
        if learning_rate <= 0.0 or not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0 or epsilon <= 0.0 or weight_decay < 0.0:
            raise ValueError("invalid Tiny graph QKV AdamW hyperparameters")
        fn = getattr(self._native(), "spaceslug_tiny_forward_train_qkv_adamw_from_gradients", None)
        if fn is None:
            raise BackendError("integrated graph QKV AdamW ABI unavailable (return code -4)")
        code = fn(handle, learning_rate, beta1, beta2, epsilon, weight_decay)
        if code != 0:
            raise BackendError(f"integrated graph QKV AdamW returned {code}")

    def train_tiny_graph_lm_head_adamw(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0) -> None:
        """Run graph-owned Tiny LM-head AdamW when all runtime symbols exist."""
        if not tokens or len(tokens) != len(targets) or len(tokens) != len(masks) or len(tokens) > 128:
            raise ValueError("integrated Tiny LM-head AdamW requires equal 1..128 token/target/mask values")
        if learning_rate <= 0.0 or not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0 or epsilon <= 0.0 or weight_decay < 0.0:
            raise ValueError("invalid Tiny LM-head AdamW hyperparameters")
        fn = getattr(self._native(), "spaceslug_tiny_forward_train_lm_head_adamw", None)
        if fn is None:
            raise BackendError("integrated graph LM-head AdamW ABI unavailable (return code -4)")
        import array
        arrays = [array.array("I", values) for values in (tokens, targets, masks)]
        pointers = [(ctypes.c_uint32 * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, len(tokens), learning_rate, beta1, beta2, epsilon, weight_decay)
        if code != 0:
            raise BackendError(f"integrated graph LM-head AdamW returned {code}")

    def train_tiny_graph_output_adamw(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], masks: list[int], learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0) -> None:
        """Run graph-owned Tiny output-projection AdamW when exported."""
        self._train_tiny_graph_group_adamw("output", handle, tokens, targets, masks, learning_rate, beta1, beta2, epsilon, weight_decay)

    def readback_tiny_graph_lm_head_adamw_state(self, handle: ctypes.c_void_p, parameter_count: int = 64 * 320) -> dict[str, Any]:
        fn = getattr(self._native(), "spaceslug_tiny_forward_readback_base_train_lm_head_adamw_state", None)
        if fn is None:
            raise BackendError("integrated graph LM-head AdamW state ABI unavailable (return code -4)")
        if parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        import array
        arrays = [array.array("f", [0.0] * parameter_count) for _ in range(3)]
        step = ctypes.c_uint64()
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, ctypes.byref(step))
        if code != 0:
            raise BackendError(f"integrated graph LM-head AdamW state readback returned {code}")
        return {"weight": arrays[0].tolist(), "m": arrays[1].tolist(), "v": arrays[2].tolist(), "step": step.value}

    def readback_tiny_graph_output_adamw_state(self, handle: ctypes.c_void_p, parameter_count: int = 64 * 64) -> dict[str, Any]:
        return self._readback_tiny_graph_group_adamw_state("output", handle, parameter_count)

    def _readback_tiny_graph_group_adamw_state(self, group: str, handle: ctypes.c_void_p, parameter_count: int) -> dict[str, Any]:
        fn = getattr(self._native(), f"spaceslug_tiny_forward_readback_base_train_{group}_adamw_state", None)
        if fn is None: raise BackendError(f"integrated graph {group} AdamW state ABI unavailable (return code -4)")
        if parameter_count <= 0: raise ValueError("parameter_count must be positive")
        import array
        arrays = [array.array("f", [0.0] * parameter_count) for _ in range(3)]
        step = ctypes.c_uint64()
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, ctypes.byref(step))
        if code != 0: raise BackendError(f"integrated graph {group} AdamW state readback returned {code}")
        return {"weight": arrays[0].tolist(), "m": arrays[1].tolist(), "v": arrays[2].tolist(), "step": step.value}

    def update_tiny_graph_output_adamw_state(self, handle: ctypes.c_void_p, state: dict[str, Any], parameter_count: int = 64 * 64) -> None:
        self._update_tiny_graph_group_adamw_state("output", handle, state, parameter_count)

    def _update_tiny_graph_group_adamw_state(self, group: str, handle: ctypes.c_void_p, state: dict[str, Any], parameter_count: int) -> None:
        fn = getattr(self._native(), f"spaceslug_tiny_forward_update_base_train_{group}_adamw_state", None)
        if fn is None: raise BackendError(f"integrated graph {group} AdamW state ABI unavailable (return code -4)")
        if parameter_count <= 0 or any(len(state.get(key, [])) != parameter_count for key in ("weight", "m", "v")):
            raise ValueError(f"invalid graph {group} AdamW state")
        import array
        arrays = [array.array("f", state[key]) for key in ("weight", "m", "v")]
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, int(state.get("step", 0)))
        if code != 0: raise BackendError(f"integrated graph {group} AdamW state update returned {code}")

    def update_tiny_graph_lm_head_adamw_state(self, handle: ctypes.c_void_p, state: dict[str, Any], parameter_count: int = 64 * 320) -> None:
        fn = getattr(self._native(), "spaceslug_tiny_forward_update_base_train_lm_head_adamw_state", None)
        if fn is None:
            raise BackendError("integrated graph LM-head AdamW state ABI unavailable (return code -4)")
        if parameter_count <= 0 or any(len(state.get(key, [])) != parameter_count for key in ("weight", "m", "v")):
            raise ValueError("invalid graph LM-head AdamW state")
        import array
        arrays = [array.array("f", state[key]) for key in ("weight", "m", "v")]
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, int(state.get("step", 0)))
        if code != 0:
            raise BackendError(f"integrated graph LM-head AdamW state update returned {code}")

    def begin_tiny_accumulation(self, handle: ctypes.c_void_p) -> None:
        code = self._native().spaceslug_tiny_forward_begin_lora_accumulation(handle)
        if code != 0: raise BackendError(f"begin Tiny accumulation returned {code}")

    def accumulate_tiny_backward(self, handle: ctypes.c_void_p, token: int, position: int, target: int, mask: int, _fn: Any | None = None) -> dict[str, Any]:
        import array
        outputs = [array.array("f", [0.0] * n) for n in (1, 320, 64, 64, 64, 64, 64, 64, 64)]
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in outputs]
        fn = _fn or self._native().spaceslug_tiny_forward_token_step_training_backward_accumulate
        code = fn(handle, token, position, target, mask, *pointers)
        if code != 0: raise BackendError(f"Tiny backward accumulation returned {code}")
        return {"loss": outputs[0].tolist(), "dlogits": outputs[1].tolist(), "dprojected": outputs[2].tolist(), "dquery": outputs[3].tolist(), "dkey": outputs[4].tolist(), "dvalue": outputs[5].tolist(), "dcontext": outputs[6].tolist(), "dstates": outputs[7].tolist()}

    def accumulate_tiny_windows(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], mask: list[int], window_length: int) -> list[float]:
        if not tokens or len(tokens) != len(targets) or len(tokens) != len(mask) or len(tokens) % window_length:
            raise ValueError("batched Tiny windows require equal, non-empty, rectangular inputs")
        import array
        fn = getattr(self._native(), "spaceslug_tiny_forward_token_windows_training_backward_accumulate", None)
        if fn is None:
            raise BackendError("batched Tiny window ABI unavailable")
        windows = len(tokens) // window_length
        token_values, target_values, mask_values = (array.array("I", values) for values in (tokens, targets, mask))
        losses = array.array("f", [0.0] * len(tokens))
        pointers = [(ctypes.c_uint32 * len(values)).from_buffer(values) for values in (token_values, target_values, mask_values)]
        loss_pointer = (ctypes.c_float * len(losses)).from_buffer(losses)
        fn.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint32)] * 3 + [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        fn.restype = ctypes.c_int
        code = fn(handle, *pointers, windows, window_length, loss_pointer)
        if code != 0: raise BackendError(f"Tiny batched backward accumulation returned {code}")
        return losses.tolist()

    def accumulate_tiny_adamw_windows(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], mask: list[int], window_length: int) -> list[float]:
        """Accumulate a rectangular AdamW batch with positions reset per window.

        The native ABI currently exposes AdamW accumulation one token at a time;
        keep the batching contract here so callers cannot accidentally use the
        position in the host batch as the model position.  A future native
        windows ABI can replace this implementation without changing trainer
        semantics.
        """
        if not tokens or len(tokens) != len(targets) or len(tokens) != len(mask) or len(tokens) % window_length:
            raise ValueError("batched Tiny AdamW windows require equal, non-empty rectangular inputs")
        losses: list[float] = []
        for start in range(0, len(tokens), window_length):
            for position in range(window_length):
                result = self.accumulate_tiny_adamw(
                    handle, tokens[start + position], position,
                    targets[start + position], mask[start + position])
                losses.append(result["loss"][0])
        return losses

    def finalize_tiny_sgd(self, handle: ctypes.c_void_p, learning_rate: float, normalizer: float) -> None:
        code = self._native().spaceslug_tiny_forward_finalize_lora_sgd(handle, learning_rate, normalizer)
        if code != 0: raise BackendError(f"finalize Tiny SGD returned {code}")

    def begin_tiny_adamw(self, handle: ctypes.c_void_p) -> None:
        fn = getattr(self._native(), "spaceslug_tiny_forward_begin_lora_adamw", None)
        if fn is None: raise BackendError("native Tiny AdamW ABI unavailable")
        code = fn(handle)
        if code != 0: raise BackendError(f"begin Tiny AdamW returned {code}")

    def accumulate_tiny_adamw(self, handle: ctypes.c_void_p, token: int, position: int, target: int, mask: int) -> dict[str, Any]:
        native = self._native()
        fn = getattr(native, "spaceslug_tiny_forward_accumulate_lora_adamw", None)
        if fn is None:
            fn = getattr(native, "spaceslug_tiny_forward_token_step_training_backward_accumulate", None)
        if fn is None: raise BackendError("native Tiny AdamW accumulation ABI unavailable")
        return self.accumulate_tiny_backward(handle, token, position, target, mask, _fn=fn)

    def finalize_tiny_adamw(self, handle: ctypes.c_void_p, learning_rate: float, beta1: float, beta2: float, epsilon: float, weight_decay: float, normalizer: float) -> None:
        fn = getattr(self._native(), "spaceslug_tiny_forward_finalize_lora_adamw", None)
        if fn is None: raise BackendError("native Tiny AdamW ABI unavailable")
        code = fn(handle, learning_rate, beta1, beta2, epsilon, weight_decay, normalizer)
        if code != 0: raise BackendError(f"finalize Tiny AdamW returned {code}")

    def readback_tiny_adamw_state(self, handle: ctypes.c_void_p, rank: int = 4) -> dict[str, Any]:
        fn = getattr(self._native(), "spaceslug_tiny_forward_readback_lora_adamw_state", None)
        if fn is None: raise BackendError("native Tiny AdamW ABI unavailable")
        import array
        size = 4 * 64 * rank + 4 * rank * 64
        arrays = [array.array("f", [0.0] * size) for _ in range(3)]
        step = ctypes.c_uint64()
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, ctypes.byref(step))
        if code != 0: raise BackendError(f"Tiny AdamW state readback returned {code}")
        return {"adapters": arrays[0].tolist(), "m": arrays[1].tolist(), "v": arrays[2].tolist(), "step": step.value}

    def restore_tiny_adamw_state(self, handle: ctypes.c_void_p, state: dict[str, Any], rank: int = 4) -> None:
        fn = getattr(self._native(), "spaceslug_tiny_forward_update_lora_adamw_state", None)
        if fn is None: raise BackendError("native Tiny AdamW ABI unavailable")
        import array
        size = 4 * 64 * rank + 4 * rank * 64
        if any(len(state.get(key, [])) != size for key in ("adapters", "m", "v")): raise ValueError("invalid Tiny AdamW state")
        arrays = [array.array("f", state[key]) for key in ("adapters", "m", "v")]
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in arrays]
        code = fn(handle, *pointers, int(state.get("step", 0)))
        if code != 0: raise BackendError(f"Tiny AdamW state restore returned {code}")

    def readback_tiny_adapters(self, handle: ctypes.c_void_p, rank: int = 4) -> list[list[float]]:
        import array
        sizes = [64 * rank, rank * 64] * 4
        outputs = [array.array("f", [0.0] * size) for size in sizes]
        pointers = [(ctypes.c_float * len(values)).from_buffer(values) for values in outputs]
        code = self._native().spaceslug_tiny_forward_readback_lora_adapters(handle, *pointers)
        if code != 0: raise BackendError(f"Tiny adapter readback returned {code}")
        return [values.tolist() for values in outputs]

    def update_tiny_adapters(self, handle: ctypes.c_void_p, values: list[list[float]], rank: int = 4) -> None:
        if len(values) != 8 or [len(v) for v in values] != [64 * rank, rank * 64] * 4:
            raise ValueError("persistent Tiny adapters require four A/B pairs")
        import array
        arrays = [array.array("f", v) for v in values]
        pointers = [(ctypes.c_float * len(v)).from_buffer(v) for v in arrays]
        code = self._native().spaceslug_tiny_forward_update_lora_adapters(handle, *pointers)
        if code != 0: raise BackendError(f"Tiny adapter restore returned {code}")

    def close_tiny_persistent(self, handle: ctypes.c_void_p) -> None:
        self._native().spaceslug_tiny_forward_destroy(handle)
        if hasattr(self, "_tiny_arrays"): self._tiny_arrays.pop(int(handle.value), None)

    def tiny_forward_capability(self) -> str:
        native = self._native()
        return native.spaceslug_tiny_forward_capability().decode("utf-8") if hasattr(native, "spaceslug_tiny_forward_capability") else "unsupported"

    def tiny_profiles(self) -> list[dict[str, int | str]]:
        native = self._native()
        if not hasattr(native, "spaceslug_tiny_profile_count"):
            return []

        class Descriptor(ctypes.Structure):
            _fields_ = [("name", ctypes.c_char_p), ("hidden", ctypes.c_uint32), ("vocab", ctypes.c_uint32),
                        ("padded_vocab", ctypes.c_uint32), ("token_capacity", ctypes.c_uint32), ("lora_rank", ctypes.c_uint32)]

        profiles = []
        for index in range(native.spaceslug_tiny_profile_count()):
            descriptor = Descriptor()
            status = native.spaceslug_tiny_profile_query(index, ctypes.byref(descriptor))
            if status != 0:
                raise BackendError(f"Tiny profile query returned {status}")
            profiles.append({"name": descriptor.name.decode("utf-8"), "hidden": descriptor.hidden,
                             "vocab": descriptor.vocab, "padded_vocab": descriptor.padded_vocab,
                             "token_capacity": descriptor.token_capacity, "lora_rank": descriptor.lora_rank})
        return profiles

    def validate_tiny_profile(self, hidden: int, vocab: int, padded_vocab: int, token_capacity: int, rank: int) -> int:
        native = self._native()
        if not hasattr(native, "spaceslug_tiny_profile_validate"):
            return 1
        return int(native.spaceslug_tiny_profile_validate(hidden, vocab, padded_vocab, token_capacity, rank))

    def execute_tiny_persistent_forward(self, handle: ctypes.c_void_p, tokens: list[int], final_only: bool = False) -> ExecutionResult:
        native = self._native()
        if not hasattr(native, "spaceslug_tiny_forward"): return ExecutionResult("not-run", "tiny_forward_persistent", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "persistent Tiny forward ABI unavailable"}, {})
        import array
        tt, out = array.array("I", tokens), array.array("f", [0.0] * ((1 if final_only else len(tokens)) * 259))
        code = native.spaceslug_tiny_forward(handle, (ctypes.c_uint32 * len(tt)).from_buffer(tt), len(tt), (ctypes.c_float * len(out)).from_buffer(out), int(final_only))
        if code != 0: raise BackendError(f"persistent Tiny forward returned {code}")
        return ExecutionResult("ok", "tiny_forward_persistent", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "persistent-forward", "gpu_execution": True, "device_resident": True}, {"logits": out.tolist()})

    def execute_tiny_fixed_retained_forward(self, handle: ctypes.c_void_p, tokens: list[int]) -> ExecutionResult:
        """Run the optional retained native forward with exactly 128 tokens."""
        native = self._native()
        if not hasattr(native, "spaceslug_tiny_forward_fixed_retained"):
            return ExecutionResult("not-run", "tiny_forward_fixed_retained", "vulkan-radv", self.runtime_revision, self.capabilities().device, False,
                                  {"parity": "not-run", "reason": "fixed retained Tiny forward ABI unavailable"}, {})
        if len(tokens) != 128:
            raise ValueError("fixed retained Tiny forward requires exactly 128 tokens")
        import array
        tt, out = array.array("I", tokens), array.array("f", [0.0] * (128 * 259))
        code = native.spaceslug_tiny_forward_fixed_retained(handle, (ctypes.c_uint32 * 128).from_buffer(tt), (ctypes.c_float * len(out)).from_buffer(out))
        if code != 0:
            raise BackendError(f"fixed retained Tiny forward returned {code}")
        return ExecutionResult("ok", "tiny_forward_fixed_retained", "vulkan-radv", self.runtime_revision, self.capabilities().device, False,
                              {"parity": "persistent-forward", "gpu_execution": True, "device_resident": True, "fixed_tokens": 128}, {"logits": out.tolist()})

    def execute_tiny_fixed_loss_metrics(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], mask: list[int]) -> ExecutionResult:
        """Read bounded GPU scalar loss/count metrics for exactly 128 rows."""
        native = self._native()
        operation = "tiny_forward_loss_fixed_metrics"
        fn = getattr(native, "spaceslug_tiny_forward_loss_fixed_metrics", None)
        if fn is None:
            return ExecutionResult("not-run", operation, "vulkan-radv", self.runtime_revision, self.capabilities().device, False,
                                   {"parity": "not-run", "reason": "fixed scalar metrics ABI unavailable"}, {})
        if len(tokens) != 128 or len(targets) != 128 or len(mask) != 128:
            raise ValueError("fixed Tiny scalar metrics requires exactly 128 tokens, targets, and mask values")
        import array
        values = [array.array("I", item) for item in (tokens, targets, mask)]
        loss, count = ctypes.c_float(), ctypes.c_uint32()
        pointers = [(ctypes.c_uint32 * 128).from_buffer(item) for item in values]
        code = fn(handle, *pointers, ctypes.byref(loss), ctypes.byref(count))
        if code != 0:
            raise BackendError(f"fixed Tiny scalar metrics returned {code}")
        return ExecutionResult("ok", operation, "vulkan-radv", self.runtime_revision, self.capabilities().device, False,
                               {"parity": "persistent-forward-loss-metrics", "gpu_execution": True, "device_resident": True,
                                "fixed_tokens": 128, "fixed_targets": 128, "fixed_mask": 128},
                               {"loss": loss.value, "count": count.value})

    def execute_tiny_fixed_retained_loss(self, handle: ctypes.c_void_p, tokens: list[int], targets: list[int], mask: list[int]) -> ExecutionResult:
        """Run the optional retained forward+masked-loss ABI with exactly 128 rows."""
        native = self._native()
        operation = "tiny_forward_loss_fixed_retained"
        if not hasattr(native, "spaceslug_tiny_forward_loss_fixed_retained"):
            return ExecutionResult("not-run", operation, "vulkan-radv", self.runtime_revision, self.capabilities().device, False,
                                   {"parity": "not-run", "reason": "fixed retained Tiny forward-loss ABI unavailable"}, {})
        if len(tokens) != 128 or len(targets) != 128 or len(mask) != 128:
            raise ValueError("fixed retained Tiny forward-loss requires exactly 128 tokens, targets, and mask values")
        import array
        tt, yy, mm = (array.array("I", values) for values in (tokens, targets, mask))
        logits = array.array("f", [0.0] * (128 * 259))
        row_losses = array.array("f", [0.0] * 128)
        code = native.spaceslug_tiny_forward_loss_fixed_retained(
            handle, (ctypes.c_uint32 * 128).from_buffer(tt), (ctypes.c_uint32 * 128).from_buffer(yy),
            (ctypes.c_uint32 * 128).from_buffer(mm), (ctypes.c_float * len(logits)).from_buffer(logits),
            (ctypes.c_float * 128).from_buffer(row_losses))
        if code != 0:
            raise BackendError(f"fixed retained Tiny forward-loss returned {code}")
        return ExecutionResult("ok", operation, "vulkan-radv", self.runtime_revision, self.capabilities().device, False,
                               {"parity": "persistent-forward-loss", "gpu_execution": True, "device_resident": True,
                                "fixed_tokens": 128, "fixed_targets": 128, "fixed_mask": 128},
                               {"logits": logits.tolist(), "row_losses": row_losses.tolist()})

    def open_lora_session(self, a: list[float], b: list[float], rank: int, learning_rate: float, rows: int) -> ctypes.c_void_p:
        native = self._native()
        if not hasattr(native, "spaceslug_lora_session_create"):
            raise BackendError("persistent LoRA session ABI unavailable")
        import array
        aa, bb = array.array("f", a), array.array("f", b)
        handle = ctypes.c_void_p()
        code = native.spaceslug_lora_session_create(rows, rank, learning_rate, (ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb), ctypes.byref(handle))
        if code != 0 or not handle.value: raise BackendError(f"persistent session create returned {code}")
        return handle

    def step_lora_session(self, handle: ctypes.c_void_p, x: list[float], dy: list[float], rows: int) -> list[float]:
        native = self._native()
        import array
        xx, dd, yy = array.array("f", x), array.array("f", dy), array.array("f", [0.0] * (rows * 64))
        code = native.spaceslug_lora_session_step(handle, (ctypes.c_float * len(xx)).from_buffer(xx), (ctypes.c_float * len(dd)).from_buffer(dd), (ctypes.c_float * len(yy)).from_buffer(yy))
        if code != 0: raise BackendError(f"persistent session step returned {code}")
        return yy.tolist()

    def readback_lora_session(self, handle: ctypes.c_void_p, rank: int) -> tuple[list[float], list[float]]:
        native = self._native()
        import array
        aa, bb = array.array("f", [0.0] * (64 * rank)), array.array("f", [0.0] * (rank * 64))
        code = native.spaceslug_lora_session_readback(handle, (ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb))
        if code != 0: raise BackendError(f"persistent session readback returned {code}")
        return aa.tolist(), bb.tolist()

    def close_lora_session(self, handle: ctypes.c_void_p) -> None:
        self._native().spaceslug_lora_session_destroy(handle)

    def execute_lora_session_step(self, x: list[float], dy: list[float], a: list[float], b: list[float], rank: int, learning_rate: float) -> ExecutionResult:
        if len(x) == 0 or len(x) % 64 or len(dy) != len(x) or len(a) != 64 * rank or len(b) != rank * 64 or rank < 1 or rank > 8:
            return ExecutionResult("not-run", "lora_session_step", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "persistent session requires X/dY [M,64] and A/B factors"}, {})
        native = self._native()
        if not hasattr(native, "spaceslug_lora_session_create"):
            return ExecutionResult("not-run", "lora_session_step", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "not-run", "reason": "persistent session ABI unavailable"}, {})
        import array, ctypes
        aa, bb, xx, dd, yy = (array.array("f", values) for values in (a, b, x, dy, [0.0] * len(x)))
        handle = ctypes.c_void_p()
        code = native.spaceslug_lora_session_create(len(x) // 64, rank, learning_rate, (ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb), ctypes.byref(handle))
        if code != 0: raise BackendError(f"persistent session create returned {code}")
        try:
            code = native.spaceslug_lora_session_step(handle, (ctypes.c_float * len(xx)).from_buffer(xx), (ctypes.c_float * len(dd)).from_buffer(dd), (ctypes.c_float * len(yy)).from_buffer(yy))
            if code != 0: raise BackendError(f"persistent session step returned {code}")
            code = native.spaceslug_lora_session_readback(handle, (ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb))
            if code != 0: raise BackendError(f"persistent session readback returned {code}")
        finally:
            native.spaceslug_lora_session_destroy(handle)
        return ExecutionResult("ok", "lora_session_step", "vulkan-radv", self.runtime_revision, self.capabilities().device, False, {"parity": "persistent-session", "gpu_execution": True, "device_resident": True}, {"y": yy.tolist(), "a": aa.tolist(), "b": bb.tolist()})

    def _native_multi_sgd(self, a: list[float], b: list[float], da: list[float], db: list[float], learning_rate: float, rank: int) -> tuple[list[float], list[float]]:
        import array, ctypes, os
        aa, bb, dda, ddb = (array.array("f", values) for values in (a, b, da, db))
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_lora_sgd_multi((ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb), (ctypes.c_float * len(dda)).from_buffer(dda), (ctypes.c_float * len(ddb)).from_buffer(ddb), learning_rate, rank)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native multi LoRA SGD returned {code}")
        return aa.tolist(), bb.tolist()

    def _native_multi_gradients(self, x: list[float], dy: list[float], a: list[float], b: list[float], rows: int, rank: int, target: int) -> tuple[list[float], list[float]]:
        import array, ctypes, os
        xx, dd, aa, bb = (array.array("f", values) for values in (x, dy, a, b))
        da = (ctypes.c_float * (4 * 64 * rank))()
        db = (ctypes.c_float * (4 * rank * 64))()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_lora_gradients_multi((ctypes.c_float * len(xx)).from_buffer(xx), (ctypes.c_float * len(dd)).from_buffer(dd), (ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb), da, db, rows, 64, rank, target)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native multi LoRA gradients returned {code}")
        return list(da), list(db)

    def _native_attention_backward(self, q: list[float], k: list[float], v: list[float], d_output: list[float], tokens: int, mode: int) -> list[float]:
        import array, ctypes, os
        qq, kk, vv, dd = (array.array("f", values) for values in (q, k, v, d_output))
        result = (ctypes.c_float * (tokens * 64))()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_attention_causal_backward((ctypes.c_float * len(qq)).from_buffer(qq), (ctypes.c_float * len(kk)).from_buffer(kk), (ctypes.c_float * len(vv)).from_buffer(vv), (ctypes.c_float * len(dd)).from_buffer(dd), result, tokens, 64, mode)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native attention backward returned {code}")
        return list(result)

    def _native_causal_loss(self, logits: list[float], targets: list[int], mask: list[int], vocab: int) -> tuple[list[float], list[float]]:
        import array, ctypes, os
        rows = len(targets)
        ll, tt, mm = array.array("f", logits), array.array("I", targets), array.array("I", mask)
        gradients = (ctypes.c_float * (rows * vocab))()
        losses = (ctypes.c_float * rows)()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_causal_loss((ctypes.c_float * len(ll)).from_buffer(ll), (ctypes.c_uint32 * len(tt)).from_buffer(tt), (ctypes.c_uint32 * len(mm)).from_buffer(mm), gradients, losses, rows, vocab)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native causal loss returned {code}")
        return list(losses), list(gradients)

    def _native_lora_delta_values(self, x: list[float], a: list[float], b: list[float], m: int, rank: int) -> list[float]:
        import array, ctypes, os
        xx, aa, bb = (array.array("f", values) for values in (x, a, b))
        yy = (ctypes.c_float * (m * 64))()
        old_icd = os.environ.get("VK_ICD_FILENAMES")
        if self.software_vulkan: os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.json"
        try:
            code = self._library.spaceslug_lora_delta((ctypes.c_float * len(xx)).from_buffer(xx), (ctypes.c_float * len(aa)).from_buffer(aa), (ctypes.c_float * len(bb)).from_buffer(bb), yy, m, rank)
        finally:
            if old_icd is None: os.environ.pop("VK_ICD_FILENAMES", None)
            else: os.environ["VK_ICD_FILENAMES"] = old_icd
        if code != 0: raise BackendError(f"native LoRA delta returned {code}")
        return list(yy)

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
